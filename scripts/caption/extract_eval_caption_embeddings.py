#!/usr/bin/env python3
"""Pre-extract Qwen3-Embedding-8B + CLIP-L text embeddings for v2 eval captions.

Why
---
Caption-conditioned v2 models (caption_local / caption_global / *_phase2) were
trained with ``LoadPreExtractedTextEmbedding`` — captions are encoded offline
into ``.pt`` files and loaded at train time. At eval time the bundle has no
``text_encoder`` config, so ``bundle.encode_text(...)`` raises RuntimeError
which was silently swallowed by the driver — captions never reached the model.

This script fixes that by pre-extracting per-caption embeddings for all eval
datalists. Output goes to ``data/eval/m2m_v2/caption_embeddings/cache.pt``
keyed by caption text. The driver then looks up each caption at eval time and
fills ``batch['text_vec_raw'] / text_ctxt_raw / text_ctxt_raw_length`` directly.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_eval_caption_embeddings.py
    CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_eval_caption_embeddings.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from hftrainer.models.motion.hymotion_m2m.network.text_encoder import (  # noqa: E402
    HYTextModel,
)

DATALIST_DIR = PROJECT_ROOT / 'data' / 'eval' / 'm2m_v2'
OUT_DIR = DATALIST_DIR / 'caption_embeddings'
OUT_FILE = OUT_DIR / 'cache.pt'

# Fields treated as caption-like
CAPTION_FIELDS = ('caption', 'caption_en', 'text_caption', 'text',
                  'text_a', 'text_b')


def _gather_unique_captions(prefer_rewritten: bool = True) -> List[str]:
    """Collect every caption that might be seen at eval time.

    Historically this only collected the ``_rewritten.json`` variants, which
    meant the original-caption datalists (``eval_e1_t2m.json`` etc.) had no
    cache entries — and since the driver defaults to the non-rewritten files,
    every caption lookup missed and every caption model silently fell back
    to unconditioned inference. This caused the visible distortion attributed
    to "caption model quality". (2026-04-20)

    We now collect captions from BOTH variants unconditionally — duplicates
    are fine because the set dedups them.
    """
    unique = set()
    # Cover EVERY eval datalist, including non eval_e*/eval_h3d* names such as
    # ``eval_motionfix_instruction.json``. A too-narrow glob here previously
    # excluded MotionFix instructions from the cache, so eval-time caption
    # lookups missed and the editfix bundle (no inference text_encoder) silently
    # fell back to UNCONDITIONED inference — making instruction editing a no-op
    # (root-caused 2026-06-22). One broad glob + set dedup avoids re-introducing
    # this class of bug whenever a new datalist is added.
    for pattern in ('eval_*.json',):
        for f in sorted(DATALIST_DIR.glob(pattern)):
            try:
                obj = json.load(open(f))
            except Exception:
                continue
            # Datalists are either {'data_list': [...]} or a bare list.
            if isinstance(obj, dict):
                items = obj.get('data_list', obj.get('data', []))
            elif isinstance(obj, list):
                items = obj
            else:
                items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                for k in CAPTION_FIELDS:
                    v = it.get(k, '')
                    if isinstance(v, str) and v.strip():
                        unique.add(v.strip())
    return sorted(unique)


def _load_existing_cache(force: bool) -> Dict[str, dict]:
    if force or not OUT_FILE.is_file():
        return {}
    try:
        data = torch.load(str(OUT_FILE), map_location='cpu', weights_only=False)
        cache = data.get('cache', {}) if isinstance(data, dict) else {}
        if not isinstance(cache, dict):
            return {}
        return cache
    except Exception as e:
        print(f'  warning: could not read existing cache: {e}')
        return {}


def _save_cache(cache: Dict[str, dict], meta: dict, out_file: Path = OUT_FILE):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix('.pt.tmp')
    torch.save({'cache': cache, 'meta': meta}, str(tmp))
    os.replace(str(tmp), str(out_file))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true',
                   help='Ignore existing cache and re-extract everything.')
    p.add_argument('--batch-size', type=int, default=4,
                   help='Texts per encoder call.')
    p.add_argument('--max-length-llm', type=int, default=512,
                   help='Max LLM token length.')
    # IMPORTANT: the deployed caption_* / *_editfix bundles were trained with
    # llm_type="qwen3" (HY-Motion-1.0-Lite text encoder, checkpoints/Qwen3-8B).
    # The existing cache.pt meta confirms llm_type=qwen3. Extracting with
    # qwen3_embedding here would produce embeddings that DO NOT match training
    # -> silent conditioning corruption. Default is therefore qwen3.
    p.add_argument('--llm-type', type=str, default='qwen3',
                   choices=['qwen3', 'qwen3_embedding'],
                   help='Text encoder LLM type (MUST match training; qwen3).')
    p.add_argument('--num-shards', type=int, default=1,
                   help='Split the TODO captions into N shards (multi-GPU).')
    p.add_argument('--shard-index', type=int, default=0,
                   help='Which shard this process handles (0-based).')
    p.add_argument('--out-file', type=str, default=str(OUT_FILE),
                   help='Output cache .pt path (per-shard for sharded runs).')
    p.add_argument('--device', type=str, default=None,
                   help="Override device, e.g. 'cpu' or 'cuda'.")
    p.add_argument('--device-map', type=str, default=None,
                   help="If 'auto', dispatch the 8B LLM across GPU+CPU via "
                        "accelerate (fits Qwen3-8B fp16 on a single 16GB V100 "
                        "by offloading a couple of layers to CPU). Avoids the "
                        "Taiji multi-GPU NVML NVLink crash entirely.")
    p.add_argument('--fp16', action='store_true',
                   help='Load encoders in float16 (recommended for GPU).')
    p.add_argument('--gpu-mem-gib', type=float, default=13.5,
                   help='Per-GPU memory budget for --device-map auto.')
    args = p.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}  llm_type={args.llm_type}')

    print('Gathering unique captions across all rewritten datalists...')
    captions = _gather_unique_captions()
    print(f'  {len(captions)} unique captions')

    # Decide which existing cache to merge against. Sharded runs write to a
    # dedicated --out-file, but still skip captions already in the MAIN cache.
    main_cache = _load_existing_cache(args.force)
    out_path = Path(args.out_file)
    if out_path == OUT_FILE:
        cache = main_cache
    else:
        cache = {} if args.force or not out_path.is_file() else (
            torch.load(str(out_path), map_location='cpu',
                       weights_only=False).get('cache', {}))
    if main_cache:
        print(f'  main cache has {len(main_cache)} entries')

    todo = [c for c in captions if c not in main_cache and c not in cache]
    # Shard the TODO list deterministically across processes.
    if args.num_shards > 1:
        todo = todo[args.shard_index::args.num_shards]
        print(f'  shard {args.shard_index}/{args.num_shards}: '
              f'{len(todo)} captions for this process')
    if not todo:
        print('  nothing to do!')
        return
    print(f'  {len(todo)} new captions to encode')

    print(f'Loading HYTextModel (llm_type={args.llm_type} + CLIP-L)...')
    t0 = time.time()
    dtype = torch.float16 if args.fp16 else None
    model = HYTextModel(
        llm_type=args.llm_type,
        max_length_llm=args.max_length_llm,
        sentence_emb_type='clipl',
        max_length_sentence_emb=77,
        torch_dtype=dtype,
    )
    model = model.eval()
    if args.device_map == 'auto':
        # Dispatch the big LLM across GPU + CPU; keep the small CLIP encoder on
        # GPU so get_module_device(self) (= first param) reports cuda and inputs
        # land on the GPU. accelerate hooks shuttle activations across devices.
        from accelerate import dispatch_model, infer_auto_device_map
        model.sentence_emb_text_encoder = model.sentence_emb_text_encoder.to(device)
        llm = model.llm_text_encoder
        no_split = list(getattr(llm, '_no_split_modules', None) or []) or ['Qwen3DecoderLayer']
        dmap = infer_auto_device_map(
            llm,
            max_memory={0: f'{args.gpu_mem_gib}GiB', 'cpu': '60GiB'},
            dtype=(dtype or torch.float16),
            no_split_module_classes=no_split,
        )
        model.llm_text_encoder = dispatch_model(llm, device_map=dmap)
        on_cpu = sum(1 for v in dmap.values() if v == 'cpu' or v == 'disk')
        print(f'  device_map=auto: {len(dmap)} blocks, {on_cpu} on cpu/disk')
    else:
        model = model.to(device)
    print(f'  loaded in {time.time() - t0:.1f}s')

    t0 = time.time()
    done = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        with torch.no_grad():
            vtxt, ctxt, ctxt_len = model.encode(batch)
        # vtxt: (B, 1, vtxt_dim); ctxt: (B, seq, ctxt_dim); ctxt_len: (B,)
        vtxt = vtxt.cpu()
        ctxt = ctxt.cpu()
        ctxt_len = ctxt_len.cpu()
        for j, text in enumerate(batch):
            # Trim ctxt to its actual length for smaller storage
            L = int(ctxt_len[j].item())
            cache[text] = {
                'text_vec_raw': vtxt[j:j + 1].clone(),   # (1, vtxt_dim)
                'text_ctxt_raw': ctxt[j:j + 1, :L].clone(),  # (1, L, ctxt_dim)
                'text_ctxt_raw_length': ctxt_len[j:j + 1].clone(),  # (1,)
            }
        done += len(batch)
        if done % 40 == 0 or done == len(todo):
            rate = done / max(1e-3, time.time() - t0)
            eta = (len(todo) - done) / max(1e-3, rate)
            print(f'    [{done}/{len(todo)}] {rate:.1f}/s ETA {eta:.0f}s')
            _meta = {
                'model': f'HYTextModel ({args.llm_type} + clipl)',
                'llm_type': args.llm_type,
                'max_length_llm': args.max_length_llm,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_entries': len(cache),
            }
            _save_cache(cache, meta=_meta, out_file=out_path)
    _save_cache(cache, meta={
        'model': f'HYTextModel ({args.llm_type} + clipl)',
        'llm_type': args.llm_type,
        'max_length_llm': args.max_length_llm,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_entries': len(cache),
    }, out_file=out_path)
    print(f'\nDone. {len(cache)} entries at {out_path}')


if __name__ == '__main__':
    main()

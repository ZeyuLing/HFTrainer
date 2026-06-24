#!/usr/bin/env python3
"""Generate HYMotion-M2M (\\ours) text-to-motion predictions on the HumanML3D
test split and export them as MotionStreamer 272-dim features for the
(validated) TMR evaluator.

    HYMotion-M2M (HunyuanMotionMMDiT, 198-dim, 30 fps; text-only / no motion cond)
      -> denormalized motion_198 -> first 135 dims (transl(3) + 22 x rot6d(132))
      -> motion135_to_272            (canonical SMPL-X-272 skeleton FK -> 272)
      -> save <out>/<id>.npy         (272, 30 fps)

This mirrors ``scripts/eval/gen_hylite_272.py`` exactly (same data iteration,
sharding, GT lengths, output format) so the \\ours rows in the NIPS2026
``tab:t2m`` are directly comparable to the HY-Motion-T2M-1.0-Lite row.  The only
difference is the model: a ``HyMotionM2MBundle`` driven through the established
T2M-from-M2M recipe (zero source motion, src_mask = length mask) used in
``scripts/eval/eval_m2m_v2_t2m.py``.

The <id>.npy files plug straight into the native validated evaluator:

    python3 ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py \
        --pred_dir <out> \
        --data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --evaluator_ckpt .../Evaluator_272/epoch=99.ckpt \
        --out_json <out>/../eval.json --n_repeats 20

ids + GT lengths come from the 272 GT set (30 fps), so gen and GT share ids,
texts and lengths (the evaluator truncates each to (len//4)*4).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

T_PAD = 360  # the context length the M2M model was trained with (clip_len)
MOTION_DIM = 198


def _read_first_caption(txt: Path):
    """First full-clip (f_tag==to_tag==0) caption from texts/<id>.txt."""
    if not txt.exists():
        return None
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap, ftag, ttag = parts[0], parts[2], parts[3]
        try:
            fv, tv = float(ftag), float(ttag)
        except ValueError:
            continue
        if (fv == 0.0 or fv != fv) and (tv == 0.0 or tv != tv):
            return cap
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="M2M training config (.py).")
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint dir/file (checkpoint-epoch_XXXX/).")
    p.add_argument("--data-root",
                   default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272",
                   help="272 GT root (motion_data/, texts/, split/test.txt).")
    p.add_argument("--out", required=True, help="Output dir for <id>.npy (272).")
    p.add_argument("--m135-dir", default=None,
                   help="Also dump raw motion_135 <id>.npy here. "
                        "Defaults to '<out>/../m135'.")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=5.0,
                   help="text_guidance_scale for caption conditioning.")
    p.add_argument("--rotation-space", choices=["local", "global"], default="local")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--min-len", type=int, default=60)
    p.add_argument("--max-len", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--text-cache", default="",
                   help="Path to a precomputed caption-feature cache (.pt). "
                        "Stage-2 generation reads features from here and SKIPS "
                        "loading the qwen3+clip text encoder, so the M2M model "
                        "fits on a 16GB GPU. Build it first with --cache-only.")
    p.add_argument("--cache-only", action="store_true",
                   help="Stage-1: only materialise the text encoder, extract "
                        "caption features for the full split into --text-cache, "
                        "then exit (no M2M bundle weights, no generation).")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from copy import deepcopy
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
    from hftrainer.motion.representation.motion272 import motion135_to_272

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    data_root = Path(args.data_root)
    motion_dir = data_root / "motion_data"
    texts_dir = data_root / "texts"
    test_ids = [s.strip() for s in (data_root / "split" / "test.txt").read_text().splitlines() if s.strip()]

    jobs = []  # (id, caption, gen_len30)
    for sid in test_ids:
        m_file = motion_dir / f"{sid}.npy"
        if not m_file.exists():
            continue
        cap = _read_first_caption(texts_dir / f"{sid}.txt")
        if not cap:
            continue
        L = int(np.load(str(m_file), mmap_mode="r").shape[0])
        if L < args.min_len or L >= args.max_len:
            continue
        jobs.append((sid, cap, L))
    if args.max_samples:
        jobs = jobs[:args.max_samples]
    # Stage-1 cache extraction covers the FULL (un-sharded) split so every
    # generation shard can reuse one cache file.
    if not args.cache_only and args.num_shards > 1:
        jobs = jobs[args.shard_index::args.num_shards]
    print(f"[+] {len(jobs)} gen jobs (shard {args.shard_index}/{args.num_shards}, "
          f"cfg={args.cfg_scale}, rot={args.rotation_space})", flush=True)

    # ---- Stage-1: extract caption features into a cache, then exit. ----
    # Qwen3-8B fp16 (~14.5GB) won't fit a 16GB V100 alongside CUDA overhead, so
    # the LLM is dispatched across GPU+CPU via accelerate (a few decoder layers
    # offloaded to CPU) — the same trick extract_eval_caption_embeddings.py uses
    # to encode on a single 16GB card. GPU runs the bulk -> far faster than pure
    # CPU. bundle stays on CPU so encode_text returns CPU tensors for caching.
    if args.cache_only:
        if not args.text_cache:
            raise SystemExit("--cache-only requires --text-cache <path>")
        from accelerate import dispatch_model, infer_auto_device_map
        cfg = Config.fromfile(args.config)
        model_cfg = dict(cfg.model)
        model_cfg["text_encoder"] = dict()
        bundle = MODEL_BUNDLES.build(model_cfg)
        bundle.eval()
        enc = HYTextModel(
            llm_type="qwen3", max_length_llm=128,
            sentence_emb_type="clipl", max_length_sentence_emb=77,
            enable_llm_padding=True, torch_dtype=torch.float16,
        ).eval()
        # Small CLIP encoder on GPU so get_module_device reports cuda; big LLM
        # auto-dispatched (GPU budget 13GiB, rest to CPU).
        enc.sentence_emb_text_encoder = enc.sentence_emb_text_encoder.to(device)
        _llm = enc.llm_text_encoder
        _no_split = list(getattr(_llm, "_no_split_modules", None) or []) \
            or ["Qwen3DecoderLayer"]
        _dmap = infer_auto_device_map(
            _llm, max_memory={0: "13.5GiB", "cpu": "40GiB"},
            dtype=torch.float16, no_split_module_classes=_no_split)
        enc.llm_text_encoder = dispatch_model(_llm, device_map=_dmap)
        _on_cpu = sum(1 for v in _dmap.values() if v in ("cpu", "disk"))
        print(f"[cache] qwen3 device_map: {len(_dmap)} blocks, "
              f"{_on_cpu} on cpu/disk", flush=True)
        bundle._text_encoder = enc
        cpath = Path(args.text_cache)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        # Resume: keep already-extracted entries so a crash/restart is cheap.
        cache = {}
        if cpath.exists():
            try:
                cache = torch.load(str(cpath), map_location="cpu")
                print(f"[cache] resume: {len(cache)} entries already cached",
                      flush=True)
            except Exception:
                cache = {}
        import time as _time
        _t0 = _time.time()
        for n, (sid, cap, L) in enumerate(jobs):
            if sid in cache:
                continue
            with torch.no_grad():
                feats = bundle.encode_text([cap])
            cache[sid] = {k: v.detach().cpu() for k, v in feats.items()}
            # Incremental checkpoint every 200 captions (atomic via .tmp swap).
            if (n + 1) % 200 == 0:
                tmp = cpath.with_suffix(".pt.tmp")
                torch.save(cache, str(tmp))
                os.replace(str(tmp), str(cpath))
                rate = (n + 1) / max(1e-3, _time.time() - _t0)
                eta = (len(jobs) - n - 1) / max(1e-3, rate)
                print(f"  [cache {n+1}/{len(jobs)}] {rate:.2f}/s "
                      f"ETA {eta/60:.0f}min saved", flush=True)
        tmp = cpath.with_suffix(".pt.tmp")
        torch.save(cache, str(tmp))
        os.replace(str(tmp), str(cpath))
        print(f"[cache] saved {len(cache)} caption feats -> {cpath}", flush=True)
        return

    # ---- build + load M2M bundle ----
    cfg = Config.fromfile(args.config)
    model_cfg = dict(cfg.model)
    # text_encoder is pre-built on GPU below; clear the lazy placeholder.
    model_cfg["text_encoder"] = dict()
    bundle = MODEL_BUNDLES.build(model_cfg)

    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    print(f"[+] loading {ckpt_path}", flush=True)
    sd = load_checkpoint(ckpt_path, map_location="cpu")
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval().to(device)

    # Pre-build the text encoder ON device (qwen3 ctxt 4096 + clip-l vtxt 768).
    # enable_llm_padding=True so a batch of captions with different token lengths
    # can be stacked (numerically irrelevant: motion model attends via ctxt_len).
    use_cache = bool(args.text_cache) and os.path.exists(args.text_cache)
    text_cache = None
    if use_cache:
        print(f"[+] loading caption-feature cache {args.text_cache} "
              f"(skip text encoder, frees ~9GB for M2M) ...", flush=True)
        text_cache = torch.load(args.text_cache, map_location="cpu")
    else:
        print("[+] building text encoder (qwen3+clipl, fp16) on GPU ...", flush=True)
        bundle._text_encoder = HYTextModel(
            llm_type="qwen3",
            max_length_llm=128,
            sentence_emb_type="clipl",
            max_length_sentence_emb=77,
            enable_llm_padding=True,
            torch_dtype=torch.float16,
        ).eval().to(device)

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.cfg_scale,
        replacement_guidance="none",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m135_dir = Path(args.m135_dir) if args.m135_dir else out.parent / "m135"
    m135_dir.mkdir(parents=True, exist_ok=True)

    done = failed = skipped = 0
    for sid, cap, L in jobs:
        if args.skip_existing and (out / f"{sid}.npy").exists():
            skipped += 1
            continue
        T = min(int(L), T_PAD)
        try:
            src_motion = torch.zeros(1, T_PAD, MOTION_DIM, device=device)
            src_mask = torch.zeros(1, T_PAD, MOTION_DIM, device=device)
            src_mask[:, :T, :] = 1.0
            batch = {
                "src_motion": src_motion,
                "src_mask": src_mask,
                "src_length": [T],
                "tgt_length": [T],
            }
            if use_cache:
                cf = text_cache.get(sid)
                if cf is None:
                    raise KeyError(f"caption feats for {sid} missing from cache")
                feats = {k: v.to(device) for k, v in cf.items()}
            else:
                feats = bundle.encode_text([cap])
            batch["text_vec_raw"] = feats["text_vec_raw"]
            batch["text_ctxt_raw"] = feats["text_ctxt_raw"]
            batch["text_ctxt_raw_length"] = feats["text_ctxt_raw_length"]

            with torch.no_grad():
                output = pipeline(batch)
            sampled = output["latent"]
            denorm = bundle.denormalize_motion(sampled)[0].cpu()[:T]
            m135 = denorm[:, :135].float().numpy().astype(np.float32)
            np.save(str(m135_dir / f"{sid}.npy"), m135)
            m272 = motion135_to_272(m135, rotation_space=args.rotation_space)
            np.save(str(out / f"{sid}.npy"), m272.astype(np.float32))
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [!] {sid}: {type(e).__name__}: {e}", flush=True)
            continue
        if done % 50 == 0:
            print(f"  [{done}/{len(jobs)}] (failed={failed} skipped={skipped})", flush=True)
    print(f"[+] done: {done} preds -> {out} (failed={failed} skipped={skipped})", flush=True)


if __name__ == "__main__":
    main()

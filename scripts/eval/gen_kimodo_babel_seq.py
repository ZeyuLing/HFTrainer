#!/usr/bin/env python3
"""KIMODO sequential-action generation on BABEL (Table 3 baseline).

KIMODO has no native multi-caption sequential T2M, so we generate each BABEL
sub-action independently with text-to-motion and concatenate the SMPL-22
positions (method A). Segments are not motion-conditioned on each other, which
reflects KIMODO's lack of an autoregressive composition mechanism.

Captions are LLM-rewritten into HumanML3D-style sentences (same as PRISM), and
text features are read from a pre-extracted LLM2Vec cache (CacheOnlyTextEncoder),
so the 8B encoder is never loaded here.

Output: ``<out-dir>/<id>.npy`` with shape ``(T,22,3)`` @30fps, consumed by
``joints_to_272_npz.py`` -> ``eval_babel_seq_ms272.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

from gen_kimodo_t2m_positions import _run_one, CacheOnlyTextEncoder  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-total", type=int, default=24)
    ap.add_argument("--max-total", type=int, default=360)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-feature-cache-dir", default="data/kimodo_text_feature")
    ap.add_argument("--text-feature-namespace", default="kimodo_soma_t2m_babel_val_llm2vec")
    ap.add_argument("--text-feature-encoder-id", default="LLM2VecEncoder")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    man = [json.loads(l) for l in open(os.path.join(REPO, args.manifest)) if l.strip()]
    man = [m for m in man if args.min_total <= m["total_frames"] <= args.max_total
           and len(m.get("segments", [])) >= 2]
    man = man[args.shard_index::args.num_shards]
    if args.max_episodes:
        man = man[:args.max_episodes]

    out = Path(args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir))
    out.mkdir(parents=True, exist_ok=True)

    os.environ["TEXT_ENCODER"] = "dummy"
    os.environ["TEXT_ENCODER_MODE"] = "local"
    from kimodo import load_model
    model = load_model("kimodo-soma-rp", device=args.device)
    model.text_encoder = CacheOnlyTextEncoder(
        namespace=args.text_feature_namespace,
        cache_dir=args.text_feature_cache_dir,
        encoder_id=args.text_feature_encoder_id,
        device=args.device,
    )
    print(f"[kimodo-babel-seq] shard {args.shard_index}/{args.num_shards} episodes={len(man)} "
          f"fps={model.fps}", flush=True)

    ok = skipped = failed = 0
    for ep_i, rec in enumerate(man):
        sid = rec["id"]
        out_file = out / f"{sid}.npy"
        if args.skip_existing and out_file.exists():
            skipped += 1
            continue
        try:
            pieces = []
            for s in rec["segments"]:
                cap = rewrite_caption(str(s["caption"]).strip())
                seg_len = int(s["end"]) - int(s["start"])
                if seg_len <= 0:
                    continue
                pos22, _ = _run_one(model, cap, seg_len, args.fps, args.postprocess)
                pieces.append(pos22)
            full = np.concatenate(pieces, axis=0).astype(np.float32)
            total = int(rec["total_frames"])
            if full.shape[0] > total:
                full = full[:total]
            elif full.shape[0] < total:
                full = np.concatenate([full, np.repeat(full[-1:], total - full.shape[0], axis=0)], axis=0)
            if not np.isfinite(full).all():
                raise ValueError("non-finite positions")
            np.save(str(out_file), full)
            ok += 1
            if ok % 25 == 0:
                print(f"[kimodo-babel-seq] ok={ok} skip={skipped} fail={failed} ({ep_i+1}/{len(man)})", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[kimodo-babel-seq] FAIL {sid}: {e}", flush=True)

    print(f"[kimodo-babel-seq] DONE ok={ok} skip={skipped} fail={failed} -> {out}", flush=True)


if __name__ == "__main__":
    main()

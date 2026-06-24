#!/usr/bin/env python3
"""Generate MotionStreamer-272 T2M outputs with the **hftrainer-native**, fully
ref_repo-independent MotionStreamer reproduction.

Pairing mirrors ``MotionStreamer272Evaluator.load_test_pairs()`` (one entry per
(name, caption) on the released ``humanml3d_272`` test split). For every pair we
generate one 272-dim motion of the GT (token-aligned) length and save it as
``<out_dir>/<idx:06d>.npy`` keyed by the *deterministic* pair index, so the
scoring step (``eval_ms_h3d272.py``) can re-pair preds with captions/GT exactly.

Generation path (independent of ``ref_repo``):
    text -> SentenceT5-XXL -> LLaMA AR (CFG, per-token diffusion) -> latent
    tokens -> causal TAE decoder -> 272-dim motion.

Example (single GPU smoke):
    python3 scripts/eval/ms_t2m_h3d272.py --out_dir outputs/evaluation/ms_h3d272/ms_272 \
        --limit 4 --device cuda

Sharded (8 GPUs): see ``scripts/eval/_run_ms_h3d272_shards.sh``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--guidance_param", type=float, default=4.0)
    p.add_argument("--tae_path", default=None, help="raw TAE .pth (ckpt['net']); default released")
    p.add_argument("--ar_path", default=None, help="raw AR .pth (ckpt['trans']); default released")
    p.add_argument("--artifact", default=None, help="hftrainer MS artifact dir (overrides tae/ar)")
    p.add_argument("--text_model_name", default="sentence-transformers/sentence-t5-xxl")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="cap #pairs (smoke); 0 = all")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- deterministic (name, caption, gt, ml) pairs from the MS evaluator --- #
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    print("[ms-gen] building MS-272 evaluator (CPU) for test pairs...", flush=True)
    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()
    n_total = len(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[ms-gen] {len(pairs)}/{n_total} pairs (limit={args.limit})", flush=True)

    # shard by pair index
    todo = [
        (i, pr)
        for i, pr in enumerate(pairs)
        if (i % args.num_shards) == args.shard_index
    ]
    print(f"[ms-gen] shard {args.shard_index}/{args.num_shards}: {len(todo)} pairs", flush=True)

    # --- build MS bundle + pipeline ----------------------------------------- #
    from hftrainer.models.motion.motionstreamer import MotionStreamerBundle
    from hftrainer.pipelines.motionstreamer import MotionStreamerPipeline

    if args.artifact:
        bundle = MotionStreamerBundle.from_pretrained(
            args.artifact, guidance_param=args.guidance_param, device=args.device,
            text_model_name=args.text_model_name,
        )
    else:
        bundle = MotionStreamerBundle(
            tae_path=args.tae_path,
            ar_path=args.ar_path,
            text_model_name=args.text_model_name,
            guidance_param=args.guidance_param,
            device=args.device,
        )
    pipe = MotionStreamerPipeline(bundle, device=args.device)
    print("[ms-gen] bundle ready; generating...", flush=True)

    written = skipped = failed = 0
    for n_done, (idx, (name, caption, gt, ml)) in enumerate(todo):
        pf = out / f"{idx:06d}.npy"
        if args.skip_existing and pf.exists():
            skipped += 1
            continue
        try:
            motion = pipe.infer_t2m([caption], [int(ml)], progress=False)[0]
            np.save(pf, motion.astype(np.float32))
            written += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ms-gen] FAIL idx={idx} name={name}: {e}", flush=True)
        if (n_done + 1) % 25 == 0:
            print(
                f"[progress] seen={n_done + 1}/{len(todo)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    print(
        f"[done] written={written} skipped={skipped} failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()

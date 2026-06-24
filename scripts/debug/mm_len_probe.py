#!/usr/bin/env python3
"""Probe MotionMillion AR generation length vs GT length.

Picks the longest-GT test pairs and samples with a *raised* token budget to check
whether the released greedy sampler is structurally capped at 50 tokens (~100
frames) or stops earlier at EOS. Prints (gt_frames, gen_tokens, gen_frames).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--max_sample_steps", type=int, default=150)
    p.add_argument("--text_model_name", default="checkpoints/flan-t5-xl")
    args = p.parse_args()

    import hftrainer.models.motion.motionmillion.bundle  # noqa: F401
    from hftrainer.models.motion.motionmillion import MotionMillionBundle
    from hftrainer.pipelines.motionmillion import MotionMillionPipeline
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()
    pairs = sorted(pairs, key=lambda x: x[3], reverse=True)[: args.n]

    bundle = MotionMillionBundle(load_text_model=True, text_model_name=args.text_model_name)
    bundle.ar.to(device=args.device, dtype=torch.bfloat16)
    bundle.vqvae.to(device=args.device)
    bundle.text_model.to(device=args.device, dtype=torch.bfloat16)
    bundle.mean = bundle.mean.to(args.device)
    bundle.std = bundle.std.to(args.device)
    bundle.eval()
    pipe = MotionMillionPipeline(bundle, max_sample_steps=args.max_sample_steps)

    import time

    print(f"max_sample_steps={args.max_sample_steps}")
    print(f"{'gt_frames':>10} {'tok_nocache':>11} {'tok_cache':>10} {'match':>6} "
          f"{'t_nc(s)':>8} {'t_c(s)':>8} {'speedup':>8}  caption")
    for name, caption, gt, ml in pairs:
        feat, y_mask = bundle.encode_text([caption])
        feat = feat.to(next(bundle.ar.parameters()).dtype)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize(); t0 = time.time()
            idx_nc = bundle.ar.sample(feat, y_mask, if_categorial=False,
                                      max_sample_steps=args.max_sample_steps)
            torch.cuda.synchronize(); t_nc = time.time() - t0
            t0 = time.time()
            idx_c = bundle.ar.sample_cached(feat, y_mask, if_categorial=False,
                                            max_sample_steps=args.max_sample_steps)
            torch.cuda.synchronize(); t_c = time.time() - t0
        same = bool(idx_nc.shape == idx_c.shape and torch.equal(idx_nc, idx_c))
        sp = t_nc / max(t_c, 1e-6)
        print(f"{ml:>10} {idx_nc.shape[1]:>11} {idx_c.shape[1]:>10} {str(same):>6} "
              f"{t_nc:>8.2f} {t_c:>8.2f} {sp:>7.1f}x  {caption[:40]}")


if __name__ == "__main__":
    main()

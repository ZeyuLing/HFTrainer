#!/usr/bin/env python3
"""Convert raw upstream MoMask checkpoints into a self-contained hftrainer artifact.

The released MoMask checkpoints live under
``ref_repo/Momask/weights/<dataset>/<name>/model/*.tar`` (one each for the
RVQ-VAE, the masked transformer, the residual transformer and the length
estimator) and rely on external ``meta/{mean,std}.npy``. This CLI loads them
through :class:`MoMaskBundle` and re-exports a diffusers-style artifact
directory that :meth:`MoMaskBundle.from_pretrained` can read with zero
dependency on ``ref_repo``::

    <out>/momask_config.json     # arch config for all sub-modules
    <out>/vq.safetensors         # RVQ-VAE weights
    <out>/t2m_trans.safetensors  # MaskTransformer (no CLIP)
    <out>/res_trans.safetensors  # ResidualTransformer (no CLIP)
    <out>/length_est.safetensors # LengthEstimator
    <out>/Mean.npy, Std.npy      # 263-dim denorm stats (embedded)

Example
-------
python3 scripts/eval/convert_momask_checkpoint.py \
    --weights_root ref_repo/Momask/weights \
    --out_dir checkpoints/momask/humanml3d \
    --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.momask import MoMaskBundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_root", default="ref_repo/Momask/weights")
    p.add_argument("--dataset_name", default="t2m")
    p.add_argument("--vq_name", default="rvq_nq6_dc512_nc512_noshare_qdp0.2")
    p.add_argument("--t2m_name", default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    p.add_argument("--res_name", default="tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw")
    p.add_argument("--len_name", default="length_estimator")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_length_estimator", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="reload artifact and assert bit-identical generation")
    args = p.parse_args()

    print(f"[convert] loading raw MoMask checkpoints from {args.weights_root}", flush=True)
    bundle = MoMaskBundle(
        weights_root=args.weights_root,
        dataset_name=args.dataset_name,
        vq_name=args.vq_name,
        t2m_name=args.t2m_name,
        res_name=args.res_name,
        len_name=args.len_name,
        load_length_estimator=not args.no_length_estimator,
    )
    bundle.save_pretrained(args.out_dir)
    print(f"[convert] wrote artifact -> {args.out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in Path(args.out_dir).iterdir())}", flush=True)

    if args.verify:
        from hftrainer.pipelines.momask import MoMaskPipeline

        reloaded = MoMaskBundle.from_pretrained(args.out_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _gen(b):
            pipe = MoMaskPipeline(b, device=device)
            torch.manual_seed(0)
            np.random.seed(0)
            return pipe.infer_t2m(["a person walks forward"], [80])[0]

        a, b = _gen(bundle), _gen(reloaded)
        diff = float(np.abs(a - b).max())
        print(f"[verify] raw vs artifact generation max-abs-diff = {diff}", flush=True)
        assert diff == 0.0, "artifact generation diverged from the raw checkpoint!"
        print("[verify] OK: artifact is bit-identical to the raw checkpoint.", flush=True)


if __name__ == "__main__":
    main()

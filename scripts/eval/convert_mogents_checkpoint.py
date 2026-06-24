#!/usr/bin/env python3
"""Convert raw upstream MoGenTS checkpoints into a self-contained hftrainer artifact.

Official MoGenTS checkpoints are expected under ``logs/<dataset>/`` after
unzipping the upstream pretrained-model archive. The length estimator follows
the official MoGenTS/MoMask convention and may live under
``checkpoints/<dataset>/length_estimator``.

Example
-------
python3 scripts/eval/convert_mogents_checkpoint.py \
    --weights_root logs \
    --length_root checkpoints \
    --out_dir checkpoints/mogents/humanml3d \
    --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.mogents import MoGenTSBundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_root", default="logs")
    p.add_argument("--length_root", default="checkpoints")
    p.add_argument("--dataset_name", default="humanml3d")
    p.add_argument("--vq_name", default="pretrain_vq")
    p.add_argument("--mtrans_name", default="pretrain_mtrans")
    p.add_argument("--rtrans_name", default="pretrain_rtrans")
    p.add_argument("--len_name", default="length_estimator")
    p.add_argument("--vq_ckpt_name", default="net_best_fid.tar")
    p.add_argument("--mask_ckpt_name", default="net_best_fid.tar")
    p.add_argument("--res_ckpt_name", default="net_best_fid.tar")
    p.add_argument("--len_ckpt_name", default="finest.tar")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_length_estimator", action="store_true")
    p.add_argument(
        "--no_clip",
        action="store_true",
        help="legacy export: do not copy CLIP ViT-B/32 into the artifact",
    )
    p.add_argument("--verify", action="store_true",
                   help="reload artifact and assert bit-identical generation")
    args = p.parse_args()

    print(f"[convert] loading raw MoGenTS checkpoints from {args.weights_root}", flush=True)
    bundle = MoGenTSBundle(
        weights_root=args.weights_root,
        length_root=args.length_root,
        dataset_name=args.dataset_name,
        vq_name=args.vq_name,
        mtrans_name=args.mtrans_name,
        rtrans_name=args.rtrans_name,
        len_name=args.len_name,
        vq_ckpt_name=args.vq_ckpt_name,
        mask_ckpt_name=args.mask_ckpt_name,
        res_ckpt_name=args.res_ckpt_name,
        len_ckpt_name=args.len_ckpt_name,
        load_length_estimator=not args.no_length_estimator,
    )
    bundle.save_pretrained(args.out_dir, include_clip=not args.no_clip)
    print(f"[convert] wrote artifact -> {args.out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in Path(args.out_dir).iterdir())}", flush=True)

    if args.verify:
        from hftrainer.pipelines.mogents import MoGenTSPipeline

        reloaded = MoGenTSBundle.from_pretrained(args.out_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _gen(b):
            pipe = MoGenTSPipeline(b, device=device)
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

#!/usr/bin/env python3
"""Convert a raw upstream MLD checkpoint into a self-contained hftrainer artifact.

The upstream MLD HumanML3D checkpoints are Lightning ``.ckpt`` files with
``vae.*`` and ``denoiser.*`` keys plus external SentenceT5 and HumanML3D
normalization stats. This CLI re-exports the model as:

    <out>/mld_config.json
    <out>/vae.safetensors
    <out>/denoiser.safetensors
    <out>/Mean.npy
    <out>/Std.npy

The SentenceT5 text encoder is resolved by name at load time, matching the
MotionLCM artifact convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.mld import MLDBundle  # noqa: E402

_DEF_CKPT = REPO / "ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml_v1.ckpt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_ckpt", default=str(_DEF_CKPT),
                   help="Lightning ckpt holding the MLD VAE and denoiser")
    p.add_argument("--mean_path", default=None,
                   help="263-dim HumanML3D Mean.npy (default: canonical Guo stats)")
    p.add_argument("--std_path", default=None)
    p.add_argument("--text_encoder_name", default="sentence-transformers/sentence-t5-large")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--verify", action="store_true",
                   help="reload artifact and assert bit-identical generation")
    args = p.parse_args()

    print(f"[convert] loading raw MLD checkpoint <- {args.model_ckpt}", flush=True)
    bundle = MLDBundle(
        model_ckpt=args.model_ckpt,
        mean_path=args.mean_path,
        std_path=args.std_path,
        text_encoder_name=args.text_encoder_name,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        load_text_encoder=args.verify,
    )
    bundle.save_pretrained(args.out_dir)
    print(f"[convert] wrote artifact -> {args.out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in Path(args.out_dir).iterdir())}", flush=True)

    if args.verify:
        from hftrainer.pipelines.mld import MLDPipeline

        reloaded = MLDBundle.from_pretrained(args.out_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _gen(b):
            pipe = MLDPipeline(b, device=device)
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

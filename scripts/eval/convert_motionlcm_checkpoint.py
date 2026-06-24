#!/usr/bin/env python3
"""Convert raw upstream MotionLCM checkpoints into a self-contained hftrainer artifact.

The released MotionLCM checkpoints live under
``ref_repo/MotionLCM/experiments_t2m/<name>/<name>.ckpt`` (lightning format,
state dict in ``["state_dict"]`` with ``vae.*`` / ``denoiser.*`` prefixes) and
rely on an external sentence-t5-large text encoder + HumanML3D Mean/Std. This
CLI loads them through :class:`MotionLCMBundle` and re-exports a diffusers-style
artifact directory that :meth:`MotionLCMBundle.from_pretrained` can read with
zero dependency on ``ref_repo``::

    <out>/motionlcm_config.json  # arch config (vae / denoiser / scheduler)
    <out>/vae.safetensors        # MLD motion VAE weights
    <out>/denoiser.safetensors   # latent consistency denoiser weights
    <out>/Mean.npy, Std.npy      # 263-dim denorm stats (embedded)

The frozen sentence-t5-large text encoder is reloaded by name and is **not**
stored (exactly like CLIP in MDM / MoMask).

Example
-------
python3 scripts/eval/convert_motionlcm_checkpoint.py \
    --vae_ckpt ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml_v1.ckpt \
    --denoiser_ckpt ref_repo/MotionLCM/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt \
    --out_dir checkpoints/motionlcm/humanml3d \
    --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.motionlcm import MotionLCMBundle

REPO = Path(__file__).resolve().parents[2]
_DEF_VAE = REPO / "ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml_v1.ckpt"
_DEF_DEN = REPO / "ref_repo/MotionLCM/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", default=str(_DEF_VAE),
                   help="lightning ckpt holding the frozen MLD VAE (vae.* keys)")
    p.add_argument("--denoiser_ckpt", default=str(_DEF_DEN),
                   help="lightning ckpt holding the LCM denoiser (denoiser.* keys)")
    p.add_argument("--mean_path", default=None,
                   help="263-dim HumanML3D Mean.npy (default: canonical Guo stats)")
    p.add_argument("--std_path", default=None)
    p.add_argument("--text_encoder_name", default="sentence-transformers/sentence-t5-large")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=1)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--verify", action="store_true",
                   help="reload artifact and assert bit-identical generation")
    args = p.parse_args()

    print(f"[convert] loading raw MotionLCM checkpoints:\n"
          f"          vae      <- {args.vae_ckpt}\n"
          f"          denoiser <- {args.denoiser_ckpt}", flush=True)
    bundle = MotionLCMBundle(
        vae_ckpt=args.vae_ckpt,
        denoiser_ckpt=args.denoiser_ckpt,
        mean_path=args.mean_path,
        std_path=args.std_path,
        text_encoder_name=args.text_encoder_name,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        load_text_encoder=args.verify,  # only needed when verifying generation
    )
    bundle.save_pretrained(args.out_dir)
    print(f"[convert] wrote artifact -> {args.out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in Path(args.out_dir).iterdir())}", flush=True)

    if args.verify:
        from hftrainer.pipelines.motionlcm import MotionLCMPipeline

        reloaded = MotionLCMBundle.from_pretrained(args.out_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _gen(b):
            pipe = MotionLCMPipeline(b, device=device)
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

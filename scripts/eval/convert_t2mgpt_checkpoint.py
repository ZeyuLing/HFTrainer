#!/usr/bin/env python3
"""Convert the raw upstream T2M-GPT checkpoints into a self-contained artifact.

The released T2M-GPT model is a pair of ``.pth`` files (a VQ-VAE
``net_last.pth`` with a ``net`` key, and a GPT ``net_best_fid.pth`` with a
``trans`` key) plus external HumanML3D ``Mean.npy`` / ``Std.npy``. This CLI loads
them through :class:`T2MGPTBundle` and re-exports a diffusers-style artifact
directory that :meth:`T2MGPTBundle.from_pretrained` can read with zero
dependency on ``ref_repo`` or the original ``.pth`` format::

    <out>/t2mgpt_config.json   # vqvae + gpt arch config + clip name
    <out>/vq.safetensors       # HumanVQVAE weights
    <out>/gpt.safetensors      # Text2Motion_Transformer weights
    <out>/Mean.npy, Std.npy    # 263-dim denorm stats (embedded, self-contained)

The CLIP ViT-B/32 text encoder is reloaded by name and is not stored.

Example
-------
python3 scripts/eval/convert_t2mgpt_checkpoint.py \
    --vq_path ref_repo/T2M-GPT/pretrained/VQVAE/net_last.pth \
    --gpt_path ref_repo/T2M-GPT/pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out_dir checkpoints/t2mgpt/humanml3d --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.t2mgpt import T2MGPTBundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vq_path", default=None, help="raw VQ-VAE .pth (key 'net')")
    p.add_argument("--gpt_path", default=None, help="raw GPT .pth (key 'trans')")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip_name", default="ViT-B/32")
    p.add_argument("--mean_path", default=None)
    p.add_argument("--std_path", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--verify",
        action="store_true",
        help="reload artifact and assert bit-identical generation",
    )
    args = p.parse_args()

    print(f"[convert] loading raw T2M-GPT checkpoints: vq={args.vq_path} gpt={args.gpt_path}", flush=True)
    bundle = T2MGPTBundle(
        vq_path=args.vq_path,
        gpt_path=args.gpt_path,
        clip_name=args.clip_name,
        mean_path=args.mean_path,
        std_path=args.std_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bundle.save_pretrained(args.out_dir)
    print(f"[convert] wrote artifact -> {args.out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in Path(args.out_dir).iterdir())}", flush=True)

    if args.verify:
        from hftrainer.pipelines.t2mgpt import T2MGPTPipeline

        reloaded = T2MGPTBundle.from_pretrained(args.out_dir)

        def _gen(b):
            pipe = T2MGPTPipeline(b, device="cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            return pipe.infer_t2m(["a person walks forward"])[0]

        a, b = _gen(bundle), _gen(reloaded)
        # T2M-GPT can emit different token counts only if RNG diverges; with the
        # same seed the shapes match and the values are bit-identical.
        if a.shape != b.shape:
            raise AssertionError(f"artifact generation shape diverged: {a.shape} vs {b.shape}")
        diff = float(np.abs(a - b).max())
        print(f"[verify] raw vs artifact generation max-abs-diff = {diff}", flush=True)
        assert diff == 0.0, "artifact generation diverged from the raw checkpoint!"
        print("[verify] OK: artifact is bit-identical to the raw checkpoint.", flush=True)


if __name__ == "__main__":
    main()

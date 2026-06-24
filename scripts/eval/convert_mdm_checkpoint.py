#!/usr/bin/env python3
"""Convert a raw upstream MDM checkpoint into a self-contained hftrainer artifact.

The released MDM checkpoint is a ``model*.pt`` next to an ``args.json`` (the
original training config) and relies on external ``Mean.npy`` / ``Std.npy``. This
CLI loads it through :class:`MDMBundle` and re-exports a diffusers-style artifact
directory that :meth:`MDMBundle.from_pretrained` can read with zero dependency on
``ref_repo`` or the original ``.pt`` format::

    <out>/mdm_config.json     # arch + diffusion config + guidance_param
    <out>/model.safetensors   # network weights (no CLIP, no fixed pe buffers)
    <out>/Mean.npy, Std.npy   # 263-dim denorm stats (embedded, self-contained)

Example
-------
python3 scripts/eval/convert_mdm_checkpoint.py \
    --model_path ref_repo/MDM/save/humanml_trans_enc_512/humanml_trans_enc_512/model000475000.pt \
    --out_dir checkpoints/mdm/humanml_trans_enc_512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.mdm import MDMBundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="raw MDM .pt (with sibling args.json)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--guidance_param", type=float, default=2.5)
    p.add_argument("--mean_path", default=None)
    p.add_argument("--std_path", default=None)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--verify", action="store_true", help="reload artifact and assert bit-identical generation")
    args = p.parse_args()

    print(f"[convert] loading raw MDM checkpoint: {args.model_path}", flush=True)
    bundle = MDMBundle(
        model_path=args.model_path,
        guidance_param=args.guidance_param,
        mean_path=args.mean_path,
        std_path=args.std_path,
        use_ema=args.use_ema,
    )
    print(f"[convert] stats_source={bundle.stats_source}", flush=True)
    bundle.save_pretrained(args.out_dir)
    print(f"[convert] wrote artifact -> {args.out_dir}", flush=True)
    print(f"          files: {sorted(Path(args.out_dir).iterdir())}", flush=True)

    if args.verify:
        from hftrainer.pipelines.mdm import MDMPipeline

        reloaded = MDMBundle.from_pretrained(args.out_dir)

        def _gen(b):
            pipe = MDMPipeline(b, device="cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(0)
            np.random.seed(0)
            return pipe.infer_t2m(["a person walks forward"], [80], progress=False)[0]

        a, b = _gen(bundle), _gen(reloaded)
        diff = float(np.abs(a - b).max())
        print(f"[verify] raw vs artifact generation max-abs-diff = {diff}", flush=True)
        assert diff == 0.0, "artifact generation diverged from the raw checkpoint!"
        print("[verify] OK: artifact is bit-identical to the raw checkpoint.", flush=True)


if __name__ == "__main__":
    main()

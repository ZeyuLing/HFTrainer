#!/usr/bin/env python3
"""Reconstruct HumanML3D-263 clips with MotionGPT's VQ-VAE."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from motiongpt_infer_hml3d263 import (  # noqa: E402
    MOTIONGPT_ROOT,
    DummyHumanML3DDataModule,
    force_untied_t5_lm_head,
    load_cfg,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-root", default="work_dirs/h3d263_eval/h3d263_test_recon_fk")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--cfg", default=str(MOTIONGPT_ROOT / "configs" / "config_h3d_stage3.yaml"))
    ap.add_argument("--checkpoint", default=str(MOTIONGPT_ROOT / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"))
    ap.add_argument("--t5-path", default="google/flan-t5-base")
    ap.add_argument("--mean", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "mean.npy"))
    ap.add_argument("--std", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "std.npy"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    recon_root = Path(args.recon_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    force_untied_t5_lm_head()
    os.chdir(MOTIONGPT_ROOT)
    from mGPT.models.base import BaseModel  # noqa: WPS433
    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    from mGPT.models.build_model import build_model  # noqa: WPS433

    cfg = load_cfg(args)
    dm = DummyHumanML3DDataModule(Path(args.mean), Path(args.std))
    model = build_model(cfg, dm).eval()
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ids = [s.strip() for s in (recon_root / "test.txt").read_text().splitlines() if s.strip()]
    written = 0
    token_lens: list[int] = []
    for sid in tqdm(ids, ncols=80):
        if args.max_samples and written >= args.max_samples:
            break
        src = recon_root / "new_joint_vecs" / f"{sid}.npy"
        if not src.exists():
            continue
        raw = np.load(src).astype(np.float32)
        if len(raw) < 40 or len(raw) >= 200:
            continue
        feat = torch.from_numpy(raw).to(device)[None]
        norm = (feat - dm.mean.to(device)) / dm.std.to(device)
        with torch.no_grad():
            tokens, _ = model.vae.encode(norm)
            recon = model.vae.decode(tokens[0])
            out = dm.denormalize(recon[:, : len(raw)]).detach().cpu().numpy()[0].astype(np.float32)
        np.save(out_dir / f"{sid}.npy", out)
        token_lens.append(int(tokens.shape[1]))
        written += 1
    print({
        "written": written,
        "token_len_min": min(token_lens) if token_lens else None,
        "token_len_median": float(np.median(token_lens)) if token_lens else None,
        "token_len_max": max(token_lens) if token_lens else None,
    }, flush=True)


if __name__ == "__main__":
    main()

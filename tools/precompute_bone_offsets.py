#!/usr/bin/env python3
"""Precompute SMPL-22 FK bone offsets used by HYMotion M2M.

The generated tensor is parent-relative ``(22, 3)`` offsets from the same
SMPL-H rest pose used by the HumanML3D conversion path.  Several M2M data
transforms lazy-load this file when converting 135-dim rotation+translation
motions to the 198-dim strict RIC representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_smplh_offsets(model_path: str | None = None) -> torch.Tensor:
    from hftrainer.datasets.motion.representation.humanml_repr import (
        DEFAULT_PATHS,
        _smplh_bone_offsets,
    )

    resolved = model_path or DEFAULT_PATHS.resolve("smplh_model")
    offsets = _smplh_bone_offsets(resolved)
    return torch.as_tensor(offsets, dtype=torch.float32)


def _load_static_offsets() -> torch.Tensor:
    from hftrainer.datasets.motion.motionhub.smpl_data import SMPL22_BONE_OFFSETS

    return torch.as_tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/hymotion_m2m_data/bone_offsets_22.pt",
        help="Output path relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--smplh-model",
        default=None,
        help="Optional SMPL-H neutral model.npz path. Defaults to HumanMLReprPaths.",
    )
    parser.add_argument(
        "--allow-static-fallback",
        action="store_true",
        help="Use the local approximate SMPL22_BONE_OFFSETS constant if SMPL-H loading fails.",
    )
    args = parser.parse_args()

    try:
        offsets = _load_smplh_offsets(args.smplh_model)
        source = "smplh_rest"
    except Exception:
        if not args.allow_static_fallback:
            raise
        offsets = _load_static_offsets()
        source = "static_smpl22"

    if tuple(offsets.shape) != (22, 3):
        raise ValueError(f"expected offsets shape (22, 3), got {tuple(offsets.shape)}")
    if not torch.isfinite(offsets).all():
        raise ValueError("offsets contain non-finite values")

    out = Path(args.output)
    if not out.is_absolute():
        out = _repo_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(offsets.cpu(), out)

    print(f"wrote {out}")
    print(f"source={source} shape={tuple(offsets.shape)} dtype={offsets.dtype}")
    print(f"root_offset={offsets[0].tolist()}")


if __name__ == "__main__":
    main()

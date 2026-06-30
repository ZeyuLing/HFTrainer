#!/usr/bin/env python3
"""Build a partial before/after PRISM length-fix T2M viewer fixture."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


SUITE_ROOT = (
    ROOT
    / "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites"
)
LENFIX = SUITE_ROOT / "prism_epoch43_official_selected_lenfix_20260628"
OLD = SUITE_ROOT / "table6_kafs_epoch43_20260627_run1"
GT_DIR = ROOT / "outputs/evaluation/ms272_tables_h3d_0607/prep/real_conv"


def _smpl_npz_to_motion135(path: Path) -> np.ndarray:
    data = np.load(path)
    transl = np.asarray(data["transl"], dtype=np.float32)
    global_orient = np.asarray(data["global_orient"], dtype=np.float32).reshape(len(transl), 1, 3)
    body_pose = np.asarray(data["body_pose"], dtype=np.float32)
    body_pose = body_pose.reshape(len(transl), 21, 3)
    aa = np.concatenate([global_orient, body_pose], axis=1)
    rotmat = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(len(transl), 22, 3, 3)
    rot6d = matrix_to_rotation_6d(rotmat, convention="row").reshape(len(transl), 132)
    return np.concatenate([transl, np.asarray(rot6d, dtype=np.float32)], axis=1).astype(np.float32)


def _write_motion135(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and "motion_135" in np.load(src).files:
        motion = np.load(src)["motion_135"].astype(np.float32)[:, :135]
    else:
        motion = _smpl_npz_to_motion135(src)
    np.savez_compressed(dst, motion_135=motion)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        default=str(LENFIX / "viewer_partial_before_after"),
    )
    ap.add_argument("--max-cases", type=int, default=82)
    ap.add_argument(
        "--ids",
        default="",
        help="Optional comma-separated ids or a text file containing one id per line.",
    )
    ap.add_argument(
        "--methods",
        default="gt,old,direct,pad360",
        help="Comma-separated subset of: gt,old,direct,pad360.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    method_pool = {
        "gt": ("GT", GT_DIR),
        "old": ("PRISM-old-table6-depth", OLD / "prep/depth_driven"),
        "direct": ("PRISM-direct-len-depth", LENFIX / "raw/direct_len/depth_driven"),
        "pad360": ("PRISM-pad360-depth", LENFIX / "raw/pad360_crop/depth_driven"),
    }
    method_keys = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(method_keys) - set(method_pool))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    methods = {method_pool[k][0]: method_pool[k][1] for k in method_keys}

    if args.ids:
        ids_arg = Path(args.ids)
        if ids_arg.exists():
            requested_ids = [ln.strip() for ln in ids_arg.read_text().splitlines() if ln.strip()]
        else:
            requested_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        ids = [
            cid for cid in requested_ids
            if all((src_dir / f"{cid}.npz").exists() for src_dir in methods.values())
        ]
    else:
        id_sets = [{p.stem for p in src_dir.glob("*.npz")} for src_dir in methods.values()]
        ids = sorted(set.intersection(*id_sets))[: args.max_cases]
    if not ids:
        raise RuntimeError("No requested/intersection cases available yet")

    for label, src_dir in methods.items():
        safe = (
            label.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
        dst_dir = out_root / safe
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for cid in ids:
            _write_motion135(src_dir / f"{cid}.npz", dst_dir / f"{cid}.npz")

    manifest = {
        "description": "Partial PRISM epoch43 before/after length-policy comparison",
        "ids": ids,
        "methods": [
            {"label": label, "dir": str((out_root / label.lower().replace(" ", "_").replace("/", "_").replace("-", "_")).relative_to(ROOT))}
            for label in methods
        ],
    }
    (out_root / "methods.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "ids.txt").write_text("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} cases to {out_root}")
    print(out_root / "methods.json")


if __name__ == "__main__":
    main()

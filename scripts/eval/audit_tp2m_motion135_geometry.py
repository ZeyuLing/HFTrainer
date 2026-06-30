#!/usr/bin/env python3
"""Geometry sanity audit for TP2M motion_135 result folders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.motion.skeleton.fk import motion135_to_fk  # noqa: E402


FOOT_JOINTS = (7, 8, 10, 11)
HEAD_JOINT = 15
PELVIS_JOINT = 0


def _load_motion135(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "motion_135" in data:
            arr = data["motion_135"]
        elif "motion" in data:
            arr = data["motion"]
        else:
            raise KeyError(f"{path} has no motion_135")
    else:
        arr = data
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 135:
        raise ValueError(f"bad shape {arr.shape}")
    return arr[:, :135]


def _score_case(
    path: Path,
    bone_offsets: torch.Tensor,
    device: str,
    include_rot6d_flags: bool = False,
) -> dict:
    m = _load_motion135(path)
    row = {
        "id": path.stem,
        "path": str(path),
        "length": int(m.shape[0]),
        "has_nan": bool(np.isnan(m).any()),
        "has_inf": bool(np.isinf(m).any()),
    }
    root = m[:, :3]
    if len(root) > 1:
        jumps = np.linalg.norm(np.diff(root, axis=0), axis=1)
        y_jumps = np.abs(np.diff(root[:, 1], axis=0))
        row["max_root_jump_m"] = float(np.max(jumps))
        row["p99_root_jump_m"] = float(np.percentile(jumps, 99))
        row["max_root_y_jump_m"] = float(np.max(y_jumps))
    else:
        row["max_root_jump_m"] = 0.0
        row["p99_root_jump_m"] = 0.0
        row["max_root_y_jump_m"] = 0.0
    row["root_y_min_m"] = float(np.min(root[:, 1]))
    row["root_y_max_m"] = float(np.max(root[:, 1]))
    row["root_y_range_m"] = float(np.ptp(root[:, 1]))
    row["root_xz_range_m"] = float(np.linalg.norm(np.ptp(root[:, [0, 2]], axis=0)))

    rot6 = m[:, 3:135].reshape(-1, 22, 6)
    a = rot6[..., :3]
    b = rot6[..., 3:]
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    row["rot6d_a_norm_p99_dev"] = float(np.percentile(np.abs(an - 1.0), 99))
    row["rot6d_b_norm_p99_dev"] = float(np.percentile(np.abs(bn - 1.0), 99))
    row["rot6d_dot_p99_abs"] = float(np.percentile(np.abs(dot), 99))

    with torch.no_grad():
        mt = torch.from_numpy(m).to(device).float()
        pos, _, _, _ = motion135_to_fk(mt, bone_offsets.to(device), rotation_space="local")
        pos_np = pos.detach().cpu().numpy()

    foot_min = pos_np[:, FOOT_JOINTS, 1].min(axis=1)
    head_y = pos_np[:, HEAD_JOINT, 1]
    pelvis_y = pos_np[:, PELVIS_JOINT, 1]
    bbox = pos_np.max(axis=1) - pos_np.min(axis=1)
    height = head_y - foot_min
    pelvis_clearance = pelvis_y - foot_min
    row["foot_min_y_mean_m"] = float(np.mean(foot_min))
    row["foot_min_y_min_m"] = float(np.min(foot_min))
    row["head_to_foot_height_mean_m"] = float(np.mean(height))
    row["head_to_foot_height_min_m"] = float(np.min(height))
    row["head_below_foot_ratio"] = float(np.mean(head_y < foot_min))
    row["pelvis_clearance_mean_m"] = float(np.mean(pelvis_clearance))
    row["pelvis_below_foot_ratio"] = float(np.mean(pelvis_y < foot_min))
    row["bbox_diag_p99_m"] = float(np.percentile(np.linalg.norm(bbox, axis=1), 99))
    row["bbox_y_p99_m"] = float(np.percentile(bbox[:, 1], 99))

    flags = []
    if row["has_nan"] or row["has_inf"]:
        flags.append("nan_or_inf")
    if row["max_root_jump_m"] > 1.0:
        flags.append("root_jump_gt_1m")
    if row["max_root_y_jump_m"] > 0.5:
        flags.append("root_y_jump_gt_0.5m")
    if row["root_y_range_m"] > 2.0:
        flags.append("root_y_range_gt_2m")
    if row["head_to_foot_height_min_m"] < 0.45:
        flags.append("collapsed_height")
    if row["head_below_foot_ratio"] > 0.01:
        flags.append("head_below_foot")
    if row["pelvis_below_foot_ratio"] > 0.01:
        flags.append("pelvis_below_foot")
    if row["bbox_diag_p99_m"] > 3.5:
        flags.append("bbox_diag_gt_3.5m")
    if include_rot6d_flags:
        if row["rot6d_a_norm_p99_dev"] > 0.5 or row["rot6d_b_norm_p99_dev"] > 0.5:
            flags.append("rot6d_norm_bad")
        if row["rot6d_dot_p99_abs"] > 0.75:
            flags.append("rot6d_ortho_bad")
    row["flags"] = flags
    row["abnormal_score"] = (
        len(flags) * 10.0
        + min(row["max_root_jump_m"], 5.0)
        + min(row["root_y_range_m"], 5.0)
        + min(row["bbox_diag_p99_m"], 5.0)
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="motion135 result dirs to audit.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=80)
    ap.add_argument(
        "--include-rot6d-flags",
        action="store_true",
        help=(
            "Also flag raw rot6d norm/orthogonality diagnostics. Off by default "
            "because convention differences can trigger these for otherwise valid "
            "motion_135 folders; FK/height/jump flags are safer for visual triage."
        ),
    )
    args = ap.parse_args()

    bone_offsets = torch.load(args.bone_offsets, map_location=args.device)
    if not isinstance(bone_offsets, torch.Tensor):
        bone_offsets = torch.as_tensor(bone_offsets)
    bone_offsets = bone_offsets.float()

    report = {}
    for root in args.roots:
        root_path = Path(root)
        rows = []
        failures = []
        for path in sorted(root_path.glob("*.npz")):
            try:
                rows.append(_score_case(path, bone_offsets, args.device, args.include_rot6d_flags))
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": path.stem, "path": str(path), "error": str(exc)})
        flagged = [r for r in rows if r["flags"]]
        top = sorted(rows, key=lambda r: r["abnormal_score"], reverse=True)[:args.top_k]
        report[str(root_path)] = {
            "count": len(rows),
            "failed": len(failures),
            "flagged": len(flagged),
            "flag_counts": {
                flag: sum(flag in r["flags"] for r in rows)
                for flag in sorted({f for r in rows for f in r["flags"]})
            },
            "top_cases": top,
            "failures": failures[:200],
        }
        print(str(root_path), "count", len(rows), "flagged", len(flagged), "failed", len(failures))

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

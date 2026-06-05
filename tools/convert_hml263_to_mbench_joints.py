#!/usr/bin/env python3
"""Convert HumanML3D-263 feature outputs to MBench raw-joint inputs.

This avoids the previous SMPL IK retargeting path for Table-3 style metrics.
MBench accepts raw joints ``(T, 22, 3)``; its own evaluator performs SMPLify
when pose/penetration metrics require SMPL parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.eval.hml263_to_smpl_ik import recover_from_ric, resample_linear  # noqa: E402


HML_YUP_TO_MBENCH_ZUP = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def expected_frame_map(eval_info_json: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in load_json(eval_info_json):
        motion_id = int(row["id"])
        frames = int(row["motion_duration"])
        old = out.get(motion_id)
        if old is not None and old != frames:
            raise ValueError(f"Conflicting frame count for id={motion_id}: {old} vs {frames}")
        out[motion_id] = frames
    return out


def parse_id_list(value: str) -> List[int] | None:
    if not value:
        return None
    ids: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(chunk))
    return sorted(set(ids))


def joint_stats(joints: np.ndarray) -> Dict[str, Any]:
    feet = joints[:, [10, 11], :]
    return {
        "shape": list(joints.shape),
        "nan_count": int(np.isnan(joints).sum()),
        "min_xyz": [float(x) for x in joints.min(axis=(0, 1))],
        "max_xyz": [float(x) for x in joints.max(axis=(0, 1))],
        "foot_min_z": float(feet[..., 2].min()),
        "foot_mean_min_z_per_frame": float(feet[..., 2].min(axis=1).mean()),
    }


def convert_one(
    in_path: str,
    out_path: str,
    source_fps: float,
    target_fps: float,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    floor_align: bool,
) -> Dict[str, Any]:
    feats = np.load(in_path).astype(np.float32)
    if feats.ndim != 2 or feats.shape[-1] != 263:
        raise ValueError(f"expected (T,263), got {feats.shape}")
    if mean is not None and std is not None:
        feats = feats * std + mean
    joints = recover_from_ric(feats, 22)
    joints = resample_linear(joints, source_fps, target_fps)
    if floor_align:
        joints = joints.copy()
        joints[..., 1] -= joints[..., 1].min()
    joints = np.einsum("ij,tvj->tvi", HML_YUP_TO_MBENCH_ZUP, joints).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, joints)
    return {"pred_frames": int(joints.shape[0]), "joint_stats": joint_stats(joints)}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = Counter(row["status"] for row in records)
    ok = [row for row in records if row["status"] == "ok"]
    frame_errors = [abs(int(row["pred_frames"]) - int(row["expected_frames"])) for row in ok]
    foot_min = [float(row["joint_stats"]["foot_min_z"]) for row in ok]
    return {
        "num_records": len(records),
        "statuses": dict(statuses),
        "ok": int(statuses.get("ok", 0)),
        "complete": int(statuses.get("ok", 0)) == len(records),
        "frame_abs_error_mean": float(np.mean(frame_errors)) if frame_errors else None,
        "frame_abs_error_max": int(max(frame_errors)) if frame_errors else None,
        "foot_min_z_mean": float(np.mean(foot_min)) if foot_min else None,
        "foot_min_z_min": float(np.min(foot_min)) if foot_min else None,
        "foot_min_z_max": float(np.max(foot_min)) if foot_min else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-info-json", default="ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json")
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--mean", default="")
    parser.add_argument("--std", default="")
    parser.add_argument("--ids", default="")
    parser.add_argument("--floor-align", action="store_true", help="Apply floor alignment before MBench coordinate conversion.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mean = np.load(args.mean).astype(np.float32) if args.mean else None
    std = np.load(args.std).astype(np.float32) if args.std else None
    if (mean is None) != (std is None):
        raise ValueError("--mean and --std must be provided together")
    if mean is not None and (mean.shape != (263,) or std.shape != (263,)):
        raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")

    frame_map = expected_frame_map(args.eval_info_json)
    selected = parse_id_list(args.ids) or sorted(frame_map)
    out_input = os.path.join(args.out_dir, "mbench_eval_input")
    os.makedirs(out_input, exist_ok=True)
    records = []
    for motion_id in selected:
        in_path = os.path.join(args.in_dir, f"{motion_id}.npy")
        out_path = os.path.join(out_input, f"{motion_id}.npy")
        record: Dict[str, Any] = {
            "id": int(motion_id),
            "input_path": in_path,
            "output_path": os.path.relpath(out_path, args.out_dir),
            "expected_frames": int(frame_map[motion_id]),
        }
        if not os.path.exists(in_path):
            record["status"] = "missing_input"
        elif os.path.exists(out_path) and not args.force:
            joints = np.load(out_path)
            record.update({"status": "skipped_existing", "pred_frames": int(joints.shape[0]), "joint_stats": joint_stats(joints)})
        else:
            try:
                record.update(
                    convert_one(
                        in_path,
                        out_path,
                        args.source_fps,
                        args.target_fps,
                        mean,
                        std,
                        args.floor_align,
                    )
                )
                record["status"] = "ok"
            except Exception as exc:
                record.update({"status": "error", "error": repr(exc)})
        records.append(record)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": os.path.abspath(args.in_dir),
        "output_dir": os.path.abspath(args.out_dir),
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "floor_align": bool(args.floor_align),
        "summary": summarize(records),
        "records": records,
    }
    write_json(os.path.join(args.out_dir, "manifest.json"), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"[convert-hml263-mbench] wrote {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()

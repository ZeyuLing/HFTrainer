#!/usr/bin/env python3
"""Validate MBench evaluator input directories.

Expected input format:

    <eval-input-dir>/{id}.npy  # float array, shape (T, 22, 3)

The script cross-checks the ids and frame counts against
``MBench_eval_info.json`` and writes a manifest that records shape, NaN, and
floor-height statistics. It does not mutate the input directory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


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
    frames: Dict[int, int] = {}
    dims_by_id: Dict[int, List[str]] = defaultdict(list)
    for row in load_json(eval_info_json):
        motion_id = int(row["id"])
        num_frames = int(row["motion_duration"])
        if motion_id in frames and frames[motion_id] != num_frames:
            raise ValueError(f"Conflicting frame count for id={motion_id}: {frames[motion_id]} vs {num_frames}")
        frames[motion_id] = num_frames
        dims_by_id[motion_id].append(row["dimension"])
    return frames


def array_record(path: str, expected_frames: int, strict_frames: bool) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"path": path, "exists": os.path.exists(path)}
    if not rec["exists"]:
        rec["status"] = "missing"
        return rec

    try:
        arr = np.load(path)
    except Exception as exc:
        rec.update({"status": "load_error", "error": repr(exc)})
        return rec

    rec["shape"] = list(arr.shape)
    rec["dtype"] = str(arr.dtype)
    rec["nan_count"] = int(np.isnan(arr).sum())
    rec["inf_count"] = int(np.isinf(arr).sum())
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        rec["status"] = "bad_shape"
        return rec

    frame_delta = int(arr.shape[0]) - int(expected_frames)
    rec["expected_frames"] = int(expected_frames)
    rec["frame_delta"] = frame_delta
    if strict_frames and frame_delta != 0:
        rec["status"] = "bad_frame_count"
    elif rec["nan_count"] or rec["inf_count"]:
        rec["status"] = "non_finite"
    else:
        rec["status"] = "ok"

    feet = arr[:, [10, 11], :]
    rec["min_xyz"] = [float(x) for x in arr.min(axis=(0, 1))]
    rec["max_xyz"] = [float(x) for x in arr.max(axis=(0, 1))]
    rec["root_start_xyz"] = [float(x) for x in arr[0, 0]]
    rec["foot_min_z"] = float(feet[..., 2].min())
    rec["foot_mean_min_z_per_frame"] = float(feet[..., 2].min(axis=1).mean())
    rec["foot_max_z"] = float(feet[..., 2].max())
    return rec


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = Counter(row["status"] for row in records)
    ok_records = [row for row in records if row["status"] == "ok"]
    frame_delta_abs = [abs(int(row.get("frame_delta", 0))) for row in ok_records]
    foot_min = [float(row["foot_min_z"]) for row in ok_records if "foot_min_z" in row]
    return {
        "num_expected": len(records),
        "status_counts": dict(statuses),
        "complete": int(statuses.get("ok", 0)) == len(records),
        "frame_delta_abs_mean": float(np.mean(frame_delta_abs)) if frame_delta_abs else None,
        "frame_delta_abs_max": int(max(frame_delta_abs)) if frame_delta_abs else None,
        "foot_min_z_mean": float(np.mean(foot_min)) if foot_min else None,
        "foot_min_z_min": float(np.min(foot_min)) if foot_min else None,
        "foot_min_z_max": float(np.max(foot_min)) if foot_min else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-input-dir", required=True)
    parser.add_argument("--eval-info-json", default="ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--strict-frames", action="store_true")
    args = parser.parse_args()

    frame_map = expected_frame_map(args.eval_info_json)
    eval_input_dir = os.path.abspath(args.eval_input_dir)
    output_json = args.output_json or os.path.join(os.path.dirname(eval_input_dir), "mbench_eval_input_manifest.json")

    records = []
    for motion_id in sorted(frame_map):
        path = os.path.join(eval_input_dir, f"{motion_id}.npy")
        rec = array_record(path, frame_map[motion_id], strict_frames=args.strict_frames)
        rec["id"] = int(motion_id)
        records.append(rec)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_input_dir": eval_input_dir,
        "eval_info_json": args.eval_info_json,
        "strict_frames": bool(args.strict_frames),
        "summary": summarize(records),
        "records": records,
    }
    write_json(output_json, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"[validate-mbench] wrote {output_json}")


if __name__ == "__main__":
    main()

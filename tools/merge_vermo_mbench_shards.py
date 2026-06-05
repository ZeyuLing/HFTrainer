#!/usr/bin/env python3
"""Merge sharded ``export_vermo_mbench.py`` outputs into one MBench directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
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


def summarize(records: List[Dict[str, Any]], expected_count: int) -> Dict[str, Any]:
    statuses = Counter(record.get("status", "unknown") for record in records)
    ok = [record for record in records if record.get("status") == "ok"]
    frame_errors = [
        abs(int(record.get("pred_frames", 0)) - int(record.get("requested_frames", 0)))
        for record in ok
    ]
    foot_min = [
        float(record["joint_stats"]["foot_min_z"])
        for record in ok
        if isinstance(record.get("joint_stats"), dict) and "foot_min_z" in record["joint_stats"]
    ]
    return {
        "expected_count": int(expected_count),
        "num_records": len(records),
        "statuses": dict(statuses),
        "complete": int(statuses.get("ok", 0)) == int(expected_count),
        "frame_abs_error_mean": float(np.mean(frame_errors)) if frame_errors else None,
        "frame_abs_error_max": int(max(frame_errors)) if frame_errors else None,
        "foot_min_z_mean": float(np.mean(foot_min)) if foot_min else None,
        "foot_min_z_min": float(np.min(foot_min)) if foot_min else None,
        "foot_min_z_max": float(np.max(foot_min)) if foot_min else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=450)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shard_root = os.path.abspath(args.shard_root)
    output_dir = os.path.abspath(args.output_dir)
    eval_input_dir = os.path.join(output_dir, "mbench_eval_input")
    motion_dir = os.path.join(output_dir, "decoded_motion135")
    os.makedirs(eval_input_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)

    shard_dirs = [
        os.path.join(shard_root, name)
        for name in sorted(os.listdir(shard_root))
        if os.path.isdir(os.path.join(shard_root, name))
    ]
    records: List[Dict[str, Any]] = []
    seen = set()
    shard_manifests = []
    errors = []

    for shard_dir in shard_dirs:
        manifest_path = os.path.join(shard_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            errors.append({"shard": shard_dir, "error": "missing manifest"})
            continue
        manifest = load_json(manifest_path)
        shard_manifests.append(os.path.relpath(manifest_path, output_dir))
        for record in manifest.get("records", []):
            motion_id = int(record["id"])
            if motion_id in seen:
                errors.append({"id": motion_id, "error": "duplicate id", "shard": shard_dir})
                continue
            seen.add(motion_id)
            merged_record = dict(record)
            src_npy = os.path.join(shard_dir, record["npy_path"])
            dst_npy = os.path.join(eval_input_dir, f"{motion_id}.npy")
            if os.path.exists(src_npy):
                if args.force or not os.path.exists(dst_npy):
                    shutil.copy2(src_npy, dst_npy)
                merged_record["npy_path"] = os.path.relpath(dst_npy, output_dir)
            src_npz = os.path.join(shard_dir, record["motion135_path"])
            dst_npz = os.path.join(motion_dir, f"{motion_id}.npz")
            if os.path.exists(src_npz):
                if args.force or not os.path.exists(dst_npz):
                    shutil.copy2(src_npz, dst_npz)
                merged_record["motion135_path"] = os.path.relpath(dst_npz, output_dir)
            records.append(merged_record)

    records.sort(key=lambda row: int(row["id"]))
    missing = sorted(set(range(args.expected_count)) - seen)
    for motion_id in missing:
        records.append({"id": motion_id, "status": "missing"})
    records.sort(key=lambda row: int(row["id"]))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "shard_root": shard_root,
        "output_dir": output_dir,
        "shard_manifests": shard_manifests,
        "errors": errors,
        "summary": summarize(records, args.expected_count),
        "records": records,
    }
    write_json(os.path.join(output_dir, "manifest.json"), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"merge_errors": errors[:20], "num_errors": len(errors)}, ensure_ascii=False, indent=2))
    print(f"[merge-vermo-mbench] wrote {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()

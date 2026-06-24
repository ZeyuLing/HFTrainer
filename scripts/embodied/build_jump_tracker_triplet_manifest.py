#!/usr/bin/env python3
"""Build an embodied_viz triplet manifest for jump tracker comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    MESHES_BY_BODY,
    motion_to_robot_frames,
)


def _with_meshes(src_json: Path, dst_json: Path) -> Path:
    data = json.loads(src_json.read_text())
    bodies = []
    for body in data.get("bodies", []):
        name = body.get("name") if isinstance(body, dict) else str(body)
        meshes = body.get("meshes") if isinstance(body, dict) else None
        if meshes is None:
            meshes = [
                {"file": mesh, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
                for mesh in MESHES_BY_BODY.get(name, [])
            ]
        bodies.append({"name": name, "meshes": meshes})
    data["bodies"] = bodies
    data["num_bodies"] = len(bodies)
    data["num_frames"] = len(data.get("frames", []))
    data.setdefault("type", "robot_frames")
    data.setdefault("robot", "g1")
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(data))
    return dst_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    bench = args.bench_root.resolve()
    out_dir = args.out_dir.resolve()
    frames_dir = out_dir / "robot_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((bench / "manifest.json").read_text())
    height_rows = {
        row["id"]: row
        for row in json.loads((bench / "jump_tracker_height_comparison.json").read_text())
    }
    rows = []
    for idx, case in enumerate(manifest):
        stem = case["id"]
        ref_json = motion_to_robot_frames(
            bench / "proto_motion" / f"{stem}.motion",
            frames_dir / f"{stem}.reference.json",
        )
        proto_json = _with_meshes(
            bench / "proto_onnx_export" / f"{stem}.json",
            frames_dir / f"{stem}.protomotions.json",
        )
        hgpt_json = _with_meshes(
            bench / "hgpt_rollout_export" / f"{stem}.json",
            frames_dir / f"{stem}.humanoidgpt.json",
        )
        h = height_rows.get(stem, {})

        rows.append({
            "iteration": 0,
            "iteration_label": "jump",
            "prompt_id": stem,
            "prompt": case.get("caption", stem),
            "category": "jump",
            "difficulty": "agile",
            "sample_idx": idx,
            "columns": {
                "reference": {
                    "status": "ready",
                    "title": "Reference",
                    "path": str(ref_json),
                    "metrics": {
                        "pelvis_z_range": h.get("ref_z"),
                        "feet_clear_ratio": h.get("ref_clear"),
                    },
                },
                "protomotions": {
                    "status": "ready",
                    "title": "ProtoMotions",
                    "path": str(proto_json),
                    "metrics": {
                        "pelvis_z_range": h.get("proto_z"),
                        "feet_clear_ratio": h.get("proto_clear"),
                        "root_err_m": h.get("proto_root"),
                    },
                },
                "humanoidgpt": {
                    "status": "ready",
                    "title": "HumanoidGPT",
                    "path": str(hgpt_json),
                    "metrics": {
                        "pelvis_z_range": h.get("hgpt_z"),
                        "feet_clear_ratio": h.get("hgpt_clear"),
                        "root_err_m": h.get("hgpt_root"),
                    },
                },
            },
        })

    out_manifest = {
        "schema_version": 1,
        "project": "Jump Tracking: Reference vs ProtoMotions vs HumanoidGPT",
        "group_label": "set",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {"bench_root": str(bench)},
        "column_order": [
            {"key": "reference", "title": "Reference", "color": "raw"},
            {"key": "protomotions", "title": "ProtoMotions", "color": "opt"},
            {"key": "humanoidgpt", "title": "HumanoidGPT", "color": "track"},
        ],
        "rows": rows,
    }
    mp = out_dir / "manifest.json"
    mp.write_text(json.dumps(out_manifest, indent=2))
    print(f"MANIFEST_DONE {mp} rows={len(rows)}")


if __name__ == "__main__":
    main()

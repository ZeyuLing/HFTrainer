#!/usr/bin/env python3
"""Build a four-column PhysFlow manifest for KIMODO and tracker before/after.

Columns:
  1. KIMODO reference before optimization
  2. KIMODO reference after optimization
  3. Baseline tracker rollout on the optimized reference
  4. Fine-tuned tracker rollout on the same optimized reference

The resulting manifest is consumed by /physflow_triplet. Despite the route name,
the viewer now reads manifest.column_order and can render more than three columns.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from physflow_triplet_manifest import motion_to_robot_frames, _resolve_existing_path


def _robot_frames_root_metrics(robot_frames_path: Path) -> dict:
    data = json.loads(robot_frames_path.read_text())
    roots = []
    for frame in data.get("frames", []):
        body_pos = frame.get("body_pos") or []
        if body_pos:
            roots.append(body_pos[0])
    fps = float(data.get("fps", 30) or 30)
    if len(roots) < 2:
        return {
            "root_displacement_m": 0.0,
            "root_xy_path_m": 0.0,
            "avg_xy_speed_mps": 0.0,
        }
    xy_path = 0.0
    for prev, cur in zip(roots, roots[1:]):
        dx = float(cur[0]) - float(prev[0])
        dy = float(cur[1]) - float(prev[1])
        xy_path += (dx * dx + dy * dy) ** 0.5
    dx = float(roots[-1][0]) - float(roots[0][0])
    dy = float(roots[-1][1]) - float(roots[0][1])
    duration = float(len(roots) / fps)
    return {
        "root_displacement_m": float((dx * dx + dy * dy) ** 0.5),
        "root_xy_path_m": float(xy_path),
        "avg_xy_speed_mps": float(xy_path / max(duration, 1e-6)),
    }


def _load_records(run_dir: Path) -> dict[tuple[str, int], dict]:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    out = {}
    for record in summary.get("records", []):
        if record.get("status") != "scored":
            continue
        key = (
            str(record.get("prompt_id", record.get("output_stem", ""))),
            int(record.get("sample_idx", 0) or 0),
        )
        out[key] = record
    return out


def _reference_column(record: dict, out_dir: Path, key: str, title: str) -> dict:
    stem = record["output_stem"]
    motion_path = _resolve_existing_path(record["motion_path"], Path.cwd())
    ref_path = out_dir / "robot_frames_reference" / f"{stem}.{key}.json"
    if not ref_path.is_file():
        ref_path = motion_to_robot_frames(motion_path, ref_path)
    else:
        ref_path = ref_path.resolve()
    root_metrics = _robot_frames_root_metrics(ref_path)
    kin = record.get("kinematic") or {}
    joint_vel_max = kin.get("joint_vel_max")
    metrics = {
        "source_score": record.get("adversarial_score"),
        "source_error": record.get("max_joint_error_rad"),
        "completion": record.get("completion_ratio"),
        "fall": record.get("fall_detected"),
        "root_displacement_m": root_metrics["root_displacement_m"],
        "root_xy_path_m": root_metrics["root_xy_path_m"],
        "avg_xy_speed_mps": root_metrics["avg_xy_speed_mps"],
        "joint_vel_max": joint_vel_max,
    }
    if joint_vel_max is not None and float(joint_vel_max) >= 30.0:
        metrics["artifact_flag"] = "joint_spike"
    foot_skate_speed = kin.get("foot_skate_speed")
    if foot_skate_speed is not None:
        metrics["foot_skate_speed"] = foot_skate_speed
    return {
        "status": "ready",
        "title": title,
        "path": str(ref_path),
        "metrics": metrics,
    }


def _tracker_column(record: dict, title: str) -> dict:
    return {
        "status": "ready",
        "title": title,
        "path": str(_resolve_existing_path(record["robot_json_path"], Path.cwd())),
        "metrics": {
            "completion": record.get("completion_ratio"),
            "fall": record.get("fall_detected"),
            "joint_err_rad": record.get("max_joint_error_rad"),
            "root_traj_err_m": record.get("root_trajectory_error_mean_m"),
            "root_height_final": record.get("root_height_final"),
        },
    }


def build(
    kimodo_before_dir: Path,
    kimodo_after_dir: Path,
    tracker_before_dir: Path,
    tracker_after_dir: Path,
    out_dir: Path,
) -> Path:
    kimodo_before = _load_records(kimodo_before_dir)
    kimodo_after = _load_records(kimodo_after_dir)
    tracker_before = _load_records(tracker_before_dir)
    tracker_after = _load_records(tracker_after_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, key in enumerate(sorted(kimodo_after)):
        after_ref = kimodo_after[key]
        before_ref = kimodo_before.get(key)
        before_tracker = tracker_before.get(key)
        after_tracker = tracker_after.get(key)
        stem = after_ref.get("output_stem", key[0])

        columns = {}
        if before_ref and before_ref.get("motion_path"):
            columns["kimodo_before"] = _reference_column(
                before_ref,
                out_dir,
                "kimodo_before",
                "KIMODO before",
            )
        else:
            columns["kimodo_before"] = {
                "status": "missing",
                "title": "KIMODO before",
                "path": "",
                "metrics": {},
            }

        columns["kimodo_after"] = _reference_column(
            after_ref,
            out_dir,
            "kimodo_after",
            "KIMODO after",
        )

        if before_tracker and before_tracker.get("robot_json_path"):
            columns["tracker_before"] = _tracker_column(before_tracker, "Tracker before")
        else:
            columns["tracker_before"] = {
                "status": "missing",
                "title": "Tracker before",
                "path": "",
                "metrics": {},
            }

        if after_tracker and after_tracker.get("robot_json_path"):
            columns["tracker_after"] = _tracker_column(after_tracker, "Tracker after")
        else:
            columns["tracker_after"] = {
                "status": "missing",
                "title": "Tracker after",
                "path": "",
                "metrics": {},
            }

        rows.append(
            {
                "iteration": idx,
                "iteration_label": f"Case {idx:02d}  ·  {after_ref.get('prompt_id', stem)}",
                "prompt_id": after_ref.get("prompt_id", stem),
                "prompt": after_ref.get("prompt", ""),
                "category": after_ref.get("category", ""),
                "difficulty": after_ref.get("difficulty"),
                "seed": after_ref.get("seed"),
                "sample_idx": after_ref.get("sample_idx"),
                "columns": columns,
            }
        )

    manifest = {
        "schema_version": 1,
        "project": "PhysFlow KIMODO-G1 — Four-way Comparison",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {
            "kimodo_before": str(kimodo_before_dir),
            "kimodo_after": str(kimodo_after_dir),
            "tracker_before": str(tracker_before_dir),
            "tracker_after": str(tracker_after_dir),
        },
        "group_label": "case",
        "column_order": [
            {"key": "kimodo_before", "title": "KIMODO before", "color": "raw"},
            {"key": "kimodo_after", "title": "KIMODO after", "color": "opt"},
            {"key": "tracker_before", "title": "Tracker before", "color": "track"},
            {"key": "tracker_after", "title": "Tracker after", "color": "track-after"},
        ],
        "rows": rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kimodo-before-dir", type=Path, required=True)
    parser.add_argument("--kimodo-after-dir", type=Path, required=True)
    parser.add_argument("--tracker-before-dir", type=Path, required=True)
    parser.add_argument("--tracker-after-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out = build(
        kimodo_before_dir=args.kimodo_before_dir.resolve(),
        kimodo_after_dir=args.kimodo_after_dir.resolve(),
        tracker_before_dir=args.tracker_before_dir.resolve(),
        tracker_after_dir=args.tracker_after_dir.resolve(),
        out_dir=args.out_dir.resolve(),
    )
    print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a PhysFlow-triplet manifest that compares tracker variants.

Reuses the existing /physflow_triplet web viewer (Three.js real G1 STL meshes,
served by motion_annot_web/embodied_viz/app.py). The 3 hardcoded column slots
(raw_reference / optimized_reference / tracked_rollout) are repurposed via the
per-column `title` override, e.g.:

    Reference Target | Baseline Tracker | A_e609 Tracker (ours)

Each tracker column points at the run's g1_tracker_json/<stem>.json rollout and
carries the run's completion/fall/error metrics. The reference target is built
from the KIMODO .motion file via forward kinematics (motion_to_robot_frames).

Open in the running 8095 server (no restart needed):
    /physflow_triplet?manifest=<out_dir>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Reuse the FK reference converter + path resolver from the existing builder.
from physflow_triplet_manifest import motion_to_robot_frames, _resolve_existing_path


def _load_records(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    out = {}
    for r in summary.get("records", []):
        if r.get("status") != "scored" or not r.get("robot_json_path"):
            continue
        out[r["output_stem"]] = r
    return out


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


def build(col_specs: list[tuple[str, Path, str, str]], out_dir: Path, iteration: int) -> Path:
    """col_specs: list of (slot_key, run_dir, title, kind) for the 3 viewer slots.

    slot_key must be one of raw_reference / optimized_reference / tracked_rollout.
    kind == "reference" -> render FK target from the run's .motion files.
    kind == "tracker"   -> render the run's g1_tracker_json rollout + metrics.
    Rows are keyed by output_stem; the first run determines the row set.
    """
    all_records = [(slot, _load_records(rd), title, kind) for slot, rd, title, kind in col_specs]
    base_records = all_records[0][1]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (stem, base_rec) in enumerate(sorted(base_records.items())):
        columns = {}
        for slot, recs, title, kind in all_records:
            rec = recs.get(stem)
            if rec is None:
                columns[slot] = {"status": "missing", "title": title, "path": "", "metrics": {}}
            elif kind == "reference":
                ref_path = motion_to_robot_frames(
                    _resolve_existing_path(rec["motion_path"], Path.cwd()),
                    out_dir / "robot_frames_reference" / f"{stem}.reference.json",
                )
                columns[slot] = {
                    "status": "ready", "title": title, "path": str(ref_path),
                    "metrics": {"frames": rec.get("duration_sec")},
                }
            else:
                columns[slot] = _tracker_column(rec, title)
        ref_rec = base_rec
        rows.append({
            "iteration": idx,
            "iteration_label": f"Case {idx:02d}  ·  {ref_rec.get('prompt_id', stem)}",
            "prompt_id": ref_rec.get("prompt_id", stem),
            "prompt": ref_rec.get("prompt", ""),
            "category": ref_rec.get("category", ""),
            "difficulty": ref_rec.get("difficulty"),
            "seed": ref_rec.get("seed"),
            "sample_idx": ref_rec.get("sample_idx"),
            "columns": columns,
        })

    manifest = {
        "schema_version": 1,
        "project": "PhysFlow KIMODO-G1 — Tracker Comparison",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {slot: str(rd) for slot, rd, _t, _k in col_specs},
        "group_label": "case",
        "rows": rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


SLOTS = ["raw_reference", "optimized_reference", "tracked_rollout"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--col", action="append", default=[], metavar="RUN_DIR:TITLE:KIND",
                   help="One per slot (max 3). KIND in {reference,tracker}. "
                        "RUN_DIR is a *_score output dir with summary.json.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iteration", type=int, default=0)
    args = p.parse_args()

    if not 2 <= len(args.col) <= 3:
        raise SystemExit("Provide 2 or 3 --col RUN_DIR:TITLE:KIND specs.")
    specs = []
    for i, raw in enumerate(args.col):
        run_dir, title, kind = raw.split(":", 2)
        specs.append((SLOTS[i], Path(run_dir).resolve(), title, kind))
    out = build(specs, args.out_dir.resolve(), args.iteration)
    print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Roll out an existing PhysFlow .motion run with a specified G1 tracker ONNX.

The output mirrors ``physflow_coevolve_viz.py`` summaries closely enough that
``cursor_tracker_compare_manifest.py`` can build a web view with columns like:

    Reference target | Tracker before | Tracker after
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "ref_repo" / "ProtoMotions"
for path in (ROOT, PROTO, ROOT / "scripts" / "embodied"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.embodied.physflow_g1_scoring import compute_g1_adversarial_score  # noqa: E402
from scripts.embodied.run_g1_rl_tracker_export import (  # noqa: E402
    DEFAULT_MJCF,
    parse_body_mesh_mapping,
    simulate_and_export,
)


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (base / p).resolve()


def _expected_frames(motion_path: Path, onnx_path: Path) -> int:
    import yaml
    from deployment.motion_utils import MotionPlayer

    with open(str(onnx_path).replace(".onnx", ".yaml")) as f:
        meta = yaml.safe_load(f)
    control_dt = meta["timing"]["control_dt"]
    return int(MotionPlayer(str(motion_path), control_dt=control_dt).total_frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True, help="Run dir with summary.json + proto/*.motion")
    ap.add_argument("--onnx", required=True, help="Tracker unified_pipeline.onnx")
    ap.add_argument("--out-dir", required=True, help="Output dir for json rollout + summary.json")
    ap.add_argument("--label", default="tracker")
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    source_run = _resolve(args.source_run)
    onnx_path = _resolve(args.onnx)
    mjcf_path = _resolve(args.mjcf)
    out_dir = _resolve(args.out_dir)
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads((source_run / "summary.json").read_text())
    records = source_summary.get("records", [])
    if args.limit > 0:
        records = records[: args.limit]

    body_mesh_mapping = parse_body_mesh_mapping(mjcf_path)
    scored_records = []
    for idx, record in enumerate(records):
        out = dict(record)
        stem = out.get("output_stem") or f"e{idx:04d}"
        motion_path = _resolve(out.get("motion_path") or source_run / "proto" / f"{stem}.motion")
        out_json = json_dir / f"{stem}.json"
        out["tracker_label"] = args.label
        out["g1_onnx_path"] = str(onnx_path)
        try:
            total_frames = _expected_frames(motion_path, onnx_path)
            stats = simulate_and_export(
                onnx_path=str(onnx_path),
                motion_file=str(motion_path),
                output_json_path=str(out_json),
                mjcf_path=str(mjcf_path),
                body_mesh_mapping=body_mesh_mapping,
                subsample_factor=args.subsample,
            )
            completion = float(stats["total_steps"] / max(total_frames, 1))
            out.update(
                {
                    "status": "scored",
                    "motion_path": str(motion_path),
                    "robot_json_path": str(out_json),
                    "completion_ratio": completion,
                    "max_joint_error_rad": float(stats.get("max_joint_error_rad", 0.0)),
                    "fall_detected": bool(stats.get("fall_detected", False)),
                    "root_height_final": stats.get("root_height_final"),
                    "root_trajectory_error_mean_m": stats.get("root_trajectory_error_mean_m"),
                    "root_trajectory_error_final_m": stats.get("root_trajectory_error_final_m"),
                    "root_displacement_ref_m": stats.get("root_displacement_ref_m"),
                    "root_displacement_track_m": stats.get("root_displacement_track_m"),
                    "root_displacement_error_m": stats.get("root_displacement_error_m"),
                }
            )
            out["adversarial_score"] = float(
                compute_g1_adversarial_score(
                    completion=completion,
                    max_joint_error_rad=out["max_joint_error_rad"],
                    root_trajectory_error_mean_m=float(out.get("root_trajectory_error_mean_m") or 0.0),
                    root_displacement_error_m=float(out.get("root_displacement_error_m") or 0.0),
                    fall_detected=out["fall_detected"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            out.update(
                {
                    "status": "failed",
                    "motion_path": str(motion_path),
                    "robot_json_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        scored_records.append(out)
        print(
            f"[{args.label}] {idx + 1:03d}/{len(records):03d} {stem} "
            f"{out['status']} comp={out.get('completion_ratio')} "
            f"err={out.get('max_joint_error_rad')}",
            flush=True,
        )

    summary = {
        "source_run": str(source_run),
        "tracker_label": args.label,
        "g1_onnx_path": str(onnx_path),
        "records": scored_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[score-tracker] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

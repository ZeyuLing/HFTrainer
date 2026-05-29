#!/usr/bin/env python3
"""Select best PhysFlow/KIMODO candidates across scored runs.

This creates a synthetic run directory with a summary.json that keeps, for each
prompt/sample pair, the record with the lowest adversarial score. It is used as
the first executable optimized-reference baseline before true checkpoint
fine-tuning is available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embodied.physflow_g1_scoring import DEFAULT_G1_SCORE_CONFIG, score_record


def key(record: dict) -> tuple[str, int]:
    return (str(record.get("prompt_id", record.get("output_stem", ""))), int(record.get("sample_idx", 0)))


def score(record: dict) -> float:
    if "root_trajectory_error_mean_m" in record or "root_displacement_error_m" in record:
        return score_record(record)
    value = record.get("adversarial_score")
    if value is None:
        value = 1.0 - float(record.get("completion_ratio", 0.0)) + float(record.get("max_joint_error_rad", 999.0))
    return float(value)


def resolve_path(path_value: str, run_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return run_dir / path


def copy_record_artifacts(record: dict, source_run_dir: Path, out_dir: Path) -> dict:
    copied = dict(record)
    artifact_fields = {
        "npz_path": out_dir / "kimodo_raw",
        "csv_path": out_dir / "kimodo_raw",
        "motion_path": out_dir / "proto",
        "robot_json_path": out_dir / "g1_tracker_json",
    }
    for field, target_dir in artifact_fields.items():
        value = record.get(field)
        if not value:
            continue
        src = resolve_path(value, source_run_dir)
        if not src.is_file():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[field] = str(dst)
    copied["selected_from_run"] = str(source_run_dir)
    copied["adversarial_score"] = score(record)
    copied["selection_method"] = "best_position_aware_adversarial_score"
    return copied


def build_bestof(run_dirs: list[Path], out_dir: Path) -> Path:
    best: dict[tuple[str, int], tuple[dict, Path]] = {}
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        for record in summary.get("records", []):
            if record.get("status") != "scored":
                continue
            pair_key = key(record)
            if pair_key not in best or score(record) < score(best[pair_key][0]):
                best[pair_key] = (record, run_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        copy_record_artifacts(record, run_dir, out_dir)
        for record, run_dir in sorted(best.values(), key=lambda item: key(item[0]))
    ]
    scored = [r for r in records if r.get("status") == "scored"]
    summary = {
        "num_records": len(records),
        "num_scored": len(scored),
        "num_errors": 0,
        "num_falls": sum(1 for r in scored if r.get("fall_detected")),
        "mean_completion": sum(float(r.get("completion_ratio", 0.0)) for r in scored) / len(scored) if scored else 0.0,
        "mean_joint_error": sum(float(r.get("max_joint_error_rad", 0.0)) for r in scored) / len(scored) if scored else 0.0,
        "mean_root_displacement_ref_m": sum(float(r.get("root_displacement_ref_m", 0.0)) for r in scored) / len(scored) if scored else 0.0,
        "mean_root_displacement_track_m": sum(float(r.get("root_displacement_track_m", 0.0)) for r in scored) / len(scored) if scored else 0.0,
        "mean_root_trajectory_error_m": sum(float(r.get("root_trajectory_error_mean_m", 0.0)) for r in scored) / len(scored) if scored else 0.0,
        "selection_method": "best_position_aware_adversarial_score",
        "score_terms": DEFAULT_G1_SCORE_CONFIG.to_dict(),
        "source_runs": [str(p) for p in run_dirs],
        "records": records,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_bestof([p.resolve() for p in args.run_dir], args.out_dir.resolve()))


if __name__ == "__main__":
    main()

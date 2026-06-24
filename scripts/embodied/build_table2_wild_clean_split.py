#!/usr/bin/env python3
"""Build the cleaned Wild-G1 split and recompute cached tracker rows.

The original Wild-G1 holdout intentionally contains hard in-the-wild references,
including motions whose names imply missing scene geometry such as stairs,
obstacles, ladders, slopes, or crawl spaces. Those cases are useful diagnostics
but are not a fair main-table measure for flat-ground humanoid tracking.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDE_KEYWORDS = (
    "stair",
    "stairs",
    "upstairs",
    "downstairs",
    "ladder",
    "climb",
    "obstacle",
    "obstacles",
    "slope",
    "step_stone",
    "step-stone",
    "stepstone",
    "crawl",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.fmean(values) if values else float("nan")


def _motion_key_from_idx(idx: int) -> str:
    return f"h{idx:03d}_gen"


def _score_rows(path: Path, keywords: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_json(path)["rows"]
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        text = f"{row.get('g1_path', '')} {row.get('name', '')}".lower()
        matched = [kw for kw in keywords if kw in text]
        out = dict(row)
        out["motion_key"] = _motion_key_from_idx(int(row["idx"]))
        out["exclude_keywords"] = matched
        if matched:
            excluded.append(out)
        else:
            kept.append(out)
    return kept, excluded


def _proto_metrics(path: Path, keep_keys: set[str]) -> dict[str, Any]:
    rows = _read_json(path)["rows"]
    selected = []
    for row in rows:
        idx = int(row.get("global_index", row.get("motion_id")))
        key = _motion_key_from_idx(idx)
        if key not in keep_keys:
            continue
        metrics = row["metrics"]["aggregator_row"]
        selected.append(metrics)
    return {
        "num_motions": len(selected),
        "success_rate": _mean([float(r["success"]) for r in selected]),
        "e_g_mpjpe_mm": _mean([float(r["aligned_global_mpjpe_mm"]) for r in selected]),
        "e_r_mpjpe_mm": _mean([float(r["local_mpjpe_mm"]) for r in selected]),
        "e_vel_mps": _mean([
            float(row["metrics"]["visual_recomputed"]["local_mpjve_mps"])
            for row in rows
            if _motion_key_from_idx(int(row.get("global_index", row.get("motion_id")))) in keep_keys
        ]),
        "e_acc_mps2": _mean([
            float(row["metrics"]["visual_recomputed"]["local_mpjae_mps2"])
            for row in rows
            if _motion_key_from_idx(int(row.get("global_index", row.get("motion_id")))) in keep_keys
        ]),
    }


def _opentrack_metrics(path: Path, keep_keys: set[str]) -> dict[str, Any]:
    rows = [r for r in _read_json(path)["motions"] if str(r["motion"]) in keep_keys]
    return {
        "num_motions": len(rows),
        "success_rate": _mean([1.0 if r.get("success") else 0.0 for r in rows]),
        "paper_success_rate": _mean([1.0 if r.get("paper_success") else 0.0 for r in rows]),
        "strict_success_rate": _mean([1.0 if r.get("strict_success") else 0.0 for r in rows]),
        "e_g_mpjpe_mm": _mean([float(r["xy_aligned_mpjpe_mm"]) for r in rows]),
        "e_r_mpjpe_mm": _mean([float(r["local_mpjpe_mm"]) for r in rows]),
        "e_vel_mps": _mean([float(r["local_mpjve_mps"]) for r in rows]),
        "e_acc_mps2": _mean([float(r["local_mpjae_mps2"]) for r in rows]),
    }


def _hgpt_metrics(path: Path, keep_keys: set[str], complete_thresh: float) -> dict[str, Any]:
    rows = {str(k): v for k, v in _read_json(path)["motions"].items()}
    selected = [rows[k] for k in sorted(keep_keys) if k in rows]
    success = [
        1.0 if float(r.get("length_ratio", 0.0)) >= complete_thresh else 0.0
        for r in selected
    ]
    return {
        "num_motions": len(selected),
        "success_rate": _mean(success),
        "completion": _mean([float(r["length_ratio"]) for r in selected]),
        "e_g_mpjpe_mm": _mean([float(r["root_pos_err_mm"]) for r in selected]),
        "e_r_mpjpe_mm": _mean([float(r["kpt_pos_mae"]) * 1000.0 for r in selected]),
        "e_vel_mps": _mean([float(r["root_vel_err_mms"]) / 1000.0 for r in selected]),
        "joint_pos_mae_rad": _mean([float(r["joint_pos_mae"]) for r in selected]),
        "joint_vel_mae_radps": _mean([float(r["joint_vel_mae"]) for r in selected]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wild-score", type=Path, default=Path("output/heldout_frozen_score/heldout_score.json"))
    ap.add_argument(
        "--proto-manifest",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/protomotions_g1_bones/viz/wild/manifest.json"),
    )
    ap.add_argument(
        "--any2track-summary",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/any2track_open/wild_v3_xyalign/summary.json"),
    )
    ap.add_argument(
        "--hgpt-summary",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/humanoid_gpt/wild_v2/summary.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/wild_g1_clean"))
    ap.add_argument("--complete-thresh", type=float, default=0.9)
    ap.add_argument("--keywords", nargs="*", default=list(DEFAULT_EXCLUDE_KEYWORDS))
    args = ap.parse_args()

    keywords = tuple(str(k).lower() for k in args.keywords)
    kept, excluded = _score_rows(args.wild_score, keywords)
    keep_keys = {str(r["motion_key"]) for r in kept}

    payload = {
        "schema_version": 1,
        "source": str(args.wild_score),
        "policy": (
            "Main Wild-G1 excludes references whose names imply unavailable scene geometry "
            "or terrain assumptions; excluded cases remain diagnostic stress tests."
        ),
        "exclude_keywords": list(keywords),
        "num_total": len(kept) + len(excluded),
        "num_main": len(kept),
        "num_excluded": len(excluded),
        "main_manifest": sorted(keep_keys),
        "excluded_manifest": sorted(str(r["motion_key"]) for r in excluded),
        "metrics": {
            "ProtoMotions G1": _proto_metrics(args.proto_manifest, keep_keys),
            "Any2Track/OpenTrack": _opentrack_metrics(args.any2track_summary, keep_keys),
            "Humanoid-GPT": _hgpt_metrics(args.hgpt_summary, keep_keys, args.complete_thresh),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps(payload["main_manifest"], indent=2) + "\n")
    (args.out_dir / "excluded_manifest.json").write_text(json.dumps(payload["excluded_manifest"], indent=2) + "\n")
    (args.out_dir / "main_rows.json").write_text(json.dumps(kept, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "excluded_rows.json").write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n")

    print(f"total={payload['num_total']} main={payload['num_main']} excluded={payload['num_excluded']}")
    for method, metrics in payload["metrics"].items():
        print(method, json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

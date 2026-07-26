#!/usr/bin/env python3
"""Export paired MotionRepair human-study preferences from score_m2m.db."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


RUN_PAIRS = ((3049, 3051), (3065, 3059), (3071, 3066))
OURS_MODEL = "HyMotion-M2M+MoGenDIT_QCSelect"
BASELINE_MODEL = "StableMotion"
DIMENSIONS = (
    "fluency",
    "naturalness",
    "pose_accuracy",
    "horizontal_support",
    "vertical_support",
    "joint_anomaly",
    "penetration",
    "repair_effectiveness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_annotations(
    conn: sqlite3.Connection, run_id: int
) -> dict[tuple[int, str], dict[str, object]]:
    rows = conn.execute(
        """
        SELECT t.sample_idx, t.prompt_id, t.model_name, a.annotator, a.scores_json
        FROM score_tasks AS t
        JOIN score_annotations AS a ON a.score_task_id = t.id
        WHERE t.eval_run_id = ?
        """,
        (run_id,),
    )
    result: dict[tuple[int, str], dict[str, object]] = {}
    for sample_idx, prompt_id, model_name, annotator, scores_json in rows:
        key = (int(sample_idx), str(annotator))
        if key in result:
            raise RuntimeError(f"duplicate annotation for run {run_id}: {key}")
        result[key] = {
            "prompt_id": prompt_id,
            "model_name": model_name,
            "scores": json.loads(scores_json),
        }
    return result


def main() -> None:
    args = parse_args()
    counts = {dimension: Counter({"better": 0, "same": 0, "worse": 0}) for dimension in DIMENSIONS}
    means = {dimension: {"ours": 0.0, "baseline": 0.0} for dimension in DIMENSIONS}
    valid_pairs = 0
    excluded_data_issues = 0
    pair_breakdown: list[dict[str, int]] = []

    with sqlite3.connect(args.db) as conn:
        for ours_run, baseline_run in RUN_PAIRS:
            ours = load_annotations(conn, ours_run)
            baseline = load_annotations(conn, baseline_run)
            common = sorted(set(ours) & set(baseline))
            pair_valid = 0
            pair_excluded = 0

            for key in common:
                ours_row = ours[key]
                baseline_row = baseline[key]
                if ours_row["model_name"] != OURS_MODEL:
                    raise RuntimeError(f"unexpected ours model in run {ours_run}")
                if baseline_row["model_name"] != BASELINE_MODEL:
                    raise RuntimeError(f"unexpected baseline model in run {baseline_run}")
                if ours_row["prompt_id"] != baseline_row["prompt_id"]:
                    raise RuntimeError(f"prompt mismatch for runs {ours_run}/{baseline_run}: {key}")

                ours_scores = ours_row["scores"]
                baseline_scores = baseline_row["scores"]
                if ours_scores.get("_data_issue") or baseline_scores.get("_data_issue"):
                    excluded_data_issues += 1
                    pair_excluded += 1
                    continue

                missing = [
                    dimension
                    for dimension in DIMENSIONS
                    if dimension not in ours_scores or dimension not in baseline_scores
                ]
                if missing:
                    raise RuntimeError(f"missing {missing} for runs {ours_run}/{baseline_run}: {key}")

                valid_pairs += 1
                pair_valid += 1
                for dimension in DIMENSIONS:
                    ours_score = float(ours_scores[dimension])
                    baseline_score = float(baseline_scores[dimension])
                    means[dimension]["ours"] += ours_score
                    means[dimension]["baseline"] += baseline_score
                    outcome = (
                        "better"
                        if ours_score > baseline_score
                        else "worse"
                        if ours_score < baseline_score
                        else "same"
                    )
                    counts[dimension][outcome] += 1

            pair_breakdown.append(
                {
                    "ours_run": ours_run,
                    "baseline_run": baseline_run,
                    "matched_annotations": len(common),
                    "valid_pairs": pair_valid,
                    "excluded_data_issues": pair_excluded,
                }
            )

    if valid_pairs != 231 or excluded_data_issues != 2:
        raise RuntimeError(
            f"expected 231 valid pairs and 2 data issues, got {valid_pairs} and {excluded_data_issues}"
        )

    summary = {
        "database": str(args.db),
        "ours_model": OURS_MODEL,
        "baseline_model": BASELINE_MODEL,
        "run_pairs": pair_breakdown,
        "valid_pairs": valid_pairs,
        "excluded_data_issues": excluded_data_issues,
        "dimensions": {},
    }
    for dimension in DIMENSIONS:
        dimension_counts = counts[dimension]
        if sum(dimension_counts.values()) != valid_pairs:
            raise RuntimeError(f"incomplete count for {dimension}")
        summary["dimensions"][dimension] = {
            "better_same_worse": [
                dimension_counts["better"],
                dimension_counts["same"],
                dimension_counts["worse"],
            ],
            "ours_mean": means[dimension]["ours"] / valid_pairs,
            "baseline_mean": means[dimension]["baseline"] / valid_pairs,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

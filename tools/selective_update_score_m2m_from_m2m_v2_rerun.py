#!/usr/bin/env python3
"""Selectively replace score_m2m paths when a new M2M v2 rerun improves physics.

The old NPZ files are never deleted. In --apply mode the score DB is copied and
the previous per-row paths are also stored in a backup table before updates.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TARGETS = {
    ("E14", "L"),
    ("E14", "M"),
    ("E15", "default"),
    ("E8", "D"),
}
MODELS = {"uncond_local", "uncond_global"}

PHYSICS_TERMS = {
    ("E14", "L"): (
        ("jitter_pos", 1000.0),
        ("foot_skating_ratio", 0.35),
        ("boundary_accel_jump", 70.0),
    ),
    ("E14", "M"): (
        ("jitter_pos", 1000.0),
        ("foot_skating_ratio", 0.35),
        ("boundary_accel_jump", 70.0),
    ),
    ("E15", "default"): (
        ("jitter_pos", 800.0),
        ("foot_skating_ratio", 0.30),
        ("boundary_accel_jump", 8.0),
    ),
    ("E8", "D"): (
        ("loop_position_error", 0.05),
        ("jitter_pos", 500.0),
        ("foot_skating_ratio", 0.30),
        ("boundary_accel_jump_loop", 8.0),
    ),
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def physics_score(task_id: str, setting: str, metrics: dict[str, Any]) -> float | None:
    terms = PHYSICS_TERMS[(task_id, setting)]
    vals: list[float] = []
    for key, denom in terms:
        value = metric_value(metrics, key)
        if value is None:
            return None
        vals.append(value / denom)
    return float(sum(vals))


def load_new_samples(rerun_root: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    samples: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for path in sorted(rerun_root.rglob("eval_v2_*.json")):
        nested = load_json(path)
        for model_name, model_block in nested.items():
            if model_name not in MODELS or not isinstance(model_block, dict):
                continue
            for task_key, entry in model_block.get("tasks", {}).items():
                task_id = entry.get("task_id")
                setting = entry.get("setting")
                if (task_id, setting) not in TARGETS:
                    continue
                for sample in entry.get("per_sample", []):
                    idx = sample.get("_sample_idx")
                    if idx is None:
                        continue
                    key = (model_name, task_id, setting, int(idx))
                    samples[key] = sample
    return samples


def load_old_metrics(eval_db: Path, keys: list[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
    if not keys:
        return {}
    conn = sqlite3.connect(str(eval_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for run_id, sample_idx in keys:
        row = cur.execute(
            "SELECT metrics_json FROM sample_results WHERE eval_run_id=? AND sample_idx=?",
            (run_id, sample_idx),
        ).fetchone()
        if row:
            out[(run_id, sample_idx)] = json.loads(row["metrics_json"])
    conn.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-root", required=True, type=Path)
    parser.add_argument("--score-db", default="motion_annot_web/score_m2m/score_m2m.db", type=Path)
    parser.add_argument("--eval-db", default="motion_annot_web/eval_dashboard/eval_dashboard.db", type=Path)
    parser.add_argument("--report", default=None, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-delta", type=float, default=1e-9)
    args = parser.parse_args()

    new_samples = load_new_samples(args.rerun_root)
    if not new_samples:
        raise SystemExit(f"No target samples found under {args.rerun_root}")

    conn = sqlite3.connect(str(args.score_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, eval_run_id, sample_idx, task_id, setting, model_name,
               gen_motion_path, num_frames
        FROM score_tasks
        WHERE task_id IN ('E14', 'E15', 'E8')
          AND model_name IN ('uncond_local', 'uncond_global')
          AND ((task_id='E14' AND setting IN ('L','M'))
            OR (task_id='E15' AND setting='default')
            OR (task_id='E8' AND setting='D'))
        ORDER BY task_id, setting, model_name, sample_idx
        """
    ).fetchall()

    old_metrics = load_old_metrics(
        args.eval_db,
        [(int(r["eval_run_id"]), int(r["sample_idx"])) for r in rows],
    )

    decisions: list[dict[str, Any]] = []
    counts = Counter()
    improved_by_group = Counter()
    missing_new = 0
    missing_metrics = 0

    for row in rows:
        key = (row["model_name"], row["task_id"], row["setting"], int(row["sample_idx"]))
        sample = new_samples.get(key)
        if sample is None:
            missing_new += 1
            continue
        old_m = old_metrics.get((int(row["eval_run_id"]), int(row["sample_idx"])))
        if old_m is None:
            missing_metrics += 1
            continue
        old_score = physics_score(row["task_id"], row["setting"], old_m)
        new_score = physics_score(row["task_id"], row["setting"], sample)
        if old_score is None or new_score is None:
            missing_metrics += 1
            continue

        improved = new_score + args.min_delta < old_score
        counts["total_compared"] += 1
        if improved:
            counts["improved"] += 1
            improved_by_group[(row["task_id"], row["setting"], row["model_name"])] += 1
        else:
            counts["kept_old"] += 1

        decisions.append({
            "score_task_id": int(row["id"]),
            "task_id": row["task_id"],
            "setting": row["setting"],
            "model_name": row["model_name"],
            "sample_idx": int(row["sample_idx"]),
            "old_eval_run_id": int(row["eval_run_id"]),
            "old_path": row["gen_motion_path"],
            "new_path": sample.get("_npz_path"),
            "old_physics_score": old_score,
            "new_physics_score": new_score,
            "delta": old_score - new_score,
            "improved": improved,
            "old_metrics": {k: metric_value(old_m, k) for k, _ in PHYSICS_TERMS[(row["task_id"], row["setting"])]},
            "new_metrics": {k: metric_value(sample, k) for k, _ in PHYSICS_TERMS[(row["task_id"], row["setting"])]},
            "new_num_frames": sample.get("_num_frames"),
        })

    report = {
        "rerun_root": str(args.rerun_root),
        "score_db": str(args.score_db),
        "eval_db": str(args.eval_db),
        "applied": bool(args.apply),
        "counts": dict(counts),
        "missing_new": missing_new,
        "missing_metrics": missing_metrics,
        "improved_by_group": {
            "|".join(map(str, k)): v for k, v in sorted(improved_by_group.items())
        },
        "decisions": decisions,
    }

    if args.apply:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.score_db.with_suffix(args.score_db.suffix + f".bak_{stamp}_m2m_v2_selective")
        shutil.copy2(args.score_db, backup_path)
        backup_table = f"score_task_path_backup_{stamp}_m2m_v2_selective"
        cur.execute(
            f"""
            CREATE TABLE {backup_table} (
                score_task_id INTEGER PRIMARY KEY,
                eval_run_id INTEGER,
                sample_idx INTEGER,
                task_id TEXT,
                setting TEXT,
                model_name TEXT,
                old_gen_motion_path TEXT,
                new_gen_motion_path TEXT,
                old_physics_score REAL,
                new_physics_score REAL,
                updated_at TEXT
            )
            """
        )
        for d in decisions:
            if not d["improved"] or not d.get("new_path"):
                continue
            cur.execute(
                f"""
                INSERT INTO {backup_table}
                (score_task_id, eval_run_id, sample_idx, task_id, setting, model_name,
                 old_gen_motion_path, new_gen_motion_path, old_physics_score,
                 new_physics_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    d["score_task_id"], d["old_eval_run_id"], d["sample_idx"],
                    d["task_id"], d["setting"], d["model_name"], d["old_path"],
                    d["new_path"], d["old_physics_score"], d["new_physics_score"],
                ),
            )
            cur.execute(
                """
                UPDATE score_tasks
                SET gen_motion_path=?, num_frames=COALESCE(?, num_frames),
                    synced_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (d["new_path"], d.get("new_num_frames"), d["score_task_id"]),
            )
        conn.commit()
        report["score_db_backup"] = str(backup_path)
        report["backup_table"] = backup_table
    conn.close()

    report_path = args.report
    if report_path is None:
        report_path = args.rerun_root / ("selective_update_report_apply.json" if args.apply else "selective_update_report_dryrun.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(json.dumps({
        "applied": bool(args.apply),
        "report": str(report_path),
        "counts": dict(counts),
        "missing_new": missing_new,
        "missing_metrics": missing_metrics,
        "improved_by_group": report["improved_by_group"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

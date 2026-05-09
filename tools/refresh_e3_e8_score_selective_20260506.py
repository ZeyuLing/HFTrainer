#!/usr/bin/env python3
"""Selectively refresh score_m2m E3/E8 pair cases for the 2026-05-06 reruns."""

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


E3_SETTINGS = ("every_5f", "every_10f", "every_15f", "every_30f", "every_60f", "adaptive")
M2M_MODELS = ("uncond_global", "uncond_local")
KIMODO = "KIMODO_uncond"

TERMS = {
    "E3": (
        ("foot_skating_ratio", 0.30),
        ("foot_avg_skate", 1.00),
        ("boundary_accel_jump", 70.0),
        ("jitter_pos", 1000.0),
    ),
    "E8": (
        ("loop_position_error", 0.05),
        ("foot_skating_ratio", 0.30),
        ("foot_avg_skate", 1.00),
        ("boundary_accel_jump_loop", 8.0),
        ("jitter_pos", 500.0),
    ),
}


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def load_metrics(eval_con: sqlite3.Connection, run_id: int, sample_idx: int) -> dict[str, Any] | None:
    row = eval_con.execute(
        "SELECT metrics_json FROM sample_results WHERE eval_run_id=? AND sample_idx=?",
        (run_id, sample_idx),
    ).fetchone()
    if not row:
        return None
    return json.loads(row["metrics_json"] or "{}")


def normalize_path(path: str | None) -> str:
    if not path:
        return ""
    return str(Path(path))


def load_old_metric_index(roots: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    json_files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            json_files.append(root)
        else:
            json_files.extend(sorted(root.rglob("eval_v2_*.json")))
            json_files.extend(sorted(root.rglob("result.json")))

    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        entries: list[dict[str, Any]] = []
        if isinstance(data, dict) and "per_sample" in data:
            entries.append(data)
        elif isinstance(data, dict):
            for model_block in data.values():
                if not isinstance(model_block, dict):
                    continue
                for entry in model_block.get("tasks", {}).values():
                    if isinstance(entry, dict):
                        entries.append(entry)
        for entry in entries:
            for sample in entry.get("per_sample", []):
                sample_path = normalize_path(sample.get("_npz_path") or sample.get("gen_motion_path"))
                if not sample_path:
                    continue
                metrics = {
                    k: v
                    for k, v in sample.items()
                    if not k.startswith("_") and isinstance(v, (int, float))
                }
                out[sample_path] = metrics
    return out


def metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def physics_score(task_id: str, metrics: dict[str, Any]) -> float | None:
    values: list[float] = []
    for key, denom in TERMS[task_id]:
        value = metric(metrics, key)
        if value is None:
            return None
        values.append(value / denom)
    return float(sum(values))


def latest_runs(eval_con: sqlite3.Connection) -> dict[tuple[str, str, str], int]:
    settings = ",".join("?" for _ in E3_SETTINGS)
    models = ",".join("?" for _ in (*M2M_MODELS, KIMODO))
    rows = eval_con.execute(
        f"""
        SELECT er.id, er.task_id, er.setting, m.name AS model_name
        FROM eval_runs er
        JOIN models m ON m.id=er.model_id
        WHERE (
            er.task_id='E3' AND er.setting IN ({settings}) AND m.name IN ({models})
        ) OR (
            er.task_id='E8' AND er.setting='D' AND m.name IN ({models})
        )
        ORDER BY er.id ASC
        """,
        (*E3_SETTINGS, *M2M_MODELS, KIMODO, *M2M_MODELS, KIMODO),
    ).fetchall()
    out: dict[tuple[str, str, str], int] = {}
    for row in rows:
        out[(row["task_id"], row["setting"], row["model_name"])] = int(row["id"])
    return out


def sync_score_tasks(score_con: sqlite3.Connection, eval_con: sqlite3.Connection, run_ids: set[int]) -> int:
    inserted = 0
    cur = score_con.cursor()
    for run_id in sorted(run_ids):
        run = eval_con.execute(
            """
            SELECT er.id, er.task_id, er.setting, m.name AS model_name,
                   m.rotation_space, m.has_caption
            FROM eval_runs er
            JOIN models m ON m.id=er.model_id
            WHERE er.id=?
            """,
            (run_id,),
        ).fetchone()
        if not run:
            continue
        for sample in eval_con.execute(
            """
            SELECT sample_idx, prompt_id, text_caption, gen_motion_path, num_frames
            FROM sample_results
            WHERE eval_run_id=?
            ORDER BY sample_idx
            """,
            (run_id,),
        ):
            cur.execute(
                """
                INSERT OR IGNORE INTO score_tasks
                (eval_run_id, sample_idx, task_id, setting, model_name, gen_motion_path,
                 text_caption, prompt_id, num_frames, rotation_space, has_caption, is_winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    run_id,
                    int(sample["sample_idx"]),
                    run["task_id"],
                    run["setting"],
                    run["model_name"],
                    sample["gen_motion_path"],
                    sample["text_caption"],
                    sample["prompt_id"],
                    sample["num_frames"],
                    run["rotation_space"],
                    run["has_caption"],
                ),
            )
            inserted += int(cur.rowcount > 0)
    return inserted


def score_rows(score_con: sqlite3.Connection, run_id: int) -> dict[int, sqlite3.Row]:
    rows = score_con.execute("SELECT * FROM score_tasks WHERE eval_run_id=?", (run_id,)).fetchall()
    return {int(row["sample_idx"]): row for row in rows}


def best_new_for_sample(
    eval_con: sqlite3.Connection,
    score_con: sqlite3.Connection,
    runs: dict[tuple[str, str, str], int],
    task_id: str,
    setting: str,
    sample_idx: int,
) -> tuple[sqlite3.Row, float, dict[str, Any]] | None:
    candidates = []
    for model in M2M_MODELS:
        run_id = runs.get((task_id, setting, model))
        if not run_id:
            continue
        row = score_con.execute(
            "SELECT * FROM score_tasks WHERE eval_run_id=? AND sample_idx=?",
            (run_id, sample_idx),
        ).fetchone()
        metrics = load_metrics(eval_con, run_id, sample_idx)
        if not row or metrics is None:
            continue
        score = physics_score(task_id, metrics)
        if score is None:
            continue
        candidates.append((score, row, metrics))
    if not candidates:
        return None
    score, row, metrics = min(candidates, key=lambda x: x[0])
    return row, score, metrics


def metric_summary(decisions: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted({k for d in decisions for k in d.get(f"{prefix}_metrics", {})}):
        vals = [d[f"{prefix}_metrics"][key] for d in decisions if d.get(f"{prefix}_metrics", {}).get(key) is not None]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def old_metrics_for_score_task(
    eval_con: sqlite3.Connection,
    old_index: dict[str, dict[str, Any]],
    row: sqlite3.Row,
) -> dict[str, Any] | None:
    metrics = load_metrics(eval_con, int(row["eval_run_id"]), int(row["sample_idx"]))
    if metrics is not None:
        return metrics
    return old_index.get(normalize_path(row["gen_motion_path"]))


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    eval_con = connect(args.eval_db)
    score_con = connect(args.score_db)
    report: dict[str, Any] = {}
    try:
        runs = latest_runs(eval_con)
        old_index = load_old_metric_index(args.old_roots)
        target_run_ids: set[int] = set()
        for setting in E3_SETTINGS:
            for model in M2M_MODELS:
                run_id = runs.get(("E3", setting, model))
                if run_id:
                    target_run_ids.add(run_id)
        for model in (*M2M_MODELS, KIMODO):
            run_id = runs.get(("E8", "D", model))
            if run_id:
                target_run_ids.add(run_id)

        backup = None
        if args.apply:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = args.score_db.with_suffix(args.score_db.suffix + f".bak_{stamp}_e3_e8_selective")
            shutil.copy2(args.score_db, backup)

        inserted_score_tasks = sync_score_tasks(score_con, eval_con, target_run_ids)
        counts = Counter()
        by_group = Counter()
        decisions: list[dict[str, Any]] = []
        cur = score_con.cursor()
        cur.execute(
            "UPDATE score_tasks SET is_winner=0 WHERE task_id IN ('E3','E8') AND model_name IN ('uncond_global','uncond_local')"
        )

        for pair in score_con.execute(
            """
            SELECT pc.*, b.eval_run_id AS old_b_run_id, b.sample_idx AS old_b_sample_idx
                   , b.gen_motion_path AS old_b_path
            FROM pair_cases pc
            JOIN score_tasks b ON b.id=pc.score_task_b_id
            WHERE pc.task_id='E3' AND pc.setting IN ({})
            ORDER BY pc.setting, pc.sample_idx
            """.format(",".join("?" for _ in E3_SETTINGS)),
            E3_SETTINGS,
        ).fetchall():
            old_score_row = score_con.execute("SELECT * FROM score_tasks WHERE id=?", (pair["score_task_b_id"],)).fetchone()
            old_metrics = old_metrics_for_score_task(eval_con, old_index, old_score_row) if old_score_row else None
            best = best_new_for_sample(eval_con, score_con, runs, "E3", pair["setting"], int(pair["sample_idx"]))
            if old_metrics is None or best is None:
                counts["missing"] += 1
                continue
            old_score = physics_score("E3", old_metrics)
            new_row, new_score, new_metrics = best
            if old_score is None:
                counts["missing"] += 1
                continue
            improved = new_score + args.min_delta < old_score
            counts["compared_e3"] += 1
            if improved:
                counts["updated_e3_m2m"] += 1
                by_group[("E3", pair["setting"], new_row["model_name"])] += 1
                cur.execute("UPDATE pair_cases SET score_task_b_id=? WHERE id=?", (new_row["id"], pair["id"]))
                cur.execute("UPDATE score_tasks SET is_winner=1 WHERE id=?", (new_row["id"],))
            else:
                counts["kept_e3_m2m"] += 1
                cur.execute("UPDATE score_tasks SET is_winner=1 WHERE id=?", (pair["score_task_b_id"],))
            decisions.append(
                {
                    "task_id": "E3",
                    "setting": pair["setting"],
                    "sample_idx": int(pair["sample_idx"]),
                    "old_score": old_score,
                    "new_score": new_score,
                    "delta": old_score - new_score,
                    "improved": improved,
                    "new_model": new_row["model_name"],
                    "old_metrics": {k: metric(old_metrics, k) for k, _ in TERMS["E3"]},
                    "new_metrics": {k: metric(new_metrics, k) for k, _ in TERMS["E3"]},
                }
            )

        e8_k_run = runs.get(("E8", "D", KIMODO))
        e8_k_rows = score_rows(score_con, e8_k_run) if e8_k_run else {}
        for pair in score_con.execute(
            """
            SELECT pc.*, b.eval_run_id AS old_b_run_id, b.sample_idx AS old_b_sample_idx
                   , b.gen_motion_path AS old_b_path
            FROM pair_cases pc
            JOIN score_tasks b ON b.id=pc.score_task_b_id
            WHERE pc.task_id='E8' AND pc.setting='D'
            ORDER BY pc.sample_idx
            """
        ).fetchall():
            sample_idx = int(pair["sample_idx"])
            kimodo = e8_k_rows.get(sample_idx)
            if kimodo:
                cur.execute("UPDATE pair_cases SET score_task_a_id=? WHERE id=?", (kimodo["id"], pair["id"]))
                counts["updated_e8_kimodo"] += 1
            old_score_row = score_con.execute("SELECT * FROM score_tasks WHERE id=?", (pair["score_task_b_id"],)).fetchone()
            old_metrics = old_metrics_for_score_task(eval_con, old_index, old_score_row) if old_score_row else None
            best = best_new_for_sample(eval_con, score_con, runs, "E8", "D", sample_idx)
            if old_metrics is None or best is None:
                counts["missing"] += 1
                continue
            old_score = physics_score("E8", old_metrics)
            new_row, new_score, new_metrics = best
            if old_score is None:
                counts["missing"] += 1
                continue
            improved = new_score + args.min_delta < old_score
            counts["compared_e8"] += 1
            if improved:
                counts["updated_e8_m2m"] += 1
                by_group[("E8", "D", new_row["model_name"])] += 1
                cur.execute("UPDATE pair_cases SET score_task_b_id=? WHERE id=?", (new_row["id"], pair["id"]))
                cur.execute("UPDATE score_tasks SET is_winner=1 WHERE id=?", (new_row["id"],))
            else:
                counts["kept_e8_m2m"] += 1
                cur.execute("UPDATE score_tasks SET is_winner=1 WHERE id=?", (pair["score_task_b_id"],))
            decisions.append(
                {
                    "task_id": "E8",
                    "setting": "D",
                    "sample_idx": sample_idx,
                    "old_score": old_score,
                    "new_score": new_score,
                    "delta": old_score - new_score,
                    "improved": improved,
                    "new_model": new_row["model_name"],
                    "old_metrics": {k: metric(old_metrics, k) for k, _ in TERMS["E8"]},
                    "new_metrics": {k: metric(new_metrics, k) for k, _ in TERMS["E8"]},
                }
            )

        report = {
            "applied": bool(args.apply),
            "score_db": str(args.score_db),
            "eval_db": str(args.eval_db),
            "score_db_backup": str(backup) if backup else None,
            "latest_runs": {"|".join(k): v for k, v in sorted(runs.items())},
            "old_metric_index_size": len(old_index),
            "inserted_score_tasks": inserted_score_tasks,
            "counts": dict(counts),
            "updated_by_group": {"|".join(map(str, k)): v for k, v in sorted(by_group.items())},
            "old_metric_means": metric_summary(decisions, "old"),
            "new_metric_means": metric_summary(decisions, "new"),
            "decisions": decisions,
        }
        if args.apply:
            score_con.commit()
        else:
            score_con.rollback()
    finally:
        eval_con.close()
        score_con.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-db", type=Path, default=Path("motion_annot_web/eval_dashboard/eval_dashboard.db"))
    parser.add_argument("--score-db", type=Path, default=Path("motion_annot_web/score_m2m/score_m2m.db"))
    parser.add_argument("--report", type=Path, default=Path("work_dirs/e3_e8_selective_report_20260506.json"))
    parser.add_argument(
        "--old-roots",
        type=Path,
        nargs="*",
        default=[
            Path("work_dirs/e3_latest_20260430_1747"),
            Path("work_dirs/e8_d_rerun_latest_20260430"),
        ],
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-delta", type=float, default=1e-9)
    args = parser.parse_args()

    report = refresh(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("applied", "inserted_score_tasks", "counts", "updated_by_group", "old_metric_means", "new_metric_means")}, ensure_ascii=False, indent=2))
    print(f"report={args.report}")


if __name__ == "__main__":
    main()

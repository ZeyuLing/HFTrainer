#!/usr/bin/env python3
"""Compare 8082 baseline metrics (saved CSV) vs latest run after rerun.

Usage:
    python scripts/eval/compare_baseline_vs_latest.py work_dirs/baseline_e3_e8d_e14_e15_20260506.csv

Prints a markdown-style table grouped by (task_id, setting), showing for each
(model, metric) the old vs new mean and the absolute / relative change.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "motion_annot_web" / "eval_dashboard" / "eval_dashboard.db"

KEY_METRICS = [
    "foot_skating_ratio",
    "foot_avg_skate",
    "foot_penetration",
    "boundary_accel_jump",
    "jitter_pos",
    "mpjpe_masked",
    "loop_position_error",
    "boundary_accel_jump_loop",
]
TASK_FILTER = ("E3", "E8", "E14", "E15")
MODEL_FILTER = ("uncond_global", "uncond_local", "caption_global_phase2", "caption_local_phase2")


def load_baseline(path: Path):
    """Parse the pipe-delimited baseline CSV exported earlier."""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            task_id, setting, model, metric, mean = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                m = float(mean)
            except ValueError:
                continue
            out[(task_id, setting, model, metric)] = m
    return out


def load_latest(db_path: Path):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    sql = """
    WITH latest AS (
      SELECT er.task_id, er.setting, m.id AS model_id, m.name AS model_name, m.epoch,
             er.id AS run_id,
             ROW_NUMBER() OVER (PARTITION BY er.task_id, er.setting, m.id ORDER BY er.id DESC) AS rn
      FROM eval_runs er
      JOIN models m ON m.id=er.model_id
    )
    SELECT l.task_id, l.setting, l.model_name, l.epoch, am.metric_name, am.mean
    FROM latest l
    JOIN agg_metrics am ON am.eval_run_id=l.run_id
    WHERE l.rn=1
    """
    out = {}
    epochs = {}
    for row in con.execute(sql):
        out[(row["task_id"], row["setting"], row["model_name"], row["metric_name"])] = float(row["mean"])
        epochs[(row["task_id"], row["setting"], row["model_name"])] = row["epoch"]
    con.close()
    return out, epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    base = load_baseline(args.baseline_csv)
    latest, epochs = load_latest(args.db)

    rows = defaultdict(dict)
    for (task, setting, model, metric), m_old in base.items():
        if task not in TASK_FILTER or model not in MODEL_FILTER or metric not in KEY_METRICS:
            continue
        m_new = latest.get((task, setting, model, metric))
        if m_new is None:
            continue
        rows[(task, setting, model)][metric] = (m_old, m_new)

    grouped = defaultdict(list)
    for (task, setting, model), metrics in rows.items():
        grouped[(task, setting)].append((model, metrics))

    def fmt(v):
        return f"{v:.4f}" if v is not None else "-"

    def delta(old, new):
        if old == 0:
            return new - old, float("inf")
        return new - old, (new - old) / abs(old) * 100

    print("# Latest M2M v2 checkpoint rerun: physics-metric diff (E3 / E8 D / E14 / E15)\n")
    print("Baseline = previous run on dashboard (uncond_*: epoch 1880/2480, caption_*phase2: epoch 1710/2090)")
    print("Latest   = rerun on lzy_debug_machine_2 (epoch 1890/2490 uncond, 2290/2650 caption_phase2)\n")

    metric_short = {
        "foot_skating_ratio": "fs_ratio",
        "foot_avg_skate": "fs_speed",
        "foot_penetration": "fs_pen",
        "boundary_accel_jump": "bnd_jump",
        "jitter_pos": "jitter",
        "mpjpe_masked": "mpjpe",
        "loop_position_error": "loop_pos",
        "boundary_accel_jump_loop": "bnd_loop",
    }

    foot_summary = []  # collect for end summary
    for (task, setting), entries in sorted(grouped.items()):
        print(f"## {task} / {setting}\n")
        col_widths = {"model": 22, "metric": 9}
        header = f"{'model':<{col_widths['model']}} | {'metric':<{col_widths['metric']}} | {'baseline':>10} | {'latest':>10} | {'Δabs':>9} | {'Δrel%':>7} |"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for model, metrics in sorted(entries):
            ep_l = epochs.get((task, setting, model), "?")
            print(f"{model + ' (ep ' + str(ep_l) + ')':<{col_widths['model']}} | {' ':<{col_widths['metric']}} |")
            for metric in KEY_METRICS:
                pair = metrics.get(metric)
                if pair is None:
                    continue
                old, new = pair
                d_abs, d_rel = delta(old, new)
                short = metric_short.get(metric, metric)
                arrow = "↓" if d_abs < 0 else ("↑" if d_abs > 0 else "·")
                print(f"  {' ':<{col_widths['model']-2}} | {short:<{col_widths['metric']}} | {fmt(old):>10} | {fmt(new):>10} | {d_abs:+9.4f} | {d_rel:+6.1f}% {arrow}|")
                if metric in ("foot_skating_ratio", "foot_avg_skate"):
                    foot_summary.append((task, setting, model, metric, old, new, d_rel))
        print()

    # Foot-skating summary
    print("\n# Foot-skating focus summary (sorted by Δrel%, improving=negative)\n")
    foot_summary.sort(key=lambda r: r[6])
    print(f"{'task/setting':<18} | {'model':<24} | {'metric':<9} | {'baseline':>10} | {'latest':>10} | {'Δrel%':>7}")
    print("-" * 90)
    for task, setting, model, metric, old, new, d_rel in foot_summary[:50]:
        short = metric_short.get(metric, metric)
        print(f"{task + '/' + setting:<18} | {model:<24} | {short:<9} | {old:>10.4f} | {new:>10.4f} | {d_rel:+6.1f}%")

    if len(foot_summary) > 50:
        print("...")
        for task, setting, model, metric, old, new, d_rel in foot_summary[-20:]:
            short = metric_short.get(metric, metric)
            print(f"{task + '/' + setting:<18} | {model:<24} | {short:<9} | {old:>10.4f} | {new:>10.4f} | {d_rel:+6.1f}%")


if __name__ == "__main__":
    main()

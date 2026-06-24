#!/usr/bin/env python3
"""Analyze PhysFlow reward-training logs before using checkpoints for comparison.

The online-adversarial G1 runs are noisy by construction: every step samples
candidates and scores them through a frozen tracker.  A checkpoint is therefore
only comparable after the reward signal is healthy (not all rejected) and the
late-window reward has stopped drifting materially.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List


ROW_RE = re.compile(
    r"step \[(?P<step>\d+)/(?P<total>\d+)\].*?"
    r"loss=(?P<loss>[0-9.eE+-]+).*?"
    r"loss_sft=(?P<loss_sft>[0-9.eE+-]+).*?"
    r"n_good=(?P<n_good>[0-9.eE+-]+).*?"
    r"loss_gt=(?P<loss_gt>[0-9.eE+-]+).*?"
    r"loss_anchor=(?P<loss_anchor>[0-9.eE+-]+).*?"
    r"reward_best_mean=(?P<reward_best>[0-9.eE+-]+).*?"
    r"reward_cand_mean=(?P<reward_cand>[0-9.eE+-]+).*?"
    r"sel_joint_std_mean=(?P<joint_std>[0-9.eE+-]+)"
)


def _mean(xs: List[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def _median(xs: List[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _slope(xs: List[float]) -> float | None:
    if len(xs) < 2:
        return None
    n = len(xs)
    xbar = (n - 1) / 2.0
    ybar = statistics.mean(xs)
    den = sum((i - xbar) ** 2 for i in range(n))
    if den <= 0:
        return 0.0
    return float(sum((i - xbar) * (y - ybar) for i, y in enumerate(xs)) / den)


def _window_stats(rows: List[Dict[str, float]], start: int, end: int) -> Dict[str, float | None]:
    seg = [r for r in rows if start <= int(r["step"]) <= end]
    out: Dict[str, float | None] = {"start": float(start), "end": float(end), "n": float(len(seg))}
    for key in ("loss", "loss_sft", "n_good", "loss_gt", "loss_anchor", "reward_best", "reward_cand", "joint_std"):
        vals = [float(r[key]) for r in seg if math.isfinite(float(r[key]))]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_median"] = _median(vals)
        out[f"{key}_slope"] = _slope(vals)
    return out


def parse_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        m = ROW_RE.search(line)
        if not m:
            continue
        item = {"step": int(m.group("step")), "total": int(m.group("total"))}
        for key in ("loss", "loss_sft", "n_good", "loss_gt", "loss_anchor", "reward_best", "reward_cand", "joint_std"):
            item[key] = float(m.group(key))
        rows.append(item)
    return rows


def analyze_one(path: Path, *, window: int, min_good: float, max_reward_rel_delta: float, max_late_slope: float) -> Dict:
    rows = parse_log(path)
    if not rows:
        return {"log": str(path), "status": "no_rows", "comparable": False, "reason": "no reward rows parsed"}
    max_step = max(int(r["step"]) for r in rows)
    windows = []
    for start in range(1, max_step + 1, window):
        windows.append(_window_stats(rows, start, min(max_step, start + window - 1)))
    late = windows[-1]
    prev = windows[-2] if len(windows) >= 2 else None
    comparable = True
    reasons = []
    n_good = float(late.get("n_good_mean") or 0.0)
    if n_good < min_good:
        comparable = False
        reasons.append(f"late n_good_mean {n_good:.3f} < {min_good:.3f}")
    if prev is not None:
        late_reward = float(late.get("reward_best_mean") or 0.0)
        prev_reward = float(prev.get("reward_best_mean") or 0.0)
        denom = max(abs(prev_reward), 1e-6)
        rel_delta = abs(late_reward - prev_reward) / denom
        if rel_delta > max_reward_rel_delta:
            comparable = False
            reasons.append(f"late reward_best relative delta {rel_delta:.3f} > {max_reward_rel_delta:.3f}")
    else:
        rel_delta = None
    late_slope = late.get("reward_best_slope")
    if late_slope is not None and abs(float(late_slope)) > max_late_slope:
        comparable = False
        reasons.append(f"late reward_best slope {float(late_slope):.5f} > {max_late_slope:.5f}")
    return {
        "log": str(path),
        "status": "ok",
        "num_rows": len(rows),
        "max_step": max_step,
        "window": window,
        "comparable": comparable,
        "reason": "; ".join(reasons) if reasons else "reward signal healthy enough by current thresholds",
        "late_reward_rel_delta": rel_delta,
        "windows": windows,
        "late": late,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, help="LABEL=/path/to/train.log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--min-good", type=float, default=0.8)
    ap.add_argument("--max-reward-rel-delta", type=float, default=0.08)
    ap.add_argument("--max-late-slope", type=float, default=0.003)
    args = ap.parse_args()

    report = {"thresholds": {
        "window": args.window,
        "min_good": args.min_good,
        "max_reward_rel_delta": args.max_reward_rel_delta,
        "max_late_slope": args.max_late_slope,
    }, "methods": {}}
    for spec in args.log:
        if "=" not in spec:
            raise SystemExit(f"--log must be LABEL=/path, got {spec}")
        label, value = spec.split("=", 1)
        report["methods"][label] = analyze_one(
            Path(value),
            window=args.window,
            min_good=args.min_good,
            max_reward_rel_delta=args.max_reward_rel_delta,
            max_late_slope=args.max_late_slope,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

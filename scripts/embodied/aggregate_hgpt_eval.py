#!/usr/bin/env python3
"""Aggregate sharded Humanoid-GPT Table-2 evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _summarize(raw: dict[str, Any], complete_thresh: float) -> dict[str, float]:
    ok_rows = [v for v in raw.values() if isinstance(v, dict) and "error" not in v]
    n = len(raw)

    def mean(key: str) -> float:
        vals = [float(v[key]) for v in ok_rows if key in v and np.isfinite(float(v[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "num_motions": float(n),
        "num_ok": float(len(ok_rows)),
        "error_rate": float((n - len(ok_rows)) / n) if n else float("nan"),
        "success_rate": float(np.mean([float(v.get("length_ratio", 0.0)) >= complete_thresh for v in ok_rows]))
        if ok_rows else float("nan"),
        "completion": mean("length_ratio"),
        "kpt_pos_mae_m": mean("kpt_pos_mae"),
        "kpt_rot_mae_rad": mean("kpt_rot_mae"),
        "joint_pos_mae_rad": mean("joint_pos_mae"),
        "joint_vel_mae_radps": mean("joint_vel_mae"),
        "root_pos_err_mm": mean("root_pos_err_mm"),
        "root_vel_err_mmps": mean("root_vel_err_mms"),
        "root_yaw_err_rad": mean("root_yaw_err"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--complete-thresh", type=float, default=0.9)
    args = parser.parse_args()

    motions: dict[str, Any] = {}
    input_info: dict[str, Any] = {"kept": [], "skipped": {}, "conversions": {}}
    for path in sorted(args.eval_root.glob("shard_*/summary.json")):
        data = json.loads(path.read_text())
        shard = path.parent.name
        for name, row in data.get("motions", {}).items():
            motions[name] = {**row, "shard": shard} if isinstance(row, dict) else row
        inp = data.get("input", {})
        input_info["kept"].extend(inp.get("kept", []))
        input_info["skipped"].update(inp.get("skipped", {}))
        input_info["conversions"].update(inp.get("conversions", {}))

    if not motions:
        raise RuntimeError(f"No successful Humanoid-GPT shard summaries found under {args.eval_root}")

    payload = {
        "summary": _summarize(motions, args.complete_thresh),
        "motions": motions,
        "input": input_info,
    }
    (args.eval_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Humanoid-GPT Table-2 Evaluation", ""]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value:.6g}")
    if input_info["skipped"]:
        lines.append("")
        lines.append(f"- skipped: {len(input_info['skipped'])}")
    (args.eval_root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

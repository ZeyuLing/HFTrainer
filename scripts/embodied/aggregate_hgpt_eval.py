#!/usr/bin/env python3
"""Aggregate sharded Humanoid-GPT Table-2 evaluation outputs."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def _load_json_retry(path: Path) -> Any:
    last_error: json.JSONDecodeError | None = None
    for _ in range(40):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(0.25)
    raise last_error  # type: ignore[misc]


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


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
        "raw_body_err_mean": mean("raw_body_err_mean"),
        "body_err_mean": mean("body_err_mean"),
        "xy_aligned_body_err_mean": mean("xy_aligned_body_err_mean"),
        "local_body_err_mean": mean("local_body_err_mean"),
        "body_vel_err_mean": mean("body_vel_err_mean"),
        "local_body_vel_err_mean": mean("local_body_vel_err_mean"),
        "body_acc_err_mean": mean("body_acc_err_mean"),
        "local_body_acc_err_mean": mean("local_body_acc_err_mean"),
        "raw_global_mpjpe_m": mean("raw_global_mpjpe_m"),
        "raw_global_mpjpe_mm": mean("raw_global_mpjpe_mm"),
        "xy_aligned_mpjpe_m": mean("xy_aligned_mpjpe_m"),
        "xy_aligned_mpjpe_mm": mean("xy_aligned_mpjpe_mm"),
        "mpjpe_m": mean("mpjpe_m"),
        "mpjpe_mm": mean("mpjpe_mm"),
        "local_mpjpe_m": mean("local_mpjpe_m"),
        "local_mpjpe_mm": mean("local_mpjpe_mm"),
        "mpjve_mps": mean("mpjve_mps"),
        "local_mpjve_mps": mean("local_mpjve_mps"),
        "mpjae_mps2": mean("mpjae_mps2"),
        "local_mpjae_mps2": mean("local_mpjae_mps2"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--complete-thresh", type=float, default=0.9)
    args = parser.parse_args()

    motions: dict[str, Any] = {}
    input_info: dict[str, Any] = {"kept": [], "skipped": {}, "conversions": {}}
    for path in sorted(args.eval_root.glob("shard_*/summary.json")):
        data = _load_json_retry(path)
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
    _write_json_atomic(args.eval_root / "summary.json", payload)
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

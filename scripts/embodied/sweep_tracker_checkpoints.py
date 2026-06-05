#!/usr/bin/env python3
"""Export and score tracker checkpoints on fixed PhysFlow source runs.

This is a benchmark-selection tool: it tests whether an intermediate tracker
checkpoint actually improves over the released tracker on the same source
motions used by the four-column visualization.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _records(path: Path) -> list[dict]:
    data = json.loads((path / "summary.json").read_text())
    return [r for r in data.get("records", []) if r.get("status") == "scored"]


def _mean(rows: list[dict], key: str, nested: str | None = None) -> float | None:
    vals = []
    for row in rows:
        obj = row.get(nested, {}) if nested else row
        val = obj.get(key)
        if isinstance(val, bool):
            val = float(val)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    return sum(vals) / len(vals) if vals else None


def _block(path: Path) -> dict[str, float | int | None]:
    rows = _records(path)
    return {
        "n_scored": len(rows),
        "completion_ratio": _mean(rows, "completion_ratio"),
        "fall_rate": _mean(rows, "fall_detected"),
        "adversarial_score": _mean(rows, "adversarial_score"),
        "max_joint_error_rad": _mean(rows, "max_joint_error_rad"),
        "root_trajectory_error_mean_m": _mean(rows, "root_trajectory_error_mean_m"),
        "root_displacement_error_m": _mean(rows, "root_displacement_error_m"),
    }


def _label_for_checkpoint(path: Path) -> str:
    parent = path.parent.name
    if parent in {"compiled_best", "compiled_models"}:
        parent = path.parent.parent.name
    label = f"{parent}__{path.stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", action="append", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--export-python", default=os.environ.get("PHYSFLOW_TRACKER_PYTHON_CMD", sys.executable))
    parser.add_argument("--score-python", default=os.environ.get("PHYSFLOW_SCORE_PYTHON", sys.executable))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_root = ROOT / "ref_repo" / "ProtoMotions"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{proto_root}:{ROOT}:{env.get('PYTHONPATH', '')}"

    results = []
    for checkpoint in args.checkpoint:
        checkpoint = checkpoint.resolve()
        label = _label_for_checkpoint(checkpoint)
        export_dir = out_dir / "compiled" / label
        onnx_path = export_dir / "unified_pipeline.onnx"
        if not (args.skip_existing and onnx_path.is_file()):
            _run(
                [
                    args.export_python,
                    "deployment/export_bm_tracker_onnx.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(export_dir),
                ],
                cwd=proto_root,
                env=env,
            )
        ckpt_result = {"checkpoint": str(checkpoint), "label": label, "onnx": str(onnx_path), "sources": []}
        for source_run in args.source_run:
            source_run = source_run.resolve()
            score_dir = out_dir / "scores" / label / source_run.name
            if not (args.skip_existing and (score_dir / "summary.json").is_file()):
                cmd = [
                    args.score_python,
                    str(ROOT / "scripts" / "embodied" / "score_tracker_on_physflow_run.py"),
                    "--source-run",
                    str(source_run),
                    "--onnx",
                    str(onnx_path),
                    "--out-dir",
                    str(score_dir),
                    "--label",
                    label,
                ]
                if args.limit > 0:
                    cmd += ["--limit", str(args.limit)]
                _run(cmd, cwd=ROOT, env=env)
            ckpt_result["sources"].append(
                {"source_run": str(source_run), "score_dir": str(score_dir), "metrics": _block(score_dir)}
            )
        results.append(ckpt_result)
        (out_dir / "sweep_metrics.json").write_text(json.dumps({"results": results}, indent=2))

    print(json.dumps({"results": results}, indent=2))
    print(f"[sweep] wrote {out_dir / 'sweep_metrics.json'}")


if __name__ == "__main__":
    main()

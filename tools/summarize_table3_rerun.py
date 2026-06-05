#!/usr/bin/env python3
"""Summarize trustworthy Table 3 rerun outputs.

This report intentionally ignores the paper's existing table values.  It only
uses artifacts produced under output/evaluation/table3_mbench by the current
rerun pipeline.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("output/evaluation/table3_mbench")
OUT_JSON = ROOT / "table3_rerun_status.json"
OUT_MD = ROOT / "table3_rerun_status.md"

METRIC_KEYS = {
    "Jit": "Jitter_Degree",
    "Dyn": "Dynamic_Degree",
    "F.Flt": "Foot_Floating",
    "F.Sld": "Foot_Sliding",
    "G.Pen": "Ground_Penetration",
    "Pose": "Pose_Quality",
    "Body": "Body_Penetration",
}


POSE_BODY_GLOBS = [
    "mbench_results_pose_full/*eval_results.json",
    "mbench_results_body_full/*eval_results.json",
    "mbench_results_pose_debug2/*eval_results.json",
    "mbench_results_body_debug2/*eval_results.json",
]


METHODS = [
    {
        "name": "MDM",
        "dir": "mdm",
        "result_glob": "mbench_results_5phys_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
    },
    {"name": "MotionDiffuse", "status": "missing_adapter_or_checkpoint"},
    {"name": "FineMoGen", "status": "missing_adapter_or_checkpoint"},
    {"name": "MotionCraft", "status": "missing_adapter_or_checkpoint"},
    {
        "name": "MotionLCM",
        "dir": "motionlcm",
        "result_glob": "mbench_results_5phys_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
    },
    {"name": "ViMoGen-light", "status": "pending_official_assets"},
    {
        "name": "ViMoGen",
        "dir": "vimogen_official",
        "result_glob": "mbench_results_5phys/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
        "status_if_done": "remeasured_official_276d_retarget",
    },
    {"name": "HYMotion", "status": "pending_adapter_or_outputs"},
    {
        "name": "T2M-GPT",
        "dir": "t2mgpt",
        "result_glob": "mbench_results_5phys_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
        "status_if_done": "remeasured_length_mismatch",
    },
    {
        "name": "MoMask",
        "dir": "momask",
        "result_glob": "mbench_results_5phys_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
    },
    {"name": "TM2T", "status": "missing_adapter_or_checkpoint"},
    {"name": "TM2D", "status": "missing_adapter_or_checkpoint"},
    {"name": "LoM", "status": "missing_adapter_or_checkpoint"},
    {
        "name": "MotionGPT3",
        "dir": "motiongpt3",
        "result_glob": "mbench_results_5phys_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
    },
    {"name": "Go-To-Zero", "status": "missing_adapter_or_checkpoint"},
    {"name": "MotionStreamer", "status": "pending_adapter"},
    {
        "name": "VerMo",
        "dir": "vermo_ckpt25000_full",
        "result_glob": "mbench_results_non_vlm_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
        "status_if_done": "remeasured_non_vlm",
    },
    {
        "name": "MotionGPT(extra)",
        "dir": "motiongpt",
        "result_glob": "mbench_results_5phys_local/*eval_results.json",
        "extra_result_globs": POSE_BODY_GLOBS,
        "status_if_done": "remeasured_not_in_table",
    },
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest(pattern: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def is_empty_eval_result(data: dict[str, Any]) -> bool:
    """Return true for evaluator outputs that finished with zero valid samples."""
    saw_metric = False
    for value in data.values():
        if not isinstance(value, dict):
            continue
        aggregate = value.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
        saw_metric = True
        if aggregate.get("num_samples", 0) != 0:
            return False
    return saw_metric


def latest_non_empty(pattern: str) -> Path | None:
    paths = sorted((Path(p) for p in glob.glob(pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        data = load_json(path)
        if not is_empty_eval_result(data):
            return path
    return None


def metric_summary(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for short, key in METRIC_KEYS.items():
        item = data.get(key)
        if isinstance(item, dict) and "aggregate" in item:
            agg = item["aggregate"]
            out[short] = {
                "mean": agg.get("mean"),
                "std": agg.get("std"),
                "num_samples": agg.get("num_samples"),
            }
        elif isinstance(item, dict) and "error" in item:
            out[short] = {"error": item["error"]}
    return out


def fmt_metric(metrics: dict[str, Any], key: str) -> str:
    item = metrics.get(key)
    if not item:
        return ""
    if "error" in item:
        return "ERR"
    value = item.get("mean")
    return "" if value is None else f"{value:.6f}"


def compact_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = load_json(path)
    summary = data.get("summary", {})
    return {
        "path": str(path),
        "num_expected": summary.get("num_expected"),
        "status_counts": summary.get("status_counts"),
        "complete": summary.get("complete"),
        "frame_delta_abs_mean": summary.get("frame_delta_abs_mean"),
        "frame_delta_abs_max": summary.get("frame_delta_abs_max"),
        "foot_min_z_mean": summary.get("foot_min_z_mean"),
        "foot_min_z_min": summary.get("foot_min_z_min"),
        "foot_min_z_max": summary.get("foot_min_z_max"),
    }


def collect_method(spec: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"status": spec.get("status", "pending")}
    method_dir = spec.get("dir")
    if not method_dir:
        return record

    base = ROOT / method_dir
    result_files: list[Path] = []
    for pattern in [spec["result_glob"], *spec.get("extra_result_globs", [])]:
        path = latest_non_empty(str(base / pattern))
        if path:
            result_files.append(path)

    metrics: dict[str, Any] = {}
    for path in result_files:
        data = load_json(path)
        for short, item in metric_summary(data).items():
            metrics[short] = item

    if metrics:
        record["status"] = spec.get("status_if_done", "remeasured")
        record["metrics"] = metrics
        record["result_files"] = [str(p) for p in result_files]
    else:
        record["status"] = spec.get("status", "pending_results")

    manifest = compact_manifest(base / "mbench_eval_input_manifest.json")
    if manifest:
        record["input_validation"] = manifest

    notes = []
    if manifest:
        frame_delta = manifest.get("frame_delta_abs_mean")
        foot_min = manifest.get("foot_min_z_mean")
        if frame_delta is not None:
            notes.append(f"frameDeltaMean={frame_delta:.2f}")
        if foot_min is not None:
            notes.append(f"footMinZMean={foot_min:.3f}m")
    errors = {
        short: item["error"]
        for short, item in metrics.items()
        if isinstance(item, dict) and "error" in item
    }
    if errors:
        for short, error in sorted(errors.items()):
            notes.append(f"{short} error: {error}")
    if notes:
        record["notes"] = "; ".join(notes)
    return record


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    methods = {spec["name"]: collect_method(spec) for spec in METHODS}
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "All previous Table 3 values are invalid for paper use until replaced by this rerun pipeline.",
        "vlm_metrics": (
            "pending: GEMINI_API_KEY is not set in current environment"
            if not os.environ.get("GEMINI_API_KEY")
            else "ready_to_run"
        ),
        "methods": methods,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    headers = ["Method", "Status", *METRIC_KEYS.keys(), "Notes"]
    lines = [
        "# Table 3 Rerun Status",
        "",
        payload["note"],
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"VLM metrics: {payload['vlm_metrics']}",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---", "---", *["---:" for _ in METRIC_KEYS], "---"]) + "|",
    ]
    for spec in METHODS:
        name = spec["name"]
        record = methods[name]
        metrics = record.get("metrics", {})
        row = [
            name,
            record.get("status", ""),
            *[fmt_metric(metrics, key) for key in METRIC_KEYS],
            record.get("notes", ""),
        ]
        lines.append("|" + "|".join(row) + "|")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

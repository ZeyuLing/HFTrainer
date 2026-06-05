#!/usr/bin/env python3
"""Merge sharded VerMo T2M/M2T viewer exports and prepare paper T2M inputs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from tools.export_vermo_overfit_viewer import summarize_cases


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_indices(path: str) -> int:
    if not os.path.exists(path):
        return 0
    text = open(path, encoding="utf-8").read().strip()
    if not text:
        return 0
    return len([item for item in text.split(",") if item.strip()])


def _expected_cases_from_index(root: str) -> Optional[int]:
    index_paths = sorted(glob.glob(os.path.join(root, "index_shards", "shard_*.txt")))
    if not index_paths:
        return None
    return sum(_count_indices(path) for path in index_paths)


def _relativize_artifacts(case: Dict[str, Any], shard_name: str) -> Dict[str, Any]:
    case = json.loads(json.dumps(case, ensure_ascii=False))
    case["case_id"] = f"{shard_name}__{case['case_id']}"
    for bucket in ("inputs", "targets", "predictions"):
        for item in case.get(bucket, []):
            if item.get("path"):
                item["path"] = f"{shard_name}/{item['path']}"
    return case


def _find_prediction_motion(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in case.get("predictions", []):
        if (
            item.get("kind") == "motion"
            and item.get("role") == "prediction"
            and item.get("source") == "decoded"
        ):
            return item
    return None


def _text_eval(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for modal_metrics in case.get("metrics", {}).values():
        text_metrics = modal_metrics.get("pred_vs_target_decoded_eval")
        if text_metrics and text_metrics.get("type") == "text":
            return text_metrics
    return None


def _extract_text_artifact(case: Dict[str, Any], bucket: str, source: str) -> str:
    for item in case.get(bucket, []):
        if item.get("kind") == "text" and item.get("source") == source:
            return item.get("text", "")
    return ""


def _copy_t2m_predictions(root: str, cases: List[Dict[str, Any]], pred_dir: str) -> Dict[str, Any]:
    os.makedirs(pred_dir, exist_ok=True)
    copied = 0
    missing = 0
    for case in cases:
        if case.get("task") != "t2m":
            continue
        source_key = case.get("overview", {}).get("source_key") or str(case.get("dataset_idx"))
        pred_item = _find_prediction_motion(case)
        if not pred_item:
            missing += 1
            continue
        src = os.path.join(root, pred_item["path"])
        if not os.path.exists(src):
            missing += 1
            continue
        arr = np.load(src, allow_pickle=True)["motion_135"].astype(np.float32)
        dst = os.path.join(pred_dir, f"{source_key}.npy")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst, arr)
        copied += 1
    return {"pred_dir": pred_dir, "copied": copied, "missing": missing}


def _summarize_m2t(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for case in cases:
        if case.get("task") != "m2t":
            continue
        metrics = _text_eval(case)
        if not metrics:
            continue
        rows.append({
            "case_id": case["case_id"],
            "dataset_idx": case.get("dataset_idx"),
            "source_key": case.get("overview", {}).get("source_key", ""),
            "exact": bool(metrics.get("exact")),
            "cer": float(metrics.get("cer", 1.0)),
            "edit_distance": int(metrics.get("edit_distance", 0)),
            "target_len": int(metrics.get("target_len", 0)),
            "pred_len": int(metrics.get("pred_len", 0)),
            "target": _extract_text_artifact(case, "targets", "token_decoded"),
            "prediction": _extract_text_artifact(case, "predictions", "decoded"),
        })
    if not rows:
        return {"count": 0, "rows": []}
    return {
        "count": len(rows),
        "exact_rate": float(np.mean([r["exact"] for r in rows])),
        "cer_mean": float(np.mean([r["cer"] for r in rows])),
        "cer_median": float(np.median([r["cer"] for r in rows])),
        "edit_distance_mean": float(np.mean([r["edit_distance"] for r in rows])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--pred-dir", default="")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    pred_dir = os.path.abspath(args.pred_dir or os.path.join(root, "paper_t2m_pred_135d"))
    manifest_paths = sorted(glob.glob(os.path.join(root, "shard_*", "manifest.json")))
    if not manifest_paths:
        raise SystemExit(f"No shard manifests found under {root}/shard_*")

    cases: List[Dict[str, Any]] = []
    config = ""
    checkpoint = ""
    shard_summaries: List[Dict[str, Any]] = []
    for manifest_path in manifest_paths:
        shard_name = os.path.basename(os.path.dirname(manifest_path))
        manifest = _load_json(manifest_path)
        shard_summaries.append(manifest.get("summary", {}))
        config = config or manifest.get("config", "")
        checkpoint = checkpoint or manifest.get("checkpoint", "")
        for case in manifest.get("cases", []):
            cases.append(_relativize_artifacts(case, shard_name))

    expected_cases = _expected_cases_from_index(root)
    if expected_cases is None:
        expected_cases = sum(int(s.get("expected_cases", 0) or 0) for s in shard_summaries) or len(cases)
    complete = bool(cases) and len(cases) >= expected_cases and all(
        bool(s.get("complete", False)) for s in shard_summaries
    )
    summary = summarize_cases(cases, expected_cases=expected_cases, complete=complete)
    summary["num_shard_manifests"] = len(manifest_paths)
    t2m_pred = _copy_t2m_predictions(root, cases, pred_dir)
    m2t_summary = _summarize_m2t(cases)
    merged = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "checkpoint": checkpoint,
        "output_dir": root,
        "summary": summary,
        "t2m_prediction_export": t2m_pred,
        "m2t_text_metrics": {k: v for k, v in m2t_summary.items() if k != "rows"},
        "cases": cases,
    }

    out_manifest = os.path.join(root, "manifest.json")
    with open(out_manifest + ".tmp", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    os.replace(out_manifest + ".tmp", out_manifest)

    m2t_rows_path = os.path.join(root, "m2t_text_rows.json")
    with open(m2t_rows_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(m2t_summary.get("rows", []), f, ensure_ascii=False, indent=2)
    os.replace(m2t_rows_path + ".tmp", m2t_rows_path)

    print(json.dumps({
        "num_cases": len(cases),
        "summary": summary,
        "t2m_prediction_export": t2m_pred,
        "m2t_text_metrics": merged["m2t_text_metrics"],
        "manifest": out_manifest,
        "m2t_rows": m2t_rows_path,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

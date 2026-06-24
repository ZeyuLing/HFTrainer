#!/usr/bin/env python3
"""Merge 2026-05-01 8082 refresh shards into flat import JSON files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUN_ROOT = ROOT / "work_dirs" / "eval_8082_refresh_20260501"
IMPORT_DIR = RUN_ROOT / "import_jsons"


EXPECTED: dict[tuple[str, str], int] = {
    ("E1", "default"): 240,
    **{("E2", s): 220 for s in ("start_1f", "end_1f", "both_1f", "pre20", "post20", "mid60", "pre20_uncond", "post20_uncond", "mid60_uncond")},
    **{("E3", s): 240 for s in ("every_5f", "every_10f", "every_15f", "every_30f", "every_60f", "adaptive")},
    **{("E4", s): 100 for s in ("A_rhand_sparse", "B_ankles_sparse", "C_rhand_lfoot", "D_both_hands", "E_all4_sparse", "F_rhand_dense")},
    **{("E5", s): 78 for s in ("A", "B", "C")},
    ("E6", "pos_contact"): 50,
    ("E7", "default"): 50,
    ("E8", "A"): 200,
    ("E8", "D"): 200,
    **{("E10", s): 50 for s in ("A_upper", "B_lower", "C_spine_only")},
    **{("E13", s): 80 for s in ("A", "B", "C")},
    ("E14", "L"): 100,
    ("E14", "M"): 100,
    ("E15", "default"): 200,
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    from hftrainer.evaluation.motion.m2m_eval_metrics import aggregate_metrics

    try:
        return aggregate_metrics(samples)
    except Exception:
        keys = set()
        for s in samples:
            keys.update(k for k, v in s.items() if not k.startswith("_") and isinstance(v, (int, float)))
        agg = {}
        for k in sorted(keys):
            vals = [float(s[k]) for s in samples if isinstance(s.get(k), (int, float))]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return agg


def _sample_idx(sample: dict[str, Any], fallback: int) -> int:
    for key in ("_sample_idx", "_eval_sample_idx"):
        val = sample.get(key)
        if isinstance(val, int):
            return val
    return fallback


def _flatten_nested_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    """Expose T2M/KIMODO nested metrics to the dashboard importer."""
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return sample
    out = dict(sample)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key not in out:
            out[key] = value
    return out


def merge_hymotion() -> list[Path]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted((RUN_ROOT / "hymotion").glob("**/eval_v2_*.json")):
        with path.open() as f:
            nested = json.load(f)
        for model_name, block in nested.items():
            if not isinstance(block, dict) or "tasks" not in block:
                continue
            for entry in block.get("tasks", {}).values():
                task_id = entry.get("task_id")
                setting = entry.get("setting")
                if (task_id, setting) not in EXPECTED:
                    continue
                key = (model_name, task_id, setting)
                group = groups.setdefault(
                    key,
                    {
                        "meta": {
                            "checkpoint": block.get("checkpoint", ""),
                            "rotation_space": block.get("rotation_space", "local"),
                            "has_caption": bool(block.get("has_caption", False)),
                        },
                        "samples": [],
                    },
                )
                group["samples"].extend(entry.get("per_sample", []))

    out_paths: list[Path] = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for (model_name, task_id, setting), group in sorted(groups.items()):
        samples = sorted(group["samples"], key=lambda s: _sample_idx(s, -1))
        expected = EXPECTED[(task_id, setting)]
        seen = [_sample_idx(s, i) for i, s in enumerate(samples)]
        if seen != list(range(expected)):
            raise SystemExit(f"{model_name}/{task_id}/{setting}: expected 0..{expected - 1}, got n={len(samples)}")
        meta = group["meta"]
        data = {
            "model": model_name,
            "checkpoint": meta["checkpoint"],
            "rotation_space": meta["rotation_space"],
            "has_caption": meta["has_caption"],
            "timestamp": timestamp,
            "task_id": task_id,
            "setting": setting,
            "num_prompts": len(samples),
            "aggregated": _aggregate(samples),
            "per_sample": samples,
        }
        out = IMPORT_DIR / "hymotion" / f"{model_name}__{task_id}_{setting}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(json.loads(json.dumps(data, default=_jsonable)), f, ensure_ascii=False, indent=2)
        out_paths.append(out)
        print(f"[hymotion] {model_name} {task_id}/{setting}: n={len(samples)} -> {out}")
    return out_paths


def merge_kimodo() -> list[Path]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted((RUN_ROOT / "kimodo").glob("**/result.json")):
        with path.open() as f:
            data = json.load(f)
        model_name = data.get("model", "KIMODO")
        # Normalize legacy T2M runner naming.
        if model_name == "KIMODO":
            model_name = "KIMODO_caption"
        task_id = data.get("task_id", "E1")
        setting = data.get("setting", "default")
        if (task_id, setting) not in EXPECTED:
            continue
        groups.setdefault((model_name, task_id, setting), []).extend(
            _flatten_nested_metrics(sample) for sample in data.get("per_sample", [])
        )

    out_paths: list[Path] = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for (model_name, task_id, setting), samples in sorted(groups.items()):
        expected = EXPECTED[(task_id, setting)]
        samples = sorted(samples, key=lambda s: _sample_idx(s, -1))
        if task_id == "E1":
            # run_kimodo_t2m.py does not write _sample_idx; preserve prompt order.
            if len(samples) != expected:
                raise SystemExit(f"{model_name}/{task_id}/{setting}: expected n={expected}, got {len(samples)}")
        else:
            seen = [_sample_idx(s, i) for i, s in enumerate(samples)]
            if seen != list(range(expected)):
                raise SystemExit(f"{model_name}/{task_id}/{setting}: expected 0..{expected - 1}, got n={len(samples)}")
        data = {
            "model": model_name,
            "checkpoint": "kimodo-soma-rp",
            "rotation_space": "global",
            "has_caption": model_name.endswith("_caption"),
            "timestamp": timestamp,
            "task_id": task_id,
            "setting": setting,
            "num_prompts": len(samples),
            "aggregated": _aggregate(samples),
            "per_sample": samples,
        }
        out = IMPORT_DIR / "kimodo" / f"{model_name}__{task_id}_{setting}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(json.loads(json.dumps(data, default=_jsonable)), f, ensure_ascii=False, indent=2)
        out_paths.append(out)
        print(f"[kimodo] {model_name} {task_id}/{setting}: n={len(samples)} -> {out}")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["hymotion", "kimodo", "all"], default="all")
    args = parser.parse_args()

    outputs: list[Path] = []
    if args.group in ("hymotion", "all"):
        outputs.extend(merge_hymotion())
    if args.group in ("kimodo", "all"):
        outputs.extend(merge_kimodo())
    print(f"[done] wrote {len(outputs)} import JSONs under {IMPORT_DIR}")


if __name__ == "__main__":
    main()

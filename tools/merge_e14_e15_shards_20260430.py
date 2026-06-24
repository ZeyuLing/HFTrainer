#!/usr/bin/env python3
"""Merge 2026-04-30 E14/E15 sharded eval JSONs into importable runs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RERUN_ROOT = ROOT / "work_dirs" / "e14_e15_rerun_latest_20260430"
MERGED_DIR = RERUN_ROOT / "merged"
IMPORT_DIR = RERUN_ROOT / "import_jsons"


def _serializable(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    from hftrainer.evaluation.motion.m2m_eval_metrics import aggregate_metrics

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(RERUN_ROOT.rglob("eval_v2_*.json")):
        if "merged" in path.parts or "import_jsons" in path.parts:
            continue
        with path.open() as f:
            nested = json.load(f)
        for model_name, model_block in nested.items():
            if not isinstance(model_block, dict):
                continue
            for task_key, entry in model_block.get("tasks", {}).items():
                task_id = entry.get("task_id")
                setting = entry.get("setting")
                if not task_id or setting is None:
                    continue
                key = (model_name, task_id, setting)
                group = groups.setdefault(
                    key,
                    {
                        "model_meta": {
                            "checkpoint": model_block.get("checkpoint", ""),
                            "model": model_name,
                            "desc": model_block.get("desc", ""),
                            "rotation_space": model_block.get("rotation_space", "local"),
                            "motion_dim": model_block.get("motion_dim", 198),
                            "num_steps": model_block.get("num_steps", 50),
                            "replacement_guidance": model_block.get("replacement_guidance", "skip_last"),
                        },
                        "task_id": task_id,
                        "setting": setting,
                        "samples": [],
                    },
                )
                group["samples"].extend(entry.get("per_sample", []))

    if not groups:
        raise SystemExit(f"No eval_v2 shard JSONs found under {RERUN_ROOT}")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    nested_out: dict[str, Any] = {}

    for (model_name, task_id, setting), group in sorted(groups.items()):
        samples = sorted(group["samples"], key=lambda s: int(s.get("_sample_idx", -1)))
        seen = [int(s.get("_sample_idx", -1)) for s in samples]
        if seen != list(range(min(seen), max(seen) + 1)):
            raise SystemExit(f"Non-contiguous sample_idx for {(model_name, task_id, setting)}: {seen[:5]}...{seen[-5:]}")
        agg = aggregate_metrics(samples)
        meta = group["model_meta"]
        task_key = f"{task_id}_{setting}"
        nested_out.setdefault(model_name, {**meta, "tasks": {}})["tasks"][task_key] = {
            "task_id": task_id,
            "setting": setting,
            "num_samples": len(samples),
            "aggregated": agg,
            "per_sample": samples,
        }
        flat = {
            "model": model_name,
            "checkpoint": meta["checkpoint"],
            "rotation_space": meta["rotation_space"],
            "has_caption": "caption" in model_name.lower(),
            "timestamp": timestamp,
            "task_id": task_id,
            "setting": setting,
            "num_prompts": len(samples),
            "aggregated": agg,
            "per_sample": samples,
        }
        out_path = IMPORT_DIR / f"{model_name}__{task_id}_{setting}.json"
        with out_path.open("w") as f:
            json.dump(json.loads(json.dumps(flat, default=_serializable)), f, ensure_ascii=False, indent=2)
        foot_ratio = agg.get("foot_skating_ratio", {}).get("mean")
        foot_avg = agg.get("foot_avg_skate", {}).get("mean")
        print(
            f"{model_name} {task_id}/{setting}: n={len(samples)} "
            f"ckpt={meta['checkpoint']} foot_ratio={foot_ratio} foot_avg={foot_avg}"
        )

    nested_path = MERGED_DIR / "eval_v2_merged.json"
    with nested_path.open("w") as f:
        json.dump(json.loads(json.dumps(nested_out, default=_serializable)), f, ensure_ascii=False, indent=2)
    print(f"merged nested: {nested_path}")
    print(f"flat import dir: {IMPORT_DIR}")


if __name__ == "__main__":
    main()

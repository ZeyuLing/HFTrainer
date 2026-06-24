#!/usr/bin/env python3
"""Merge active E3 latest-checkpoint shard outputs into importable JSONs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RERUN_ROOT = ROOT / "work_dirs" / "e3_latest_20260430_1747"
IMPORT_DIR = RERUN_ROOT / "import_jsons"
MERGED_DIR = RERUN_ROOT / "merged"
SETTINGS = {"every_10f", "every_15f", "every_30f", "every_60f", "adaptive"}


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

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(RERUN_ROOT.rglob("eval_v2_*.json")):
        if "import_jsons" in path.parts or "merged" in path.parts:
            continue
        with path.open() as f:
            nested = json.load(f)
        for model_name, block in nested.items():
            if not isinstance(block, dict):
                continue
            for _task_key, entry in block.get("tasks", {}).items():
                if entry.get("task_id") != "E3" or entry.get("setting") not in SETTINGS:
                    continue
                setting = entry["setting"]
                group = groups.setdefault(
                    (model_name, setting),
                    {
                        "meta": {
                            "checkpoint": block.get("checkpoint", ""),
                            "model": model_name,
                            "desc": block.get("desc", ""),
                            "rotation_space": block.get("rotation_space", "local"),
                            "motion_dim": block.get("motion_dim", 198),
                            "num_steps": block.get("num_steps", 50),
                            "replacement_guidance": block.get("replacement_guidance", "skip_last"),
                        },
                        "samples": [],
                    },
                )
                group["samples"].extend(entry.get("per_sample", []))

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    nested_out: dict[str, Any] = {}
    for (model_name, setting), group in sorted(groups.items()):
        samples = sorted(group["samples"], key=lambda s: int(s.get("_sample_idx", -1)))
        seen = [int(s.get("_sample_idx", -1)) for s in samples]
        if seen != list(range(240)):
            raise SystemExit(f"{model_name}/{setting}: expected sample_idx 0..239, got n={len(seen)}")
        agg = aggregate_metrics(samples)
        meta = group["meta"]
        task_key = f"E3_{setting}"
        nested_out.setdefault(model_name, {**meta, "tasks": {}})["tasks"][task_key] = {
            "task_id": "E3",
            "setting": setting,
            "num_samples": len(samples),
            "aggregated": agg,
            "per_sample": samples,
        }
        flat = {
            "model": model_name,
            "checkpoint": meta["checkpoint"],
            "rotation_space": meta["rotation_space"],
            "has_caption": False,
            "timestamp": timestamp,
            "task_id": "E3",
            "setting": setting,
            "num_prompts": len(samples),
            "aggregated": agg,
            "per_sample": samples,
        }
        out_path = IMPORT_DIR / f"{model_name}__E3_{setting}.json"
        with out_path.open("w") as f:
            json.dump(json.loads(json.dumps(flat, default=_serializable)), f, ensure_ascii=False, indent=2)
        print(
            f"{model_name} E3/{setting}: n={len(samples)} ckpt={meta['checkpoint']} "
            f"foot_ratio={agg.get('foot_skating_ratio', {}).get('mean')} "
            f"jitter={agg.get('jitter_pos', {}).get('mean')}"
        )
    with (MERGED_DIR / "eval_v2_merged.json").open("w") as f:
        json.dump(json.loads(json.dumps(nested_out, default=_serializable)), f, ensure_ascii=False, indent=2)
    print(f"flat import dir: {IMPORT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge sharded KIMODO rerun outputs into flat result.json files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "work_dirs" / "kimodo_swin_fix_20260430"
IMPORT_DIR = RUN_ROOT / "import_jsons"
TARGETS = {
    ("E3", "every_30f"): 240,
    ("E3", "every_60f"): 240,
    ("E3", "adaptive"): 240,
    ("E14", "M"): 100,
    ("E14", "L"): 100,
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
    keys = set()
    for s in samples:
        keys.update(k for k, v in s.items() if not k.startswith("_") and isinstance(v, (int, float)))
    agg = {}
    for k in sorted(keys):
        vals = [float(s[k]) for s in samples if isinstance(s.get(k), (int, float))]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main() -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {k: [] for k in TARGETS}
    for p in sorted(RUN_ROOT.glob("kimodo_*/*/result.json")):
        with p.open() as f:
            data = json.load(f)
        key = (data.get("task_id"), data.get("setting"))
        if key in groups:
            groups[key].extend(data.get("per_sample", []))

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for key, expected in sorted(TARGETS.items()):
        task_id, setting = key
        samples = sorted(groups[key], key=lambda s: int(s.get("_sample_idx", -1)))
        seen = [int(s.get("_sample_idx", -1)) for s in samples]
        if seen != list(range(expected)):
            raise SystemExit(f"{task_id}/{setting}: expected 0..{expected-1}, got n={len(seen)}")
        data = {
            "model": "KIMODO_uncond",
            "task_id": task_id,
            "setting": setting,
            "retarget_method": "rotation_based",
            "has_caption": False,
            "timestamp": timestamp,
            "num_prompts": len(samples),
            "aggregated": _aggregate(samples),
            "per_sample": samples,
        }
        out = IMPORT_DIR / f"KIMODO_uncond__{task_id}_{setting}.json"
        with out.open("w") as f:
            json.dump(json.loads(json.dumps(data, default=_jsonable)), f, ensure_ascii=False, indent=2)
        print(f"{task_id}/{setting}: n={len(samples)} -> {out}")
    print(f"flat import dir: {IMPORT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select physically bad before-cases for fixed-noise PhysFlow comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


SCENE_DEPENDENT_PATTERNS = (
    "chair",
    "seat",
    "bench",
    "sofa",
    "stool",
    "stairs",
    "stair",
    "steps",
    "step over an object",
    "step over a",
    "ladder",
    "table",
    "desk",
    "door",
    "wall",
    "box",
    "basketball",
    "ball",
    "pick up an object",
    "pick up a",
    "place an object",
    "place a",
    "put an object",
    "put a",
)


def is_scene_dependent(prompt: str | None) -> bool:
    """Heuristic filter for actions that need unsupported scene/prop contact."""
    if not prompt:
        return False
    text = prompt.lower()
    if "sit" in text and "floor" not in text and "ground" not in text:
        return True
    return any(pattern in text for pattern in SCENE_DEPENDENT_PATTERNS)


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def badness(record: Dict[str, Any]) -> float:
    kin = record.get("kinematic") or {}
    score = 0.0
    score += 2.0 * _num(record.get("adversarial_score"), 0.0)
    score += 2.0 * max(0.0, 1.0 - _num(record.get("completion_ratio"), 1.0))
    score += 1.5 * _num(record.get("fall_detected"), 0.0)
    score += 1.0 * max(0.0, _num(record.get("max_joint_error_rad"), 0.0) - 0.7)
    score += 1.0 * _num(record.get("root_trajectory_error_mean_m"), 0.0)
    score += 1.5 * _num(kin.get("foot_skate_speed"), 0.0)
    score += 0.001 * _num(kin.get("jerk"), 0.0)
    score += 0.5 * _num(kin.get("float_ratio"), 0.0)
    score += 0.5 * _num(kin.get("penetration_ratio"), 0.0)
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="before run summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument(
        "--exclude-scene-dependent",
        action="store_true",
        help="Drop prompts that require unavailable scene objects/supports.",
    )
    args = ap.parse_args()

    src = Path(args.summary)
    blob = json.loads(src.read_text())
    rows: List[Dict[str, Any]] = []
    skipped_scene_dependent: List[Dict[str, Any]] = []
    for rec in blob.get("records", []):
        b = badness(rec)
        if b < args.min_score:
            continue
        prompt = rec.get("prompt") or rec.get("caption")
        if args.exclude_scene_dependent and is_scene_dependent(prompt):
            skipped_scene_dependent.append({
                "source_index": int(rec["source_index"]),
                "badness": b,
                "prompt": prompt,
            })
            continue
        rows.append({
            "source_index": int(rec["source_index"]),
            "badness": b,
            "prompt": prompt,
            "prompt_id": rec.get("prompt_id"),
            "source_motion_path": rec.get("source_motion_path"),
            "metrics": {
                "adversarial_score": rec.get("adversarial_score"),
                "completion": rec.get("completion_ratio"),
                "fall": rec.get("fall_detected"),
                "max_joint_error": rec.get("max_joint_error_rad"),
                "root_traj": rec.get("root_trajectory_error_mean_m"),
                "foot_skate": (rec.get("kinematic") or {}).get("foot_skate_speed"),
                "jerk": (rec.get("kinematic") or {}).get("jerk"),
            },
        })
    rows.sort(key=lambda x: x["badness"], reverse=True)
    rows = rows[: args.topk]
    out = {
        "source_summary": str(src),
        "topk": int(args.topk),
        "exclude_scene_dependent": bool(args.exclude_scene_dependent),
        "indices": [r["source_index"] for r in rows],
        "items": rows,
        "skipped_scene_dependent_top": sorted(
            skipped_scene_dependent,
            key=lambda x: x["badness"],
            reverse=True,
        )[: args.topk],
    }
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

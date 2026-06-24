#!/usr/bin/env python3
"""Add Humanoid-GPT rollout actors to the Table 2 same-scene case viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _hgpt_key(row: dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset", ""))
    stem = str(row.get("stem") or row.get("match_key"))
    if dataset.startswith("LAFAN"):
        return "lafan1", stem
    if dataset.startswith("Wild"):
        return "wild", f"{stem.split('_', 1)[0]}_gen"
    raise ValueError(f"Unsupported dataset for Humanoid-GPT case merge: {dataset}")


def _frames_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    data = _load(path)
    return int(data.get("num_frames") or len(data.get("frames") or []))


def _metric_payload(metric: dict[str, Any], frames: int | None, complete_thresh: float) -> dict[str, Any]:
    if not metric:
        return {"missing": True}
    out = {
        "success": float(metric.get("length_ratio", 0.0)) >= complete_thresh,
        "completion": metric.get("length_ratio"),
        "kpt_pos_mae_mm": (
            float(metric["kpt_pos_mae"]) * 1000.0 if metric.get("kpt_pos_mae") is not None else None
        ),
        "root_pos_err_mm": metric.get("root_pos_err_mm"),
        "root_vel_err_mmps": metric.get("root_vel_err_mms"),
        "root_yaw_err_rad": metric.get("root_yaw_err"),
        "joint_pos_mae_rad": metric.get("joint_pos_mae"),
        "joint_vel_mae_radps": metric.get("joint_vel_mae"),
        "frames": frames,
    }
    return out


def _normalize_actor_offsets(row: dict[str, Any]) -> None:
    offsets = {
        "reference": [-3.6, 0, 0],
        "protomotions": [-1.2, 0, 0],
        "any2track": [1.2, 0, 0],
        "humanoid_gpt": [3.6, 0, 0],
    }
    for actor in row.get("actors", []):
        key = actor.get("key")
        if key in offsets:
            actor["offset"] = offsets[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/case_compare_viz/manifest.json"),
    )
    ap.add_argument(
        "--frames-root",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/case_compare_viz/humanoid_gpt"),
    )
    ap.add_argument(
        "--lafan-summary",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/humanoid_gpt/lafan1_v2/summary.json"),
    )
    ap.add_argument(
        "--wild-summary",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/humanoid_gpt/wild_v2/summary.json"),
    )
    ap.add_argument("--complete-thresh", type=float, default=0.9)
    args = ap.parse_args()

    manifest = _load(args.manifest)
    summaries = {
        "lafan1": _load(args.lafan_summary).get("motions", {}),
        "wild": _load(args.wild_summary).get("motions", {}),
    }

    for row in manifest.get("rows", []):
        split, key = _hgpt_key(row)
        frames_path = (args.frames_root / split / f"{key}.json").resolve()
        frames = _frames_count(frames_path)
        metric = _metric_payload(summaries[split].get(key, {}), frames, args.complete_thresh)
        row.setdefault("metrics", {}).update(
            {
                "hgpt_success": metric.get("success"),
                "hgpt_completion": metric.get("completion"),
                "hgpt_kpt_pos_mae_mm": metric.get("kpt_pos_mae_mm"),
                "hgpt_root_pos_err_mm": metric.get("root_pos_err_mm"),
                "hgpt_frames": metric.get("frames"),
            }
        )
        row["actors"] = [actor for actor in row.get("actors", []) if actor.get("key") != "humanoid_gpt"]
        if frames_path.is_file():
            row["actors"].append(
                {
                    "key": "humanoid_gpt",
                    "title": "Humanoid-GPT",
                    "group": "released zero-shot tracker",
                    "path": str(frames_path),
                    "color": "#8d65d8",
                    "offset": [3.6, 0, 0],
                    "metrics": metric,
                }
            )
        else:
            row.setdefault("notes", []).append(f"missing Humanoid-GPT frames: {frames_path}")
        _normalize_actor_offsets(row)

    manifest.setdefault("sources", {})["humanoid_gpt"] = {
        "frames_root": str(args.frames_root.resolve()),
        "lafan_summary": str(args.lafan_summary.resolve()),
        "wild_summary": str(args.wild_summary.resolve()),
        "complete_thresh": args.complete_thresh,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(args.manifest)


if __name__ == "__main__":
    main()

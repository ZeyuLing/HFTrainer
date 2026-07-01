#!/usr/bin/env python3
"""Add SONIC rollout actors to the Table 2 same-scene case viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_TO_SPLIT = {
    "lafan1": "lafan1_fixed600",
    "wild": "wild_clean_fixed600",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _split_and_name(row: dict[str, Any]) -> tuple[str, str]:
    key = str(row.get("dataset_key") or "").lower()
    if key not in DATASET_TO_SPLIT:
        dataset = str(row.get("dataset") or "").lower()
        if "lafan" in dataset:
            key = "lafan1"
        elif "wild" in dataset:
            key = "wild"
    if key not in DATASET_TO_SPLIT:
        raise ValueError(f"Unsupported dataset for SONIC case merge: {row.get('dataset')}")
    return DATASET_TO_SPLIT[key], str(row.get("stem") or row.get("match_key"))


def _metric_payload(path: Path, frames: int | None) -> dict[str, Any]:
    if not path.is_file():
        return {"missing": True}
    metric = _load(path)
    return {
        "success": metric.get("paper_success", metric.get("success")),
        "completion": metric.get("completion"),
        "local_mpjpe_mm": metric.get("local_mpjpe_mm"),
        "xy_aligned_mpjpe_mm": metric.get("xy_aligned_mpjpe_mm"),
        "root_height_err_mm": (
            float(metric["root_height_err_mean"]) * 1000.0
            if metric.get("root_height_err_mean") is not None
            else None
        ),
        "evel_mps": metric.get("local_mpjve_mps", metric.get("mpjve_mps")),
        "eacc_mps2": metric.get("local_mpjae_mps2", metric.get("mpjae_mps2")),
        "frames": frames,
        "covered_frames": metric.get("covered_frames"),
        "control_start_alignment_err": metric.get("control_start_alignment_err"),
    }


def _frames_count(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    data = _load(path)
    frames = int(data.get("num_frames") or len(data.get("frames") or []))
    fps = data.get("fps")
    return frames, int(round(float(fps))) if fps is not None else None


def _normalize_offsets(row: dict[str, Any]) -> None:
    offsets = {
        "reference": [-4.8, 0, 0],
        "protomotions": [-2.4, 0, 0],
        "any2track": [0.0, 0, 0],
        "humanoid_gpt": [2.4, 0, 0],
        "sonic": [4.8, 0, 0],
    }
    for actor in row.get("actors", []):
        key = actor.get("key")
        if key in offsets:
            actor["offset"] = offsets[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/case_compare_viz/manifest.json"),
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1"),
    )
    args = parser.parse_args()

    manifest = _load(args.manifest)
    added = 0
    missing = []
    for row in manifest.get("rows", []):
        split, name = _split_and_name(row)
        run_dir = args.protocol_root / "runs" / "sonic" / split / name
        frames_path = (run_dir / "robot_frames.json").resolve()
        metrics_path = run_dir / "metrics.json"
        frames, fps = _frames_count(frames_path)
        metric = _metric_payload(metrics_path, frames)
        if fps is not None:
            metric["fps"] = fps
        row.setdefault("metrics", {}).update(
            {
                "sonic_success": metric.get("success"),
                "sonic_completion": metric.get("completion"),
                "sonic_local_mpjpe_mm": metric.get("local_mpjpe_mm"),
                "sonic_frames": frames,
            }
        )
        row["actors"] = [actor for actor in row.get("actors", []) if actor.get("key") != "sonic"]
        if frames_path.is_file():
            row["actors"].append(
                {
                    "key": "sonic",
                    "title": "SONIC",
                    "group": "released local-motion tracker",
                    "path": str(frames_path),
                    "color": "#2fb8a2",
                    "offset": [4.8, 0, 0],
                    "metrics": metric,
                }
            )
            added += 1
        else:
            missing.append(f"{split}/{name}")
        _normalize_offsets(row)

    manifest.setdefault("sources", {})["sonic"] = {
        "protocol_root": str(args.protocol_root.resolve()),
        "runner": str(Path("scripts/embodied/run_table2_sonic_eval_shards.sh").resolve()),
        "missing": missing,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "added": added, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()

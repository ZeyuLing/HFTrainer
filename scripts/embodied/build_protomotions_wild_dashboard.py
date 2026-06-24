#!/usr/bin/env python3
"""Build the in-the-wild ProtoMotions G1 reference-vs-tracker dashboard.

This consumes the frozen heldout judge directory produced for Table 2:
  - heldout_score.json contains per-case completion/fall/judge metrics.
  - judge/proto/hNNN.motion contains the reference motion.
  - judge/json/hNNN.json contains the actual tracker rollout.

The output schema matches build_protomotions_ref_track_dashboard.py so all
datasets can share the same static viewer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.build_protomotions_ref_track_dashboard import (  # noqa: E402
    _align_xy_to_reference,
    _copy_assets,
    _html_doc,
    _quat_to_mat_wxyz,
    _rel,
    _series,
    _summary,
)
from scripts.embodied.physflow_triplet_manifest import motion_to_robot_frames  # noqa: E402


def _load_robot_frames(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    pos = np.asarray([frame["body_pos"] for frame in data["frames"]], dtype=np.float32)
    quat_wxyz = np.asarray([frame["body_quat"] for frame in data["frames"]], dtype=np.float32)
    return data, pos, quat_wxyz


def _local_pos_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    root = pos[:, :1, :]
    root_rot = _quat_to_mat_wxyz(quat_wxyz[:, 0, :])
    return np.einsum("tbc,tcd->tbd", pos - root, root_rot)


def _metrics_wxyz(
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    pred_pos: np.ndarray,
    pred_quat: np.ndarray,
    dt: float,
    judge_success: bool,
) -> dict[str, float]:
    frames = min(len(ref_pos), len(pred_pos), len(ref_quat), len(pred_quat))
    ref_pos = ref_pos[:frames]
    pred_pos = pred_pos[:frames]
    ref_quat = ref_quat[:frames]
    pred_quat = pred_quat[:frames]
    ref_local = _local_pos_wxyz(ref_pos, ref_quat)
    pred_local = _local_pos_wxyz(pred_pos, pred_quat)
    aligned_body_err = np.linalg.norm(pred_pos - ref_pos, axis=-1)
    local_err = np.linalg.norm(pred_local - ref_local, axis=-1)
    root_err = np.linalg.norm(pred_pos[:, 0, :] - ref_pos[:, 0, :], axis=-1)
    root_height_err = np.abs(pred_pos[:, 0, 2] - ref_pos[:, 0, 2])
    local_step_vel = np.linalg.norm(np.diff(pred_local, axis=0) - np.diff(ref_local, axis=0), axis=-1)
    if frames > 2:
        local_step_acc = np.linalg.norm(
            np.diff(pred_local, n=2, axis=0) - np.diff(ref_local, n=2, axis=0),
            axis=-1,
        )
    else:
        local_step_acc = np.asarray([float("nan")], dtype=np.float32)
    safe_dt = max(float(dt), 1e-9)
    local_mpjpe_m = float(local_err.mean()) if frames else float("nan")
    root_height_m = float(root_height_err.mean()) if frames else float("nan")
    local_gate_success = local_mpjpe_m <= 0.2 and root_height_m <= 0.2
    return {
        "frames": float(frames),
        "success": float(judge_success),
        "local_gate_success": float(local_gate_success),
        "aligned_global_mpjpe_mm": float(aligned_body_err.mean() * 1000.0) if frames else float("nan"),
        "local_mpjpe_mm": float(local_mpjpe_m * 1000.0),
        "root_err_m": float(root_err.mean()) if frames else float("nan"),
        "root_err_max_m": float(root_err.max()) if frames else float("nan"),
        "root_height_err_m": root_height_m,
        "local_mpjve_mps": float(np.nanmean(local_step_vel) / safe_dt),
        "local_mpjae_mps2": float(np.nanmean(local_step_acc) / (safe_dt * safe_dt)),
        "ref_disp_m": float(np.linalg.norm(ref_pos[-1, 0, :2] - ref_pos[0, 0, :2])) if frames > 1 else 0.0,
        "track_disp_m": float(np.linalg.norm(pred_pos[-1, 0, :2] - pred_pos[0, 0, :2])) if frames > 1 else 0.0,
    }


def _write_robot_frames(path: Path, template: dict[str, Any], pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    out = dict(template)
    out["fps"] = 30
    out["source_fps"] = template.get("fps")
    out["source_fps_note"] = "normalized_to_30fps_for_visual_inspection"
    out["num_frames"] = int(len(pos))
    out["frames"] = [
        {"body_pos": pos[i].astype(float).tolist(), "body_quat": quat_wxyz[i].astype(float).tolist()}
        for i in range(len(pos))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))


def _copy_reference_motion(motion_path: Path, out_path: Path) -> None:
    motion_to_robot_frames(motion_path, out_path)
    data, pos, quat = _load_robot_frames(out_path)
    _write_robot_frames(out_path, data, pos, quat)


def _build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    score = json.loads(args.score_json.read_text())
    complete_thresh = float(score.get("complete_thresh", 0.9))
    rows = []
    for row in score["rows"]:
        idx = int(row["idx"])
        stem = f"h{idx:03d}"
        ref_src = args.heldout_root / "judge" / "proto" / f"{stem}.motion"
        track_src = args.heldout_root / "judge" / "json" / f"{stem}.json"
        if not ref_src.is_file() or not track_src.is_file():
            raise FileNotFoundError(f"Missing wild case artifacts for {stem}")

        case_id = f"{args.dataset_key}_g{idx:05d}_{Path(row['name']).stem}"
        case_dir = args.out_dir / "data" / case_id
        ref_json = case_dir / "reference.json"
        track_json = case_dir / "tracked.json"
        _copy_reference_motion(ref_src, ref_json)
        ref_data, ref_pos, ref_quat = _load_robot_frames(ref_json)
        track_data, pred_pos_raw, pred_quat = _load_robot_frames(track_src)
        frames = min(len(ref_pos), len(pred_pos_raw), len(ref_quat), len(pred_quat))
        pred_pos = _align_xy_to_reference(pred_pos_raw, ref_pos)
        judge_success = float(row.get("completion", 0.0)) >= complete_thresh and not bool(row.get("fall", False))
        metrics = _metrics_wxyz(ref_pos[:frames], ref_quat[:frames], pred_pos[:frames], pred_quat[:frames], 1.0 / 30.0, judge_success)
        _write_robot_frames(ref_json, ref_data, ref_pos, ref_quat)
        _write_robot_frames(track_json, track_data, pred_pos, pred_quat)

        agg = dict(row)
        agg["success"] = float(judge_success)
        agg["local_mpjpe_mm"] = metrics["local_mpjpe_mm"]
        agg["aligned_global_mpjpe_mm"] = metrics["aligned_global_mpjpe_mm"]
        agg["reference_frames"] = int(len(ref_pos))
        agg["tracker_frames"] = int(len(pred_pos))
        agg["overlap_frames"] = int(frames)
        rows.append(
            {
                "id": case_id,
                "dataset": args.dataset_name,
                "stem": stem + "_" + Path(row["name"]).stem,
                "global_index": idx,
                "shard": "heldout",
                "motion_id": idx,
                "source_motion": row.get("g1_path", ""),
                "paths": {
                    "reference": _rel(ref_json, args.out_dir),
                    "tracked": _rel(track_json, args.out_dir),
                },
                "metrics": {
                    "visual_recomputed": metrics,
                    "aggregator_row": agg,
                    "series": _series(ref_pos, pred_pos),
                },
            }
        )
    rows.sort(
        key=lambda item: (
            float(item["metrics"]["visual_recomputed"]["success"]),
            float(item["metrics"]["aggregator_row"].get("completion", 0.0)),
            -float(item["metrics"]["visual_recomputed"]["local_mpjpe_mm"]),
            int(item["global_index"]),
        )
    )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.heldout_root = args.heldout_root.resolve()
    args.score_json = args.score_json.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _copy_assets(args.out_dir)
    rows = _build_rows(args)
    data = {
        "dataset": args.dataset_name,
        "dataset_key": args.dataset_key,
        "method": args.method,
        "source": {
            "heldout_root": str(args.heldout_root),
            "score_json": str(args.score_json),
        },
        "summary": _summary(rows),
        "rows": rows,
        "sibling_links": [],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.out_dir / "index.html").write_text(_html_doc(data), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-root", type=Path, default=Path("output/heldout_frozen_score"))
    parser.add_argument("--score-json", type=Path, default=Path("output/heldout_frozen_score/heldout_score.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="In-the-Wild G1 Heldout")
    parser.add_argument("--dataset-key", default="wild")
    parser.add_argument("--method", default="protomotions_g1_bones")
    args = parser.parse_args()
    data = build(args)
    print(args.out_dir / "index.html")
    print(json.dumps(data["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

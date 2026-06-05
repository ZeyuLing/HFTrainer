#!/usr/bin/env python3
"""Aggregate metrics from ProtoMotions saved predicted MotionLib shards.

The ProtoMotions evaluator can save closed-loop rollout trajectories as
MotionLib-compatible ``predicted_motion_lib_epoch_*.pt`` files.  This script
compares those trajectories against the reference MotionLib shards used for
evaluation and reports OpenTrack-style SR/MPJPE/MPJVE metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-9)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    ).astype(np.float32)


def _starts_and_lens(lib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    starts = _as_np(lib["length_starts"]).astype(np.int64)
    lens = _as_np(lib["motion_num_frames"]).astype(np.int64)
    return starts, lens


def _motion_slice(lib: dict[str, Any], starts: np.ndarray, lens: np.ndarray, motion_id: int) -> slice:
    start = int(starts[motion_id])
    return slice(start, start + int(lens[motion_id]))


def _local_pos(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    root = pos[:, :1, :]
    root_rot = _quat_to_mat_wxyz(rot[:, 0, :])
    rel = pos - root
    return np.einsum("tbc,tcd->tbd", rel, root_rot)


def _per_motion_metrics(ref: dict[str, Any], pred: dict[str, Any]) -> list[dict[str, float]]:
    ref_starts, ref_lens = _starts_and_lens(ref)
    pred_starts, pred_lens = _starts_and_lens(pred)
    n = min(len(ref_lens), len(pred_lens))

    ref_pos_all = _as_np(ref["gts"]).astype(np.float32)
    pred_pos_all = _as_np(pred["gts"]).astype(np.float32)
    ref_rot_all = _as_np(ref["grs"]).astype(np.float32)
    pred_rot_all = _as_np(pred["grs"]).astype(np.float32)
    ref_dt = _as_np(ref["motion_dt"]).astype(np.float32).reshape(-1)
    pred_dt = _as_np(pred["motion_dt"]).astype(np.float32).reshape(-1)

    rows = []
    for motion_id in range(n):
        ref_slice = _motion_slice(ref, ref_starts, ref_lens, motion_id)
        pred_slice = _motion_slice(pred, pred_starts, pred_lens, motion_id)
        frames = min(ref_lens[motion_id], pred_lens[motion_id])
        if frames <= 1:
            continue

        ref_pos = ref_pos_all[ref_slice][:frames]
        pred_pos = pred_pos_all[pred_slice][:frames]
        ref_rot = ref_rot_all[ref_slice][:frames]
        pred_rot = pred_rot_all[pred_slice][:frames]

        ref_local = _local_pos(ref_pos, ref_rot)
        pred_local = _local_pos(pred_pos, pred_rot)

        global_mpjpe_m = float(np.linalg.norm(pred_pos - ref_pos, axis=-1).mean())
        local_mpjpe_m = float(np.linalg.norm(pred_local - ref_local, axis=-1).mean())
        root_height_err_m = float(np.abs(pred_pos[:, 0, 2] - ref_pos[:, 0, 2]).mean())

        global_step_vel_err_m = float(
            np.linalg.norm(np.diff(pred_pos, axis=0) - np.diff(ref_pos, axis=0), axis=-1).mean()
        )
        local_step_vel_err_m = float(
            np.linalg.norm(
                np.diff(pred_local, axis=0) - np.diff(ref_local, axis=0), axis=-1
            ).mean()
        )
        if frames > 2:
            global_step_acc_err_m = float(
                np.linalg.norm(
                    np.diff(pred_pos, n=2, axis=0) - np.diff(ref_pos, n=2, axis=0),
                    axis=-1,
                ).mean()
            )
            local_step_acc_err_m = float(
                np.linalg.norm(
                    np.diff(pred_local, n=2, axis=0) - np.diff(ref_local, n=2, axis=0),
                    axis=-1,
                ).mean()
            )
        else:
            global_step_acc_err_m = float("nan")
            local_step_acc_err_m = float("nan")

        dt = float(max(ref_dt[min(motion_id, len(ref_dt) - 1)], pred_dt[min(motion_id, len(pred_dt) - 1)]))
        local_vel_mps = local_step_vel_err_m / max(dt, 1e-9)
        global_vel_mps = global_step_vel_err_m / max(dt, 1e-9)
        local_acc_mps2 = local_step_acc_err_m / max(dt * dt, 1e-9)
        global_acc_mps2 = global_step_acc_err_m / max(dt * dt, 1e-9)

        paper_failed = local_mpjpe_m > 0.2 or root_height_err_m > 0.2
        rows.append(
            {
                "motion_id": float(motion_id),
                "frames": float(frames),
                "success": float(not paper_failed),
                "mpjpe_mm": global_mpjpe_m * 1000.0,
                "local_mpjpe_mm": local_mpjpe_m * 1000.0,
                "mpjve_mm_per_frame": global_step_vel_err_m * 1000.0,
                "local_mpjve_mm_per_frame": local_step_vel_err_m * 1000.0,
                "mpjve_mps": global_vel_mps,
                "local_mpjve_mps": local_vel_mps,
                "mpjae_mm_per_frame2": global_step_acc_err_m * 1000.0,
                "local_mpjae_mm_per_frame2": local_step_acc_err_m * 1000.0,
                "mpjae_mps2": global_acc_mps2,
                "local_mpjae_mps2": local_acc_mps2,
                "root_height_err_m": root_height_err_m,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {"num_motions": float(len(rows))}
    if not rows:
        return out
    keys = [k for k in rows[0].keys() if k not in {"motion_id", "frames"}]
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(np.nanmean(vals))
    out["success_rate"] = out.pop("success")
    return out


def _latest_predicted(root: Path) -> Path | None:
    candidates = sorted((root / "results").glob("predicted_motion_lib_epoch_*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--motion-base", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--shard-file-template",
        default="amass_g1_full_shard_{shard}.pt",
    )
    args = parser.parse_args()

    all_results: dict[str, dict[str, Any]] = {}
    missing: dict[str, list[str]] = {}

    for eval_dir in sorted(args.eval_root.glob("eval_*")):
        name = eval_dir.name[len("eval_") :]
        rows: list[dict[str, float]] = []
        missing_paths: list[str] = []
        for shard in range(args.num_shards):
            ref_path = args.motion_base / args.shard_file_template.format(shard=shard)
            pred_root = eval_dir / f"predicted_shard_{shard}"
            pred_path = _latest_predicted(pred_root)
            if not ref_path.exists() or pred_path is None:
                missing_paths.append(str(pred_root / "results"))
                continue
            rows.extend(_per_motion_metrics(_load(ref_path), _load(pred_path)))

        if rows:
            all_results[name] = {
                "summary": _aggregate(rows),
                "motions": rows,
            }
        if missing_paths:
            missing[name] = missing_paths

    payload = {"results": all_results, "missing_predicted_motion_libs": missing}
    (args.eval_root / "predicted_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    metric_order = [
        "num_motions",
        "success_rate",
        "local_mpjpe_mm",
        "local_mpjve_mm_per_frame",
        "mpjpe_mm",
        "mpjve_mm_per_frame",
        "root_height_err_m",
        "local_mpjve_mps",
        "local_mpjae_mps2",
    ]
    lines = ["| baseline | " + " | ".join(metric_order) + " |"]
    lines.append("|---|" + "|".join(["---:"] * len(metric_order)) + "|")
    for name, data in sorted(all_results.items()):
        summary = data["summary"]
        vals = [summary.get(key, float("nan")) for key in metric_order]
        lines.append(f"| {name} | " + " | ".join(f"{v:.6g}" for v in vals) + " |")
    if missing:
        lines.append("")
        lines.append("Missing predicted motion libs:")
        for name, paths in sorted(missing.items()):
            lines.append(f"- {name}: {len(paths)} shard(s)")

    (args.eval_root / "predicted_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

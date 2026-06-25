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


DEFAULT_LOCAL_SUCCESS_THRESH_M = 0.2
DEFAULT_ROOT_HEIGHT_SUCCESS_THRESH_M = 0.2
DEFAULT_ROOT_TRAJ_SUCCESS_THRESH_M = 0.5
DEFAULT_COMPLETION_SUCCESS_THRESH = 0.95


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


def _motion_lengths(lib: dict[str, Any], lens: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if "motion_lengths" in lib:
        return _as_np(lib["motion_lengths"]).astype(np.float32).reshape(-1)
    return np.maximum(lens.astype(np.float32) - 1.0, 0.0) * dt.astype(np.float32)


def _interp_linear(values: np.ndarray, src_dt: float, query_times: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    src_times = np.arange(values.shape[0], dtype=np.float32) * float(src_dt)
    query = np.minimum(query_times.astype(np.float32), src_times[-1])
    out = np.stack([np.interp(query, src_times, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape((len(query_times),) + values.shape[1:]).astype(np.float32)


def _interp_quat_xyzw(quat: np.ndarray, src_dt: float, query_times: np.ndarray) -> np.ndarray:
    if quat.shape[0] == 1:
        return np.repeat(quat, len(query_times), axis=0).astype(np.float32)

    src_times = np.arange(quat.shape[0], dtype=np.float32) * float(src_dt)
    query = np.minimum(query_times.astype(np.float32), src_times[-1])
    idx0 = np.floor(query / float(src_dt)).astype(np.int64)
    idx0 = np.clip(idx0, 0, quat.shape[0] - 1)
    idx1 = np.clip(idx0 + 1, 0, quat.shape[0] - 1)
    blend = ((query - idx0.astype(np.float32) * float(src_dt)) / max(float(src_dt), 1e-9)).reshape(-1, 1)

    q0 = quat[idx0].astype(np.float32)
    q1 = quat[idx1].astype(np.float32)
    while blend.ndim < q0.ndim:
        blend = np.expand_dims(blend, axis=-1)
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True).clip(min=1e-9)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True).clip(min=1e-9)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot).clip(0.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = sin_theta < 1e-5
    s0 = np.sin((1.0 - blend) * theta) / np.where(linear, 1.0, sin_theta)
    s1 = np.sin(blend * theta) / np.where(linear, 1.0, sin_theta)
    out = np.where(linear, (1.0 - blend) * q0 + blend * q1, s0 * q0 + s1 * q1)
    out = out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-9)
    return out.astype(np.float32)


def _local_pos(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    root = pos[:, :1, :]
    # ProtoMotions MotionLib stores rigid-body rotations as XYZW.  Convert the
    # root quaternion before building the root-relative frame used by MPJPE.
    root_rot = _quat_to_mat_wxyz(rot[:, 0, [3, 0, 1, 2]])
    rel = pos - root
    return np.einsum("tbc,tcd->tbd", rel, root_rot)


def _align_xy_to_reference(pred_pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """Remove IsaacGym/VectorEnv XY grid offsets before global MPJPE.

    ProtoMotions saved rollouts can be translated to per-environment grid
    origins.  That offset is not tracking error, but height/root drift should
    still count, so only the initial XY difference is stripped.
    """
    aligned = pred_pos.copy()
    aligned[..., :2] -= aligned[0, 0, :2] - ref_pos[0, 0, :2]
    return aligned


def _per_motion_metrics(
    ref: dict[str, Any],
    pred: dict[str, Any],
    *,
    local_success_thresh_m: float,
    root_height_success_thresh_m: float,
    root_traj_success_thresh_m: float,
    completion_success_thresh: float,
) -> list[dict[str, float]]:
    ref_starts, ref_lens = _starts_and_lens(ref)
    pred_starts, pred_lens = _starts_and_lens(pred)
    n = min(len(ref_lens), len(pred_lens))

    ref_pos_all = _as_np(ref["gts"]).astype(np.float32)
    pred_pos_all = _as_np(pred["gts"]).astype(np.float32)
    ref_rot_all = _as_np(ref["grs"]).astype(np.float32)
    pred_rot_all = _as_np(pred["grs"]).astype(np.float32)
    ref_dt = _as_np(ref["motion_dt"]).astype(np.float32).reshape(-1)
    pred_dt = _as_np(pred["motion_dt"]).astype(np.float32).reshape(-1)
    ref_lengths = _motion_lengths(ref, ref_lens, ref_dt)

    rows = []
    for motion_id in range(n):
        ref_slice = _motion_slice(ref, ref_starts, ref_lens, motion_id)
        pred_slice = _motion_slice(pred, pred_starts, pred_lens, motion_id)
        ref_motion_dt = float(ref_dt[min(motion_id, len(ref_dt) - 1)])
        pred_motion_dt = float(pred_dt[min(motion_id, len(pred_dt) - 1)])
        ref_motion_length = float(ref_lengths[min(motion_id, len(ref_lengths) - 1)])
        max_ref_frames_at_pred_dt = int(np.floor(ref_motion_length / max(pred_motion_dt, 1e-9))) + 1
        frames = min(int(pred_lens[motion_id]), max_ref_frames_at_pred_dt)
        if frames <= 1:
            continue

        query_times = np.arange(frames, dtype=np.float32) * pred_motion_dt
        ref_pos_src = ref_pos_all[ref_slice]
        ref_rot_src = ref_rot_all[ref_slice]
        ref_pos = _interp_linear(ref_pos_src, ref_motion_dt, query_times)
        ref_rot = _interp_quat_xyzw(ref_rot_src.reshape(ref_rot_src.shape[0], -1, 4), ref_motion_dt, query_times)
        pred_pos = pred_pos_all[pred_slice][:frames]
        pred_pos_aligned = _align_xy_to_reference(pred_pos, ref_pos)
        pred_rot = pred_rot_all[pred_slice][:frames]

        ref_local = _local_pos(ref_pos, ref_rot)
        pred_local = _local_pos(pred_pos, pred_rot)

        raw_global_mpjpe_m = float(np.linalg.norm(pred_pos - ref_pos, axis=-1).mean())
        aligned_global_mpjpe_m = float(np.linalg.norm(pred_pos_aligned - ref_pos, axis=-1).mean())
        local_mpjpe_m = float(np.linalg.norm(pred_local - ref_local, axis=-1).mean())
        root_err = np.linalg.norm(pred_pos_aligned[:, 0, :] - ref_pos[:, 0, :], axis=-1)
        root_height_err_m = float(np.abs(pred_pos[:, 0, 2] - ref_pos[:, 0, 2]).mean())

        global_step_vel_err_m = float(
            np.linalg.norm(np.diff(pred_pos_aligned, axis=0) - np.diff(ref_pos, axis=0), axis=-1).mean()
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

        dt = pred_motion_dt
        local_vel_mps = local_step_vel_err_m / max(dt, 1e-9)
        global_vel_mps = global_step_vel_err_m / max(dt, 1e-9)
        local_acc_mps2 = local_step_acc_err_m / max(dt * dt, 1e-9)
        global_acc_mps2 = global_step_acc_err_m / max(dt * dt, 1e-9)

        root_err_m = float(root_err.mean())
        pred_duration = float(pred_lens[motion_id]) * pred_motion_dt
        completion = min(pred_duration / max(ref_motion_length, pred_motion_dt), 1.0)
        paper_failed = (
            completion < completion_success_thresh
            or
            local_mpjpe_m > local_success_thresh_m
            or root_height_err_m > root_height_success_thresh_m
            or root_err_m > root_traj_success_thresh_m
        )
        rows.append(
            {
                "motion_id": float(motion_id),
                "frames": float(frames),
                "ref_dt": ref_motion_dt,
                "pred_dt": pred_motion_dt,
                "ref_motion_length_s": ref_motion_length,
                "pred_motion_length_s": pred_duration,
                "completion": completion,
                "success": float(not paper_failed),
                "mpjpe_mm": aligned_global_mpjpe_m * 1000.0,
                "aligned_global_mpjpe_mm": aligned_global_mpjpe_m * 1000.0,
                "raw_global_mpjpe_mm": raw_global_mpjpe_m * 1000.0,
                "local_mpjpe_mm": local_mpjpe_m * 1000.0,
                "mpjve_mm_per_frame": global_step_vel_err_m * 1000.0,
                "local_mpjve_mm_per_frame": local_step_vel_err_m * 1000.0,
                "mpjve_mps": global_vel_mps,
                "local_mpjve_mps": local_vel_mps,
                "mpjae_mm_per_frame2": global_step_acc_err_m * 1000.0,
                "local_mpjae_mm_per_frame2": local_step_acc_err_m * 1000.0,
                "mpjae_mps2": global_acc_mps2,
                "local_mpjae_mps2": local_acc_mps2,
                "root_err_m": root_err_m,
                "root_err_max_m": float(root_err.max()),
                "root_height_err_m": root_height_err_m,
                "success_local_thresh_m": float(local_success_thresh_m),
                "success_root_height_thresh_m": float(root_height_success_thresh_m),
                "success_root_traj_thresh_m": float(root_traj_success_thresh_m),
                "success_completion_thresh": float(completion_success_thresh),
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
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated eval method names to aggregate, e.g. protomotions_g1_bones.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <eval-root>/predicted_metrics.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output Markdown path. Defaults to <eval-root>/predicted_metrics.md.",
    )
    parser.add_argument(
        "--local-success-thresh-m",
        type=float,
        default=DEFAULT_LOCAL_SUCCESS_THRESH_M,
        help="Mean root-frame MPJPE threshold for strict rollout success.",
    )
    parser.add_argument(
        "--root-height-success-thresh-m",
        type=float,
        default=DEFAULT_ROOT_HEIGHT_SUCCESS_THRESH_M,
        help="Mean root-height error threshold for strict rollout success.",
    )
    parser.add_argument(
        "--root-traj-success-thresh-m",
        type=float,
        default=DEFAULT_ROOT_TRAJ_SUCCESS_THRESH_M,
        help="Mean root trajectory error threshold for strict rollout success.",
    )
    parser.add_argument(
        "--completion-success-thresh",
        type=float,
        default=DEFAULT_COMPLETION_SUCCESS_THRESH,
        help="Minimum executed/reference duration ratio for strict rollout success.",
    )
    args = parser.parse_args()

    all_results: dict[str, dict[str, Any]] = {}
    missing: dict[str, list[str]] = {}
    method_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    for eval_dir in sorted(args.eval_root.glob("eval_*")):
        name = eval_dir.name[len("eval_") :]
        if method_filter and name not in method_filter:
            continue
        rows: list[dict[str, float]] = []
        missing_paths: list[str] = []
        for shard in range(args.num_shards):
            ref_path = args.motion_base / args.shard_file_template.format(shard=shard)
            pred_root = eval_dir / f"predicted_shard_{shard}"
            pred_path = _latest_predicted(pred_root)
            if not ref_path.exists() or pred_path is None:
                missing_paths.append(str(pred_root / "results"))
                continue
            rows.extend(
                _per_motion_metrics(
                    _load(ref_path),
                    _load(pred_path),
                    local_success_thresh_m=args.local_success_thresh_m,
                    root_height_success_thresh_m=args.root_height_success_thresh_m,
                    root_traj_success_thresh_m=args.root_traj_success_thresh_m,
                    completion_success_thresh=args.completion_success_thresh,
                )
            )

        if rows:
            all_results[name] = {
                "summary": _aggregate(rows),
                "motions": rows,
            }
        if missing_paths:
            missing[name] = missing_paths

    payload = {"results": all_results, "missing_predicted_motion_libs": missing}
    output_json = args.output_json or (args.eval_root / "predicted_metrics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    metric_order = [
        "num_motions",
        "success_rate",
        "aligned_global_mpjpe_mm",
        "raw_global_mpjpe_mm",
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

    output_md = args.output_md or (args.eval_root / "predicted_metrics.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

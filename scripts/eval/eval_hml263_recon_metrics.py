#!/usr/bin/env python3
"""Evaluate reconstruction metrics directly in HumanML3D-263 space.

This is a retarget-free sanity check for HML3D tokenizers.  The official Table 3
SMPL-space comparison is produced after HML263->SMPL conversion, but these
numbers are useful for detecting tokenizer or retargeting failures separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.evaluation.motion.m2m_eval_metrics import compute_pa_mpjpe
from hftrainer.motion.representation.rotation import rotation_6d_to_matrix
from scripts.eval.hml263_to_smpl_ik import (
    N_JOINTS,
    recover_from_ric,
    resample_linear,
)


T2M22_EDGES = [
    [0, 2], [2, 5], [5, 8], [8, 11],
    [0, 1], [1, 4], [4, 7], [7, 10],
    [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
    [9, 14], [14, 17], [17, 19], [19, 21],
    [9, 13], [13, 16], [16, 18], [18, 20],
]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_samples": int(arr.size)}


def _geodesic_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    rel = np.matmul(np.swapaxes(pred, -1, -2), gt)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = [q[..., i] for i in range(4)]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _recover_hml263_local_rotations(data: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Recover root yaw + 21 HumanML3D cont6d local rotations as matrices."""
    data = resample_linear(np.asarray(data, dtype=np.float32), source_fps, target_fps)
    rot_vel = data[:, 0]
    root_ang = np.zeros_like(rot_vel)
    root_ang[1:] = rot_vel[:-1]
    root_ang = np.cumsum(root_ang, axis=0)
    root_quat = np.zeros((len(data), 4), dtype=np.float32)
    root_quat[:, 0] = np.cos(root_ang)
    root_quat[:, 2] = np.sin(root_ang)
    root_rot = _quat_to_matrix(root_quat)

    rot_start = 4 + (N_JOINTS - 1) * 3
    rot_end = rot_start + (N_JOINTS - 1) * 6
    body_6d = data[:, rot_start:rot_end].reshape(len(data), N_JOINTS - 1, 6)
    body_rot = rotation_6d_to_matrix(body_6d, convention="column").astype(np.float32)
    return np.concatenate([root_rot[:, None], body_rot], axis=1)


def _root_aware_position_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    raw = np.linalg.norm(pred - gt, axis=-1)
    root_delta = pred[:, 0, :] - gt[:, 0, :]
    root0_shift = gt[0:1, 0:1, :] - pred[0:1, 0:1, :]
    root0 = np.linalg.norm((pred + root0_shift) - gt, axis=-1)
    rootframe = np.linalg.norm(
        (pred - pred[:, 0:1, :]) - (gt - gt[:, 0:1, :]),
        axis=-1,
    )
    rootframe_mm = float(rootframe.mean() * 1000.0)
    return {
        "mpjpe_mm": rootframe_mm,
        "raw_mpjpe_mm": float(raw.mean() * 1000.0),
        "root0_mpjpe_mm": float(root0.mean() * 1000.0),
        "rootframe_mpjpe_mm": rootframe_mm,
        "root_mpjpe_mm": float(np.linalg.norm(root_delta, axis=-1).mean() * 1000.0),
    }


def _gt_quality_issue(gt_pos: np.ndarray) -> str | None:
    if not np.isfinite(gt_pos).all():
        return "non_finite"
    flat = gt_pos.reshape(-1, 3)
    y_min = float(flat[:, 1].min())
    y_span = float(flat[:, 1].max() - y_min)
    root_y_max = float(gt_pos[:, 0, 1].max())
    bones = [
        np.linalg.norm(gt_pos[:, a] - gt_pos[:, b], axis=-1)
        for a, b in T2M22_EDGES
    ]
    bone_max = float(np.stack(bones, axis=-1).max()) if bones else 0.0
    if abs(y_min) > 0.08:
        return f"floor_y={y_min:.3f}"
    if y_span > 2.50:
        return f"span_y={y_span:.3f}"
    if root_y_max > 2.40:
        return f"root_y_max={root_y_max:.3f}"
    if bone_max > 0.80:
        return f"bone_max={bone_max:.3f}"
    return None


def _load_ids(gt_dir: Path, pred_dir: Path, ids_file: str | None) -> list[str]:
    if ids_file:
        raw = [line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = sorted(path.stem for path in gt_dir.glob("*.npy"))
    return [sid for sid in raw if (gt_dir / f"{sid}.npy").exists() and (pred_dir / f"{sid}.npy").exists()]


def _load_features(path: Path) -> np.ndarray:
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 263:
        raise ValueError(f"expected (T,263), got {arr.shape}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--floor-align", action="store_true")
    parser.add_argument(
        "--skip-bad-gt",
        action="store_true",
        help="Exclude HML GT clips with implausible recovered height/root/bone ranges.",
    )
    parser.add_argument(
        "--quality-report-json",
        default="",
        help="Optional path to write bad-GT exclusions from this metric run.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    ids = _load_ids(gt_dir, pred_dir, args.ids or None)
    if args.limit:
        ids = ids[: args.limit]

    values: dict[str, list[float]] = {
        "mpjpe_mm": [],
        "raw_mpjpe_mm": [],
        "root0_mpjpe_mm": [],
        "rootframe_mpjpe_mm": [],
        "root_mpjpe_mm": [],
        "pa_mpjpe_mm": [],
        "mpjre_deg": [],
    }
    per_case: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    excluded_quality: list[dict[str, str]] = []
    frame_deltas: list[int] = []

    for sid in ids:
        try:
            gt_feat = _load_features(gt_dir / f"{sid}.npy")
            gt_pos = resample_linear(recover_from_ric(gt_feat, N_JOINTS), args.source_fps, args.target_fps)
            quality_issue = _gt_quality_issue(gt_pos)
            if args.skip_bad_gt and quality_issue is not None:
                excluded_quality.append({"key": sid, "reason": quality_issue})
                continue
            pred_feat = _load_features(pred_dir / f"{sid}.npy")
            pred_pos = resample_linear(recover_from_ric(pred_feat, N_JOINTS), args.source_fps, args.target_fps)
            gt_rot = _recover_hml263_local_rotations(gt_feat, args.source_fps, args.target_fps)
            pred_rot = _recover_hml263_local_rotations(pred_feat, args.source_fps, args.target_fps)
            t = min(len(gt_pos), len(pred_pos), len(gt_rot), len(pred_rot))
            if t <= 0:
                raise ValueError("empty sequence")
            frame_deltas.append(int(len(pred_pos) - len(gt_pos)))
            gt_pos = gt_pos[:t]
            pred_pos = pred_pos[:t]
            gt_rot = gt_rot[:t]
            pred_rot = pred_rot[:t]
            if args.floor_align:
                gt_pos = gt_pos.copy()
                pred_pos = pred_pos.copy()
                gt_pos[..., 1] -= gt_pos[..., 1].min()
                pred_pos[..., 1] -= pred_pos[..., 1].min()
            pos_metrics = _root_aware_position_metrics(pred_pos, gt_pos)
            mpjpe = pos_metrics["mpjpe_mm"]
            pa = float(compute_pa_mpjpe(pred_pos, gt_pos)["pa_mpjpe_mean"] * 1000.0)
            mpjre = _geodesic_deg(pred_rot, gt_rot)
            values["mpjpe_mm"].append(mpjpe)
            values["raw_mpjpe_mm"].append(pos_metrics["raw_mpjpe_mm"])
            values["root0_mpjpe_mm"].append(pos_metrics["root0_mpjpe_mm"])
            values["rootframe_mpjpe_mm"].append(pos_metrics["rootframe_mpjpe_mm"])
            values["root_mpjpe_mm"].append(pos_metrics["root_mpjpe_mm"])
            values["pa_mpjpe_mm"].append(pa)
            values["mpjre_deg"].append(mpjre)
            per_case.append(
                {
                    "key": sid,
                    "frames": int(t),
                    "gt_frames": int(len(gt_feat)),
                    "pred_frames": int(len(pred_feat)),
                    "mpjpe_mm": mpjpe,
                    "raw_mpjpe_mm": pos_metrics["raw_mpjpe_mm"],
                    "root0_mpjpe_mm": pos_metrics["root0_mpjpe_mm"],
                    "rootframe_mpjpe_mm": pos_metrics["rootframe_mpjpe_mm"],
                    "root_mpjpe_mm": pos_metrics["root_mpjpe_mm"],
                    "pa_mpjpe_mm": pa,
                    "mpjre_deg": mpjre,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": sid, "error": repr(exc)})

    payload = {
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "ids": args.ids,
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "floor_align": bool(args.floor_align),
        "selected": len(ids),
        "summary": {
            "mpjpe_mm": _summary(values["mpjpe_mm"]),
            "raw_mpjpe_mm": _summary(values["raw_mpjpe_mm"]),
            "root0_mpjpe_mm": _summary(values["root0_mpjpe_mm"]),
            "rootframe_mpjpe_mm": _summary(values["rootframe_mpjpe_mm"]),
            "root_mpjpe_mm": _summary(values["root_mpjpe_mm"]),
            "pa_mpjpe_mm": _summary(values["pa_mpjpe_mm"]),
            "mpjre_deg": _summary(values["mpjre_deg"]),
            "frame_delta_abs_mean": float(np.mean(np.abs(frame_deltas))) if frame_deltas else None,
            "frame_delta_abs_max": int(np.max(np.abs(frame_deltas))) if frame_deltas else None,
            "num_failures": len(failures),
            "num_quality_excluded": len(excluded_quality),
        },
        "quality_filter": {
            "enabled": bool(args.skip_bad_gt),
            "excluded": excluded_quality,
        },
        "failures": failures,
        "per_case": per_case,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.quality_report_json:
        quality_path = Path(args.quality_report_json)
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(
            json.dumps(payload["quality_filter"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"[hml263-recon-metrics] wrote {out}")


if __name__ == "__main__":
    main()

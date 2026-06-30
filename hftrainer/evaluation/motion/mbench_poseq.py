"""Reusable MBench Pose_Quality / PoseQ evaluator.

PoseQ is the MBench NRDF pose-naturalness distance:
``mean(NRDF.dist_pred) * 10`` averaged over clips.  MBench labels Pose Quality
with a down arrow, so lower values are better.

The official MBench render path saves full SMPL poses from SMPLify, then feeds
the saved ``pose`` tensor directly to NRDF.  The NRDF encoder reads only 21
parts, which effectively scores ``global_orient + body_pose[:20]``.  This module
mirrors that actual behavior for repository-native ``motion_135`` and
MotionStreamer-272 files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from hftrainer.evaluation.motion.mbench_physics import (
    MotionMode,
    list_metric_files,
    load_manifest,
)
from hftrainer.datasets.motion.representation.humanml_repr import (
    recover_local_rotations_and_root,
)

POSEQ_KEY = "PoseQuality"
POSEQ_DIRECTION: Literal["lower"] = "lower"
POSEQ_LOWER_IS_BETTER = True

_ROOT = Path(__file__).resolve().parents[3]
_VIMOGEN = _ROOT / "ref_repo" / "ViMoGen"
DEFAULT_NRDF_DIR = (
    _VIMOGEN
    / "checkpoints/nrdf/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1"
)


def _ensure_vimogen_on_path() -> None:
    for path in (str(_ROOT), str(_VIMOGEN)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _rot6d_to_rotmat_rowmajor(d6: np.ndarray) -> np.ndarray:
    """Convert row-major two-column 6D rotations to matrices."""

    x = np.asarray(d6, dtype=np.float32).reshape(*d6.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _reshape_body_pose(body_pose: np.ndarray) -> np.ndarray | None:
    body = np.asarray(body_pose, dtype=np.float32)
    if body.ndim == 2 and body.shape[1] == 63:
        return body.reshape(-1, 21, 3)
    if body.ndim == 3 and body.shape[1:] == (21, 3):
        return body
    return None


def poseq_axis_angle_from_motion135(motion_135: np.ndarray) -> np.ndarray | None:
    """Extract MBench-compatible first-21 axis-angle pose from ``motion_135``."""

    motion = np.asarray(motion_135, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 135:
        return None
    rot6d = motion[:, 3:135].reshape(-1, 22, 6)[:, :21]
    rotmat = _rot6d_to_rotmat_rowmajor(rot6d)
    return R.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 21, 3).astype(np.float32)


def poseq_axis_angle_from_smpl_npz(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Extract MBench-compatible first-21 axis-angle pose from SMPL-param NPZ."""

    if "body_pose" in data:
        body = _reshape_body_pose(np.asarray(data["body_pose"], dtype=np.float32))
        if body is None:
            return None
        if "global_orient" in data:
            root = np.asarray(data["global_orient"], dtype=np.float32).reshape(body.shape[0], 1, 3)
            return np.concatenate([root, body[:, :20]], axis=1).astype(np.float32)
        return body.astype(np.float32)

    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        if poses.ndim == 2 and poses.shape[1] >= 63:
            return poses[:, :63].reshape(-1, 21, 3).astype(np.float32)
    return None


def poseq_axis_angle_from_272(motion_272: np.ndarray) -> np.ndarray | None:
    """Extract MBench-compatible first-21 axis-angle pose from native 272 motion."""

    motion = np.asarray(motion_272, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272:
        return None
    rot, _ = recover_local_rotations_and_root(motion)
    rotmat = np.asarray(rot, dtype=np.float32)[:, :21]
    return R.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 21, 3).astype(np.float32)


def poseq_axis_angle_for_file(path: str | os.PathLike[str], mode: MotionMode) -> np.ndarray | None:
    """Load one supported motion file and return the first-21 NRDF axis-angle pose."""

    p = str(path)
    try:
        if mode == "m135":
            data = np.load(p, allow_pickle=True)
            if "motion_135" in data:
                return poseq_axis_angle_from_motion135(np.asarray(data["motion_135"], dtype=np.float32))
            return poseq_axis_angle_from_smpl_npz(data)
        if mode == "gt272":
            if p.endswith(".npz"):
                data = np.load(p, allow_pickle=True)
                if "motion_272" not in data:
                    return None
                motion_272 = np.asarray(data["motion_272"], dtype=np.float32)
            else:
                motion_272 = np.asarray(np.load(p), dtype=np.float32)
            return poseq_axis_angle_from_272(motion_272)
    except Exception:
        return None
    raise ValueError(f"Unknown mode {mode!r}")


def _resolve_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_poseq_model(
    model_dir: str | os.PathLike[str] = DEFAULT_NRDF_DIR,
    device: str | None = None,
):
    """Load the NRDF model used by MBench Pose_Quality."""

    import torch

    _ensure_vimogen_on_path()
    from mbench.third_party.NRDF.nrdf import NRDF, load_config

    model_path = Path(model_dir)
    resolved_device = _resolve_device(device)
    cfg = load_config(str(model_path / "config.yaml"))
    cfg.setdefault("train", {})["device"] = resolved_device
    model = NRDF(cfg)
    checkpoint_path = str(model_path / "checkpoints" / "checkpoint_epoch_best.tar")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=True,
        )["model_state_dict"]
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(resolved_device)
    model.eval()
    return model


def compute_poseq_from_axis_angle(
    axis_angle: np.ndarray,
    model,
    batch: int = 8192,
    device: str | None = None,
) -> float:
    """Compute PoseQ for one clip represented as ``(T,21,3)`` axis-angle."""

    import torch

    _ensure_vimogen_on_path()
    from mbench.third_party.NRDF import axis_angle_to_quaternion

    aa = np.asarray(axis_angle, dtype=np.float32)
    if aa.ndim != 3 or aa.shape[1:] != (21, 3):
        raise ValueError(f"Expected axis-angle shape (T,21,3), got {aa.shape}")
    resolved_device = _resolve_device(device)
    t = torch.from_numpy(aa).to(resolved_device)
    q = axis_angle_to_quaternion(t)
    dists = []
    with torch.no_grad():
        for start in range(0, q.shape[0], batch):
            dists.append(model(q[start : start + batch], train=False)["dist_pred"].flatten())
    return float(torch.cat(dists).mean().item() * 10.0)


def compute_poseq_for_file(
    path: str | os.PathLike[str],
    mode: MotionMode,
    model=None,
    model_dir: str | os.PathLike[str] = DEFAULT_NRDF_DIR,
    batch: int = 8192,
    device: str | None = None,
) -> float | None:
    """Compute MBench PoseQ for one file, returning ``None`` on unsupported data."""

    aa = poseq_axis_angle_for_file(path, mode)
    if aa is None or aa.shape[0] < 1:
        return None
    resolved_device = _resolve_device(device)
    nrdf = model if model is not None else load_poseq_model(model_dir, resolved_device)
    return compute_poseq_from_axis_angle(aa, nrdf, batch=batch, device=resolved_device)


def aggregate_poseq_scores(scores: list[float | None]) -> dict[str, float]:
    """Average per-clip PoseQ scores."""

    valid = [float(v) for v in scores if v is not None]
    return {
        "n": len(valid),
        POSEQ_KEY: float(np.mean(valid)) if valid else 0.0,
    }


def evaluate_poseq_dir(
    src: str | os.PathLike[str],
    mode: MotionMode,
    limit: int = 0,
    seed: int = 0,
    model=None,
    model_dir: str | os.PathLike[str] = DEFAULT_NRDF_DIR,
    batch: int = 8192,
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate MBench PoseQ over all metric files in one directory."""

    resolved_device = _resolve_device(device)
    nrdf = model if model is not None else load_poseq_model(model_dir, resolved_device)
    files = list_metric_files(src, mode, limit=limit, seed=seed)
    scores = [
        compute_poseq_for_file(
            path,
            mode,
            model=nrdf,
            model_dir=model_dir,
            batch=batch,
            device=resolved_device,
        )
        for path in files
    ]
    return aggregate_poseq_scores(scores)


def dump_results_json(results: dict[str, dict[str, float]], out_json: str | os.PathLike[str]) -> None:
    """Write PoseQ results to JSON, creating parent directories."""

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))


__all__ = [
    "DEFAULT_NRDF_DIR",
    "POSEQ_DIRECTION",
    "POSEQ_KEY",
    "POSEQ_LOWER_IS_BETTER",
    "aggregate_poseq_scores",
    "compute_poseq_for_file",
    "compute_poseq_from_axis_angle",
    "dump_results_json",
    "evaluate_poseq_dir",
    "list_metric_files",
    "load_manifest",
    "load_poseq_model",
    "poseq_axis_angle_for_file",
    "poseq_axis_angle_from_272",
    "poseq_axis_angle_from_motion135",
    "poseq_axis_angle_from_smpl_npz",
]

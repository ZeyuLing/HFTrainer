#!/usr/bin/env python3
"""Shared protocol helpers for HumanML3D mixed position control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


PAPER_FPS = 30
BASELINE_FPS = 20
POSITION_SETTINGS = ("P1", "P2", "P3", "P4")
HELDOUT_POSITION_SETTINGS = ("I1", "H1", "I2", "H2", "I3", "H3")
ALL_POSITION_SETTINGS = POSITION_SETTINGS + HELDOUT_POSITION_SETTINGS
MASKCONTROL_JOINTS = frozenset((0, 10, 11, 15, 20, 21))
_AXES = {"x": 0, "y": 1, "z": 2}


def _scale_interval(value: int, fps: int) -> int:
    return max(1, int(round(int(value) * int(fps) / PAPER_FPS)))


def setting_params(setting: str, fps: int = BASELINE_FPS) -> dict[str, Any]:
    """Return an E18/E20 position setting mapped to ``fps``."""
    if setting not in ALL_POSITION_SETTINGS:
        raise ValueError(
            f"setting must be one of {ALL_POSITION_SETTINGS}, got {setting!r}"
        )
    from hftrainer.evaluation.motion.m2m_eval_tasks import get_task

    task_id = "E18" if setting in POSITION_SETTINGS else "E20"
    params = deepcopy(get_task(task_id).settings[setting].mask_kwargs)
    if task_id == "E20":
        params["fps"] = int(fps)
        return params
    for key in ("waypoint_interval", "rotation_interval", "position_interval"):
        if key in params:
            params[key] = _scale_interval(params[key], fps)
    return params


def _src198_to_position_atoms(src_mask: np.ndarray) -> np.ndarray:
    """Map known HY-Motion-198 translation/position cells to joint XYZ atoms."""
    known = np.asarray(src_mask) < 0.5
    atoms = np.zeros((len(known), 22, 3), dtype=bool)
    atoms[:, 0, :] = known[:, :3]
    atoms[:, 1:, :] = known[:, 135:198].reshape(len(known), 21, 3)
    return atoms


def build_position_protocol(
    length: int,
    setting: str,
    *,
    fps: int = BASELINE_FPS,
    sample_seed: int = 0,
) -> dict[str, Any]:
    """Build requested atoms and clause masks for one mixed-control sample."""
    from hftrainer.evaluation.motion.m2m_eval_tasks import (
        build_heldout_layout_mask,
        build_mixed_condition_mask,
    )

    params = setting_params(setting, fps=fps)
    mask_builder = (
        build_mixed_condition_mask
        if setting in POSITION_SETTINGS
        else build_heldout_layout_mask
    )
    src_mask, info = mask_builder(
        int(length), D=198, sample_seed=int(sample_seed), **params)
    requested = _src198_to_position_atoms(src_mask)

    clauses: dict[str, np.ndarray] = {
        name: np.zeros_like(requested)
        for name in ("frame", "trajectory", "body", "joint")
    }
    frames = np.asarray(info["full_frame_indices"], dtype=np.int64)
    clauses["frame"][frames, :, :] = True

    traj_axes = [_AXES[value] for value in str(info["trajectory_axes"]).lower()]
    traj_frames = np.asarray(info["trajectory_frames"], dtype=np.int64)
    if len(traj_frames) and traj_axes:
        clauses["trajectory"][np.ix_(traj_frames, [0], traj_axes)] = True

    body_frames = np.asarray(
        info.get("body_position_frames", []), dtype=np.int64)
    body_joints = np.asarray(
        info.get("body_position_joints", []), dtype=np.int64)
    if len(body_frames) and len(body_joints):
        clauses["body"][np.ix_(body_frames, body_joints, range(3))] = True

    if setting in POSITION_SETTINGS:
        joint_frames = np.asarray(info["frames"], dtype=np.int64)
        joint_joints = np.asarray(info["joints"], dtype=np.int64)
        joint_axes = [_AXES[value] for value in str(info["axes"]).lower()]
        if len(joint_frames) and len(joint_joints) and joint_axes:
            clauses["joint"][
                np.ix_(joint_frames, joint_joints, joint_axes)
            ] = True
    else:
        assigned = (
            clauses["frame"] | clauses["trajectory"] | clauses["body"])
        clauses["joint"] = requested & ~assigned

    return {
        "setting": setting,
        "fps": int(fps),
        "requested": requested,
        "clauses": clauses,
        "info": info,
        "params": params,
    }


def method_protocol(requested: np.ndarray, method: str) -> dict[str, Any]:
    """Project requested atoms onto a released baseline's native interface."""
    requested = np.asarray(requested, dtype=bool)
    if method in {"omnicontrol", "projflow"}:
        provided = requested.copy()
        status = "native_axis_level"
    elif method == "condmdi":
        provided = np.repeat(requested.any(axis=-1, keepdims=True), 3, axis=-1)
        status = "joint_level_xyz_extra_axes"
    elif method == "maskcontrol":
        provided = np.repeat(requested.any(axis=-1, keepdims=True), 3, axis=-1)
        supported = np.zeros((1, 22, 1), dtype=bool)
        supported[:, list(MASKCONTROL_JOINTS), :] = True
        provided &= supported
        status = "six_anchor_joint_level_partial"
    elif method == "motioncanvas":
        provided = requested.copy()
        status = "native_axis_level"
    else:
        raise ValueError(f"unknown mixed-control method {method!r}")

    requested_count = int(requested.sum())
    supplied_requested = int((provided & requested).sum())
    extra_count = int((provided & ~requested).sum())
    return {
        "method": method,
        "status": status,
        "provided": provided,
        "requested_atoms": requested_count,
        "supplied_requested_atoms": supplied_requested,
        "extra_atoms": extra_count,
        "coverage": supplied_requested / max(1, requested_count),
    }


__all__ = [
    "BASELINE_FPS",
    "ALL_POSITION_SETTINGS",
    "HELDOUT_POSITION_SETTINGS",
    "MASKCONTROL_JOINTS",
    "POSITION_SETTINGS",
    "build_position_protocol",
    "method_protocol",
    "setting_params",
]

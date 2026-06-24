"""MBench-style physical plausibility metrics on shared SMPL-22 joints.

This module contains the protocol used by the PRISM HumanML3D paper tables.  It
ports the non-VLM MBench ``motion_quality`` dimensions to the repository's
canonical SMPL-22 forward-kinematics joints so every method is measured on the
same skeleton and axis convention.

The metrics intentionally preserve MBench's per-frame finite differences: no
fps normalization is applied.  Inputs are expected to be in metres.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

MotionMode = Literal["m135", "gt272"]

FOOT_IDX = [10, 11]
HEIGHT_AXIS = 1
HORIZONTAL_AXES = [0, 2]
METRIC_KEYS = ["Jitter", "Dynamic", "Penet", "Float", "Slide"]

_ROOT = Path(__file__).resolve().parents[3]
_wp_m135 = None
_wp_272 = None


@dataclass(frozen=True)
class MBenchPhysicsConfig:
    """Configuration for the joint-based MBench protocol.

    ``floor_mode`` is currently fixed to ``"min_foot"`` to match the existing
    Table 1 numbers.  With this policy, ``Penet`` is degenerate for most clips
    because no foot point is below the per-clip minimum foot height.
    """

    vel_threshold: float = 0.01
    contact_height_threshold: float = 0.02
    floor_mode: Literal["min_foot"] = "min_foot"


DEFAULT_CONFIG = MBenchPhysicsConfig()


def _pad_vel(x: np.ndarray) -> np.ndarray:
    """MBench repeats the final velocity so velocity arrays have length ``T``."""

    v = np.diff(x, axis=0)
    return np.concatenate([v, v[-1:]], axis=0)


def _get_contact(
    foot_pos: np.ndarray,
    floor: float,
    cfg: MBenchPhysicsConfig,
) -> np.ndarray:
    foot_vel = _pad_vel(foot_pos)
    delta = np.linalg.norm(foot_vel, axis=-1)
    height = foot_pos[:, :, HEIGHT_AXIS] - floor
    return ((delta < cfg.vel_threshold) | (height < cfg.contact_height_threshold)).astype(np.int32)


def _ranges(contact: np.ndarray, state: int) -> list[list[list[int]]]:
    """Contiguous index ranges where ``contact[:, i] == state``, per foot."""

    out = []
    for i in range(contact.shape[1]):
        ranges, start, end = [], -1, -1
        for idx in range(contact.shape[0]):
            if contact[idx, i] != state:
                continue
            if start == -1:
                start = end = idx
            elif idx - end == 1:
                end += 1
            else:
                ranges.append([start, end])
                start = end = idx
        if end != -1:
            ranges.append([start, end])
        out.append(ranges)
    return out


def _calc_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def compute_mbench_physics_from_joints(
    joints: np.ndarray,
    cfg: MBenchPhysicsConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Compute joint-based MBench physical metrics for one clip.

    Args:
        joints: World joints with shape ``(T, 22, 3)`` in metres, Y-up.
        cfg: Contact/floor configuration.  Defaults match Table 1.

    Returns:
        Dict with ``Jitter``, ``Dynamic``, ``Penet``, ``Float``, and ``Slide``.
    """

    pos = np.asarray(joints, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints shape (T,22,3), got {pos.shape}")
    if pos.shape[0] < 4:
        raise ValueError("At least 4 frames are required for MBench physics metrics")

    from scipy.signal import find_peaks

    T = pos.shape[0]
    foot_pos = pos[:, FOOT_IDX]
    floor = float(foot_pos[:, :, HEIGHT_AXIS].min())

    def _accel_mean(p: np.ndarray) -> float:
        a = np.diff(p, n=2, axis=0)
        return float(np.linalg.norm(a, axis=2).mean()) if a.shape[0] else 0.0

    local = pos - pos[:, 0:1, :]
    jitter = _accel_mean(pos) + _accel_mean(local)

    def _vel_mean(p: np.ndarray) -> float:
        v = np.diff(p, axis=0)
        return float(np.linalg.norm(v, axis=2).mean()) if v.shape[0] else 0.0

    dynamic = _vel_mean(pos) + _vel_mean(local)

    height = foot_pos[:, :, HEIGHT_AXIS] - floor
    below = np.abs(height[height < -0.005])
    penet = float(below.mean()) if below.size else 0.0

    contact = _get_contact(foot_pos, floor, cfg)
    foot_vel = _pad_vel(foot_pos)
    foot_delta = np.linalg.norm(foot_vel[:, :, HORIZONTAL_AXES], axis=-1)
    left_slide = (foot_delta[:, 0] * contact[:, 0]).sum() / (contact[:, 0].sum() + 1e-6)
    right_slide = (foot_delta[:, 1] * contact[:, 1]).sum() / (contact[:, 1].sum() + 1e-6)
    sliding = float((left_slide + right_slide) / 2)

    delta_ts, rate_ts, rate_high_ts = 0.001, 0.6, 1.75
    root_pos = pos[:, 0]
    root_vel = _pad_vel(root_pos)
    rel_foot = foot_pos - root_pos[:, None]
    rel_foot_vel = _pad_vel(rel_foot)
    left_rates = np.zeros(T)
    right_rates = np.zeros(T)
    invalid = np.ones((T, 2))
    for f in range(T):
        root_dis = np.linalg.norm(root_vel[f])
        left_pd = np.linalg.norm(rel_foot_vel[f, 0])
        right_pd = np.linalg.norm(rel_foot_vel[f, 1])
        rate_l = left_pd / (root_dis + 1e-6)
        rate_r = right_pd / (root_dis + 1e-6)
        left_rates[f] = rate_l
        right_rates[f] = rate_r
        left_fd = np.linalg.norm(foot_vel[f, 0])
        right_fd = np.linalg.norm(foot_vel[f, 1])
        if root_dis < delta_ts:
            continue
        left_invalid = (
            (rate_l < rate_ts and left_fd > 1.2e-4)
            or (rate_l > rate_high_ts and left_fd > 1.2e-4 and root_dis > 1.2e-4)
        )
        right_invalid = (
            (rate_r < rate_ts and right_fd > 1.2e-4)
            or (rate_r > rate_high_ts and right_fd > 1.2e-4 and root_dis > 1.2e-4)
        )
        if contact[f].sum() == 2 and left_invalid and right_invalid:
            invalid[f, 0] = invalid[f, 1] = 0
        elif contact[f, 0] == 1 and contact[f, 1] == 0 and left_invalid:
            invalid[f, 0] = 0
        elif contact[f, 1] == 1 and contact[f, 0] == 0 and right_invalid:
            invalid[f, 1] = 0

    all_rates = np.stack([left_rates, right_rates], axis=-1)
    no_contact = _ranges(contact, 0)
    floating_lens = [0]
    for foot_i, ranges in enumerate(no_contact):
        for start, end in ranges:
            rates = all_rates[start : end + 1, foot_i]
            if len(rates) < 4:
                continue
            skip_n = sum(
                1 for f in range(start, end + 1) if np.linalg.norm(root_vel[f]) < delta_ts
            )
            if skip_n / (end - start + 1) > 0.5:
                continue
            cur = (rates < (rate_ts - 0.2)).astype(np.float32)
            diff = np.diff(np.concatenate([[0.0], cur, [0.0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if len(starts):
                floating_lens.extend((ends - starts).tolist())

    mass_lens = []
    if len(no_contact[0]) and len(no_contact[1]):
        for s0, e0 in no_contact[0]:
            for s1, e1 in no_contact[1]:
                start, end = max(s0, s1), min(e0, e1)
                if end - start + 1 < 4:
                    continue
                base = foot_pos[end, 0] - foot_pos[start, 0]
                angles = [
                    np.rad2deg(abs(_calc_angle(foot_pos[f, 0] - foot_pos[start, 0], base)))
                    for f in range(start + 1, end + 1)
                ]
                peaks, _ = find_peaks(angles)
                if len(peaks) > 2:
                    mass_lens.append(end - start + 1)

    merge_invalid = (invalid[:, 0] + invalid[:, 1]) <= 1
    invalid_n = int(merge_invalid.sum()) + sum(floating_lens) / 2 + sum(mass_lens)
    floating = float(invalid_n / T)

    return {
        "Jitter": float(jitter),
        "Dynamic": float(dynamic),
        "Penet": float(penet),
        "Float": float(floating),
        "Slide": float(sliding),
    }


def _ensure_fk_loaded() -> None:
    global _wp_m135, _wp_272
    if _wp_m135 is not None and _wp_272 is not None:
        return

    import torch
    from hftrainer.datasets.motion.representation.humanml_repr import (
        recover_272_stored_positions,
    )
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    bone22 = torch.load(
        str(_ROOT / "data/hymotion_m2m_data/bone_offsets_22.pt"),
        map_location="cpu",
    ).float()

    def _m135_to_joints(m135: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(m135[:, :135], dtype=np.float32))
        wp, _, _, _ = motion135_to_fk(t, bone22, rotation_space="local")
        return wp.numpy()

    def _m272_to_joints(m272: np.ndarray) -> np.ndarray:
        return np.asarray(recover_272_stored_positions(m272), dtype=np.float32)

    _wp_m135 = _m135_to_joints
    _wp_272 = _m272_to_joints


def motion135_to_joints(motion_135: np.ndarray) -> np.ndarray:
    """Convert ``motion_135`` to shared SMPL-22 FK joints."""

    _ensure_fk_loaded()
    return _wp_m135(np.asarray(motion_135, dtype=np.float32))


def motion272_to_joints(motion_272: np.ndarray) -> np.ndarray:
    """Recover stored MotionStreamer-272 joint positions."""

    _ensure_fk_loaded()
    return _wp_272(np.asarray(motion_272, dtype=np.float32))


def compute_mbench_physics_for_file(
    path: str | os.PathLike[str],
    mode: MotionMode,
    cfg: MBenchPhysicsConfig = DEFAULT_CONFIG,
) -> dict[str, float] | None:
    """Compute metrics for one ``motion_135`` npz or native 272 npy/npz file."""

    try:
        p = str(path)
        if mode == "m135":
            data = np.load(p, allow_pickle=True)
            if "motion_135" not in data:
                return None
            joints = motion135_to_joints(np.asarray(data["motion_135"], dtype=np.float32))
        elif mode == "gt272":
            if p.endswith(".npz"):
                data = np.load(p, allow_pickle=True)
                if "motion_272" not in data:
                    return None
                motion_272 = np.asarray(data["motion_272"], dtype=np.float32)
            else:
                motion_272 = np.asarray(np.load(p), dtype=np.float32)
            if motion_272.ndim != 2 or motion_272.shape[1] != 272:
                return None
            joints = motion272_to_joints(motion_272)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        if joints.shape[0] < 4:
            return None
        return compute_mbench_physics_from_joints(joints, cfg)
    except Exception:
        return None


def list_metric_files(src: str | os.PathLike[str], mode: MotionMode, limit: int = 0, seed: int = 0) -> list[str]:
    """List files participating in a metric run."""

    suffixes = [".npz"] if mode == "m135" else [".npy", ".npz"]
    files = sorted(
        str(e.path)
        for e in os.scandir(src)
        if any(e.name.endswith(suffix) for suffix in suffixes)
    )
    if limit and len(files) > limit:
        rng = np.random.RandomState(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), limit, False))]
    return files


def aggregate_mbench_physics(rows: Iterable[dict[str, float] | None]) -> dict[str, float]:
    """Average per-clip metric dictionaries."""

    valid = [r for r in rows if r is not None]
    out: dict[str, float] = {"n": len(valid)}
    if not valid:
        out.update({k: 0.0 for k in METRIC_KEYS})
        return out
    arr = np.asarray([[r[k] for k in METRIC_KEYS] for r in valid], dtype=np.float64)
    mean = arr.mean(axis=0)
    out.update({k: float(mean[i]) for i, k in enumerate(METRIC_KEYS)})
    return out


def _init_worker() -> None:
    _ensure_fk_loaded()


def _one(task: tuple[str, MotionMode]) -> dict[str, float] | None:
    path, mode = task
    return compute_mbench_physics_for_file(path, mode)


def evaluate_mbench_physics_dir(
    src: str | os.PathLike[str],
    mode: MotionMode,
    limit: int = 0,
    seed: int = 0,
    workers: int = 16,
) -> dict[str, float]:
    """Evaluate all metric files in one directory."""

    files = list_metric_files(src, mode, limit=limit, seed=seed)
    if workers <= 1:
        rows = [compute_mbench_physics_for_file(path, mode) for path in files]
        return aggregate_mbench_physics(rows)

    with mp.Pool(workers, initializer=_init_worker) as pool:
        rows = list(pool.imap_unordered(_one, [(f, mode) for f in files], chunksize=4))
    return aggregate_mbench_physics(rows)


def table_scaled_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Scale raw metric values to the paper-table display units."""

    return {
        "Slide": float(metrics.get("Slide", 0.0)) * 1000.0,
        "Float": float(metrics.get("Float", 0.0)) * 100.0,
        "Jitter": float(metrics.get("Jitter", 0.0)) * 1000.0,
        "Dynamic": float(metrics.get("Dynamic", 0.0)) * 1000.0,
        "Penet": float(metrics.get("Penet", 0.0)) * 1000.0,
    }


def load_manifest(path: str | os.PathLike[str]) -> list[tuple[str, MotionMode, str]]:
    """Read ``tag<TAB>{m135|gt272}<TAB>dir`` manifest files."""

    methods = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, mode, directory = line.split("\t")
        if mode not in ("m135", "gt272"):
            raise ValueError(f"Bad manifest mode {mode!r} in {path}")
        methods.append((tag, mode, directory))
    return methods


def dump_results_json(results: dict[str, dict[str, float]], out_json: str | os.PathLike[str]) -> None:
    """Write metric results to JSON, creating parent directories."""

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))


__all__ = [
    "DEFAULT_CONFIG",
    "FOOT_IDX",
    "HEIGHT_AXIS",
    "HORIZONTAL_AXES",
    "MBenchPhysicsConfig",
    "METRIC_KEYS",
    "aggregate_mbench_physics",
    "compute_mbench_physics_for_file",
    "compute_mbench_physics_from_joints",
    "dump_results_json",
    "evaluate_mbench_physics_dir",
    "list_metric_files",
    "load_manifest",
    "motion135_to_joints",
    "motion272_to_joints",
    "table_scaled_metrics",
]

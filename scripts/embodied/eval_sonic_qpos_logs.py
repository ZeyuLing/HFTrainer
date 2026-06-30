#!/usr/bin/env python3
"""Compute unified G1 tracking metrics from SONIC simulator qpos logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.outer(q, q)
    return np.array(
        [
            [q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3], 2 * (q[1, 2] - q[0, 3]), 2 * (q[1, 3] + q[0, 2])],
            [2 * (q[1, 2] + q[0, 3]), q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3], 2 * (q[2, 3] - q[0, 1])],
            [2 * (q[1, 3] - q[0, 2]), 2 * (q[2, 3] + q[0, 1]), q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3]],
        ],
        dtype=np.float64,
    )


def _load_sim_qpos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        data = data.reshape(1)
    times = np.asarray(data["time"], dtype=np.float64)
    qpos = np.stack([np.asarray(data[f"qpos_{i}"], dtype=np.float64) for i in range(36)], axis=1)
    return times, qpos


def _load_q_log(path: Path) -> np.ndarray | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        data = data.reshape(1)
    return np.stack([np.asarray(data[f"q_{i}"], dtype=np.float64) for i in range(29)], axis=1)


def _control_start_time(sim_times: np.ndarray, sim_qpos: np.ndarray, q_log: np.ndarray | None) -> float:
    if q_log is None or len(q_log) == 0:
        return float(sim_times[0])
    target = q_log[0]
    d = np.linalg.norm(sim_qpos[:, 7:] - target[None], axis=1)
    return float(sim_times[int(np.argmin(d))])


def _sample_exec(ref_len: int, fps: float, sim_times: np.ndarray, sim_qpos: np.ndarray, start_time: float) -> tuple[np.ndarray, int]:
    desired = start_time + np.arange(ref_len, dtype=np.float64) / fps
    valid = desired <= sim_times[-1]
    if not np.any(valid):
        return sim_qpos[:0], 0
    desired = desired[valid]
    idx = np.searchsorted(sim_times, desired, side="left")
    idx = np.clip(idx, 0, len(sim_times) - 1)
    prev = np.clip(idx - 1, 0, len(sim_times) - 1)
    use_prev = np.abs(sim_times[prev] - desired) < np.abs(sim_times[idx] - desired)
    idx[use_prev] = prev[use_prev]
    return sim_qpos[idx], int(valid.sum())


def _body_positions(model: mujoco.MjModel, qpos: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    out = []
    for q in qpos:
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        out.append(data.xpos[body_ids].copy())
    return np.asarray(out)


def _metrics(ref_qpos: np.ndarray, exec_qpos: np.ndarray, fps: float, model: mujoco.MjModel) -> dict:
    n = min(len(ref_qpos), len(exec_qpos))
    if n < 2:
        return {
            "success": False,
            "paper_success": False,
            "strict_success": False,
            "legacy_success": False,
            "steps": n,
            "completion": float(n / len(ref_qpos)) if len(ref_qpos) else 0.0,
        }
    ref = ref_qpos[:n].astype(np.float64)
    exe = exec_qpos[:n].astype(np.float64)
    body_ids = np.array([i for i in range(1, model.nbody)])
    ref_body = _body_positions(model, ref, body_ids)
    exe_body = _body_positions(model, exe, body_ids)

    raw_body = np.linalg.norm(exe_body - ref_body, axis=-1).mean(axis=1)
    xy_ref = ref_body.copy()
    xy_ref[:, :, :2] += exe[:, None, :2] - ref[:, None, :2]
    xy_body = np.linalg.norm(exe_body - xy_ref, axis=-1).mean(axis=1)
    local_body = []
    for i in range(n):
        exe_local = (exe_body[i] - exe[i, :3][None]) @ quat_to_mat(exe[i, 3:7])
        ref_local = (ref_body[i] - ref[i, :3][None]) @ quat_to_mat(ref[i, 3:7])
        local_body.append(float(np.linalg.norm(exe_local - ref_local, axis=-1).mean()))
    local_body = np.asarray(local_body)

    exe_vel = np.diff(exe_body, axis=0) * fps
    ref_vel = np.diff(ref_body, axis=0) * fps
    vel_err = np.linalg.norm(exe_vel - ref_vel, axis=-1).mean(axis=1)
    exe_acc = np.diff(exe_vel, axis=0) * fps
    ref_acc = np.diff(ref_vel, axis=0) * fps
    acc_err = np.linalg.norm(exe_acc - ref_acc, axis=-1).mean(axis=1) if len(exe_acc) else np.array([np.nan])

    root_err = np.linalg.norm(exe[:, :3] - ref[:, :3], axis=1)
    root_height_err = np.abs(exe[:, 2] - ref[:, 2])
    joint_abs = np.abs(exe[:, 7:] - ref[:, 7:])
    max_joint = joint_abs.max(axis=1)
    finite = bool(np.isfinite(exe).all())
    raw_global_m = float(raw_body.mean())
    xy_m = float(xy_body.mean())
    local_m = float(local_body.mean())
    root_m = float(root_err.mean())
    root_h_m = float(root_height_err.mean())
    completion = float(n / len(ref_qpos)) if len(ref_qpos) else 0.0
    fall_failed = float(exe[:, 2].min()) < 0.25
    paper_failed = (not finite) or completion < 0.95 or fall_failed or local_m > 0.2 or root_h_m > 0.2
    strict_failed = paper_failed or root_m > 1.0 or float(max_joint.max()) > 0.7
    legacy_failed = (
        (not finite)
        or float(local_body.max()) > 0.75
        or fall_failed
        or float(max_joint.max()) > 2.5
    )
    return {
        "steps": int(n),
        "completion": completion,
        "success": bool(not paper_failed),
        "paper_success": bool(not paper_failed),
        "strict_success": bool(not strict_failed),
        "legacy_success": bool(not legacy_failed),
        "fall": bool(fall_failed),
        "root_err_mean": root_m,
        "root_err_max": float(root_err.max()),
        "root_height_err_mean": root_h_m,
        "root_height_err_max": float(root_height_err.max()),
        "raw_body_err_mean": raw_global_m,
        "raw_body_err_max": float(raw_body.max()),
        "body_err_mean": xy_m,
        "body_err_max": float(xy_body.max()),
        "xy_aligned_body_err_mean": xy_m,
        "xy_aligned_body_err_max": float(xy_body.max()),
        "local_body_err_mean": local_m,
        "local_body_err_max": float(local_body.max()),
        "body_vel_err_mean": float(vel_err.mean()),
        "local_body_vel_err_mean": float(vel_err.mean()),
        "body_acc_err_mean": float(np.nanmean(acc_err)),
        "local_body_acc_err_mean": float(np.nanmean(acc_err)),
        "raw_global_mpjpe_m": raw_global_m,
        "raw_global_mpjpe_mm": raw_global_m * 1000.0,
        "xy_aligned_mpjpe_m": xy_m,
        "xy_aligned_mpjpe_mm": xy_m * 1000.0,
        "mpjpe_m": xy_m,
        "mpjpe_mm": xy_m * 1000.0,
        "local_mpjpe_m": local_m,
        "local_mpjpe_mm": local_m * 1000.0,
        "mpjve_mps": float(vel_err.mean()),
        "local_mpjve_mps": float(vel_err.mean()),
        "mpjae_mps2": float(np.nanmean(acc_err)),
        "local_mpjae_mps2": float(np.nanmean(acc_err)),
        "joint_err_mean": float(joint_abs.mean()),
        "max_joint_err_mean": float(max_joint.mean()),
        "max_joint_err_max": float(max_joint.max()),
        "min_height": float(exe[:, 2].min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-npz", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    pack = np.load(args.ref_npz, allow_pickle=True)
    ref_qpos = np.asarray(pack["qpos"], dtype=np.float64)
    fps = float(np.asarray(pack["frequency"]).reshape(-1)[0]) if "frequency" in pack.files else 30.0
    sim_times, sim_qpos = _load_sim_qpos(args.run_dir / "sim_qpos.csv")
    q_log = _load_q_log(args.run_dir / "deploy_logs" / "q.csv")
    start = _control_start_time(sim_times, sim_qpos, q_log)
    exec_qpos, covered = _sample_exec(len(ref_qpos), fps, sim_times, sim_qpos, start)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    row = _metrics(ref_qpos, exec_qpos, fps, model)
    row.update(
        {
            "motion": args.ref_npz.stem,
            "control_start_time": start,
            "sim_qpos_rows": int(len(sim_qpos)),
            "covered_frames": int(covered),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

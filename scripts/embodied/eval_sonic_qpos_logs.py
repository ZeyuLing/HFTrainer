#!/usr/bin/env python3
"""Compute unified G1 tracking metrics from SONIC simulator qpos logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


MESHES_BY_BODY = {
    "pelvis": ["pelvis.stl", "pelvis_contour_link.stl"],
    "head": [],
    "left_hip_pitch_link": ["left_hip_pitch_link.stl"],
    "left_hip_roll_link": ["left_hip_roll_link.stl"],
    "left_hip_yaw_link": ["left_hip_yaw_link.stl"],
    "left_knee_link": ["left_knee_link.stl"],
    "left_ankle_pitch_link": ["left_ankle_pitch_link.stl"],
    "left_ankle_roll_link": ["left_ankle_roll_link.stl"],
    "right_hip_pitch_link": ["right_hip_pitch_link.stl"],
    "right_hip_roll_link": ["right_hip_roll_link.stl"],
    "right_hip_yaw_link": ["right_hip_yaw_link.stl"],
    "right_knee_link": ["right_knee_link.stl"],
    "right_ankle_pitch_link": ["right_ankle_pitch_link.stl"],
    "right_ankle_roll_link": ["right_ankle_roll_link.stl"],
    "waist_yaw_link": ["waist_yaw_link_rev_1_0.stl"],
    "waist_roll_link": ["waist_roll_link_rev_1_0.stl"],
    "torso_link": ["torso_link_rev_1_0.stl", "logo_link.stl", "head_link.stl"],
    "left_shoulder_pitch_link": ["left_shoulder_pitch_link.stl"],
    "left_shoulder_roll_link": ["left_shoulder_roll_link.stl"],
    "left_shoulder_yaw_link": ["left_shoulder_yaw_link.stl"],
    "left_elbow_link": ["left_elbow_link.stl"],
    "left_wrist_roll_link": ["left_wrist_roll_link.stl"],
    "left_wrist_pitch_link": ["left_wrist_pitch_link.stl"],
    "left_wrist_yaw_link": ["left_wrist_yaw_link.stl", "left_rubber_hand.stl"],
    "left_rubber_hand": [],
    "right_shoulder_pitch_link": ["right_shoulder_pitch_link.stl"],
    "right_shoulder_roll_link": ["right_shoulder_roll_link.stl"],
    "right_shoulder_yaw_link": ["right_shoulder_yaw_link.stl"],
    "right_elbow_link": ["right_elbow_link.stl"],
    "right_wrist_roll_link": ["right_wrist_roll_link.stl"],
    "right_wrist_pitch_link": ["right_wrist_pitch_link.stl"],
    "right_wrist_yaw_link": ["right_wrist_yaw_link.stl", "right_rubber_hand.stl"],
    "right_rubber_hand": [],
}


def _body_descriptors(body_names: list[str]) -> list[dict]:
    return [
        {
            "name": name,
            "meshes": [
                {"file": mesh, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
                for mesh in MESHES_BY_BODY.get(name, [])
            ],
        }
        for name in body_names
    ]


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
    qpos_keys = [f"qpos_{i}" for i in range(36)]
    times: list[float] = []
    qpos_rows: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                times.append(float(row["time"]))
                qpos_rows.append([float(row[key]) for key in qpos_keys])
            except (KeyError, TypeError, ValueError):
                if times:
                    times.pop()
                continue
    if not qpos_rows:
        raise ValueError(f"no valid qpos rows in {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(qpos_rows, dtype=np.float64)


def _load_q_log(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    q_keys = [f"q_{i}" for i in range(29)]
    times: list[float] = []
    q_rows: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                times.append(float(row["time_ms"]) / 1000.0)
                q_rows.append([float(row[key]) for key in q_keys])
            except (KeyError, TypeError, ValueError):
                if times:
                    times.pop()
                continue
    if not q_rows:
        return None
    times_arr = np.asarray(times, dtype=np.float64)
    times_arr = times_arr - float(times_arr[0])
    return times_arr, np.asarray(q_rows, dtype=np.float64)


def _control_start_time(sim_times: np.ndarray, sim_qpos: np.ndarray, q_log: tuple[np.ndarray, np.ndarray] | None) -> tuple[float, float]:
    if q_log is None:
        return float(sim_times[0]), float("nan")
    q_times, q_values = q_log
    if len(q_values) == 0:
        return float(sim_times[0]), float("nan")

    # SONIC starts the simulator before the deploy controller is in CONTROL.
    # Align the logged robot joints against the MuJoCo qpos trace as a short
    # sequence; matching only q_log[0] can snap to the standing warm-up phase.
    max_points = min(len(q_values), 240)
    if max_points < 2:
        target = q_values[0]
        d = np.linalg.norm(sim_qpos[:, 7:] - target[None], axis=1)
        best = int(np.argmin(d))
        return float(sim_times[best]), float(d[best])

    sample_ids = np.linspace(0, len(q_values) - 1, max_points, dtype=np.int64)
    target_times = q_times[sample_ids]
    target_q = q_values[sample_ids]
    max_start = sim_times[-1] - target_times[-1]
    candidates = np.flatnonzero(sim_times <= max_start)
    if len(candidates) == 0:
        return float(sim_times[0]), float("nan")

    step = max(1, int(round(0.02 / max(np.median(np.diff(sim_times)), 1e-6))))
    best_time = float(sim_times[0])
    best_err = float("inf")
    for ci in candidates[::step]:
        desired = sim_times[ci] + target_times
        idx = np.searchsorted(sim_times, desired, side="left")
        idx = np.clip(idx, 0, len(sim_times) - 1)
        prev = np.clip(idx - 1, 0, len(sim_times) - 1)
        use_prev = np.abs(sim_times[prev] - desired) < np.abs(sim_times[idx] - desired)
        idx[use_prev] = prev[use_prev]
        err = float(np.mean(np.linalg.norm(sim_qpos[idx, 7:] - target_q, axis=1)))
        if err < best_err:
            best_err = err
            best_time = float(sim_times[ci])
    return best_time, best_err


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


def _write_robot_frames(path: Path, model: mujoco.MjModel, qpos: np.ndarray, fps: float, note: str) -> None:
    data = mujoco.MjData(model)
    body_ids = np.array([i for i in range(1, model.nbody)])
    body_names = [model.body(i).name for i in body_ids]
    frames = []
    for q in qpos:
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        frames.append(
            {
                "body_pos": data.xpos[body_ids].copy().tolist(),
                "body_quat": data.xquat[body_ids].copy().tolist(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "robot_frames",
                "robot": "g1",
                "fps": int(round(fps)),
                "source_note": note,
                "num_frames": len(frames),
                "num_bodies": len(body_names),
                "bodies": _body_descriptors(body_names),
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )


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
    parser.add_argument(
        "--output-robot-frames",
        type=Path,
        default=None,
        help="Optional path for the time-aligned SONIC execution as G1 robot_frames JSON.",
    )
    parser.add_argument(
        "--output-reference-frames",
        type=Path,
        default=None,
        help="Optional path for the SONIC 50 Hz reference as G1 robot_frames JSON.",
    )
    parser.add_argument(
        "--require-q-log",
        action="store_true",
        help="Mark the case as failed instead of scoring simulator warm-up when deploy q.csv is missing.",
    )
    args = parser.parse_args()

    pack = np.load(args.ref_npz, allow_pickle=True)
    ref_qpos = np.asarray(pack["qpos"], dtype=np.float64)
    fps = float(np.asarray(pack["frequency"]).reshape(-1)[0]) if "frequency" in pack.files else 30.0
    sim_times, sim_qpos = _load_sim_qpos(args.run_dir / "sim_qpos.csv")
    q_log = _load_q_log(args.run_dir / "deploy_logs" / "q.csv")
    if args.require_q_log and q_log is None:
        row = {
            "success": False,
            "paper_success": False,
            "strict_success": False,
            "legacy_success": False,
            "fall": False,
            "steps": 0,
            "completion": 0.0,
            "motion": args.ref_npz.stem,
            "runtime_error": "missing_or_empty_deploy_q_log",
            "control_start_time": float(sim_times[0]) if len(sim_times) else float("nan"),
            "control_start_alignment_err": float("nan"),
            "sim_qpos_rows": int(len(sim_qpos)),
            "covered_frames": 0,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(json.dumps(row, indent=2, sort_keys=True))
        return
    start, align_err = _control_start_time(sim_times, sim_qpos, q_log)
    exec_qpos, covered = _sample_exec(len(ref_qpos), fps, sim_times, sim_qpos, start)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    row = _metrics(ref_qpos, exec_qpos, fps, model)
    if args.output_robot_frames is not None:
        _write_robot_frames(args.output_robot_frames, model, exec_qpos, fps, "sonic_execution_aligned_to_reference")
    if args.output_reference_frames is not None:
        _write_robot_frames(args.output_reference_frames, model, ref_qpos, fps, "sonic_reference")
    row.update(
        {
            "motion": args.ref_npz.stem,
            "control_start_time": start,
            "control_start_alignment_err": align_err,
            "sim_qpos_rows": int(len(sim_qpos)),
            "covered_frames": int(covered),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

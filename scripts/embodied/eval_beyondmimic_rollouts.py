#!/usr/bin/env python3
"""Evaluate dumped BeyondMimic rollouts against Table-2 G1 references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_ROOT / "ref_repo/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    if key not in data.files:
        return float(default)
    arr = np.asarray(data[key]).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _as_string_list(value: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(value, dtype=object).reshape(-1).tolist()]


def _qpos_qvel_slices(joint_names: list[str], jnt_type: np.ndarray) -> tuple[dict[str, slice], dict[str, slice]]:
    qpos_slices: dict[str, slice] = {}
    qvel_slices: dict[str, slice] = {}
    qpos_i = 0
    qvel_i = 0
    for name, typ in zip(joint_names, jnt_type):
        typ = int(typ)
        if typ == mujoco.mjtJoint.mjJNT_FREE:
            qpos_slices[name] = slice(qpos_i, qpos_i + 7)
            qvel_slices[name] = slice(qvel_i, qvel_i + 6)
            qpos_i += 7
            qvel_i += 6
        elif typ in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qpos_slices[name] = slice(qpos_i, qpos_i + 1)
            qvel_slices[name] = slice(qvel_i, qvel_i + 1)
            qpos_i += 1
            qvel_i += 1
        else:
            raise ValueError(f"Unsupported joint type {typ} for {name}")
    return qpos_slices, qvel_slices


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)


def _resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(float(source_fps) - float(target_fps)) < 1e-6 or qpos.shape[0] < 2:
        out = qpos.astype(np.float32, copy=True)
        out[:, 3:7] = _normalize_quat(out[:, 3:7])
        return out
    n = qpos.shape[0]
    duration = (n - 1) / float(source_fps)
    src_t = np.arange(n, dtype=np.float64) / float(source_fps)
    out_n = int(round(duration * float(target_fps))) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / float(target_fps)
    dst_t[-1] = min(dst_t[-1], src_t[-1])
    out = np.empty((out_n, qpos.shape[1]), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    src_xyzw = _normalize_quat(qpos[:, 3:7])[:, [1, 2, 3, 0]]
    out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
    out[:, 3:7] = out_xyzw[:, [3, 0, 1, 2]]
    for i in range(7, qpos.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    return out


def _quat_step_angvel_wxyz(qpos: np.ndarray, fps: float) -> np.ndarray:
    quat = _normalize_quat(qpos[:, 3:7])
    inv = np.concatenate([quat[:, :1], -quat[:, 1:]], axis=1)
    q1 = inv[:-1]
    q2 = quat[1:]
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    s = 2 * (w**2) - 1
    angle = np.arccos(np.clip(s, -1, 1))
    axis = np.stack([x, y, z], axis=1)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(min=1e-9)
    return axis * angle[:, None] * fps


def _build_qvel(qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros((qpos.shape[0], qpos.shape[1] - 1), dtype=np.float32)
    if qpos.shape[0] > 1:
        qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) * fps
        qvel[1:, 3:6] = _quat_step_angvel_wxyz(qpos, fps)
        qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) * fps
    return qvel


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.outer(q, q)
    return np.array(
        [
            [
                q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                2 * (q[1, 2] - q[0, 3]),
                2 * (q[1, 3] + q[0, 2]),
            ],
            [
                2 * (q[1, 2] + q[0, 3]),
                q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                2 * (q[2, 3] - q[0, 1]),
            ],
            [
                2 * (q[1, 3] - q[0, 2]),
                2 * (q[2, 3] + q[0, 1]),
                q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
            ],
        ],
        dtype=np.float32,
    )


def _load_qpos(path: Path, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    data = np.load(path, allow_pickle=True)
    if "qpos" not in data.files:
        raise ValueError(f"{path}: missing qpos")
    qpos_src = np.asarray(data["qpos"], dtype=np.float32)
    fps = _scalar(data, "frequency", _scalar(data, "fps", 50.0))
    if qpos_src.shape[1] == model.nq:
        qpos = qpos_src.copy()
    else:
        if "joint_names" not in data.files:
            raise ValueError(f"{path}: qpos dim {qpos_src.shape[1]} != model.nq {model.nq}, no joint_names")
        source_joint_names = _as_string_list(data["joint_names"])
        if len(source_joint_names) == qpos_src.shape[1] - 6:
            source_joint_names = ["root", *source_joint_names]
        jnt_type = (
            np.asarray(data["jnt_type"])
            if "jnt_type" in data.files
            else np.array([mujoco.mjtJoint.mjJNT_FREE] + [mujoco.mjtJoint.mjJNT_HINGE] * (len(source_joint_names) - 1))
        )
        src_qpos_slices, _ = _qpos_qvel_slices(source_joint_names, jnt_type)
        model_joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
        ]
        model_qpos_slices, _ = _qpos_qvel_slices(model_joint_names, model.jnt_type)
        qpos = np.zeros((qpos_src.shape[0], model.nq), dtype=np.float32)
        for name in model_joint_names:
            if name in src_qpos_slices:
                qpos[:, model_qpos_slices[name]] = qpos_src[:, src_qpos_slices[name]]
    qpos[:, 3:7] = _normalize_quat(qpos[:, 3:7])
    return qpos, fps


def _body_positions(model: mujoco.MjModel, qpos: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    out = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float32)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        out[i] = data.xpos[body_ids]
    return out


def _evaluate_pair(reference: Path, execution: Path, xml: Path, motion_name: str | None = None) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml))
    ref_qpos, ref_fps = _load_qpos(reference, model)
    exec_qpos, exec_fps = _load_qpos(execution, model)
    target_fps = exec_fps
    ref_qpos = _resample_qpos(ref_qpos, ref_fps, target_fps)
    n_ref = int(ref_qpos.shape[0])
    n = min(n_ref, int(exec_qpos.shape[0]))
    if n < 2:
        raise ValueError(f"Too few frames: reference={n_ref}, execution={exec_qpos.shape[0]}")
    ref_qpos = ref_qpos[:n]
    exec_qpos = exec_qpos[:n]
    body_ids = np.array([i for i in range(1, model.nbody)], dtype=np.int32)
    ref_body = _body_positions(model, ref_qpos, body_ids)
    exec_body = _body_positions(model, exec_qpos, body_ids)

    root_err = np.linalg.norm(exec_qpos[:, :3] - ref_qpos[:, :3], axis=-1)
    root_height_err = np.abs(exec_qpos[:, 2] - ref_qpos[:, 2])
    joint_abs = np.abs(exec_qpos[:, 7:] - ref_qpos[:, 7:])
    body_err = np.linalg.norm(exec_body - ref_body, axis=-1).mean(axis=-1)

    xy_aligned = ref_body.copy()
    xy_aligned[:, :, :2] += exec_qpos[:, None, :2] - ref_qpos[:, None, :2]
    xy_body_err = np.linalg.norm(exec_body - xy_aligned, axis=-1).mean(axis=-1)

    local_body_err = []
    for i in range(n):
        current_root_mat = quat_to_mat(exec_qpos[i, 3:7])
        ref_root_mat = quat_to_mat(ref_qpos[i, 3:7])
        current_local = (exec_body[i] - exec_qpos[i, None, :3]) @ current_root_mat
        ref_local = (ref_body[i] - ref_qpos[i, None, :3]) @ ref_root_mat
        local_body_err.append(float(np.linalg.norm(current_local - ref_local, axis=-1).mean()))
    local_body_err_arr = np.asarray(local_body_err, dtype=np.float32)

    dt = 1.0 / float(target_fps)
    body_vel = np.diff(exec_body, axis=0) / dt
    ref_body_vel = np.diff(ref_body, axis=0) / dt
    body_vel_err = np.linalg.norm(body_vel - ref_body_vel, axis=-1).mean(axis=-1)
    local_vel_err = []
    local_body_vel = []
    ref_local_body_vel = []
    for i in range(1, n):
        current_root_mat = quat_to_mat(exec_qpos[i - 1, 3:7])
        ref_root_mat = quat_to_mat(ref_qpos[i - 1, 3:7])
        current_local_vel = ((exec_body[i] - exec_body[i - 1]) / dt) @ current_root_mat
        ref_local_vel = ((ref_body[i] - ref_body[i - 1]) / dt) @ ref_root_mat
        local_body_vel.append(current_local_vel)
        ref_local_body_vel.append(ref_local_vel)
        local_vel_err.append(float(np.linalg.norm(current_local_vel - ref_local_vel, axis=-1).mean()))
    local_vel_err_arr = np.asarray(local_vel_err, dtype=np.float32)

    body_acc_err = np.linalg.norm(np.diff(body_vel, axis=0) / dt - np.diff(ref_body_vel, axis=0) / dt, axis=-1).mean(axis=-1)
    if len(local_body_vel) > 1:
        local_body_vel_arr = np.stack(local_body_vel, axis=0)
        ref_local_body_vel_arr = np.stack(ref_local_body_vel, axis=0)
        local_acc_err = np.linalg.norm(
            np.diff(local_body_vel_arr, axis=0) / dt - np.diff(ref_local_body_vel_arr, axis=0) / dt,
            axis=-1,
        ).mean(axis=-1)
    else:
        local_acc_err = np.asarray([np.nan])

    completion = float(n / max(n_ref, 1))
    exec_pack = np.load(execution, allow_pickle=True)
    done_step = int(np.asarray(exec_pack["done_step"]).reshape(-1)[0]) if "done_step" in exec_pack.files else -1
    paper_failed = (
        completion < 0.95
        or float(np.mean(local_body_err_arr)) > 0.2
        or float(np.mean(root_height_err)) > 0.2
        or not np.all(np.isfinite(exec_qpos))
    )
    strict_failed = paper_failed or float(np.mean(root_err)) > 1.0 or float(np.max(joint_abs)) > 0.7
    loose_failed = (
        paper_failed
        or float(np.max(local_body_err_arr)) > 0.75
        or float(np.min(exec_qpos[:, 2])) < 0.25
        or float(np.max(joint_abs)) > 2.5
    )
    name = motion_name or reference.stem
    return {
        "motion": name,
        "reference": str(reference),
        "execution": str(execution),
        "fps": float(target_fps),
        "reference_frames": int(n_ref),
        "execution_frames": int(exec_qpos.shape[0]),
        "steps": int(n),
        "completion": completion,
        "done_step": done_step,
        "success": bool(not loose_failed),
        "paper_success": bool(not paper_failed),
        "strict_success": bool(not strict_failed),
        "root_err_mean": float(np.mean(root_err)),
        "root_err_max": float(np.max(root_err)),
        "root_height_err_mean": float(np.mean(root_height_err)),
        "root_height_err_max": float(np.max(root_height_err)),
        "raw_body_err_mean": float(np.mean(body_err)),
        "raw_body_err_max": float(np.max(body_err)),
        "body_err_mean": float(np.mean(xy_body_err)),
        "body_err_max": float(np.max(xy_body_err)),
        "xy_aligned_body_err_mean": float(np.mean(xy_body_err)),
        "xy_aligned_body_err_max": float(np.max(xy_body_err)),
        "local_body_err_mean": float(np.mean(local_body_err_arr)),
        "local_body_err_max": float(np.max(local_body_err_arr)),
        "body_vel_err_mean": float(np.mean(body_vel_err)),
        "local_body_vel_err_mean": float(np.mean(local_vel_err_arr)),
        "body_acc_err_mean": float(np.mean(body_acc_err)) if body_acc_err.size else float("nan"),
        "local_body_acc_err_mean": float(np.nanmean(local_acc_err)) if local_acc_err.size else float("nan"),
        "raw_global_mpjpe_m": float(np.mean(body_err)),
        "raw_global_mpjpe_mm": float(np.mean(body_err) * 1000.0),
        "xy_aligned_mpjpe_m": float(np.mean(xy_body_err)),
        "xy_aligned_mpjpe_mm": float(np.mean(xy_body_err) * 1000.0),
        "mpjpe_m": float(np.mean(xy_body_err)),
        "mpjpe_mm": float(np.mean(xy_body_err) * 1000.0),
        "local_mpjpe_m": float(np.mean(local_body_err_arr)),
        "local_mpjpe_mm": float(np.mean(local_body_err_arr) * 1000.0),
        "mpjve_mps": float(np.mean(body_vel_err)),
        "local_mpjve_mps": float(np.mean(local_vel_err_arr)),
        "mpjae_mps2": float(np.mean(body_acc_err)) if body_acc_err.size else float("nan"),
        "local_mpjae_mps2": float(np.nanmean(local_acc_err)) if local_acc_err.size else float("nan"),
        "joint_err_mean": float(np.mean(joint_abs)),
        "max_joint_err_mean": float(np.mean(np.max(joint_abs, axis=-1))),
        "max_joint_err_max": float(np.max(joint_abs)),
        "min_height": float(np.min(exec_qpos[:, 2])),
    }


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-npz", type=Path, action="append", default=[])
    parser.add_argument("--execution-npz", type=Path, action="append", default=[])
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--pairs-json", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    pairs: list[tuple[Path, Path, str | None]] = []
    if args.pairs_json:
        for row in json.loads(args.pairs_json.read_text()):
            pairs.append((Path(row["reference"]), Path(row["execution"]), row.get("name")))
    else:
        if len(args.reference_npz) != len(args.execution_npz):
            raise SystemExit("--reference-npz and --execution-npz counts must match")
        for idx, (ref, exe) in enumerate(zip(args.reference_npz, args.execution_npz)):
            pairs.append((ref, exe, args.name[idx] if idx < len(args.name) else None))
    if not pairs:
        raise SystemExit("No reference/execution pairs.")

    rows = [_evaluate_pair(ref, exe, args.xml, name) for ref, exe, name in pairs]
    summary = {
        "num_motions": len(rows),
        "success_rate": _mean(rows, "success"),
        "paper_success_rate": _mean(rows, "paper_success"),
        "strict_success_rate": _mean(rows, "strict_success"),
        "completion": _mean(rows, "completion"),
        "root_err_mean": _mean(rows, "root_err_mean"),
        "root_height_err_mean": _mean(rows, "root_height_err_mean"),
        "raw_global_mpjpe_mm": _mean(rows, "raw_global_mpjpe_mm"),
        "mpjpe_mm": _mean(rows, "mpjpe_mm"),
        "local_mpjpe_mm": _mean(rows, "local_mpjpe_mm"),
        "mpjve_mps": _mean(rows, "mpjve_mps"),
        "local_mpjve_mps": _mean(rows, "local_mpjve_mps"),
        "mpjae_mps2": _mean(rows, "mpjae_mps2"),
        "local_mpjae_mps2": _mean(rows, "local_mpjae_mps2"),
        "joint_err_mean": _mean(rows, "joint_err_mean"),
        "max_joint_err_mean": _mean(rows, "max_joint_err_mean"),
        "max_joint_err_max": _mean(rows, "max_joint_err_max"),
        "min_height": _mean(rows, "min_height"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"summary": summary, "motions": rows}, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

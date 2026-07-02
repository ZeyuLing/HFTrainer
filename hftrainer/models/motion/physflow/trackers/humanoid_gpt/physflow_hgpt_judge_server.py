#!/usr/bin/env python3
"""Persistent Humanoid-GPT tracking judge for the PhysFlow online loop.

The PhysFlow trainer (KIMODO py3.x + torch) cannot import Humanoid-GPT's
jax/mujoco-mjx stack, and re-loading the ONNX policy + MuJoCo model every
train step would dominate wall-time. This long-lived worker loads the policy,
the convert model and the env config ONCE, then serves scoring jobs over a
line-based stdin/stdout JSON protocol so the per-step cost is just the rollout.

Run (from the Humanoid-GPT repo root, in its py3.11 venv):

    .venv311/bin/python physflow_hgpt_judge_server.py \
        --load_path storage/ckpts/pns_wo_priv216.onnx --freq 50

Protocol (one JSON object per line):
  -> {"job_dir": "/abs/dir/with/qpos_npz", "out": "/abs/metrics.json"}
  -> {"job_dir": "/abs/dir/with/qpos_npz", "out": "/abs/metrics.json",
      "frames_out_dir": "/abs/dir/with/robot_frames_json"}
  <- {"status": "ok", "job_dir": ..., "out": ..., "n": N}
  -> {"cmd": "shutdown"}
  <- {"status": "bye"}
On startup the worker prints {"status": "ready"} once the model is loaded.

Each *.npz in job_dir must carry `qpos` [T, 36] (root_xyz + quat_wxyz + 29 dof)
and a scalar `frequency` (KIMODO-G1 is 30). The worker resamples to --freq,
runs the HGPT tracker in MuJoCo and writes {stem: per-clip metrics} to `out`.
All HGPT/absl chatter is forced to stderr so stdout carries only protocol lines.
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SCRIPTS = os.path.join(_HERE, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_PROJECT_ROOT = None
for _candidate in [Path(_HERE).resolve(), *Path(_HERE).resolve().parents]:
    if (_candidate / "scripts" / "embodied" / "physflow_canonical_rollouts.py").exists():
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.embodied.physflow_canonical_rollouts import (
    save_body,
    save_qpos,
    write_reference_from_qpos,
)


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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
    "waist_pitch_link": [],
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


def _quat_from_rotmat_wxyz(m):
    """Convert a 3x3 row-major rotation matrix to a MuJoCo-style WXYZ quat."""
    tr = float(m[0][0] + m[1][1] + m[2][2])
    if tr > 0.0:
        s = (tr + 1.0) ** 0.5 * 2.0
        return [
            float(0.25 * s),
            float((m[2][1] - m[1][2]) / s),
            float((m[0][2] - m[2][0]) / s),
            float((m[1][0] - m[0][1]) / s),
        ]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = (1.0 + m[0][0] - m[1][1] - m[2][2]) ** 0.5 * 2.0
        return [
            float((m[2][1] - m[1][2]) / s),
            float(0.25 * s),
            float((m[0][1] + m[1][0]) / s),
            float((m[0][2] + m[2][0]) / s),
        ]
    if m[1][1] > m[2][2]:
        s = (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5 * 2.0
        return [
            float((m[0][2] - m[2][0]) / s),
            float((m[0][1] + m[1][0]) / s),
            float(0.25 * s),
            float((m[1][2] + m[2][1]) / s),
        ]
    s = (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5 * 2.0
    return [
        float((m[1][0] - m[0][1]) / s),
        float((m[0][2] + m[2][0]) / s),
        float((m[1][2] + m[2][1]) / s),
        float(0.25 * s),
    ]


def _body_meta(body_name):
    return {
        "name": body_name,
        "meshes": [
            {"file": mesh_file, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
            for mesh_file in MESHES_BY_BODY.get(body_name, [])
        ],
    }


def _write_robot_frames_from_state_history(history, model, out_path, fps):
    import mujoco

    body_ids = []
    bodies = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        body_ids.append(body_id)
        bodies.append(_body_meta(name))

    frames = []
    for state in history:
        xpos = state["xpos"]
        xmat = state["xmat"].reshape(model.nbody, 3, 3)
        frames.append(
            {
                "body_pos": [xpos[i].tolist() for i in body_ids],
                "body_quat": [_quat_from_rotmat_wxyz(xmat[i]) for i in body_ids],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "type": "robot_frames",
                "robot": "g1",
                "fps": int(fps),
                "source_fps_note": "humanoid_gpt_rollout_resampled_to_policy_freq",
                "num_frames": len(frames),
                "num_bodies": len(bodies),
                "bodies": bodies,
                "frames": frames,
            }
        )
    )


def _body_arrays_from_state_history(history, model):
    import mujoco

    body_ids = np.arange(1, model.nbody, dtype=np.int32)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) or f"body_{int(i)}"
        for i in body_ids
    ]
    body_pos = np.asarray(
        [np.asarray(state["xpos"], dtype=np.float32)[body_ids] for state in history],
        dtype=np.float32,
    )
    if history and "xquat" in history[0]:
        body_quat = np.asarray(
            [np.asarray(state["xquat"], dtype=np.float32)[body_ids] for state in history],
            dtype=np.float32,
        )
    else:
        body_quat = np.asarray(
            [
                [_quat_from_rotmat_wxyz(state["xmat"].reshape(model.nbody, 3, 3)[int(i)]) for i in body_ids]
                for state in history
            ],
            dtype=np.float32,
        )
    return body_pos, body_quat, body_names


def _quat_to_mat(q):
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


def _mean_norm(x):
    return float(np.mean(np.linalg.norm(x, axis=-1)))


def _rollout_body_tracking_metrics(history, ref_traj, model, fps):
    import mujoco

    if not history or "qpos" not in ref_traj:
        return {}

    first_xpos = np.asarray(history[0]["xpos"])
    num_bodies = min(int(first_xpos.shape[0]), int(model.nbody))
    if num_bodies <= 1:
        return {}
    body_ids = np.arange(1, num_bodies, dtype=np.int32)

    ref_qpos_all = np.asarray(ref_traj["qpos"], dtype=np.float32)
    ref_qvel_all = np.asarray(ref_traj.get("qvel", np.zeros((len(ref_qpos_all), model.nv))), dtype=np.float32)
    num_steps = min(len(history), len(ref_qpos_all))
    if num_steps <= 0:
        return {}

    exec_qpos = np.asarray([np.asarray(state["qpos"], dtype=np.float32) for state in history[:num_steps]])
    exec_pos = np.asarray([np.asarray(state["xpos"], dtype=np.float32)[body_ids] for state in history[:num_steps]])
    ref_qpos = ref_qpos_all[:num_steps]
    ref_pos = np.zeros_like(exec_pos, dtype=np.float32)
    ref_data = mujoco.MjData(model)
    for t in range(num_steps):
        ref_data.qpos[:] = ref_qpos_all[t]
        if t < len(ref_qvel_all) and ref_qvel_all.shape[1] == model.nv:
            ref_data.qvel[:] = ref_qvel_all[t]
        mujoco.mj_forward(model, ref_data)
        ref_pos[t] = ref_data.xpos[body_ids]

    local_exec_pos = np.zeros_like(exec_pos, dtype=np.float32)
    local_ref_pos = np.zeros_like(ref_pos, dtype=np.float32)
    for t in range(num_steps):
        local_exec_pos[t] = (exec_pos[t] - exec_qpos[t, :3][None]) @ _quat_to_mat(exec_qpos[t, 3:7])
        local_ref_pos[t] = (ref_pos[t] - ref_qpos[t, :3][None]) @ _quat_to_mat(ref_qpos[t, 3:7])

    ref_xy_aligned = ref_pos.copy()
    ref_xy_aligned[:, :, :2] += exec_qpos[:, None, :2] - ref_qpos[:, None, :2]
    root_err = np.linalg.norm(exec_qpos[:, :3] - ref_qpos[:, :3], axis=1)
    root_height_err = np.abs(exec_qpos[:, 2] - ref_qpos[:, 2])
    joint_dim = min(exec_qpos.shape[1], ref_qpos.shape[1])
    if joint_dim > 7:
        joint_abs = np.abs(exec_qpos[:, 7:joint_dim] - ref_qpos[:, 7:joint_dim])
        max_joint = joint_abs.max(axis=1)
    else:
        max_joint = np.array([np.inf], dtype=np.float32)
    completion = float(num_steps / len(ref_qpos_all)) if len(ref_qpos_all) else 0.0
    min_height = float(np.min(exec_qpos[:, 2])) if len(exec_qpos) else float("-inf")
    finite = bool(np.isfinite(exec_qpos).all() and np.isfinite(ref_qpos).all())
    out = {
        "completion": completion,
        "fall": float(min_height < 0.25),
        "root_err_mean": float(root_err.mean()),
        "root_err_max": float(root_err.max()),
        "root_height_err_mean": float(root_height_err.mean()),
        "root_height_err_max": float(root_height_err.max()),
        "max_joint_err_mean": float(max_joint.mean()),
        "max_joint_err_max": float(max_joint.max()),
        "min_height": min_height,
        "raw_body_err_mean": _mean_norm(exec_pos - ref_pos),
        "body_err_mean": _mean_norm(exec_pos - ref_xy_aligned),
        "xy_aligned_body_err_mean": _mean_norm(exec_pos - ref_xy_aligned),
        "local_body_err_mean": _mean_norm(local_exec_pos - local_ref_pos),
    }
    paper_failed = (
        (not finite)
        or completion < 0.95
        or bool(out["fall"])
        or out["local_body_err_mean"] > 0.2
        or out["root_height_err_mean"] > 0.2
    )
    strict_failed = paper_failed or out["root_err_mean"] > 1.0 or out["max_joint_err_max"] > 0.7
    out["paper_success"] = float(not paper_failed)
    out["strict_success"] = float(not strict_failed)
    out.update(
        {
            "raw_global_mpjpe_m": out["raw_body_err_mean"],
            "raw_global_mpjpe_mm": out["raw_body_err_mean"] * 1000.0,
            "xy_aligned_mpjpe_m": out["xy_aligned_body_err_mean"],
            "xy_aligned_mpjpe_mm": out["xy_aligned_body_err_mean"] * 1000.0,
            "mpjpe_m": out["xy_aligned_body_err_mean"],
            "mpjpe_mm": out["xy_aligned_body_err_mean"] * 1000.0,
            "local_mpjpe_m": out["local_body_err_mean"],
            "local_mpjpe_mm": out["local_body_err_mean"] * 1000.0,
        }
    )

    if num_steps >= 2:
        dt = 1.0 / float(fps)
        body_vel = np.diff(exec_pos, axis=0) / dt
        ref_body_vel = np.diff(ref_pos, axis=0) / dt
        local_body_vel = np.zeros_like(body_vel, dtype=np.float32)
        ref_local_body_vel = np.zeros_like(ref_body_vel, dtype=np.float32)
        for t in range(1, num_steps):
            local_body_vel[t - 1] = ((exec_pos[t] - exec_pos[t - 1]) / dt) @ _quat_to_mat(exec_qpos[t, 3:7])
            ref_local_body_vel[t - 1] = ((ref_pos[t] - ref_pos[t - 1]) / dt) @ _quat_to_mat(ref_qpos[t, 3:7])
        out.update(
            {
                "body_vel_err_mean": _mean_norm(body_vel - ref_body_vel),
                "local_body_vel_err_mean": _mean_norm(local_body_vel - ref_local_body_vel),
                "mpjve_mps": _mean_norm(body_vel - ref_body_vel),
                "local_mpjve_mps": _mean_norm(local_body_vel - ref_local_body_vel),
            }
        )

        if num_steps >= 3:
            body_acc = np.diff(body_vel, axis=0) / dt
            ref_body_acc = np.diff(ref_body_vel, axis=0) / dt
            local_body_acc = np.diff(local_body_vel, axis=0) / dt
            ref_local_body_acc = np.diff(ref_local_body_vel, axis=0) / dt
            out.update(
                {
                    "body_acc_err_mean": _mean_norm(body_acc - ref_body_acc),
                    "local_body_acc_err_mean": _mean_norm(local_body_acc - ref_local_body_acc),
                    "mpjae_mps2": _mean_norm(body_acc - ref_body_acc),
                    "local_mpjae_mps2": _mean_norm(local_body_acc - ref_local_body_acc),
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True, help="HGPT tracker .onnx")
    ap.add_argument("--freq", type=int, default=50, help="control/eval freq (Hz)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--policy_type", default="mlp")
    ap.add_argument("--privileged", action="store_true")
    cli = ap.parse_args()

    # This worker never renders. Some Taiji V100 containers expose no usable EGL,
    # so default to MuJoCo's no-GL path unless the caller explicitly overrides it.
    os.environ.setdefault("MUJOCO_GL", "disable")

    # ---- load everything ONCE (redirect any stdout noise to stderr) ----------
    with contextlib.redirect_stdout(sys.stderr):
        import mujoco
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.WARNING)

        # Import from Humanoid-GPT's local scripts/ directory directly. The
        # enclosing hf_trainer repository also has a top-level "scripts"
        # namespace, so "from scripts.eval_parallel" can resolve to the wrong
        # package on Taiji.
        from eval_parallel import (
            ParallelEvalArgs,
            _build_policy,
            _convert_traj_to_kpt,
            _evaluate_single_traj,
            _load_npz_with_qpos,
        )
        from tracking.infer_utils import g1_infer_env_config

        args = ParallelEvalArgs(
            load_path=cli.load_path,
            mocap_path="",
            freq=cli.freq,
            device=cli.device,
            policy_type=cli.policy_type,
            privileged=cli.privileged,
            convert=True,
            workers=1,
        )
        policy = _build_policy(args)
        convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
        env_cfg = g1_infer_env_config(ctrl_dt=1.0 / args.freq)

    _emit({"status": "ready", "load_path": cli.load_path, "freq": cli.freq})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            _emit({"status": "error", "error": f"bad json: {exc}"})
            continue
        if req.get("cmd") == "shutdown":
            _emit({"status": "bye"})
            break

        job_dir = Path(req["job_dir"])
        out_path = Path(req.get("out", job_dir / "metrics.json"))
        frames_out_dir = Path(req["frames_out_dir"]) if req.get("frames_out_dir") else None
        canonical_root = Path(req["canonical_root"]) if req.get("canonical_root") else None
        canonical_split = req.get("canonical_split")
        canonical_method = req.get("canonical_method", "humanoid_gpt")
        canonical_output_fps = float(req.get("canonical_output_fps", 30.0))
        results = {}
        npzs = sorted(job_dir.glob("*.npz"))
        for i, f in enumerate(npzs):
            stem = f.stem
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    raw = _load_npz_with_qpos(f)
                    ref = _convert_traj_to_kpt(raw, convert_mj_model, args.freq)
                    m = _evaluate_single_traj(
                        traj_id=i,
                        ref_traj=ref,
                        file_name=f.name,
                        args=args,
                        env_cfg=env_cfg,
                        policy=policy,
                        capture_state_history=True,
                    )
                state_history = m.pop("state_history", None)
                if state_history is not None:
                    m.update(_rollout_body_tracking_metrics(state_history, ref, convert_mj_model, args.freq))
                if frames_out_dir is not None and state_history is not None:
                    _write_robot_frames_from_state_history(
                        state_history,
                        convert_mj_model,
                        frames_out_dir / f"{stem}.json",
                        args.freq,
                    )
                if canonical_root is not None and canonical_split and state_history is not None:
                    source_fps = float(np.asarray(raw.get("frequency", raw.get("fps", args.freq))).reshape(-1)[0])
                    meta = {
                        "source": str(f),
                        "runner": "hftrainer/models/motion/physflow/trackers/humanoid_gpt/physflow_hgpt_judge_server.py",
                        "onnx": str(cli.load_path),
                        "control_fps": args.freq,
                        "output_fps": canonical_output_fps,
                    }
                    write_reference_from_qpos(
                        canonical_root,
                        canonical_split,
                        stem,
                        np.asarray(raw["qpos"], dtype=np.float32),
                        source_fps=source_fps,
                        model=convert_mj_model,
                        target_fps=canonical_output_fps,
                        metadata=meta,
                    )
                    exec_qpos = np.asarray(
                        [np.asarray(state["qpos"], dtype=np.float32) for state in state_history],
                        dtype=np.float32,
                    )
                    save_qpos(
                        canonical_root,
                        canonical_split,
                        canonical_method,
                        stem,
                        exec_qpos,
                        source_fps=float(args.freq),
                        target_fps=canonical_output_fps,
                        metadata={**meta, "execution_frames_source": int(exec_qpos.shape[0])},
                    )
                    body_pos, body_quat, body_names = _body_arrays_from_state_history(state_history, convert_mj_model)
                    save_body(
                        canonical_root,
                        canonical_split,
                        canonical_method,
                        stem,
                        body_pos,
                        body_quat,
                        body_names,
                        source_fps=float(args.freq),
                        target_fps=canonical_output_fps,
                        metadata={**meta, "execution_frames_source": int(body_pos.shape[0])},
                    )
                results[stem] = {k: float(v) for k, v in m.items() if k not in ("file_name",)}
            except Exception as exc:  # noqa: BLE001
                results[stem] = {"error": str(exc)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results))
        _emit({"status": "ok", "job_dir": str(job_dir), "out": str(out_path), "n": len(results)})


if __name__ == "__main__":
    main()

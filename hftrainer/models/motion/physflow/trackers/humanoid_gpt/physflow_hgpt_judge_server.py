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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SCRIPTS = os.path.join(_HERE, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


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
                        capture_state_history=frames_out_dir is not None,
                    )
                state_history = m.pop("state_history", None)
                if frames_out_dir is not None and state_history is not None:
                    _write_robot_frames_from_state_history(
                        state_history,
                        convert_mj_model,
                        frames_out_dir / f"{stem}.json",
                        args.freq,
                    )
                results[stem] = {k: float(v) for k, v in m.items() if k not in ("file_name",)}
            except Exception as exc:  # noqa: BLE001
                results[stem] = {"error": str(exc)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results))
        _emit({"status": "ok", "job_dir": str(job_dir), "out": str(out_path), "n": len(results)})


if __name__ == "__main__":
    main()

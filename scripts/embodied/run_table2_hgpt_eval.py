#!/usr/bin/env python3
"""Run Humanoid-GPT Table-2 tracker evaluation on 36-dim G1 qpos npz files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HGPT_ROOT = PROJECT_ROOT / "hftrainer/models/motion/physflow/trackers/humanoid_gpt"
HGPT_ONNX = HGPT_ROOT / "storage/ckpts/pns_wo_priv216.onnx"
TARGET_G1_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _readline_timeout(proc: subprocess.Popen, timeout_s: float) -> str:
    out = {"line": ""}

    def _read() -> None:
        out["line"] = proc.stdout.readline()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout_s)
    return out["line"] if not t.is_alive() else ""


def _names_from_score(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return [f"h{int(row['idx']):03d}_gen" for row in data["rows"]]


def _names_from_manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} must be a JSON list")
    return [str(x) for x in data]


def _safe_output_stem(name: str) -> str:
    return "__".join(Path(name).with_suffix("").parts).replace(" ", "_")


def _resolve_source(motion_dir: Path, name: str) -> Path:
    direct = motion_dir / f"{name}.npz"
    if direct.exists():
        return direct
    rel = motion_dir / name
    if rel.suffix == ".npz" and rel.exists():
        return rel
    rel_npz = rel.with_suffix(".npz")
    if rel_npz.exists():
        return rel_npz
    return direct


def _as_list(value: np.ndarray) -> list[str]:
    return [str(x) for x in value.tolist()]


def _scalar(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in data.files:
            arr = np.asarray(data[key]).reshape(-1)
            if arr.size:
                return float(arr[0])
    return float(default)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-8)


def _qpos_slices(joint_names: list[str], jnt_type: np.ndarray | None) -> dict[str, slice]:
    if jnt_type is None:
        jnt_type = np.array([0] + [3] * (len(joint_names) - 1), dtype=np.int32)
    qpos_slices: dict[str, slice] = {}
    qpos_i = 0
    for name, typ in zip(joint_names, jnt_type):
        typ = int(typ)
        if typ == 0:
            qpos_slices[name] = slice(qpos_i, qpos_i + 7)
            qpos_i += 7
        elif typ in (2, 3):
            qpos_slices[name] = slice(qpos_i, qpos_i + 1)
            qpos_i += 1
        else:
            raise ValueError(f"Unsupported joint type {typ} for {name}")
    return qpos_slices


def _expand_named_qpos_to_g1(
    qpos: np.ndarray,
    joint_names: list[str],
    jnt_type: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2D, got {qpos.shape}")
    if qpos.shape[1] == 36:
        return qpos.astype(np.float32, copy=False), []

    source_slices = _qpos_slices(joint_names, jnt_type)
    out = np.zeros((qpos.shape[0], 36), dtype=np.float32)
    if "root" in source_slices:
        out[:, :7] = qpos[:, source_slices["root"]]
    elif qpos.shape[1] >= 7:
        out[:, :7] = qpos[:, :7]
    else:
        raise ValueError(f"qpos shape {qpos.shape} cannot provide free root")

    missing: list[str] = []
    for i, name in enumerate(TARGET_G1_DOF_NAMES):
        sl = source_slices.get(name)
        if sl is None:
            missing.append(name)
            continue
        out[:, 7 + i] = qpos[:, sl].reshape(qpos.shape[0])
    return out, missing


def _amp_source_to_qpos(
    data: np.lib.npyio.NpzFile,
    body_quat_order: str,
) -> tuple[np.ndarray, float, list[str]]:
    body_names = _as_list(data["body_names"])
    dof_names = _as_list(data["dof_names"])
    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    root_pos = np.asarray(data["body_positions"][:, pelvis_idx, :], dtype=np.float32)
    root_quat = np.asarray(data["body_rotations"][:, pelvis_idx, :], dtype=np.float32)
    if body_quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    elif body_quat_order != "wxyz":
        raise ValueError(f"Unsupported body_quat_order={body_quat_order}")
    root_quat = _normalize_quat(root_quat).astype(np.float32)

    dof_src = np.asarray(data["dof_positions"], dtype=np.float32)
    dof = np.zeros((dof_src.shape[0], len(TARGET_G1_DOF_NAMES)), dtype=np.float32)
    missing: list[str] = []
    for i, name in enumerate(TARGET_G1_DOF_NAMES):
        if name not in dof_names:
            missing.append(name)
            continue
        dof[:, i] = dof_src[:, dof_names.index(name)]
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    fps = _scalar(data, ("frequency", "fps"), 30.0)
    return qpos, fps, missing


def _materialize_inputs(
    motion_dir: Path,
    names: list[str],
    job_dir: Path,
    input_fps: float,
    body_quat_order: str,
) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, str] = {}
    kept: list[str] = []
    conversions: dict[str, Any] = {}
    for name in names:
        src = _resolve_source(motion_dir, name)
        if not src.exists():
            skipped[name] = f"missing {src}"
            continue
        data = np.load(src, allow_pickle=True)
        missing_joints: list[str] = []
        if "qpos" in data.files:
            qpos_src = np.asarray(data["qpos"], dtype=np.float32)
            if "joint_names" in data.files:
                joint_names = _as_list(data["joint_names"])
                jnt_type = np.asarray(data["jnt_type"]) if "jnt_type" in data.files else None
                qpos, missing_joints = _expand_named_qpos_to_g1(qpos_src, joint_names, jnt_type)
            elif qpos_src.ndim == 2 and qpos_src.shape[1] == 36:
                qpos = qpos_src
            else:
                skipped[name] = f"qpos shape {qpos_src.shape}, expected (T, 36) or joint_names"
                continue
            freq = _scalar(data, ("frequency", "fps"), input_fps)
        elif {"body_positions", "body_rotations", "dof_positions", "dof_names", "body_names"}.issubset(data.files):
            qpos, freq, missing_joints = _amp_source_to_qpos(data, body_quat_order)
        else:
            skipped[name] = "missing qpos or AMP body/dof fields"
            continue

        if qpos.ndim != 2 or qpos.shape[1] != 36:
            skipped[name] = f"converted qpos shape {qpos.shape}, expected (T, 36)"
            continue

        out_stem = _safe_output_stem(name)
        np.savez_compressed(
            job_dir / f"{out_stem}.npz",
            qpos=qpos.astype(np.float32, copy=False),
            frequency=np.float32(freq),
            source=str(src),
        )
        kept.append(out_stem)
        conversions[out_stem] = {
            "source_name": name,
            "source_path": str(src),
            "num_frames": int(qpos.shape[0]),
            "frequency": float(freq),
            "missing_joints_filled_zero": missing_joints,
        }
    return {"kept": kept, "skipped": skipped, "conversions": conversions}


def _write_manifest_from_recursive_npz(root: Path, out_path: Path) -> None:
    names = [str(p.relative_to(root).with_suffix("")) for p in sorted(root.glob("**/*.npz"))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(names, indent=2) + "\n")


def _run_worker(
    hgpt_python: Path,
    job_dir: Path,
    out_json: Path,
    onnx_path: Path,
    freq: int,
    device: str,
    timeout_s: float,
    frames_out_dir: Path | None = None,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(HGPT_ROOT), str(HGPT_ROOT / "scripts"), env.get("PYTHONPATH", "")])
    env.setdefault("MUJOCO_GL", "disable")
    cmd = [
        str(hgpt_python),
        "physflow_hgpt_judge_server.py",
        "--load_path",
        str(onnx_path),
        "--freq",
        str(freq),
        "--device",
        device,
        "--policy_type",
        "mlp",
    ]
    log_path = out_json.with_suffix(".worker.log")
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(HGPT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        try:
            ready = _readline_timeout(proc, timeout_s)
            if not ready:
                raise RuntimeError(f"worker did not become ready, see {log_path}")
            ready_obj = json.loads(ready)
            if ready_obj.get("status") != "ready":
                raise RuntimeError(f"bad worker startup: {ready_obj}")
            req = {"job_dir": str(job_dir.resolve()), "out": str(out_json.resolve())}
            if frames_out_dir is not None:
                req["frames_out_dir"] = str(frames_out_dir.resolve())
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            resp = _readline_timeout(proc, timeout_s)
            if not resp:
                raise RuntimeError(f"worker timed out, see {log_path}")
            resp_obj = json.loads(resp)
            if resp_obj.get("status") != "ok":
                raise RuntimeError(f"worker error: {resp_obj}")
            proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            proc.stdin.flush()
            _readline_timeout(proc, 30)
        finally:
            if proc.poll() is None:
                proc.kill()


def _summarize(raw: dict[str, Any], complete_thresh: float) -> dict[str, float]:
    ok_rows = [v for v in raw.values() if isinstance(v, dict) and "error" not in v]
    n = len(raw)

    def mean(key: str) -> float:
        vals = [float(v[key]) for v in ok_rows if key in v and np.isfinite(float(v[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "num_motions": float(n),
        "num_ok": float(len(ok_rows)),
        "error_rate": float((n - len(ok_rows)) / n) if n else float("nan"),
        "success_rate": float(np.mean([float(v.get("length_ratio", 0.0)) >= complete_thresh for v in ok_rows]))
        if ok_rows else float("nan"),
        "completion": mean("length_ratio"),
        "kpt_pos_mae_m": mean("kpt_pos_mae"),
        "kpt_rot_mae_rad": mean("kpt_rot_mae"),
        "joint_pos_mae_rad": mean("joint_pos_mae"),
        "joint_vel_mae_radps": mean("joint_vel_mae"),
        "root_pos_err_mm": mean("root_pos_err_mm"),
        "root_vel_err_mmps": mean("root_vel_err_mms"),
        "root_yaw_err_rad": mean("root_yaw_err"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--score-json", type=Path)
    group.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hgpt-python", type=Path, default=Path("/dev/shm/hgpt_venv311/bin/python"))
    ap.add_argument("--onnx", type=Path, default=HGPT_ONNX)
    ap.add_argument("--input-fps", type=float, default=30.0)
    ap.add_argument("--body-quat-order", choices=["xyzw", "wxyz"], default="xyzw")
    ap.add_argument("--write-recursive-manifest", type=Path)
    ap.add_argument("--freq", type=int, default=50)
    ap.add_argument("--device", default=os.environ.get("HGPT_DEVICE", "cpu"))
    ap.add_argument("--complete-thresh", type=float, default=0.9)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--frames-out-dir", type=Path)
    args = ap.parse_args()

    if args.write_recursive_manifest:
        _write_manifest_from_recursive_npz(args.motion_dir, args.write_recursive_manifest)
        return

    names = _names_from_score(args.score_json) if args.score_json else _names_from_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    job_dir = args.out_dir / "hgpt_inputs"
    prep = _materialize_inputs(args.motion_dir, names, job_dir, args.input_fps, args.body_quat_order)
    if not prep["kept"]:
        raise RuntimeError(f"no 36-dim qpos inputs; skipped={prep['skipped']}")

    raw_json = args.out_dir / "hgpt_raw_metrics.json"
    _run_worker(
        args.hgpt_python,
        job_dir,
        raw_json,
        args.onnx,
        args.freq,
        args.device,
        args.timeout_s,
        args.frames_out_dir,
    )
    raw = json.loads(raw_json.read_text())
    payload = {
        "summary": _summarize(raw, args.complete_thresh),
        "motions": raw,
        "input": {
            "motion_dir": str(args.motion_dir),
            "num_requested": len(names),
            "kept": prep["kept"],
            "skipped": prep["skipped"],
            "conversions": prep["conversions"],
            "hgpt_python": str(args.hgpt_python),
            "onnx": str(args.onnx),
            "device": args.device,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Humanoid-GPT Table-2 Evaluation", ""]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value:.6g}")
    if prep["skipped"]:
        lines.append("")
        lines.append(f"- skipped: {len(prep['skipped'])}")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

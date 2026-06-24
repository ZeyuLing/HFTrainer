#!/usr/bin/env python3
"""PyRoki retargeted output -> AMP/ProtoMotions NPZ (same schema as data/g1).

Input  : <id>_retargeted.npz with base_frame_pos[T,3], base_frame_wxyz[T,4],
         joint_angles[T,29]  (g1.urdf actuated-joint order == DOF_NAMES order).
Output : AMP npz (MuJoCo FK on g1_holo_compat.xml, 30 std bodies, ground-aligned,
         finite-diff velocities) identical in schema to batch_retarget_g1_gmr.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = PROJECT_ROOT / "ref_repo" / "ProtoMotions" / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"

DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
STD_BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def patch_mjcf(xml_path: Path) -> str:
    tree = ET.parse(str(xml_path)); root = tree.getroot()
    for s in root.findall("sensor"):
        root.remove(s)
    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            if "floor" in pair.get("geom1", "") or "floor" in pair.get("geom2", ""):
                contact.remove(pair)
    wb = root.find("worldbody")
    if wb is not None and not any("floor" in g.get("name", "").lower() or g.get("type", "") == "plane" for g in wb.findall("geom")):
        g = ET.SubElement(wb, "geom"); g.set("name", "floor"); g.set("type", "plane"); g.set("size", "0 0 0.05")
    return ET.tostring(root, encoding="unicode")


def load_model(mjcf: Path):
    import mujoco
    patched = patch_mjcf(mjcf)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=str(mjcf.parent), delete=False) as f:
        f.write(patched); tmp = f.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)
    data = mujoco.MjData(model)
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in STD_BODY_NAMES])
    return mujoco, model, data, body_ids


def fk_amp(mujoco, model, data, body_ids, qpos_zup):
    T = qpos_zup.shape[0]; nb = len(body_ids)
    min_z = np.inf
    for t in range(T):
        data.qpos[:36] = qpos_zup[t]; mujoco.mj_forward(model, data)
        min_z = min(min_z, float(data.xpos[body_ids, 2].min()))
    z_off = (-min_z) if np.isfinite(min_z) else 0.0
    body_pos = np.zeros((T, nb, 3), np.float32); body_rot = np.zeros((T, nb, 4), np.float32)
    q = qpos_zup.copy(); q[:, 2] += z_off
    for t in range(T):
        data.qpos[:36] = q[t]; mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[body_ids]
        body_rot[t] = data.xquat[body_ids][:, [1, 2, 3, 0]]
    return body_pos, body_rot


def ang_vel(rot_xyzw, fps):
    from scipy.spatial.transform import Rotation as R
    T, nb = rot_xyzw.shape[0], rot_xyzw.shape[1]
    av = np.zeros((T, nb, 3), np.float32)
    if T < 2:
        return av
    for b in range(nb):
        r = R.from_quat(rot_xyzw[:, b, :])
        rel = r[:-1].inv() * r[1:]
        world = r[:-1].apply(rel.as_rotvec() * fps)
        av[:-1, b, :] = world; av[-1, b, :] = av[-2, b, :]
    return av


def convert_one(in_npz, mujoco, model, data, body_ids, fps):
    d = np.load(in_npz, allow_pickle=True)
    root_pos = d["base_frame_pos"].astype(np.float64)        # (T,3)
    root_wxyz = d["base_frame_wxyz"].astype(np.float64)      # (T,4)
    dof = d["joint_angles"].astype(np.float64)               # (T,29)
    T = dof.shape[0]
    qpos = np.zeros((T, 36))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_wxyz
    qpos[:, 7:] = dof
    body_pos, body_rot = fk_amp(mujoco, model, data, body_ids, qpos)
    dof_pos = dof.astype(np.float32)
    dof_vel = (np.gradient(dof_pos, axis=0) * fps).astype(np.float32)
    body_lin = (np.gradient(body_pos, axis=0) * fps).astype(np.float32)
    body_ang = ang_vel(body_rot, fps).astype(np.float32)
    return dict(fps=np.array([fps], np.float32), dof_names=np.array(DOF_NAMES),
                body_names=np.array(STD_BODY_NAMES), dof_positions=dof_pos,
                dof_velocities=dof_vel, body_positions=body_pos, body_rotations=body_rot,
                body_linear_velocities=body_lin, body_angular_velocities=body_ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir of <id>_retargeted.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mujoco, model, data, body_ids = load_model(Path(args.mjcf))

    for f in sorted(in_dir.glob("*_retargeted.npz")):
        stem = f.name.replace("_retargeted.npz", "")
        try:
            fields = convert_one(f, mujoco, model, data, body_ids, args.fps)
        except Exception as e:
            import traceback
            print(f"[fail] {stem}: {e}\n{traceback.format_exc()[-400:]}", flush=True)
            continue
        out = out_dir / f"{stem}.npz"
        np.savez(str(out) + ".tmp", **fields); os.replace(str(out) + ".tmp.npz", out)
        print(f"[ok] {stem}: {fields['dof_positions'].shape[0]} frames", flush=True)


if __name__ == "__main__":
    main()

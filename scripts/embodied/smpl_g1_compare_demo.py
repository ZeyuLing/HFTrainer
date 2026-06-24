#!/usr/bin/env python3
"""Produce SMPL-mesh vs G1-mesh side-by-side comparison demo data.

For each chosen high_quality HYMotion AMASS npz this script emits two viewer
payloads consumed by motion_annot_web/embodied_viz:

  * SMPL side  -> {type:"frames", fps, frames:[[{Rh,Th,poses,shapes,...}]]}
                  (axis-angle, smplh; rendered by /viewer load_smpl.js)
  * G1 side    -> {type:"robot_frames", ...}  rendered by /robot_viewer

The G1 side is produced fully kinematically (no IsaacGym / no ONNX policy):
  AMASS axis-angle poses -> rot6d -> SMPLToG1Retargeter.retarget
  -> qpos[T,36] -> Y-up->Z-up fix -> ground align -> MuJoCo mj_forward FK
  -> per-body world xpos/xquat (wxyz).

It also writes a manifest.json consumed by the new /smpl_vs_g1 compare page.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from math import cos, sin
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUALITY = PROJECT_ROOT / "data" / "hymotion_m2m_refine_data" / "data_quality_list" / "high_quality.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "hymotion_data"
DEFAULT_MJCF = PROJECT_ROOT / "ref_repo" / "ProtoMotions" / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"
DEFAULT_OUT = PROJECT_ROOT / "output" / "smpl_g1_compare"

# Keyword buckets used to pick visually-diverse demo motions.
DIVERSE_KEYWORDS = [
    "walk", "run", "jump", "kick", "dance", "wave", "punch",
    "turn", "throw", "jog", "squat", "spin",
]


def rx90_quat() -> np.ndarray:
    """wxyz quaternion of a +90 deg rotation about X (maps Y-up -> Z-up)."""
    c = np.cos(np.pi / 4.0)
    s = np.sin(np.pi / 4.0)
    return np.array([c, s, 0.0, 0.0], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions (q1 applied after q2)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def build_smpl_json(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray,
                    fps: int, gender: str) -> dict:
    """AMASS axis-angle (smplh, poses[T,156]) -> SMPL-mesh viewer JSON."""
    T = poses.shape[0]
    root_orient = poses[:, :3]
    poses_smplh = poses[:, :156]
    shape = betas.reshape(-1)[:16]
    if shape.shape[0] < 16:
        shape = np.concatenate([shape, np.zeros(16 - shape.shape[0])])
    shapes = [shape.astype(float).tolist()]
    frames = []
    for t in range(T):
        frames.append([{
            "id": 0,
            "gender": gender,
            "smpl_type": "smplh",
            "Rh": [root_orient[t].astype(float).tolist()],
            "Th": [trans[t].astype(float).tolist()],
            "poses": [poses_smplh[t].astype(float).tolist()],
            "shapes": shapes,
            "mocap_framerate": fps,
        }])
    return {"type": "frames", "fps": fps, "frames": frames}


def retarget_to_qpos(poses: np.ndarray, trans: np.ndarray, fps: int):
    """AMASS axis-angle -> G1 qpos[T,36] in MuJoCo Z-up frame, ground-aligned later."""
    import torch
    from hftrainer.motion.retarget.smpl_g1 import SMPLToG1Retargeter
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
    )

    T = poses.shape[0]
    aa = torch.from_numpy(poses[:, :66].reshape(T, 22, 3).astype(np.float32))
    rotmats = axis_angle_to_matrix(aa)  # (T,22,3,3)
    # Row-major rot6d expected by the retargeter == first two columns flattened.
    rot6d_row = rotmats[..., :2].reshape(T, 22, 6).numpy()
    ret = SMPLToG1Retargeter()
    res = ret.retarget(rot6d_row, trans.astype(np.float32), fps=float(fps))
    qpos = ret.to_mujoco_qpos(res).astype(np.float64)  # (T,36)

    # Y-up (SMPL/AMASS) -> Z-up (MuJoCo) is a world basis change (R_x +90deg).
    # Positions transform as vectors: p' = R_fix @ p  =>  (x,y,z) -> (x,-z,y).
    # Root orientation transforms by SIMILARITY (conjugation): q' = qfix * q * qfix^-1.
    # Conjugation keeps identity -> identity, so an upright SMPL pose (q~=I) maps to
    # an upright G1 base (q~=I) instead of being tipped over 90deg.
    qfix = rx90_quat()
    qfix_inv = np.array([qfix[0], -qfix[1], -qfix[2], -qfix[3]], dtype=np.float64)
    pos = qpos[:, 0:3]
    pos_z = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], axis=1)
    qpos[:, 0:3] = pos_z
    for t in range(T):
        qpos[t, 3:7] = quat_mul(quat_mul(qfix, qpos[t, 3:7]), qfix_inv)
    return qpos


def _patch_mjcf_xml(xml_path: Path) -> str:
    """Strip sensors and add a ground plane so the model loads standalone."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or "ground" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("rgba", "0.7 0.7 0.7 1")
    return ET.tostring(root, encoding="unicode")


def parse_body_mesh_mapping(mjcf_path: Path) -> list:
    """Parse MJCF -> [{name, meshes:[{file,pos,quat}]}] preserving XML geom xforms."""
    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()
    mesh_name_to_file = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh_elem in asset.findall("mesh"):
            name = mesh_elem.get("name", "")
            filename = mesh_elem.get("file", "")
            if name and filename:
                mesh_name_to_file[name] = filename

    def parse_float_list(value, default):
        if value is None:
            return list(default)
        return [float(x) for x in value.split()]

    def quat_from_axis_angle(axis_angle):
        if axis_angle is None:
            return None
        values = [float(x) for x in axis_angle.split()]
        if len(values) != 4:
            return None
        axis = np.asarray(values[:3], dtype=np.float64)
        angle = values[3]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return [1.0, 0.0, 0.0, 0.0]
        axis = axis / norm
        half = 0.5 * angle
        xyz = axis * sin(half)
        return [float(cos(half)), float(xyz[0]), float(xyz[1]), float(xyz[2])]

    def quat_from_euler(euler):
        if euler is None:
            return None
        values = [float(x) for x in euler.split()]
        if len(values) != 3:
            return None
        cx, cy, cz = (cos(v * 0.5) for v in values)
        sx, sy, sz = (sin(v * 0.5) for v in values)
        return [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ]

    def geom_mesh_record(geom):
        if geom.get("type") not in (None, "mesh") and not geom.get("mesh"):
            return None
        mesh_name = geom.get("mesh", "")
        if mesh_name not in mesh_name_to_file:
            return None
        quat = (
            parse_float_list(geom.get("quat"), [1.0, 0.0, 0.0, 0.0])
            if geom.get("quat") is not None
            else quat_from_axis_angle(geom.get("axisangle"))
            or quat_from_euler(geom.get("euler"))
            or [1.0, 0.0, 0.0, 0.0]
        )
        return {
            "file": mesh_name_to_file[mesh_name],
            "pos": parse_float_list(geom.get("pos"), [0.0, 0.0, 0.0]),
            "quat": quat,
        }

    bodies = []

    def walk_body(elem):
        body_name = elem.get("name", "unnamed")
        mesh_records = []
        seen_files = set()
        for geom in elem.findall("geom"):
            mesh_record = geom_mesh_record(geom)
            if mesh_record is None:
                continue
            stl_file = mesh_record["file"]
            if stl_file in seen_files:
                continue
            seen_files.add(stl_file)
            mesh_records.append(mesh_record)
        bodies.append({"name": body_name, "meshes": mesh_records})
        for child in elem.findall("body"):
            walk_body(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            walk_body(top_body)
    return bodies


GMR_HEADLESS = PROJECT_ROOT / "scripts" / "embodied" / "gmr_retarget_headless.py"


def gmr_retarget_to_qpos(poses, trans, betas, gender, fps, workdir, stem):
    """High-quality SMPL->G1 via GMR (mink IK). Returns qpos[T,36] (Z-up, grounded)."""
    T = poses.shape[0]
    smplx_npz = workdir / f"{stem}.smplx.npz"
    np.savez(
        smplx_npz,
        gender=str(gender),
        pose_body=poses[:, 3:66].astype(np.float32),   # 21 body joints * 3
        root_orient=poses[:, :3].astype(np.float32),
        trans=trans.astype(np.float32),
        betas=betas.reshape(-1).astype(np.float32),
        mocap_frame_rate=np.int64(fps),
    )
    pkl = workdir / f"{stem}.g1.pkl"
    env = dict(os.environ)
    env["MUJOCO_GL"] = "disable"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, str(GMR_HEADLESS),
        "--smplx_file", str(smplx_npz),
        "--robot", "unitree_g1",
        "--save_path", str(pkl),
        "--tgt_fps", str(int(fps)),
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0 or not pkl.is_file():
        raise RuntimeError(f"GMR failed for {stem}:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")
    from scipy.spatial.transform import Rotation as Rot

    d = pickle.load(open(pkl, "rb"))
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)       # (T,3) SMPL-X Y-up frame
    root_rot_xyzw = np.asarray(d["root_rot"], dtype=np.float64)  # (T,4) xyzw, rot_offset baked
    dof = np.asarray(d["dof_pos"], dtype=np.float64)             # (T,29)

    # GMR output lives in the SMPL-X Y-up frame with a 120deg rot_offset
    # (smplx_to_g1.json pelvis rot_offset, wxyz [0.5,-0.5,-0.5,-0.5]) baked into
    # the pelvis quaternion. Undo it to land in MuJoCo's native Z-up frame -- this
    # mirrors gmr_to_protomotions.py::{convert_root_pos_to_zup, remove_gmr_root_offset}.
    rot_offset = Rot.from_quat([-0.5, -0.5, -0.5, 0.5])  # xyzw
    # Y-up -> Z-up is a WORLD basis change: LEFT-multiply rotations (right-multiply
    # is a body-frame op and flips the pelvis up-axis whenever facing changes).
    root_pos_zup = rot_offset.inv().apply(root_pos)
    root_rot_zup_xyzw = (rot_offset.inv() * Rot.from_quat(root_rot_xyzw)).as_quat()

    n = root_pos.shape[0]
    qpos = np.zeros((n, 36), dtype=np.float64)
    qpos[:, 0:3] = root_pos_zup
    qpos[:, 3:7] = root_rot_zup_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    qpos[:, 7:36] = dof
    return qpos


def load_g1_model(mjcf: Path):
    import mujoco

    patched = _patch_mjcf_xml(mjcf)
    asset_dir = str(mjcf.parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    parsed = {b["name"]: b["meshes"] for b in parse_body_mesh_mapping(mjcf)}
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]
    bodies = [{"name": n, "meshes": parsed.get(n, [])} for n in body_names]
    return model, bodies


def qpos_to_robot_frames(model, bodies, qpos: np.ndarray, fps: int) -> dict:
    import mujoco
    data = mujoco.MjData(model)
    nb = model.nbody
    T = qpos.shape[0]

    # First pass: ground-align so the lowest body sits near z=0.
    min_z = np.inf
    for t in range(T):
        data.qpos[: qpos.shape[1]] = qpos[t]
        mujoco.mj_forward(model, data)
        min_z = min(min_z, float(data.xpos[1:nb, 2].min()))
    z_off = -min_z + 0.02 if np.isfinite(min_z) else 0.0
    qpos = qpos.copy()
    qpos[:, 2] += z_off

    frames = []
    for t in range(T):
        data.qpos[: qpos.shape[1]] = qpos[t]
        mujoco.mj_forward(model, data)
        frames.append({
            "body_pos": data.xpos[1:nb].astype(float).tolist(),
            "body_quat": data.xquat[1:nb].astype(float).tolist(),
        })
    return {
        "type": "robot_frames",
        "robot": "g1",
        "fps": int(fps),
        "num_frames": len(frames),
        "num_bodies": len(bodies),
        "bodies": bodies,
        "frames": frames,
    }


def pick_motions(items, data_dir: Path, num: int, seed: int):
    rng = random.Random(seed)
    by_kw = {k: [] for k in DIVERSE_KEYWORDS}
    others = []
    for it in items:
        p = it["path"] if isinstance(it, dict) else it
        low = p.lower()
        placed = False
        for k in DIVERSE_KEYWORDS:
            if k in low:
                by_kw[k].append(p)
                placed = True
                break
        if not placed:
            others.append(p)
    picked = []
    for k in DIVERSE_KEYWORDS:
        if by_kw[k]:
            picked.append(rng.choice(by_kw[k]))
        if len(picked) >= num:
            break
    rng.shuffle(others)
    i = 0
    while len(picked) < num and i < len(others):
        picked.append(others[i])
        i += 1
    return picked[:num]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-list", type=str, default=str(DEFAULT_QUALITY))
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--method", type=str, default="analytic", choices=["analytic", "gmr"],
                    help="analytic = fast Euler decomposition; gmr = mink IK (high quality).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--names", type=str, nargs="*", default=None,
                    help="Explicit relative npz paths to use instead of auto-pick.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    smpl_dir = out_dir / "smpl_mesh"
    g1_dir = out_dir / "robot_frames"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    g1_dir.mkdir(parents=True, exist_ok=True)

    if args.names:
        rels = args.names
    else:
        print(f"[pick] loading quality list {args.quality_list} ...")
        ql = json.load(open(args.quality_list))
        items = ql["items"] if isinstance(ql, dict) and "items" in ql else ql
        rels = pick_motions(items, data_dir, args.num, args.seed)
    print(f"[pick] {len(rels)} motions")

    model, bodies = load_g1_model(Path(args.mjcf))
    print(f"[g1] model loaded: {model.nbody} bodies, nq={model.nq}")

    rows = []
    for idx, rel in enumerate(rels):
        npz = data_dir / rel
        if not npz.is_file():
            print(f"[skip] missing {npz}")
            continue
        d = np.load(npz, allow_pickle=True)
        poses = d["poses"].astype(np.float32)
        trans = d["trans"].astype(np.float32)
        betas = d["betas"].astype(np.float32) if "betas" in d.files else np.zeros((1, 16), np.float32)
        gender = str(d["gender"]) if "gender" in d.files else "neutral"
        src_fps = int(np.asarray(d.get("mocap_framerate", 30)).reshape(-1)[0])
        stride = max(1, round(src_fps / args.target_fps))
        poses = poses[::stride]
        trans = trans[::stride]
        if poses.shape[0] > args.max_frames:
            poses = poses[: args.max_frames]
            trans = trans[: args.max_frames]
        out_fps = max(1, round(src_fps / stride))

        stem = f"{idx:02d}_" + Path(rel).stem
        smpl_json = build_smpl_json(poses, trans, betas, out_fps, gender)
        if args.method == "gmr":
            try:
                qpos = gmr_retarget_to_qpos(poses, trans, betas, gender, out_fps, g1_dir, stem)
            except Exception as e:
                print(f"[gmr-fail] {stem}: {e}")
                continue
        else:
            qpos = retarget_to_qpos(poses, trans, out_fps)
        g1_json = qpos_to_robot_frames(model, bodies, qpos, out_fps)

        smpl_path = (smpl_dir / f"{stem}.json").resolve()
        g1_path = (g1_dir / f"{stem}.json").resolve()
        json.dump(smpl_json, open(smpl_path, "w"))
        json.dump(g1_json, open(g1_path, "w"))
        rows.append({
            "name": stem,
            "source": rel,
            "frames": int(poses.shape[0]),
            "fps": out_fps,
            "smpl_path": str(smpl_path),
            "g1_path": str(g1_path),
        })
        print(f"[ok] {stem}: {poses.shape[0]} frames @ {out_fps}fps")

    manifest = {
        "title": "SMPL (human) vs G1 (robot) retarget comparison",
        "data_dir": str(data_dir),
        "rows": rows,
    }
    man_path = out_dir / "manifest.json"
    json.dump(manifest, open(man_path, "w"), indent=2)
    print(f"[done] {len(rows)} rows -> {man_path}")


if __name__ == "__main__":
    main()

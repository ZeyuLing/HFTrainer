#!/usr/bin/env python3
"""Batch SMPL/AMASS -> Unitree G1 retargeting via GMR (mink IK).

For every entry in a quality list (e.g. high_quality.json) this produces one
IsaacLab-AMP / ProtoMotions compatible npz with the SAME schema as
``data/AMASS_Retarged_for_G1/g1``:

    fps                       (1,)        float32
    dof_names                 (29,)       <U
    body_names                (30,)       <U
    dof_positions             (T, 29)     float32
    dof_velocities            (T, 29)     float32
    body_positions            (T, 30, 3)  float32   (MuJoCo Z-up, ground-aligned)
    body_rotations            (T, 30, 4)  float32   (xyzw)
    body_linear_velocities    (T, 30, 3)  float32
    body_angular_velocities   (T, 30, 3)  float32

Pipeline per motion:
    AMASS axis-angle npz -> SMPL-X FK targets -> GMR mink IK (frame-by-frame, with
       in-solver temporal posture regularization; NO output smoothing)
    -> joint-limit clamp (hard clip only)
    -> undo GMR 120deg pelvis rot_offset (Y-up -> Z-up)
    -> MuJoCo forward kinematics (30 standard bodies) -> finite-diff velocities.

Sharding:
    --world-size N --rank R   : process items[R::N]  (multi-node)
    --workers W               : in-node multiprocessing pool

Resumable: skips motions whose output npz already exists unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
GMR_ROOT = PROJECT_ROOT / "ref_repo" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

DEFAULT_QUALITY = PROJECT_ROOT / "data" / "hymotion_m2m_refine_data" / "data_quality_list" / "high_quality.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "hymotion_data"
DEFAULT_MJCF = PROJECT_ROOT / "ref_repo" / "ProtoMotions" / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"
DEFAULT_OUT = PROJECT_ROOT / "data" / "g1"

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
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
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

# GMR pelvis rot_offset (smplx_to_g1.json wxyz [0.5,-0.5,-0.5,-0.5]); undo to Z-up.
_ROT_OFFSET_XYZW = np.array([-0.5, -0.5, -0.5, 0.5], dtype=np.float64)

# Per-process singletons (filled by _worker_init).
_G = {}


def _patch_mjcf_xml(xml_path: Path) -> str:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)
    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            if "floor" in pair.get("geom1", "") or "floor" in pair.get("geom2", ""):
                contact.remove(pair)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower() or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
    return ET.tostring(root, encoding="unicode")


def _worker_init(mjcf_path: str, tgt_fps: int, posture_cost: float = 20.0):
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa
    from general_motion_retargeting.utils.smpl import (  # noqa
        get_smplx_data_offline_fast,
    )
    from gmr_retarget_headless import (  # noqa
        clamp_joint_limits, compute_ground_offset,
    )

    patched = _patch_mjcf_xml(Path(mjcf_path))
    asset_dir = str(Path(mjcf_path).parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in STD_BODY_NAMES]
    if any(bid < 0 for bid in body_ids):
        missing = [n for n, bid in zip(STD_BODY_NAMES, body_ids) if bid < 0]
        raise RuntimeError(f"MJCF missing standard bodies: {missing}")

    _G.update(
        mujoco=mujoco,
        model=model,
        data=data,
        body_ids=np.asarray(body_ids, dtype=int),
        tgt_fps=int(tgt_fps),
        GMR=GMR,
        get_smplx_data_offline_fast=get_smplx_data_offline_fast,
        clamp_joint_limits=clamp_joint_limits,
        compute_ground_offset=compute_ground_offset,
        body_model_cache={},
        smplx_folder=str(GMR_ROOT / "assets" / "body_models"),
        posture_cost=float(posture_cost),
    )


def _get_body_model(gender: str):
    import smplx
    cache = _G["body_model_cache"]
    g = str(gender)
    if g not in cache:
        cache[g] = smplx.create(_G["smplx_folder"], "smplx", gender=g, use_pca=False)
    return cache[g]


def _smplx_fk(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray, gender: str):
    """Run SMPL-X FK; return (smplx_output, body_model, human_height)."""
    import torch
    from general_motion_retargeting.utils.smpl import estimate_human_height_from_joints

    body_model = _get_body_model(gender)
    T = poses.shape[0]
    num_betas = getattr(body_model, "num_betas", 10)
    b = betas.reshape(-1).astype(np.float32)
    if b.shape[0] > num_betas:
        b = b[:num_betas]
    elif b.shape[0] < num_betas:
        b = np.concatenate([b, np.zeros(num_betas - b.shape[0], np.float32)])
    betas_t = torch.tensor(b).float().view(1, -1).expand(T, -1)
    with torch.no_grad():
        out = body_model(
            betas=betas_t,
            global_orient=torch.tensor(poses[:, :3]).float(),
            body_pose=torch.tensor(poses[:, 3:66]).float(),
            transl=torch.tensor(trans).float(),
            left_hand_pose=torch.zeros(T, 45).float(),
            right_hand_pose=torch.zeros(T, 45).float(),
            jaw_pose=torch.zeros(T, 3).float(),
            leye_pose=torch.zeros(T, 3).float(),
            reye_pose=torch.zeros(T, 3).float(),
            expression=torch.zeros(T, 10).float(),
            return_full_pose=True,
        )
    joints = out.joints.detach().numpy()
    s, e = T // 4, 3 * T // 4
    height, _ = estimate_human_height_from_joints(
        joints, frame_indices=slice(s, e), head_joint_idx=15, foot_joint_indices=(10, 11)
    )
    height = float(max(1.4, min(2.2, height)))
    return out, body_model, height


def _fk_amp_fields(qpos_zup: np.ndarray, fps: int):
    """MuJoCo FK -> (body_pos[T,30,3], body_rot_xyzw[T,30,4]) ground-aligned."""
    mujoco = _G["mujoco"]
    model, data, body_ids = _G["model"], _G["data"], _G["body_ids"]
    T = qpos_zup.shape[0]
    nb = len(body_ids)

    # global ground align: lowest std-body z over all frames -> ~0.
    min_z = np.inf
    for t in range(T):
        data.qpos[:36] = qpos_zup[t]
        mujoco.mj_forward(model, data)
        min_z = min(min_z, float(data.xpos[body_ids, 2].min()))
    z_off = (-min_z) if np.isfinite(min_z) else 0.0

    body_pos = np.zeros((T, nb, 3), np.float32)
    body_rot = np.zeros((T, nb, 4), np.float32)  # xyzw
    q = qpos_zup.copy()
    q[:, 2] += z_off
    for t in range(T):
        data.qpos[:36] = q[t]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[body_ids]
        xq = data.xquat[body_ids]              # wxyz
        body_rot[t] = xq[:, [1, 2, 3, 0]]      # -> xyzw
    return body_pos, body_rot


def _quat_angular_velocity(rot_xyzw: np.ndarray, fps: int) -> np.ndarray:
    """Angular velocity (T,nb,3) from xyzw quaternion sequence via finite diff."""
    from scipy.spatial.transform import Rotation as R
    T, nb = rot_xyzw.shape[0], rot_xyzw.shape[1]
    av = np.zeros((T, nb, 3), np.float32)
    if T < 2:
        return av
    for b in range(nb):
        r = R.from_quat(rot_xyzw[:, b, :])
        rel = r[:-1].inv() * r[1:]
        rotvec = rel.as_rotvec() * fps  # body frame
        # rotate into world frame
        world = r[:-1].apply(rotvec)
        av[:-1, b, :] = world
        av[-1, b, :] = av[-2, b, :] if T >= 2 else 0.0
    return av


def retarget_one(rel: str, data_dir: Path, out_dir: Path, overwrite: bool):
    out_path = out_dir / (Path(rel).with_suffix("").as_posix() + ".npz")
    if out_path.is_file() and not overwrite:
        return "skip", rel, None
    npz = data_dir / rel
    if not npz.is_file():
        return "missing", rel, None
    try:
        d = np.load(npz, allow_pickle=True)
        poses = d["poses"].astype(np.float32)
        trans = d["trans"].astype(np.float32)
        betas = d["betas"].astype(np.float32) if "betas" in d.files else np.zeros((1, 16), np.float32)
        gender = str(d["gender"]) if "gender" in d.files else "neutral"
        src_fps = int(np.asarray(d.get("mocap_framerate", 30)).reshape(-1)[0])
        if poses.shape[0] < 2:
            return "tooshort", rel, None

        tgt_fps = _G["tgt_fps"]
        out, body_model, height = _smplx_fk(poses, trans, betas, gender)
        smplx_data = {
            "pose_body": poses[:, 3:66],
            "root_orient": poses[:, :3],
            "trans": trans,
            "betas": betas.reshape(-1),
            "mocap_frame_rate": np.int64(src_fps),
        }
        frames, aligned_fps = _G["get_smplx_data_offline_fast"](
            smplx_data, body_model, out, tgt_fps=tgt_fps
        )

        # posture_cost>0 + posture_temporal(default): anchors redundant DOFs to the
        # previous frame's solution -> ~57% lower joint-acceleration jitter while the
        # pelvis/foot trajectory is preserved (validated sweep). This is an IK-level
        # temporal regularizer, NOT output smoothing.
        gmr = _G["GMR"](actual_human_height=height, src_human="smplx",
                        tgt_robot="unitree_g1", posture_cost=_G["posture_cost"])
        gmr.set_ground_offset(_G["compute_ground_offset"](gmr, frames))

        # offset_to_ground=False: GMR's per-frame offset_human_data_to_ground wrongly
        # treats Z (a horizontal axis in GMR's Y-up working frame) as the ground/up axis
        # and re-grounds it every frame, which COLLAPSES the Z translation (corr Z-Z
        # drops to ~0, path becomes a back-and-forth line). We instead rely on the global
        # set_ground_offset (apply_ground_offset) + the global min-z align in _fk_amp_fields
        # for vertical placement, which preserves the full horizontal trajectory.
        qpos_list = [gmr.retarget(f, offset_to_ground=False) for f in frames]
        if len(qpos_list) < 2:
            return "tooshort", rel, None
        qpos = np.asarray(qpos_list, dtype=np.float64)  # (T,36) wxyz root, Y-up + rot_offset

        # dof joint-limit clamp only (enforces valid G1 joint ranges; NOT smoothing).
        # No Savitzky-Golay / low-pass output filtering: jitter is handled inside the
        # IK via temporal posture regularization (posture_cost>0), not by post-hoc
        # smoothing of the solved trajectory.
        dof = qpos[:, 7:].copy()
        dof, _ = _G["clamp_joint_limits"](dof, soft=False)  # hard clip: only touches
        # out-of-limit frames (the tanh "soft" variant warps the whole range every frame).

        # undo GMR pelvis rot_offset -> Z-up
        from scipy.spatial.transform import Rotation as R
        rot_off = R.from_quat(_ROT_OFFSET_XYZW)
        root_pos = qpos[:, 0:3]
        root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]  # wxyz->xyzw
        # World-frame (Y-up -> Z-up) basis change: LEFT-multiply for rotation,
        # rotate the vector for translation. Using a right-multiply here is the
        # body-frame operation and tilts/flips the pelvis up-axis for any clip
        # whose facing changes (walk/jog/turn), corrupting ~45% of frames.
        root_pos_z = rot_off.inv().apply(root_pos)
        root_rot_z = (rot_off.inv() * R.from_quat(root_rot_xyzw)).as_quat()
        qpos_z = np.zeros_like(qpos)
        qpos_z[:, 0:3] = root_pos_z
        qpos_z[:, 3:7] = root_rot_z[:, [3, 0, 1, 2]]  # xyzw->wxyz for MuJoCo
        qpos_z[:, 7:] = dof

        fps = int(round(float(aligned_fps)))
        body_pos, body_rot = _fk_amp_fields(qpos_z, fps)

        dof_pos = dof.astype(np.float32)
        dof_vel = (np.gradient(dof_pos, axis=0) * fps).astype(np.float32)
        body_lin_vel = (np.gradient(body_pos, axis=0) * fps).astype(np.float32)
        body_ang_vel = _quat_angular_velocity(body_rot, fps).astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" when the path does not end in it, so write to
        # "<out>.tmp" (-> "<out>.tmp.npz") then atomically rename to <out>.
        tmp = str(out_path) + ".tmp"
        np.savez(
            tmp,
            fps=np.array([fps], np.float32),
            dof_names=np.array(DOF_NAMES),
            body_names=np.array(STD_BODY_NAMES),
            dof_positions=dof_pos,
            dof_velocities=dof_vel,
            body_positions=body_pos,
            body_rotations=body_rot,
            body_linear_velocities=body_lin_vel,
            body_angular_velocities=body_ang_vel,
        )
        os.replace(tmp + ".npz", out_path)
        return "ok", rel, int(dof_pos.shape[0])
    except Exception as exc:
        return "error", rel, f"{exc}\n{traceback.format_exc()[-800:]}"


def _mp_worker(rel, data_dir, out_dir, overwrite):
    return retarget_one(rel, Path(data_dir), Path(out_dir), overwrite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-list", default=str(DEFAULT_QUALITY))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--posture-cost", type=float, default=20.0,
                    help="IK temporal-consistency regularizer (posture target = previous "
                         "frame). ~57%% less joint-accel jitter, trajectory preserved. "
                         "0 disables (legacy/baseline). (default: 20.0)")
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=None, help="process at most N items (debug)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--names", nargs="*", default=None, help="explicit rel npz paths (debug)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.names:
        rels = list(args.names)
    else:
        ql = json.load(open(args.quality_list))
        items = ql["items"] if isinstance(ql, dict) and "items" in ql else ql
        rels = [it["path"] if isinstance(it, dict) else it for it in items]
        rels = rels[args.rank::args.world_size]
    if args.limit:
        rels = rels[: args.limit]
    print(f"[rank {args.rank}/{args.world_size}] {len(rels)} motions, workers={args.workers}", flush=True)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "tooshort": 0}
    err_log = out_dir / f"_errors_rank{args.rank}.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        _worker_init(args.mjcf, args.target_fps, args.posture_cost)
        it = (retarget_one(r, data_dir, out_dir, args.overwrite) for r in rels)
        for i, (status, rel, info) in enumerate(it):
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                with open(err_log, "a") as f:
                    f.write(f"{rel}\t{info}\n")
            if (i + 1) % 50 == 0 or i + 1 == len(rels):
                el = time.time() - t0
                print(f"  {i+1}/{len(rels)} ok={counts['ok']} skip={counts['skip']} "
                      f"err={counts['error']} miss={counts['missing']} "
                      f"({(i+1)/el:.2f} mot/s)", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=(args.mjcf, args.target_fps, args.posture_cost)) as pool:
            tasks = [(r, str(data_dir), str(out_dir), args.overwrite) for r in rels]
            for i, (status, rel, info) in enumerate(
                pool.starmap(_mp_worker, tasks, chunksize=4)
            ):
                counts[status] = counts.get(status, 0) + 1
                if status == "error":
                    with open(err_log, "a") as f:
                        f.write(f"{rel}\t{info}\n")
                if (i + 1) % 200 == 0 or i + 1 == len(rels):
                    el = time.time() - t0
                    print(f"  {i+1}/{len(rels)} ok={counts['ok']} skip={counts['skip']} "
                          f"err={counts['error']} miss={counts['missing']} "
                          f"({(i+1)/el:.2f} mot/s)", flush=True)

    el = time.time() - t0
    print(f"[rank {args.rank}] DONE {counts} in {el:.1f}s "
          f"({counts['ok']/max(el,1e-9):.2f} ok/s)", flush=True)


if __name__ == "__main__":
    main()

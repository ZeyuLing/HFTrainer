#!/usr/bin/env python3
"""Diagnose persistent oscillation in physics sim for worst-case motions.

Analyzes per-joint tracking error, oscillation frequency, and identifies
which joints are the worst offenders in v4_crouch_001 and v4_turn_004.
"""
import numpy as np
import mujoco
import json
import os
import sys
from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ_DIR = f"{CEPH}/output/embodied_t2m_v4/data/npz"
PHYS_DIR = f"{CEPH}/output/embodied_t2m_v4/data/smpl_mesh_physics"
KIN_DIR = f"{CEPH}/output/embodied_t2m_v4/data/smpl_mesh"

# Import from the main script
sys.path.insert(0, f"{CEPH}/scripts/embodied")
from run_smpl_physics_sim import (
    rot6d_to_rotmat, decode_motion_135, yup_to_zup,
    smpl_to_qpos, compute_ground_offset, load_mujoco_model,
    run_physics_sim, SMPL_2_MUJOCO, MUJOCO_2_SMPL,
    SMPL_JOINT_NAMES, _YUP_TO_ZUP
)

# MuJoCo joint names (body order, 3 hinge joints each)
def get_mujoco_joint_info(model):
    """Get per-joint info: name, body, limits."""
    joints = []
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"j{jid}"
        bid = model.jnt_bodyid[jid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"b{bid}"
        qi = model.jnt_qposadr[jid]
        limited = bool(model.jnt_limited[jid])
        lo, hi = model.jnt_range[jid] if limited else (None, None)
        joints.append({
            "jid": jid, "name": jname, "body": bname, "qi": qi,
            "limited": limited, "lo": lo, "hi": hi,
        })
    return joints


def compute_per_joint_metrics(ref_qpos, sim_qpos, fps=30):
    """Compute per-joint tracking error, jerk, acceleration."""
    T = min(ref_qpos.shape[0], sim_qpos.shape[0])
    dt = 1.0 / fps
    n_joints = 69  # body joints only (qpos[7:76])

    ref = ref_qpos[:T, 7:]  # (T, 69)
    sim = sim_qpos[:T, 7:]  # (T, 69)

    # Tracking error per joint
    tracking_err = np.abs(ref - sim)  # (T, 69)

    # Compute angular velocity, acceleration, jerk for sim
    vel_sim = np.diff(sim, axis=0) / dt                  # (T-1, 69)
    acc_sim = np.diff(vel_sim, axis=0) / dt               # (T-2, 69)
    jerk_sim = np.diff(acc_sim, axis=0) / dt              # (T-3, 69)

    # Same for reference
    vel_ref = np.diff(ref, axis=0) / dt
    acc_ref = np.diff(vel_ref, axis=0) / dt
    jerk_ref = np.diff(acc_ref, axis=0) / dt

    # Per-DOF metrics
    results = {}
    for i in range(n_joints):
        body_idx = i // 3
        axis_idx = i % 3
        axis_name = ['X', 'Y', 'Z'][axis_idx]

        results[i] = {
            "dof_index": i,
            "body_mj_idx": body_idx,
            "axis": axis_name,
            "mean_tracking_err_rad": float(np.mean(tracking_err[:, i])),
            "max_tracking_err_rad": float(np.max(tracking_err[:, i])),
            "mean_jerk_sim": float(np.mean(np.abs(jerk_sim[:, i]))) if jerk_sim.shape[0] > 0 else 0,
            "mean_jerk_ref": float(np.mean(np.abs(jerk_ref[:, i]))) if jerk_ref.shape[0] > 0 else 0,
            "mean_acc_sim": float(np.mean(np.abs(acc_sim[:, i]))) if acc_sim.shape[0] > 0 else 0,
            "mean_acc_ref": float(np.mean(np.abs(acc_ref[:, i]))) if acc_ref.shape[0] > 0 else 0,
        }
        jerk_ref_val = results[i]["mean_jerk_ref"]
        jerk_sim_val = results[i]["mean_jerk_sim"]
        results[i]["jerk_ratio"] = jerk_sim_val / max(jerk_ref_val, 1e-6)

    return results, tracking_err, vel_sim, acc_sim, jerk_sim


def analyze_oscillation(sim_qpos, fps=30):
    """Detect oscillation by looking at sign changes in velocity."""
    T = sim_qpos.shape[0]
    dt = 1.0 / fps
    sim = sim_qpos[:, 7:]  # (T, 69)
    vel = np.diff(sim, axis=0) / dt  # (T-1, 69)

    # Count sign changes per DOF (oscillation indicator)
    sign_changes = np.sum(np.diff(np.sign(vel), axis=0) != 0, axis=0)  # (69,)
    # Normalize by number of possible sign changes
    max_sign_changes = max(1, T - 2)
    oscillation_ratio = sign_changes / max_sign_changes  # 0=constant, 1=every-frame flip

    return oscillation_ratio, sign_changes


def analyze_motion(stem, verbose=True):
    """Full analysis for a single motion."""
    npz_path = f"{NPZ_DIR}/{stem}.npz"
    if not os.path.exists(npz_path):
        print(f"  NPZ not found: {npz_path}")
        return None

    # Load and convert motion
    smpl_pose_yup, transl_yup, fps = decode_motion_135(npz_path)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose_yup, transl_yup)

    model, data = load_mujoco_model(MJCF)
    body_pos_1 = model.body_pos[1].copy()

    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1, model=model)
    ground_offset = compute_ground_offset(model, data, ref_qpos)
    ref_qpos[:, 2] -= ground_offset

    # Run sim
    sim_qpos, stats = run_physics_sim(model, data, ref_qpos, fps)
    T_sim = sim_qpos.shape[0]

    # Per-joint analysis
    per_joint, tracking_err, vel_sim, acc_sim, jerk_sim = compute_per_joint_metrics(
        ref_qpos, sim_qpos, fps
    )

    # Oscillation analysis
    osc_ratio, sign_changes = analyze_oscillation(sim_qpos, fps)

    # Get MuJoCo joint info
    joint_info = get_mujoco_joint_info(model)
    # Skip the root free joint (first 7 dofs)
    hinge_joints = [j for j in joint_info if j["qi"] >= 7]

    if verbose:
        print(f"\n{'='*80}")
        print(f"  {stem}: T={ref_qpos.shape[0]}, T_sim={T_sim}, fps={fps}")
        print(f"  Tracking error: mean={stats['joint_tracking_error_rad']:.4f} rad")
        print(f"{'='*80}")

        # Sort DOFs by jerk ratio (worst first)
        sorted_dofs = sorted(per_joint.keys(), key=lambda k: per_joint[k]["jerk_ratio"], reverse=True)

        print(f"\n  TOP 15 WORST DOFs by jerk ratio (sim/ref):")
        print(f"  {'DOF':>4s}  {'Joint':25s}  {'Axis':>4s}  {'JerkRatio':>10s}  {'MeanErr':>10s}  "
              f"{'MaxErr':>10s}  {'OscRatio':>10s}  {'Limits':15s}")
        for rank, dof_i in enumerate(sorted_dofs[:15]):
            d = per_joint[dof_i]
            # Find the matching joint
            qi_abs = 7 + dof_i
            matching = [j for j in hinge_joints if j["qi"] == qi_abs]
            jname = matching[0]["name"] if matching else f"dof{dof_i}"
            jbody = matching[0]["body"] if matching else "?"
            limited = matching[0]["limited"] if matching else False
            lo = matching[0]["lo"] if matching else None
            hi = matching[0]["hi"] if matching else None
            limits_str = f"[{np.degrees(lo):.1f},{np.degrees(hi):.1f}]" if limited else "unlimited"

            print(f"  {dof_i:4d}  {jname:25s}  {d['axis']:>4s}  "
                  f"{d['jerk_ratio']:10.2f}  {d['mean_tracking_err_rad']:10.4f}  "
                  f"{d['max_tracking_err_rad']:10.4f}  {osc_ratio[dof_i]:10.4f}  {limits_str}")

        # Also show DOFs with highest oscillation ratio
        sorted_by_osc = np.argsort(osc_ratio)[::-1]
        print(f"\n  TOP 10 MOST OSCILLATING DOFs (sign-change ratio):")
        print(f"  {'DOF':>4s}  {'Joint':25s}  {'OscRatio':>10s}  {'JerkRatio':>10s}  {'MeanErr':>10s}")
        for dof_i in sorted_by_osc[:10]:
            d = per_joint[dof_i]
            qi_abs = 7 + dof_i
            matching = [j for j in hinge_joints if j["qi"] == qi_abs]
            jname = matching[0]["name"] if matching else f"dof{dof_i}"
            print(f"  {dof_i:4d}  {jname:25s}  {osc_ratio[dof_i]:10.4f}  "
                  f"{d['jerk_ratio']:10.2f}  {d['mean_tracking_err_rad']:10.4f}")

        # Analyze reference qpos: are there large jumps in ref after clamping?
        ref_body = ref_qpos[:T_sim, 7:]
        ref_vel = np.diff(ref_body, axis=0) / (1.0/fps)
        print(f"\n  REFERENCE qpos analysis (after clamping):")
        for dof_i in sorted_dofs[:10]:
            qi_abs = 7 + dof_i
            matching = [j for j in hinge_joints if j["qi"] == qi_abs]
            jname = matching[0]["name"] if matching else f"dof{dof_i}"
            limited = matching[0]["limited"] if matching else False
            lo = matching[0]["lo"] if matching else -999
            hi = matching[0]["hi"] if matching else 999

            vals = ref_body[:, dof_i]
            ref_v = ref_vel[:, dof_i]
            max_vel = float(np.max(np.abs(ref_v)))
            mean_vel = float(np.mean(np.abs(ref_v)))
            # Count frames at limits
            at_lo = np.sum(np.abs(vals - lo) < 1e-6)
            at_hi = np.sum(np.abs(vals - hi) < 1e-6)

            print(f"    {jname:25s}: range=[{np.degrees(vals.min()):.1f},{np.degrees(vals.max()):.1f}]° "
                  f"mean_vel={np.degrees(mean_vel):.1f}°/s max_vel={np.degrees(max_vel):.1f}°/s "
                  f"at_lo={at_lo} at_hi={at_hi}")

    return {
        "stem": stem,
        "per_joint": per_joint,
        "osc_ratio": osc_ratio,
        "tracking_err": tracking_err,
        "stats": stats,
    }


if __name__ == "__main__":
    motions_to_analyze = [
        "v4_crouch_001",
        "v4_turn_004",
        "v4_balance_003",  # 2.56x
        "v4_walk_001",     # ~1.0x (good case for comparison)
    ]

    for stem in motions_to_analyze:
        print(f"\n{'#'*80}")
        print(f"# Analyzing: {stem}")
        print(f"{'#'*80}")
        result = analyze_motion(stem)

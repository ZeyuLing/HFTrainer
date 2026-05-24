#!/usr/bin/env python3
"""Quick test: compare NO transform vs CONJUGATION (vector rotation) approach.

This directly tests the fixed yup_to_zup() which uses aa_zup = Rx @ aa_yup.
"""
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.embodied.run_smpl_physics_sim import (
    MUJOCO_BODY_NAMES, decode_motion_135, yup_to_zup, smpl_to_qpos,
)


def load_model(xml_path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def set_qpos_and_forward(model, data, qpos):
    import mujoco
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.xpos.copy()


def print_body_pos(xpos, label=""):
    print(f"\n  --- {label} ---")
    for i, name in enumerate(MUJOCO_BODY_NAMES):
        x, y, z = xpos[i + 1]
        print(f"    [{i:2d}] {name:15s}  x={x:8.4f}  y={y:8.4f}  z={z:8.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=str, required=True)
    parser.add_argument("--npz-file", type=str, required=True)
    args = parser.parse_args()

    model, data = load_model(args.xml_path)
    body_pos_1 = model.body_pos[1].copy()

    smpl_pose, transl, fps = decode_motion_135(args.npz_file)
    T = smpl_pose.shape[0]
    print(f"Decoded: {T} frames @ {fps}fps")
    print(f"Frame 0 root aa (Y-up):  {smpl_pose[0, :3]}, mag={np.degrees(np.linalg.norm(smpl_pose[0, :3])):.1f}°")
    print(f"Frame 0 transl (Y-up):   {transl[0]}")

    # -----------------------------------------------
    # Approach A: NO coordinate transform (like PHC)
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("APPROACH A: No coordinate transform (Y-up data → MuJoCo directly)")
    print("=" * 70)
    qpos_a = smpl_to_qpos(smpl_pose[:1], transl[:1], body_pos_1)
    print(f"qpos root trans: {qpos_a[0, :3]}")
    print(f"qpos root quat:  {qpos_a[0, 3:7]}")
    xpos_a = set_qpos_and_forward(model, data, qpos_a[0])
    pelvis_z_a = xpos_a[1, 2]
    print(f"Pelvis world pos: {xpos_a[1]}")
    issues_a = sum(1 for i, n in enumerate(MUJOCO_BODY_NAMES)
                   if n in ["R_Knee", "R_Ankle", "R_Toe", "L_Knee", "L_Ankle", "L_Toe"]
                   and xpos_a[i + 1, 2] > pelvis_z_a)
    print(f"Leg joints above pelvis (Z): {issues_a}")
    print_body_pos(xpos_a, "Approach A: No transform")

    # -----------------------------------------------
    # Approach B: Conjugation / vector rotation (FIXED yup_to_zup)
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("APPROACH B: Conjugation (aa_zup = Rx @ aa_yup) — FIXED yup_to_zup")
    print("=" * 70)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose[:1], transl[:1])
    print(f"Frame 0 root aa (Z-up):  {smpl_pose_zup[0, :3]}, mag={np.degrees(np.linalg.norm(smpl_pose_zup[0, :3])):.1f}°")
    print(f"Frame 0 transl (Z-up):   {transl_zup[0]}")

    qpos_b = smpl_to_qpos(smpl_pose_zup[:1], transl_zup[:1], body_pos_1)
    print(f"qpos root trans: {qpos_b[0, :3]}")
    print(f"qpos root quat:  {qpos_b[0, 3:7]}")
    xpos_b = set_qpos_and_forward(model, data, qpos_b[0])
    pelvis_z_b = xpos_b[1, 2]
    print(f"Pelvis world pos: {xpos_b[1]}")
    issues_b = sum(1 for i, n in enumerate(MUJOCO_BODY_NAMES)
                   if n in ["R_Knee", "R_Ankle", "R_Toe", "L_Knee", "L_Ankle", "L_Toe"]
                   and xpos_b[i + 1, 2] > pelvis_z_b)
    print(f"Leg joints above pelvis (Z): {issues_b}")
    print_body_pos(xpos_b, "Approach B: Conjugation")

    # -----------------------------------------------
    # Approach C: Only transform translation (not rotation)
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("APPROACH C: Only transform translation, NOT rotation")
    print("=" * 70)
    _YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    transl_c = (transl[:1].astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
    smpl_pose_c = smpl_pose[:1].copy()  # rotations unchanged

    qpos_c = smpl_to_qpos(smpl_pose_c, transl_c, body_pos_1)
    print(f"qpos root trans: {qpos_c[0, :3]}")
    print(f"qpos root quat:  {qpos_c[0, 3:7]}")
    xpos_c = set_qpos_and_forward(model, data, qpos_c[0])
    pelvis_z_c = xpos_c[1, 2]
    print(f"Pelvis world pos: {xpos_c[1]}")
    issues_c = sum(1 for i, n in enumerate(MUJOCO_BODY_NAMES)
                   if n in ["R_Knee", "R_Ankle", "R_Toe", "L_Knee", "L_Ankle", "L_Toe"]
                   and xpos_c[i + 1, 2] > pelvis_z_c)
    print(f"Leg joints above pelvis (Z): {issues_c}")
    print_body_pos(xpos_c, "Approach C: Transl only")

    # -----------------------------------------------
    # Approach D: T-pose sanity — verify identity root maps correctly
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("APPROACH D: T-pose verification")
    print("=" * 70)
    tpose = np.zeros((1, 72), dtype=np.float32)
    tpose_t = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # Y-up: height in Y

    # D1: No transform — identity root, Y-up translation
    qpos_d1 = smpl_to_qpos(tpose, tpose_t, body_pos_1)
    xpos_d1 = set_qpos_and_forward(model, data, qpos_d1[0])
    print(f"D1 (no transform, Y-up trans): pelvis={xpos_d1[1]}")

    # D2: yup_to_zup — identity root should stay identity, translation should swap
    tpose_zup, tpose_t_zup = yup_to_zup(tpose, tpose_t)
    print(f"D2 after yup_to_zup: root_aa={tpose_zup[0, :3]}, transl={tpose_t_zup[0]}")
    qpos_d2 = smpl_to_qpos(tpose_zup, tpose_t_zup, body_pos_1)
    xpos_d2 = set_qpos_and_forward(model, data, qpos_d2[0])
    print(f"D2 (conjugation, Z-up trans): pelvis={xpos_d2[1]}")
    issues_d2 = sum(1 for i, n in enumerate(MUJOCO_BODY_NAMES)
                    if n in ["R_Knee", "R_Ankle", "R_Toe", "L_Knee", "L_Ankle", "L_Toe"]
                    and xpos_d2[i + 1, 2] > xpos_d2[1, 2])
    print(f"D2 leg joints above pelvis (Z): {issues_d2}")
    print_body_pos(xpos_d2, "D2: T-pose + conjugation yup_to_zup")

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Approach A (no transform):     {issues_a} leg joints above pelvis")
    print(f"Approach B (conjugation):       {issues_b} leg joints above pelvis")
    print(f"Approach C (transl only):       {issues_c} leg joints above pelvis")
    print(f"T-pose D2 (conjugation):       {issues_d2} leg joints above pelvis")


if __name__ == "__main__":
    main()

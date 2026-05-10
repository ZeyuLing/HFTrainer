#!/usr/bin/env python3
"""Normalize all NPZ files in npz_split/ in-place:
  1. First frame faces Z+
  2. First frame centered at XZ origin
  3. Floor at y=0 (via FK foot joints)

For raw+cleaned pairs, uses RAW to compute normalization params,
then applies the same transform to both.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

NPZ_DIR = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/lightai_data/CJGame_MB/npz_split")

# ─── SMPLH FK ─────────────────────────────────────────────────────────────────
SMPLH_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
    16, 17, 18, 19,
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
]
FOOT_JOINTS = [7, 8, 10, 11]

SCRIPT_DIR = Path(__file__).resolve().parent
J_TEMPLATE_PATH = SCRIPT_DIR / "motion_annot_web/npz_compare/static/assets/dump_smplh/j_template.bin"
# Fallback
if not J_TEMPLATE_PATH.is_file():
    J_TEMPLATE_PATH = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/motion_annot_web/m2m_database/static/assets/dump_smplh/j_template.bin")

_j_template = None

def get_j_template():
    global _j_template
    if _j_template is None:
        data = np.frombuffer(J_TEMPLATE_PATH.read_bytes(), dtype=np.float32)
        _j_template = data.reshape(-1, 3).copy()
    return _j_template


def axis_angle_to_matrix(rv):
    angle = np.linalg.norm(rv)
    if angle < 1e-8:
        return np.eye(3)
    return Rotation.from_rotvec(rv).as_matrix()


def fk_joint_positions(poses, trans, j_template):
    n = j_template.shape[0]
    world_pos = np.zeros((n, 3))
    world_rot = np.zeros((n, 3, 3))
    root_rot = axis_angle_to_matrix(poses[:3])
    world_rot[0] = root_rot
    world_pos[0] = trans + root_rot @ j_template[0]
    for j in range(1, min(n, len(SMPLH_PARENTS))):
        p = SMPLH_PARENTS[j]
        local_rot = axis_angle_to_matrix(poses[j*3:j*3+3])
        world_rot[j] = world_rot[p] @ local_rot
        offset = j_template[j] - j_template[p]
        world_pos[j] = world_pos[p] + world_rot[p] @ offset
    return world_pos


def compute_floor_y(poses, trans):
    """Global min foot Y across all frames."""
    jt = get_j_template()
    min_y = float('inf')
    for i in range(len(poses)):
        joints = fk_joint_positions(poses[i], trans[i], jt)
        for fj in FOOT_JOINTS:
            if fj < len(joints):
                min_y = min(min_y, joints[fj, 1])
    return min_y if min_y != float('inf') else 0.0


def compute_facing_yaw(poses_frame0):
    root_rot = axis_angle_to_matrix(poses_frame0[:3])
    fwd = root_rot @ np.array([0, 0, 1.0])
    return math.atan2(fwd[0], fwd[2])


def normalize_inplace(poses, trans):
    """Compute yaw, xz_offset, floor_y from this motion. Apply in-place. Return params."""
    yaw = compute_facing_yaw(poses[0])

    # 1. Yaw rotation
    if abs(yaw) > 1e-6:
        R_yaw = Rotation.from_euler('y', -yaw).as_matrix()
        for i in range(len(poses)):
            root_R = axis_angle_to_matrix(poses[i, :3])
            poses[i, :3] = Rotation.from_matrix(R_yaw @ root_R).as_rotvec()
        trans[:] = (R_yaw @ trans.T).T

    # 2. Center XZ
    xz_offset = trans[0, [0, 2]].copy()
    trans[:, 0] -= xz_offset[0]
    trans[:, 2] -= xz_offset[1]

    # 3. Floor
    floor_y = compute_floor_y(poses, trans)
    trans[:, 1] -= floor_y

    return yaw, xz_offset, floor_y


def apply_normalization(poses, trans, yaw, xz_offset, floor_y):
    """Apply pre-computed normalization params (for cleaned pair)."""
    if abs(yaw) > 1e-6:
        R_yaw = Rotation.from_euler('y', -yaw).as_matrix()
        for i in range(len(poses)):
            root_R = axis_angle_to_matrix(poses[i, :3])
            poses[i, :3] = Rotation.from_matrix(R_yaw @ root_R).as_rotvec()
        trans[:] = (R_yaw @ trans.T).T
    trans[:, 0] -= xz_offset[0]
    trans[:, 2] -= xz_offset[1]
    trans[:, 1] -= floor_y


def process_file(raw_path, clean_path=None):
    """Normalize raw (and optionally cleaned) NPZ in-place."""
    # Load raw
    raw_data = dict(np.load(raw_path, allow_pickle=True))
    poses = raw_data['poses'].astype(np.float64)
    if poses.ndim == 3:
        poses = poses.reshape(poses.shape[0], -1)
    trans = raw_data.get('trans', np.zeros((poses.shape[0], 3))).astype(np.float64)

    # Compute & apply normalization from raw
    yaw, xz_offset, floor_y = normalize_inplace(poses, trans)

    # Save raw
    raw_data['poses'] = poses.astype(np.float32)
    raw_data['trans'] = trans.astype(np.float32)
    np.savez(raw_path, **raw_data)

    # Process cleaned with same params
    if clean_path and os.path.isfile(clean_path):
        clean_data = dict(np.load(clean_path, allow_pickle=True))
        c_poses = clean_data['poses'].astype(np.float64)
        if c_poses.ndim == 3:
            c_poses = c_poses.reshape(c_poses.shape[0], -1)
        c_trans = clean_data.get('trans', np.zeros((c_poses.shape[0], 3))).astype(np.float64)

        apply_normalization(c_poses, c_trans, yaw, xz_offset, floor_y)

        clean_data['poses'] = c_poses.astype(np.float32)
        clean_data['trans'] = c_trans.astype(np.float32)
        np.savez(clean_path, **clean_data)


def main():
    import json
    report_path = NPZ_DIR.parent / "quality_report_2_post_slice.json"
    with open(report_path) as f:
        segments = json.load(f)

    total = len(segments)
    done = 0
    errors = 0

    for seg in segments:
        raw_name = seg['raw_out']
        clean_name = seg.get('clean_out')
        raw_path = str(NPZ_DIR / raw_name)
        clean_path = str(NPZ_DIR / clean_name) if clean_name else None

        if not os.path.isfile(raw_path):
            print(f'[SKIP] {raw_name} not found')
            errors += 1
            continue

        try:
            process_file(raw_path, clean_path)
            done += 1
            if done % 100 == 0:
                print(f'  [{done}/{total}] processed...')
        except Exception as e:
            print(f'[ERROR] {raw_name}: {e}')
            errors += 1

    print(f'\nDone! {done}/{total} processed, {errors} errors.')


if __name__ == '__main__':
    main()

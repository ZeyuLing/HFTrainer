"""Diagnose StableMotion encode/decode round-trip on a CLEAN motion.

If the round-trip on a clean (uncorrupted) motion is significantly lossy
(e.g., trajectory misaligned, rotations differ visibly), then the bug lies
in the M2M↔smpldata + axis-swap + canonicalization pipeline rather than
in the diffusion detect/fix pass.

The detect/fix pass is BYPASSED here — we just go through:

    motion_135 → smpldata_24 → y-up→z-up → encode (feats_232) →
    decode (smpldata) → decanon → z-up→y-up → motion_135'

and compare motion_135' against motion_135.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SM_ROOT = PROJECT_ROOT / 'ref_repo' / 'StableMotion'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SM_ROOT))

# Reuse the production helpers from run_stablemotion_e9.py
from scripts.run_stablemotion_e9 import (  # noqa: E402
    m2m135_to_smpldata_24, smpldata_to_m2m135,
    smpldata_y_up_to_z_up, smpldata_z_up_to_y_up,
)
from data_loaders.amasstools.globsmplrifke_feats import (  # noqa: E402
    smpldata_to_alignglobsmplrifkefeats, globsmplrifkefeats_to_smpldata,
)
from data_loaders.amasstools.geometry import (  # noqa: E402
    axis_angle_to_matrix as _aa2mat,
    matrix_to_euler_angles as _mat2euler,
)
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (  # noqa: E402
    process_transl, process_smplx_pose,
)
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    matrix_to_axis_angle as _mat2aa,
)


def diagnose(motion_path: str, bone_offsets: torch.Tensor) -> None:
    print(f'\n=== Round-trip on {motion_path} ===')

    d = np.load(motion_path, allow_pickle=True)
    tk = 'trans' if 'trans' in d.files else 'transl'
    pk = 'poses' if 'poses' in d.files else 'body_pose'
    raw_trans = d[tk].astype(np.float32)
    raw_poses = d[pk].astype(np.float32)
    motion_135 = np.concatenate([
        process_transl(raw_trans, 'abs'),
        process_smplx_pose(raw_poses, 'rotation_6d', 'smpl_22'),
    ], axis=-1).astype(np.float32)
    T = motion_135.shape[0]
    print(f'  T={T}, motion_135 shape={motion_135.shape}')

    # ─── stage 1: m2m → smpldata_24 → smpldata_24 → m2m (pure identity) ───
    smpldata = m2m135_to_smpldata_24(motion_135, bone_offsets)
    motion_135_a = smpldata_to_m2m135(smpldata)
    err_a_trans = float(np.abs(motion_135_a[:, :3] - motion_135[:, :3]).max())
    err_a_rot = float(np.abs(motion_135_a[:, 3:] - motion_135[:, 3:]).max())
    print(f'  [stage 1] m2m→smpldata→m2m  trans_max_err={err_a_trans:.6f}  '
          f'rot_max_err={err_a_rot:.6f}')

    # ─── stage 2: + y-up ↔ z-up swap ───
    smpldata_z = smpldata_y_up_to_z_up(smpldata)
    smpldata_y = smpldata_z_up_to_y_up(smpldata_z)
    motion_135_b = smpldata_to_m2m135(smpldata_y)
    err_b_trans = float(np.abs(motion_135_b[:, :3] - motion_135[:, :3]).max())
    err_b_rot = float(np.abs(motion_135_b[:, 3:] - motion_135[:, 3:]).max())
    print(f'  [stage 2] +yz swap         trans_max_err={err_b_trans:.6f}  '
          f'rot_max_err={err_b_rot:.6f}')

    # ─── stage 3: + canonicalization (encode + inverse) ───
    # Compute the canonicalization parameters that the encoder applies
    # internally so we can invert them.
    j0 = smpldata_z['joints'].clone()
    ground_shift_z = float(j0[..., 2].min())
    traj0_xy = j0[0, 0, :2].clone()
    poses_z = smpldata_z['poses'].reshape(-1, 22, 3)
    R0 = _aa2mat(poses_z[0, 0].unsqueeze(0))[0]
    euler0 = _mat2euler(R0.unsqueeze(0), "ZYX")[0]
    rotZ0_angle = euler0[0].item()

    bo_np = bone_offsets.numpy() if hasattr(bone_offsets, 'numpy') else np.asarray(bone_offsets)
    pelvis_offset_y_smpl = float(bo_np[0, 1])
    trans_gravity_correction = -pelvis_offset_y_smpl
    trans_xy_correction = (-float(bo_np[0, 0]), -float(bo_np[0, 2]))

    # Encode and decode (no diffusion, pure feats round-trip)
    feats = smpldata_to_alignglobsmplrifkefeats(smpldata_z)  # (T, 232)
    smpldata_canon = globsmplrifkefeats_to_smpldata(feats)
    print(f'  [encode] feats={tuple(feats.shape)}  '
          f'feats_canon_traj0={smpldata_canon["trans"][0].numpy()}')

    cos = np.cos(rotZ0_angle); sin = np.sin(rotZ0_angle)
    R_z_inv = torch.tensor([
        [cos, -sin, 0.0],
        [sin,  cos, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=smpldata_canon['joints'].dtype)

    smpldata_dec_z = {}
    j = smpldata_canon['joints'].clone()
    j = j @ R_z_inv.T
    j[..., 0] += traj0_xy[0]; j[..., 1] += traj0_xy[1]; j[..., 2] += ground_shift_z
    smpldata_dec_z['joints'] = j

    t = smpldata_canon['trans'].clone()
    t = t @ R_z_inv.T
    t[..., 0] += traj0_xy[0] + trans_xy_correction[0]
    t[..., 1] += traj0_xy[1] + trans_xy_correction[1]
    t[..., 2] += ground_shift_z + trans_gravity_correction
    smpldata_dec_z['trans'] = t

    poses_dec = smpldata_canon['poses'].reshape(-1, 22, 3)
    go_aa = poses_dec[:, 0]
    go_mat = _aa2mat(go_aa)
    go_mat = R_z_inv.to(go_mat.dtype) @ go_mat
    go_aa_new = _mat2aa(go_mat)
    poses_dec[:, 0] = go_aa_new
    smpldata_dec_z['poses'] = poses_dec.reshape(-1, 66)

    smpldata_dec_y = smpldata_z_up_to_y_up(smpldata_dec_z)
    motion_135_c = smpldata_to_m2m135(smpldata_dec_y)
    err_c_trans = float(np.abs(motion_135_c[:, :3] - motion_135[:, :3]).max())
    err_c_rot = float(np.abs(motion_135_c[:, 3:] - motion_135[:, 3:]).max())
    print(f'  [stage 3] +full canon RTT  trans_max_err={err_c_trans:.6f}  '
          f'rot_max_err={err_c_rot:.6f}')

    # Per-axis trans error
    err_xyz = np.abs(motion_135_c[:, :3] - motion_135[:, :3]).max(axis=0)
    print(f'           trans per-axis max err: x={err_xyz[0]:.4f}  '
          f'y={err_xyz[1]:.4f}  z={err_xyz[2]:.4f}')
    # Verify frame-0 alignment
    f0_diff = motion_135_c[0, :3] - motion_135[0, :3]
    print(f'           frame-0 trans diff: {f0_diff}')
    # Mean traj diff
    diff_traj = motion_135_c[:, :3] - motion_135[:, :3]
    print(f'           mean trans diff: {diff_traj.mean(axis=0)}')


def main() -> None:
    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()
    print(f'bone_offsets shape: {tuple(bone_offsets.shape)}')
    print(f'bone_offsets[0]: {bone_offsets[0].numpy()}')

    import json
    with open(PROJECT_ROOT / 'data/eval/m2m_v2/eval_e9_repair_v2.json') as f:
        dl = json.load(f)
    items = dl.get('data_list', dl)
    for i in (0, 1, 2):
        mp = items[i]['motion_path']
        if not str(mp).startswith('/'):
            mp = str(PROJECT_ROOT / mp)
        diagnose(mp, bone_offsets)


if __name__ == '__main__':
    main()

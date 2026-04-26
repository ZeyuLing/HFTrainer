"""StableMotion E9 Baseline Inference Wrapper (2026-04-23).

End-to-end detect-and-fix pipeline for the E9 Motion Repair benchmark:

    M2M 135-dim LQ motion
      → SMPL-24 smpldata (poses 66 + trans 3 + joints 24×3)
      → y-up → z-up axis swap (StableMotion uses z as gravity)
      → Global SMPL RIFKE feats (232-dim)
      → normalize + append label=0 channel (233-dim)
      → StableMotion detect pass (predict per-frame corruption label)
      → StableMotion fix pass (inpaint corrupted frames)
      → denormalize → drop label → 232-dim body feats
      → globsmplrifkefeats_to_smpldata (23 joints minus pelvis reconstruction)
      → z-up → y-up → 135-dim M2M motion
      → NPZ output for dashboard ingestion

Notes:
- StableMotion's `smpldata_to_alignglobsmplrifkefeats` expects 24 input
  joints (SMPL+H "smpljoints" extractor: body 22 + left_hand + right_hand).
  M2M only has SMPL body 22 joints. We synthesize joints 22 (left_hand)
  and 23 (right_hand) by duplicating the wrists (joints 20, 21). The
  feature pipeline mostly uses joints_local for lower body and pelvis
  geometry; hand joints only enter via joints_local and are not used
  for foot_global, rotZ, or trajectory. Duplicating wrists therefore
  introduces a small bias in the hand-joint RIFKE channels but does
  not break the canonical frame or root trajectory.
- StableMotion training used AMASS canonicalized with specific noise
  kinds. Our E9 LQ motions are HyMotion-domain — a known domain gap,
  but the StableMotion baseline is meant to serve as a reference
  ceiling/floor, not as a drop-in solution.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_stablemotion_e9.py \
        --eval-datalist data/eval/m2m_v2/eval_e9_repair.json \
        --output-dir output/eval_v2_e9_stablemotion_20260423 \
        --max-samples 9999
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SM_ROOT = PROJECT_ROOT / 'ref_repo' / 'StableMotion'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SM_ROOT))


# ───────────────────────── StableMotion imports ─────────────────────────
from data_loaders.amasstools.globsmplrifke_feats import (  # noqa: E402
    smpldata_to_alignglobsmplrifkefeats, globsmplrifkefeats_to_smpldata,
)
from diffusion import gaussian_diffusion as gd  # noqa: E402
from diffusion.respace import SpacedDiffusion, space_timesteps  # noqa: E402
from model.stablemotion import StableMotionDiTModel  # noqa: E402
from utils.normalizer import Normalizer  # noqa: E402


# ───────────────────────── M2M imports ─────────────────────────
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (  # noqa: E402
    process_transl, process_smplx_pose,
)
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk  # noqa: E402
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


# ───────────────────────── Rot6d convention helper ─────────────────────────
def _rot6d_row_to_col(rot6d_row: torch.Tensor) -> torch.Tensor:
    """M2M 135-dim uses row-major rot6d ([R00,R01,R10,R11,R20,R21]); the
    project-local rotation_convert.py is column-major
    ([R00,R10,R20,R01,R11,R21]). Swap order before feeding into
    rotation_6d_to_matrix."""
    return rot6d_row[..., [0, 2, 4, 1, 3, 5]]


def _rot6d_col_to_row(rot6d_col: torch.Tensor) -> torch.Tensor:
    """Inverse of _rot6d_row_to_col."""
    return rot6d_col[..., [0, 3, 1, 4, 2, 5]]


# ───────────────────────── M2M ↔ smpldata conversion ─────────────────────────
def m2m135_to_smpldata_24(
    motion_135: np.ndarray,
    bone_offsets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convert M2M 135-dim motion to smpldata for StableMotion's 24-joint
    topology. Returns {poses: (T,66), trans: (T,3), joints: (T,24,3)}
    where joints[:, 22] = joints[:, 20] (l_wrist copy for l_hand) and
    joints[:, 23] = joints[:, 21] (r_wrist copy for r_hand).
    """
    T = motion_135.shape[0]
    motion_t = torch.from_numpy(motion_135).float()
    trans = motion_t[:, :3].clone()
    rot6d_row = motion_t[:, 3:].reshape(T, 22, 6)
    rot6d_col = _rot6d_row_to_col(rot6d_row)
    R = rotation_6d_to_matrix(rot6d_col)
    poses_aa = matrix_to_axis_angle(R).reshape(T, 66)
    with torch.no_grad():
        joints_world, _, _, _ = motion135_to_fk(
            motion_t.unsqueeze(0), bone_offsets, 'local'
        )
    joints_22 = joints_world.squeeze(0)  # (T, 22, 3)
    # Synthesize 2 hand joints by copying the wrists
    joints_24 = torch.cat([joints_22, joints_22[:, 20:21], joints_22[:, 21:22]], dim=1)
    return {'poses': poses_aa, 'trans': trans, 'joints': joints_24}


def smpldata_to_m2m135(
    smpldata: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Convert smpldata back to M2M 135-dim. Only uses poses + trans
    (the first 22 joints' rotations); hand-joint positions are discarded."""
    poses = smpldata['poses']
    trans = smpldata['trans']
    T = poses.shape[0]
    poses_aa = poses.reshape(T, -1, 3)[:, :22]  # drop hand rotations if any
    R = axis_angle_to_matrix(poses_aa)
    rot6d_col = matrix_to_rotation_6d(R)
    rot6d_row = _rot6d_col_to_row(rot6d_col)
    motion_135 = torch.cat([trans[:, :3], rot6d_row.reshape(T, 132)], dim=-1)
    return motion_135.numpy().astype(np.float32)


# ───────────────────────── Axis swap y-up ↔ z-up ─────────────────────────
# StableMotion's globsmplrifke_feats assumes joints[:, :, 2] is the
# gravity axis (z-up). M2M uses y-up (joints[:, :, 1] = height). We swap
# y ↔ z on both joints and trans, and pre-rotate the global orient by
# +90° around the world x axis so FK stays consistent.
_R_X90 = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
_R_X_NEG90 = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


def _swap_yz(tensor_xyz: torch.Tensor) -> torch.Tensor:
    """Swap y and z components of the last axis. Last dim must be 3."""
    out = tensor_xyz.clone()
    out[..., 1] = tensor_xyz[..., 2]
    out[..., 2] = tensor_xyz[..., 1]
    return out


def _rotate_global_orient(
    poses_aa: torch.Tensor,
    R_pre: torch.Tensor,
) -> torch.Tensor:
    """Pre-multiply the global_orient (joint 0 axis-angle) by R_pre."""
    out = poses_aa.clone()
    global_aa = poses_aa.reshape(-1, 22, 3)[:, 0]  # (T, 3)
    R_old = axis_angle_to_matrix(global_aa)
    R_new = R_pre.to(R_old.dtype).to(R_old.device) @ R_old
    global_new = matrix_to_axis_angle(R_new)
    out_flat = out.reshape(-1, 22, 3)
    out_flat[:, 0] = global_new
    return out_flat.reshape(poses_aa.shape)


def smpldata_y_up_to_z_up(
    smpldata: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        'poses': _rotate_global_orient(smpldata['poses'], _R_X90),
        'trans': _swap_yz(smpldata['trans']),
        'joints': _swap_yz(smpldata['joints']),
    }


def smpldata_z_up_to_y_up(
    smpldata: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        'poses': _rotate_global_orient(smpldata['poses'], _R_X_NEG90),
        'trans': _swap_yz(smpldata['trans']),
        'joints': _swap_yz(smpldata['joints']),
    }


# ───────────────────────── StableMotion model load ─────────────────────────
def load_stablemotion(device: str = 'cuda'):
    """Load the published StableMotion EMA checkpoint.

    The OneDrive 'stablemotion_brokenamass.pt' is actually a tar.gz bundle
    containing save/stablemotion/ema*.pt + model*.pt + args.json. If the
    archive has not been extracted yet, do it now.
    """
    save_dir = SM_ROOT / 'save'
    ema_ckpt = save_dir / 'stablemotion' / 'ema001000000.pt'
    args_path = save_dir / 'stablemotion' / 'args.json'
    if not ema_ckpt.exists():
        bundle = save_dir / 'stablemotion_brokenamass.pt'
        assert bundle.exists(), f'Missing bundle: {bundle}'
        print(f'[load] Extracting bundle {bundle} → {save_dir}')
        import tarfile
        with tarfile.open(bundle, 'r:gz') as tar:
            tar.extractall(path=save_dir)
    assert ema_ckpt.exists(), f'Missing EMA ckpt after extract: {ema_ckpt}'
    with open(args_path) as f:
        train_args = json.load(f)

    # Build model exactly as utils/model_util.py::create_model_and_diffusion
    model = StableMotionDiTModel(
        in_channels=233,           # 232 body + 1 label
        out_channels=233,
        num_layers=train_args['layers'],       # 8
        num_attention_heads=train_args['heads'],  # 8
        attention_head_dim=64,
        class_cond=True,
        zero_init=train_args.get('zero_init', False),
    )
    state = torch.load(str(ema_ckpt), map_location='cpu', weights_only=False)
    # The EMA ckpt stores keys prefixed with 'ema_model.'; strip it.
    body = {}
    for k, v in state.items():
        if k.startswith('ema_model.'):
            body[k[len('ema_model.'):]] = v
    missing, unexpected = model.load_state_dict(body, strict=False)
    # Report (expect only the 'initted' / 'step' scalars missing)
    print(f'[load] missing={len(missing)}, unexpected={len(unexpected)}')
    model.to(device).eval()

    # Diffusion — DDPM 50 steps (predict_xstart=1, cosine beta, FIXED_SMALL)
    steps = train_args['diffusion_steps']  # 50
    betas = gd.get_named_beta_schedule(train_args['noise_schedule'], steps, 1.0)
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )

    # Normalizer (232-d body); append label mean/std per StableMotion convention
    normalizer = Normalizer(str(SM_ROOT / train_args['normalizer_dir']))
    normalizer.add_label_channel()
    normalizer = normalizer.to(device)

    return model, diffusion, normalizer


# ───────────────────────── StableMotion detect + fix ─────────────────────────
@torch.no_grad()
def run_stablemotion_detect_fix(
    feats_232: torch.Tensor,          # (T, 232) body feats (un-normalized)
    model: torch.nn.Module,
    diffusion: SpacedDiffusion,
    normalizer: Normalizer,
    prob_det_th: float = 0.5,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """Mirror of ref_repo/StableMotion/sample/fix_globsmpl.py with:
      - batch size 1
      - no MC averaging (ProbDetNum=0 → single detection pass)
      - no ensemble, no SITS, no foot-lock guidance
      - DDPM 50-step ancestral sampling (ts_respace=None)
    Returns {'feats_fixed': (T, 232), 'label': (T,) bool}.
    """
    T = feats_232.shape[0]
    # Append label=0 channel and normalize
    x_full = torch.cat([feats_232, torch.zeros(T, 1)], dim=-1)  # (T, 233)
    x_norm = normalizer(x_full.to(device))                      # (T, 233)
    x_norm = x_norm.transpose(0, 1).unsqueeze(0)                # (1, 233, T)

    length = torch.tensor([T], device=device)
    attention_mask = torch.ones(1, T, device=device, dtype=torch.bool)

    # ---- Detection pass ---------------------------------------------------
    inp_det = x_norm.clone()
    inp_det[:, -1] = 1.0                         # "corrupt" label channel
    mask_det = torch.ones_like(x_norm, dtype=torch.bool)
    mask_det[:, -1] = False                      # predict label only
    inpaint_cond_det = (~mask_det) & attention_mask.unsqueeze(-2)
    kw_det = {
        'y': {'inpainting_mask': mask_det, 'inpainted_motion': inp_det},
        'inpaint_cond': inpaint_cond_det,
        'length': length,
        'attention_mask': attention_mask,
    }
    shape = (1, 233, T)
    re_sample = diffusion.p_sample_loop(
        model, shape,
        clip_denoised=False, model_kwargs=kw_det,
        skip_timesteps=0, init_image=None,
        progress=False, dump_steps=None, noise=None, const_noise=False,
    )
    # De-normalize, read label
    det_full = normalizer.inverse(re_sample.transpose(1, 2))  # (1, T, 233), still on device
    det_full = det_full.cpu()
    label = (det_full[..., -1] > prob_det_th).squeeze(0)            # (T,) bool

    # ---- Dilate ±1 and force last frame clean -----------------------------
    temp = label.clone()
    label[1:] = label[1:] | temp[:-1]
    label[:-1] = label[:-1] | temp[1:]
    label[-1] = False

    # ---- Fix pass ---------------------------------------------------------
    inpaint_mask_fix = torch.zeros_like(x_norm, dtype=torch.bool)
    good = (~label).nonzero(as_tuple=False).squeeze(-1).tolist()
    if len(good) > 0:
        inpaint_mask_fix[0, :, good] = True        # keep good frames
    inpaint_mask_fix[:, -1] = True                 # always keep label channel
    inp_fix = x_norm.clone()
    inp_fix[:, -1] = -1.0                          # tell model "this is clean"
    inpaint_cond_fix = (~inpaint_mask_fix) & attention_mask.unsqueeze(-2)
    kw_fix = {
        'y': {'inpainting_mask': inpaint_mask_fix, 'inpainted_motion': inp_fix},
        'inpaint_cond': inpaint_cond_fix,
        'length': length,
        'attention_mask': attention_mask,
    }
    sample_fix = diffusion.p_sample_loop(
        model, shape,
        clip_denoised=False, model_kwargs=kw_fix,
        skip_timesteps=0, init_image=inp_fix,
        progress=False, dump_steps=None, noise=None, const_noise=False,
    )
    fix_full = normalizer.inverse(sample_fix.transpose(1, 2)).cpu()  # (1, T, 233)
    feats_fixed = fix_full[0, :, :-1]                                # (T, 232)
    return {'feats_fixed': feats_fixed, 'label': label}


# ───────────────────────── One-sample pipeline ─────────────────────────
def process_one_sample(
    motion_path: str,
    bone_offsets: torch.Tensor,
    model: torch.nn.Module,
    diffusion: SpacedDiffusion,
    normalizer: Normalizer,
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """Run the full M2M → StableMotion → M2M roundtrip on one motion npz.
    Returns a dict with 'lq_135', 'hq_135', 'label' (np.ndarray per key).

    ── Preserving world coords (2026-04-23) ──
    StableMotion's `smpldata_to_alignglobsmplrifkefeats` canonicalizes
    the input by (1) removing ground (z_min → 0), (2) shifting trajectory
    so frame 0 is at origin, (3) zeroing out init_rotZ (so frame 0 faces
    canonical +X). If we invert feats naively we get HQ in canonical
    space.

    Also, the decoder returns ``trans = [trajectory, root_grav_axis]``
    where ``root_grav_axis`` is the **pelvis joint Z** (= trans.z +
    bone_offsets[0].y in SMPL y-up space), NOT the original SMPL root
    translation. That means decoded trans sits ~0.22m below the input
    trans (since SMPL's bone_offsets[0].y ≈ -0.22 — pelvis joint is
    offset downward from the root translation). We compensate for this
    by adding the SMPL pelvis offset back to the gravity axis of the
    decoded trans.
    """
    d = np.load(motion_path, allow_pickle=True)
    tk = 'trans' if 'trans' in d.files else 'transl'
    pk = 'poses' if 'poses' in d.files else 'body_pose'

    # ── fps resampling (2026-04-23) ──
    # StableMotion was trained exclusively on 20 fps AMASS data. E9 LQ data is
    # 30 fps, which is a hard OOD domain shift for the model (the per-frame
    # velocity distribution the diffusion learned is 1.5× slower than what
    # it sees here). Downsample LQ → 20 fps before encoding, upsample HQ
    # back to 30 fps after decoding via linear-interpolation on trans and
    # slerp-like interpolation on rot6d (approximated by linear on 6d then
    # re-orthogonalize downstream — diffusion output is already noisy so
    # simple linear is fine).
    src_fps = 30.0
    try:
        if 'mocap_framerate' in d.files:
            src_fps = float(np.asarray(d['mocap_framerate']).item())
    except Exception:
        pass
    tgt_fps = 20.0

    raw_trans = d[tk].astype(np.float32)
    raw_poses = d[pk].astype(np.float32)
    T_orig = raw_trans.shape[0]

    def _resample_time(arr: np.ndarray, src_fps_: float, tgt_fps_: float) -> np.ndarray:
        """Linear-interpolation along axis 0. Preserves values at frame 0
        (no phase shift); length becomes max(1, round(T*tgt/src))."""
        if abs(src_fps_ - tgt_fps_) < 1e-6:
            return arr
        T_in = arr.shape[0]
        T_out = max(2, int(round(T_in * tgt_fps_ / src_fps_)))
        t_in = np.linspace(0.0, 1.0, T_in, dtype=np.float32)
        t_out = np.linspace(0.0, 1.0, T_out, dtype=np.float32)
        out = np.empty((T_out,) + arr.shape[1:], dtype=arr.dtype)
        for c in range(arr.shape[1]) if arr.ndim == 2 else [None]:
            if c is None:
                out[...] = np.interp(t_out, t_in, arr)
            else:
                out[:, c] = np.interp(t_out, t_in, arr[:, c])
        return out

    # Downsample to 20 fps for StableMotion
    trans_20 = _resample_time(raw_trans, src_fps, tgt_fps)
    poses_20 = _resample_time(raw_poses, src_fps, tgt_fps)

    motion_135 = np.concatenate([
        process_transl(trans_20, 'abs'),
        process_smplx_pose(poses_20, 'rotation_6d', 'smpl_22'),
    ], axis=-1).astype(np.float32)

    # Stash original length so we can upsample back at the end
    _orig_T = T_orig
    _orig_fps = src_fps
    _model_fps = tgt_fps

    # Convert to smpldata (24-joint topology) and y-up → z-up
    smpldata_y = m2m135_to_smpldata_24(motion_135, bone_offsets)
    smpldata_z = smpldata_y_up_to_z_up(smpldata_y)

    # ── Record canonicalization transforms BEFORE feats forward ──
    j0 = smpldata_z['joints'].clone()                          # (T, 24, 3)
    ground_shift_z = float(j0[..., 2].min())                   # scalar (≈ 0)
    traj0_xy = j0[0, 0, :2].clone()                            # (2,)  pelvis XY at frame 0

    # bone_offsets[0] in SMPL y-up = (bo_x, bo_y, bo_z); pelvis offset from
    # root trans. In z-up after y↔z swap, the "downward" component is
    # now in the y axis (not z). But the scalar length we need to add to
    # the decoded trans (in z-up) is the *gravity* component of the
    # pelvis→root offset. In y-up it was bo_y (≈ -0.22). After y↔z swap,
    # bo_y becomes the new bo_z (the new gravity axis). We add its
    # magnitude back so that decoded trans.z matches the original root-
    # translation z (not the pelvis-joint z that the decoder returns).
    bo_np = bone_offsets.numpy() if hasattr(bone_offsets, 'numpy') else np.asarray(bone_offsets)
    pelvis_offset_y_smpl = float(bo_np[0, 1])                  # SMPL y-up
    # trans_shift_z: amount the decoder "lost" in the gravity axis.
    # Since decoded trans.z = pelvis_joint_z, and we want it to be the
    # root_trans_z = pelvis_joint_z - pelvis_offset_y, we ADD
    # (-pelvis_offset_y_smpl) = +0.22 to decoded trans.z.
    trans_gravity_correction = -pelvis_offset_y_smpl           # ≈ +0.22

    # Compute init_rotZ from global_orient at frame 0.
    from data_loaders.amasstools.geometry import (  # noqa: E402
        axis_angle_to_matrix as _aa2mat,
        matrix_to_euler_angles as _mat2euler,
    )
    poses_z = smpldata_z['poses'].reshape(-1, 22, 3)
    global_orient_0 = poses_z[0, 0]                            # (3,)
    R0 = _aa2mat(global_orient_0.unsqueeze(0))[0]              # (3, 3)
    euler0 = _mat2euler(R0.unsqueeze(0), "ZYX")[0]             # (3,)  ZYX order
    rotZ0_angle = euler0[0].item()                             # scalar

    # Feats forward
    feats = smpldata_to_alignglobsmplrifkefeats(smpldata_z)        # (T, 232)

    # StableMotion detect + fix
    result = run_stablemotion_detect_fix(
        feats, model, diffusion, normalizer, device=device,
    )
    feats_fixed = result['feats_fixed']                            # (T, 232)
    label = result['label'].cpu().numpy().astype(np.uint8)

    # Feats inverse (returns canonical-space smpldata; trans.z = pelvis_z)
    smpldata_fixed_z_canon = globsmplrifkefeats_to_smpldata(feats_fixed)

    # ── Apply inverse of the forward canonicalization ──
    cos = np.cos(rotZ0_angle)
    sin = np.sin(rotZ0_angle)
    R_z_inv = torch.tensor([
        [cos, -sin, 0.0],
        [sin,  cos, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=smpldata_fixed_z_canon['joints'].dtype)

    def _decanon_joints(j):
        # j: (T, N, 3) in canonical space
        j = j @ R_z_inv.T   # rotate back
        j[..., 0] += traj0_xy[0]
        j[..., 1] += traj0_xy[1]
        j[..., 2] += ground_shift_z
        return j

    def _decanon_trans(t):
        # t: (T, 3)
        t = t @ R_z_inv.T
        t[..., 0] += traj0_xy[0]
        t[..., 1] += traj0_xy[1]
        # decoder returns pelvis_z; we want root_trans_z (= pelvis_z - pelvis_offset)
        t[..., 2] += ground_shift_z + trans_gravity_correction
        return t

    smpldata_fixed_z = {
        'joints': _decanon_joints(smpldata_fixed_z_canon['joints'].clone()),
        'trans':  _decanon_trans(smpldata_fixed_z_canon['trans'].clone()),
        'poses':  smpldata_fixed_z_canon['poses'].clone(),
    }
    # Rotate global_orient (pelvis axis-angle) back by +rotZ0_angle:
    poses_fixed = smpldata_fixed_z['poses'].reshape(-1, 22, 3)
    go_aa = poses_fixed[:, 0]                                  # (T, 3)
    go_mat = _aa2mat(go_aa)                                    # (T, 3, 3)
    go_mat = R_z_inv.to(go_mat.dtype) @ go_mat
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        matrix_to_axis_angle as _mat2aa,
    )
    go_aa_new = _mat2aa(go_mat)
    poses_fixed[:, 0] = go_aa_new
    smpldata_fixed_z['poses'] = poses_fixed.reshape(-1, 66)

    smpldata_fixed_y = smpldata_z_up_to_y_up(smpldata_fixed_z)

    motion_135_fixed = smpldata_to_m2m135(smpldata_fixed_y)

    # Clip time dim (decoder might change T slightly on edge cases — assert here)
    T = motion_135.shape[0]
    if motion_135_fixed.shape[0] != T:
        motion_135_fixed = motion_135_fixed[:T]

    # ── fps upsample back to original (30 fps) ──
    # Encoder/decoder worked at 20 fps. We need to upsample both the label
    # and the hq motion back so the final output matches the original time
    # axis expected by downstream (dashboard, metrics).
    #
    # 2026-04-26: per-channel linear interp on rot6d corrupts unit-norm
    # (we measured col norms in [0.19, 1.33] on real outputs — clearly
    # non-orthonormal). Switched to a rot6d-aware resampler:
    #   * trans channels use linear interp (smooth, preserves position).
    #   * rot6d channels are converted to rotation matrices, slerp-
    #     interpolated, then converted back. Slerp preserves unit norm
    #     AND bounds discontinuities to the shortest arc, which damps
    #     the per-frame jumps the diffusion model produces at corrupt/
    #     clean boundaries.
    def _resample_time_np(arr: np.ndarray, T_out: int) -> np.ndarray:
        T_in = arr.shape[0]
        if T_in == T_out:
            return arr
        t_in = np.linspace(0.0, 1.0, T_in, dtype=np.float32)
        t_out = np.linspace(0.0, 1.0, T_out, dtype=np.float32)
        if arr.ndim == 1:
            return np.interp(t_out, t_in, arr.astype(np.float32)).astype(arr.dtype)
        out = np.empty((T_out,) + arr.shape[1:], dtype=arr.dtype)
        for c in range(arr.shape[1]):
            out[:, c] = np.interp(t_out, t_in, arr[:, c].astype(np.float32))
        return out

    def _resample_motion135_slerp(motion_135_in: np.ndarray, T_out: int) -> np.ndarray:
        """Resample (T, 135) to (T_out, 135). Trans: linear. Rot6d: slerp.

        Implementation: convert rot6d row-major (M2M layout) → col-major →
        rotation matrices, then slerp each joint independently across the
        new time axis, then convert back. Identity at endpoints, shortest-
        arc elsewhere, unit-norm preserved.
        """
        T_in = motion_135_in.shape[0]
        if T_in == T_out:
            return motion_135_in.astype(np.float32)
        # Trans
        trans_in = motion_135_in[:, :3]
        trans_out = _resample_time_np(trans_in, T_out)
        # Rot6d → rotmat (T_in, 22, 3, 3)
        rot6d_row = torch.from_numpy(
            motion_135_in[:, 3:].reshape(T_in, 22, 6)).float()
        rot6d_col = _rot6d_row_to_col(rot6d_row)
        R_in = rotation_6d_to_matrix(rot6d_col)        # (T_in, 22, 3, 3)
        # Slerp via quaternion logmap on adjacent pairs
        from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
            matrix_to_quaternion as _mat2quat,
            quaternion_to_matrix as _quat2mat,
        )
        quat = _mat2quat(R_in)                          # (T_in, 22, 4) wxyz
        # Float64 for stable slerp
        q = quat.double().numpy()
        # Sample T_out positions along [0, T_in - 1]
        s = np.linspace(0.0, T_in - 1, T_out, dtype=np.float64)
        i0 = np.floor(s).astype(np.int64)
        i0 = np.clip(i0, 0, T_in - 2)
        i1 = i0 + 1
        u = (s - i0).astype(np.float64)
        # Per-output-frame slerp for each joint
        q0 = q[i0]                                      # (T_out, 22, 4)
        q1 = q[i1]
        # Make q1 take the short-arc side
        d = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(d < 0, -q1, q1)
        d = np.abs(d)
        d = np.clip(d, -1.0, 1.0)
        theta = np.arccos(d)                            # (T_out, 22, 1)
        sin_theta = np.sin(theta)
        eps = 1e-7
        small = sin_theta < eps
        a = np.where(small, 1.0 - u[:, None, None],
                     np.sin((1.0 - u[:, None, None]) * theta) / np.maximum(sin_theta, eps))
        b = np.where(small, u[:, None, None],
                     np.sin(u[:, None, None] * theta) / np.maximum(sin_theta, eps))
        q_out = a * q0 + b * q1                         # (T_out, 22, 4)
        # Renormalize (small numerical drift)
        q_out = q_out / np.maximum(
            np.linalg.norm(q_out, axis=-1, keepdims=True), 1e-7)
        R_out = _quat2mat(torch.from_numpy(q_out).float())
        rot6d_out_col = matrix_to_rotation_6d(R_out)
        rot6d_out_row = _rot6d_col_to_row(rot6d_out_col)
        rot6d_flat = rot6d_out_row.reshape(T_out, 132).numpy().astype(np.float32)
        return np.concatenate([trans_out.astype(np.float32), rot6d_flat], axis=-1)

    def _smooth_motion135(motion_135_in: np.ndarray, win: int = 5,
                          polyorder: int = 2) -> np.ndarray:
        """Light Savitzky-Golay smoothing on a (T, 135) motion.

        Trans channels: smoothed channel-wise. Rot6d channels: converted
        to quaternions, smoothed via per-component Savgol, renormalized,
        and converted back to rot6d. This prevents residual per-frame
        spikes (typical at corrupt/clean inpaint boundaries) without
        flattening real motion content.
        """
        T_in = motion_135_in.shape[0]
        if T_in < win:
            return motion_135_in.astype(np.float32)
        try:
            from scipy.signal import savgol_filter as _sg
        except Exception:
            return motion_135_in.astype(np.float32)
        out = motion_135_in.astype(np.float32).copy()
        # Trans
        out[:, :3] = _sg(out[:, :3], win, polyorder, axis=0, mode='nearest')
        # Rot6d → quat → smooth → quat → rot6d
        rot6d_row = torch.from_numpy(out[:, 3:].reshape(T_in, 22, 6)).float()
        rot6d_col = _rot6d_row_to_col(rot6d_row)
        R = rotation_6d_to_matrix(rot6d_col)
        from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
            matrix_to_quaternion as _mat2quat,
            quaternion_to_matrix as _quat2mat,
        )
        quat = _mat2quat(R).numpy()                    # (T, 22, 4)
        # Make quaternion sign continuous (avoid q vs -q flips before smoothing)
        for j in range(quat.shape[1]):
            for t in range(1, T_in):
                if np.dot(quat[t, j], quat[t-1, j]) < 0:
                    quat[t, j] = -quat[t, j]
        flat = quat.reshape(T_in, -1)
        flat_s = _sg(flat, win, polyorder, axis=0, mode='nearest')
        quat_s = flat_s.reshape(T_in, 22, 4)
        quat_s = quat_s / np.maximum(
            np.linalg.norm(quat_s, axis=-1, keepdims=True), 1e-7)
        R_s = _quat2mat(torch.from_numpy(quat_s).float())
        rot6d_col_s = matrix_to_rotation_6d(R_s)
        rot6d_row_s = _rot6d_col_to_row(rot6d_col_s)
        out[:, 3:] = rot6d_row_s.reshape(T_in, 132).numpy()
        return out.astype(np.float32)

    # Rebuild LQ at original fps from raw data so metrics are computed on
    # the ORIGINAL 30-fps reference (avoids comparing 20-fps-downsampled LQ
    # vs upsampled-HQ — which would bake in the fps artifact into LQ too).
    motion_135_lq_orig = np.concatenate([
        process_transl(raw_trans, 'abs'),
        process_smplx_pose(raw_poses, 'rotation_6d', 'smpl_22'),
    ], axis=-1).astype(np.float32)

    # Slerp-aware upsample (preserves rot6d unit-norm; prevents the
    # 0.19-1.33 col-norm corruption observed under linear interp).
    motion_135_fixed_orig = _resample_motion135_slerp(motion_135_fixed, _orig_T)
    # Final 5-frame Savgol pass to damp residual diffusion-output spikes
    # at corrupt/clean boundaries. Cheap, no-op if scipy is missing.
    motion_135_fixed_orig = _smooth_motion135(
        motion_135_fixed_orig, win=5, polyorder=2)
    label_np = label.numpy() if hasattr(label, 'numpy') else np.asarray(label)
    label_orig = _resample_time_np(label_np.astype(np.float32), _orig_T) > 0.5
    label_orig = torch.from_numpy(label_orig.astype(np.uint8)).bool()

    return {
        'lq_135': motion_135_lq_orig,
        'hq_135': motion_135_fixed_orig.astype(np.float32),
        'label': label_orig,
    }


# ───────────────────────── CLI ─────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-datalist', type=str,
                        default='data/eval/m2m_v2/eval_e9_repair.json')
    parser.add_argument('--output-dir', type=str,
                        default='output/eval_v2_e9_stablemotion_20260423')
    parser.add_argument('--max-samples', type=int, default=9999)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--feasibility-only', action='store_true',
                        help='Only run 1 sample end-to-end and print stats.')
    args = parser.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()

    # Load model (extracts bundle on first run)
    t0 = time.time()
    model, diffusion, normalizer = load_stablemotion(args.device)
    print(f'[init] StableMotion loaded in {time.time()-t0:.1f}s '
          f'({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)')

    # Load datalist
    with open(args.eval_datalist) as f:
        dl = json.load(f)
    items = dl.get('data_list', dl)
    N = min(len(items), args.max_samples)
    print(f'[run] processing {N} / {len(items)} samples → {args.output_dir}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'npz').mkdir(exist_ok=True)

    results_summary: List[Dict] = []
    for i, item in enumerate(items[:N]):
        mp = item['motion_path']
        if not os.path.isabs(mp):
            mp = os.path.join(str(PROJECT_ROOT), mp)
        t_s = time.time()
        try:
            out = process_one_sample(
                mp, bone_offsets, model, diffusion, normalizer, args.device,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[{i+1}/{N}] {os.path.basename(mp)[:50]}  FAIL: {e}')
            if args.feasibility_only:
                break
            continue
        dt = time.time() - t_s
        n_corrupted = int(out['label'].sum())
        T = out['hq_135'].shape[0]
        # Save NPZ for dashboard
        npz_out = out_dir / 'npz' / f'{i:05d}.npz'
        np.savez(
            npz_out,
            lq_motion_135=out['lq_135'],
            hq_motion_135=out['hq_135'],
            stablemotion_label=out['label'],
            prompt_id=item.get('prompt_id', ''),
            defect_type=item.get('defect_type', ''),
            source_path=os.path.basename(mp),
        )
        results_summary.append({
            'idx': i,
            'prompt_id': item.get('prompt_id', ''),
            'defect_type': item.get('defect_type', ''),
            'T': T,
            'n_detected': n_corrupted,
            'frac_detected': n_corrupted / max(T, 1),
            'elapsed_s': dt,
            'npz': str(npz_out.relative_to(out_dir)),
        })
        print(f'[{i+1}/{N}] {os.path.basename(mp)[:50]}  '
              f'T={T} detected={n_corrupted}/{T} ({n_corrupted/T*100:.1f}%) '
              f'time={dt:.1f}s')
        if args.feasibility_only:
            break

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump({'results': results_summary, 'n_samples': len(results_summary)},
                  f, indent=2)
    print(f'\n✓ Done. {len(results_summary)} samples → {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

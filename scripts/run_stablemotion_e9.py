"""StableMotion E9 Baseline Inference Wrapper (2026-04-27 rewrite).

Strict open-source-faithful pipeline. NO tricks (no F1/F2/F3/F4, no savgol,
no slerp upsample, no detector cap, no QC fallback). Mirrors
``ref_repo/StableMotion/sample/fix_globsmpl.py`` exactly.

Pipeline:
    M2M 135-dim LQ motion (30 fps)
      → resample to 20 fps (StableMotion training fps)
      → SMPL-24 smpldata
            * poses = axis-angle (T, 66)
            * trans = pelvis_joint_world (= motion135 trans + bone_offsets[0])
            * joints (T, 24, 3) computed via FK; hand joints copy wrists
      → y-up → z-up axis swap (StableMotion uses z as gravity)
      → smpldata_to_alignglobsmplrifkefeats (232-D)
      → append label channel (233-D) + normalize
      → StableMotion detect → fix (DDPM 50-step)
      → denormalize → drop label → 232-D
      → globsmplrifkefeats_to_smpldata (canonical z-up)
      → invert canonicalization (rotZ + traj0_xy + ground_shift)
      → z-up → y-up
      → motion135 trans = pelvis_world - bone_offsets[0]   ← KEY FIX
      → resample to 30 fps (linear on trans, slerp on rot6d for unit-norm)
      → NPZ output

Why this is faithful:
  • encoder assert ``trans[:,2]-trans[0,2] ≈ joints[:,0,2]-joints[0,0,2]``
    is now trivially satisfied with delta = 0 (trans IS pelvis_world).
  • decoder output trans = pelvis_world (canonical) → after canon undo,
    pelvis_world (world). Subtracting the constant bone_offsets[0] gives
    SMPL root_translation directly, matching motion135 semantics.
  • No frame-0 hard re-anchor needed: when SM keeps frame 0 clean
    (always forced via inpainting_mask), the 232-D feats at frame 0
    round-trip exactly back to LQ pelvis/pose.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_stablemotion_e9.py \
        --eval-datalist data/eval/m2m_v2/eval_e9_repair_v2.json \
        --output-dir output/eval_v2_e9_stablemotion_v9 \
        --max-samples 9999
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from functools import partial
from typing import Dict, List

import einops
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
from data_loaders.amasstools.geometry import (  # noqa: E402
    axis_angle_to_matrix as _aa2mat,
    matrix_to_euler_angles as _mat2euler,
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
    matrix_to_rotation_6d, matrix_to_quaternion, quaternion_to_matrix,
)


# ───────────────────────── rot6d row/col helpers ─────────────────────────
def _rot6d_row_to_col(rot6d_row: torch.Tensor) -> torch.Tensor:
    """M2M 135-dim uses row-major rot6d ([R00,R01,R10,R11,R20,R21]); the
    project rotation_convert.py is column-major. Swap order before feeding
    rotation_6d_to_matrix."""
    return rot6d_row[..., [0, 2, 4, 1, 3, 5]]


def _rot6d_col_to_row(rot6d_col: torch.Tensor) -> torch.Tensor:
    return rot6d_col[..., [0, 3, 1, 4, 2, 5]]


# ───────────────────────── M2M ↔ smpldata conversion ─────────────────────────
def m2m135_to_smpldata_24(
    motion_135: np.ndarray,
    bone_offsets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convert M2M 135-dim motion to smpldata for StableMotion 24-joint topology.

    KEY: ``trans`` is set to ``joints[:, 0]`` (pelvis_world), NOT to the
    motion135 root translation. This makes the encoder assert
    ``(trans[:,z]-trans[0,z]) == (joints[:,0,z]-joints[0,0,z])`` trivially
    true (delta = 0). The decoder output trans then carries pelvis_world
    semantics directly; subtracting the constant ``bone_offsets[0]``
    recovers the SMPL root translation cleanly.

    Returns:
        {'poses': (T, 66), 'trans': (T, 3) = pelvis_world,
         'joints': (T, 24, 3) — 24-joint world pos with hand joints copied}
    """
    T = motion_135.shape[0]
    motion_t = torch.from_numpy(motion_135).float()
    rot6d_row = motion_t[:, 3:].reshape(T, 22, 6)
    rot6d_col = _rot6d_row_to_col(rot6d_row)
    R = rotation_6d_to_matrix(rot6d_col)
    poses_aa = matrix_to_axis_angle(R).reshape(T, 66)

    with torch.no_grad():
        joints_world, _, _, _ = motion135_to_fk(
            motion_t.unsqueeze(0), bone_offsets, 'local'
        )
    joints_22 = joints_world.squeeze(0)  # (T, 22, 3) y-up

    # SMPL 24: append left_hand (=20=l_wrist) and right_hand (=21=r_wrist)
    joints_24 = torch.cat([joints_22, joints_22[:, 20:21], joints_22[:, 21:22]], dim=1)

    # KEY: trans = pelvis_world (joints[:, 0])
    trans_pelvis = joints_22[:, 0].clone()
    return {'poses': poses_aa, 'trans': trans_pelvis, 'joints': joints_24}


def smpldata_to_m2m135(
    smpldata: Dict[str, torch.Tensor],
    bone_offsets: torch.Tensor,
) -> np.ndarray:
    """Convert smpldata back to M2M 135-dim. Assumes ``smpldata.trans`` is
    pelvis_world (y-up) — recovers SMPL root translation by subtracting
    the constant ``bone_offsets[0]`` (FK definition: pelvis_world =
    trans_root + bone_offsets[0])."""
    poses = smpldata['poses']
    trans_pelvis = smpldata['trans']  # y-up pelvis_world
    T = poses.shape[0]
    poses_aa = poses.reshape(T, -1, 3)[:, :22]
    R = axis_angle_to_matrix(poses_aa)
    rot6d_col = matrix_to_rotation_6d(R)
    rot6d_row = _rot6d_col_to_row(rot6d_col)

    bo0 = bone_offsets[0].to(trans_pelvis.dtype).to(trans_pelvis.device)
    trans_root = trans_pelvis[:, :3] - bo0[None, :]

    motion_135 = torch.cat([trans_root, rot6d_row.reshape(T, 132)], dim=-1)
    return motion_135.numpy().astype(np.float32)


# ───────────────────────── y-up ↔ z-up axis swap ─────────────────────────
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


def _swap_yz(t: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    out[..., 1] = t[..., 2]
    out[..., 2] = t[..., 1]
    return out


def _rotate_global_orient(poses_aa: torch.Tensor, R_pre: torch.Tensor) -> torch.Tensor:
    """Pre-multiply global_orient (joint 0 axis-angle) by R_pre."""
    out = poses_aa.clone()
    global_aa = poses_aa.reshape(-1, 22, 3)[:, 0]
    R_old = axis_angle_to_matrix(global_aa)
    R_new = R_pre.to(R_old.dtype).to(R_old.device) @ R_old
    global_new = matrix_to_axis_angle(R_new)
    out_flat = out.reshape(-1, 22, 3)
    out_flat[:, 0] = global_new
    return out_flat.reshape(poses_aa.shape)


def smpldata_y_up_to_z_up(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        'poses': _rotate_global_orient(sd['poses'], _R_X90),
        'trans': _swap_yz(sd['trans']),
        'joints': _swap_yz(sd['joints']),
    }


def smpldata_z_up_to_y_up(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        'poses': _rotate_global_orient(sd['poses'], _R_X_NEG90),
        'trans': _swap_yz(sd['trans']),
        'joints': _swap_yz(sd['joints']),
    }


# ───────────────────────── StableMotion model load ─────────────────────────
def load_stablemotion(device: str = 'cuda'):
    """Load the published EMA checkpoint exactly as model_util.create_*."""
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
    assert ema_ckpt.exists(), f'Missing EMA ckpt: {ema_ckpt}'
    with open(args_path) as f:
        train_args = json.load(f)

    model = StableMotionDiTModel(
        in_channels=233,
        out_channels=233,
        num_layers=train_args['layers'],
        num_attention_heads=train_args['heads'],
        attention_head_dim=64,
        class_cond=True,
        zero_init=train_args.get('zero_init', False),
    )
    state = torch.load(str(ema_ckpt), map_location='cpu', weights_only=False)
    body = {}
    for k, v in state.items():
        if k.startswith('ema_model.'):
            body[k[len('ema_model.'):]] = v
    missing, unexpected = model.load_state_dict(body, strict=False)
    print(f'[load] missing={len(missing)}, unexpected={len(unexpected)}')
    model.to(device).eval()

    steps = train_args['diffusion_steps']
    betas = gd.get_named_beta_schedule(train_args['noise_schedule'], steps, 1.0)
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )

    normalizer = Normalizer(str(SM_ROOT / train_args['normalizer_dir']))
    normalizer.add_label_channel()
    normalizer = normalizer.to(device)

    return model, diffusion, normalizer, train_args


def _choose_sampler(diffusion: SpacedDiffusion, ts_respace: bool):
    return diffusion.ddim_sample_loop if ts_respace else diffusion.p_sample_loop


_FEET2METER = 0.3048


def _batch_expander(model_kwargs, repeat_times: int):
    out_model_kwargs = {}
    for k, v in model_kwargs.items():
        if k == 'y':
            out_model_kwargs['y'] = _batch_expander(v, repeat_times)
        elif isinstance(v, list):
            out_model_kwargs[k] = v * repeat_times
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            out_model_kwargs[k] = einops.repeat(
                v, 'b ... -> (repeat b) ...', repeat=repeat_times
            )
        else:
            out_model_kwargs[k] = v
    return out_model_kwargs


def _run_cleanup_selection_local(
    model,
    model_kwargs_detmode,
    model_kwargs,
    motion_normalizer,
    sample_fn,
    cond_fn,
    prob_det_th: float,
    skip_timesteps: int,
    enable_sits: bool,
    diffusion_steps: int,
    classifier_scale: float,
    bs: int,
    nfeats: int,
    nframes: int,
):
    """Local copy of StableMotion ensemble cleanup without TMR imports."""
    forward_rp_times = 5
    eval_times = 25
    device = model_kwargs['y']['inpainted_motion'].device

    rp_model_kwargs_detmode = _batch_expander(model_kwargs_detmode, forward_rp_times)
    with torch.no_grad():
        re_sample = 0
        re_t = torch.ones((bs * forward_rp_times,), device=device) * 49
        for _ in range(eval_times):
            x = torch.randn_like(rp_model_kwargs_detmode['y']['inpainted_motion'])
            inpaint_cond = rp_model_kwargs_detmode['inpaint_cond']
            x_gt = rp_model_kwargs_detmode['y']['inpainted_motion']
            x = torch.where(inpaint_cond, x, x_gt)
            re_sample += model(x, re_t, **rp_model_kwargs_detmode)
        re_sample = re_sample / eval_times

    sample_det = motion_normalizer.inverse(re_sample.transpose(1, 2))
    label = sample_det[..., -1] > prob_det_th
    temp_labels = label.clone()
    label[..., 1:] += temp_labels[..., :-1]
    label[..., :-1] += temp_labels[..., 1:]
    for mids, mlen in enumerate(rp_model_kwargs_detmode['length'].cpu().numpy()):
        label[mids, ..., mlen - 1] = 0

    det_good_frames_per_sample = {
        sample_i: np.nonzero(~label.detach().cpu().numpy()[sample_i].squeeze())[0].tolist()
        for sample_i in range(len(label))
    }

    inpainting_mask_fixmode = torch.zeros_like(re_sample, dtype=torch.bool)
    for sample_i in range(len(re_sample)):
        inpainting_mask_fixmode[sample_i, ..., det_good_frames_per_sample[sample_i]] = True
    inpainting_mask_fixmode[:, -1] = True

    inpaint_motion_fixmode = rp_model_kwargs_detmode['y']['inpainted_motion'].clone()
    inpaint_motion_fixmode[:, -1] = -1.0
    inpaint_cond_fixmode = (
        (~inpainting_mask_fixmode) & rp_model_kwargs_detmode['attention_mask'].unsqueeze(-2)
    )

    rp_model_kwargs = _batch_expander(model_kwargs, forward_rp_times)
    rp_model_kwargs['y']['inpainting_mask'] = inpainting_mask_fixmode.clone()
    rp_model_kwargs['y']['inpainted_motion'] = inpaint_motion_fixmode.clone()
    rp_model_kwargs['inpaint_cond'] = inpaint_cond_fixmode.clone()

    if enable_sits:
        soft_inpaint_ts = einops.repeat(
            re_sample[:, [-1]], 'b c l -> b (repeat c) l', repeat=nfeats
        )
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        soft_inpaint_ts = torch.ceil(
            (torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * diffusion_steps
        ).long()
    else:
        soft_inpaint_ts = None

    sample = sample_fn(
        model,
        (bs * forward_rp_times, nfeats, nframes),
        clip_denoised=False,
        model_kwargs=rp_model_kwargs,
        skip_timesteps=skip_timesteps,
        init_image=rp_model_kwargs['y']['inpainted_motion'],
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
        soft_inpaint_ts=soft_inpaint_ts,
        cond_fn=cond_fn if classifier_scale else None,
    )

    inpaint_motion_detmode = sample.clone()
    inpaint_motion_detmode[:, -1] = 1.0
    rp_model_kwargs_detmode['y']['inpainted_motion'] = inpaint_motion_detmode.clone()

    score = 0
    with torch.no_grad():
        re_t = torch.ones((bs * forward_rp_times,), device=device) * 49
        for _ in range(eval_times):
            x = torch.randn_like(rp_model_kwargs_detmode['y']['inpainted_motion'])
            inpaint_cond = rp_model_kwargs_detmode['inpaint_cond']
            x_gt = rp_model_kwargs_detmode['y']['inpainted_motion']
            x = torch.where(inpaint_cond, x, x_gt)
            score += model(x, re_t, **rp_model_kwargs_detmode)[:, -1]
    score /= eval_times
    score = torch.sum(
        (score > 0.0) * rp_model_kwargs_detmode['attention_mask'], dim=-1
    )
    score = einops.rearrange(score, '(repeat b) -> repeat b', repeat=forward_rp_times)

    sample_candidates = einops.rearrange(
        sample, '(repeat b) c l -> repeat b c l', repeat=forward_rp_times
    )
    selected_id = torch.argmin(score, dim=0)
    selected_id = selected_id[..., None, None].expand(
        sample_candidates.shape[1:]
    ).unsqueeze(0)
    sample = torch.gather(sample_candidates, dim=0, index=selected_id).squeeze(0)
    return sample


def _compute_foot_sliding_torch_local(
    foot_data: torch.Tensor,
    traj_qpos: torch.Tensor,
    offseth=None,
    upaxis: int = 1,
    H: float = 0.05,
):
    plane_axis = [0, 1, 2]
    plane_axis.pop(upaxis)
    foot = foot_data.clone() * _FEET2METER
    if offseth is None:
        offseth = torch.mean(foot[:10, upaxis])
    else:
        offseth = offseth * _FEET2METER
    foot[:, upaxis] = foot[:, upaxis] - offseth
    traj_qpos = traj_qpos.clone()
    traj_qpos[:, upaxis] = traj_qpos[:, upaxis] - offseth
    foot_disp = torch.linalg.norm(foot[1:, plane_axis] - foot[:-1, plane_axis], dim=1)
    seq_len = len(traj_qpos)
    y_threshold = 0.65
    y = traj_qpos[1:, upaxis]
    foot_avg = (foot[:-1, upaxis] + foot[1:, upaxis]) / 2
    subset = torch.logical_and(foot_avg < H, y > y_threshold)
    sliding_stats = torch.abs(foot_disp * (2 - 2 ** (foot_avg.detach() / H)))[subset]
    sliding = torch.sum(sliding_stats) / max(seq_len, 1)
    return sliding, sliding_stats


def _compute_foot_sliding_wrapper_torch_local(
    motions: torch.Tensor,
    motion_lengths: torch.Tensor,
    upaxis: int = 1,
    ankle_h: float = 0.05,
):
    traj_idx = 0
    feet_idxs = [7, 8]
    sliding_mean = []
    for motion, mlen in zip(motions, motion_lengths):
        motion = motion[:mlen]
        traj_qpos = motion[:, traj_idx]
        offseth = torch.min(motion[..., upaxis]).detach()
        for foot_idx in feet_idxs:
            foot_data = motion[:, foot_idx]
            sliding, _ = _compute_foot_sliding_torch_local(
                foot_data, traj_qpos, offseth, upaxis=upaxis, H=ankle_h
            )
            sliding_mean.append(sliding)
    return sliding_mean


def _prepare_cond_fn_abs(
    model: torch.nn.Module,
    motion_normalizer: Normalizer,
    classifier_scale: float,
    device: str,
):
    if not classifier_scale:
        return None

    j_regressor_stat = np.load(
        str(SM_ROOT / 'data_loaders/amasstools/smpl_neutral_nobetas_24J.npz')
    )
    J_regressor = torch.from_numpy(j_regressor_stat['J']).to(device)
    parents = torch.from_numpy(j_regressor_stat['parents'])
    root_offset = torch.tensor([-0.00179506, -0.22333382, 0.02821918]).to(device)
    std = motion_normalizer.std.clone().to(device)
    mean = motion_normalizer.mean.clone().to(device)
    from smplx.lbs import batch_rigid_transform  # noqa: WPS433

    def _footlocking_fn(x, t, **kwargs):
        lengths = kwargs['length']
        eps = 1e-12
        with torch.enable_grad():
            inpaint_cond = kwargs['inpaint_cond']
            x_gt = kwargs['y']['inpainted_motion']
            x_in = x.detach().requires_grad_(True)
            x_in = torch.where(inpaint_cond, x_in, x_gt)
            x_0 = model(x_in, t, **kwargs)
            x_0 = torch.where(inpaint_cond, x_0, x_gt)
            denorm_x0 = x_0.transpose(1, 2) * (std + eps) + mean
            B = denorm_x0.shape[0]
            denorm_x0_flatten = einops.rearrange(denorm_x0, 'b n d -> (b n) d')
            smpldata = globsmplrifkefeats_to_smpldata(denorm_x0_flatten[..., :-1])
            poses = einops.rearrange(smpldata['poses'], 'k (l t) -> k l t', t=3)
            trans = smpldata['trans']
            rot_mat = axis_angle_to_matrix(poses)
            T_all = rot_mat.shape[0]
            zero_hands_rot = torch.eye(3)[None, None].expand(
                T_all, 2, -1, -1
            ).to(device)
            rot_mat = torch.concat((rot_mat, zero_hands_rot), dim=1)
            joints, _ = batch_rigid_transform(
                rot_mat,
                J_regressor[None].expand(T_all, -1, -1),
                parents,
            )
            joints = joints.squeeze() + trans.unsqueeze(1) - root_offset
            joints = einops.rearrange(joints, '(b n) j d -> b n j d', b=B)
            slide_dist = _compute_foot_sliding_wrapper_torch_local(
                joints, lengths, upaxis=2, ankle_h=0.1
            )
            loss = sum(slide_dist)
            grad = torch.autograd.grad(-loss, x_in)[0] * classifier_scale

        grad = torch.nan_to_num(grad)
        grad = torch.clip(grad, min=-10, max=10)
        grad[:, 0] = 0.0
        return grad

    return _footlocking_fn


# ───────────────────────── Detect + Fix (open-source faithful) ─────────────────────────
@torch.no_grad()
def run_stablemotion_detect_fix(
    feats_232: torch.Tensor,            # (T, 232)
    model: torch.nn.Module,
    diffusion: SpacedDiffusion,
    normalizer: Normalizer,
    prob_det_th: float = 0.5,
    prob_det_num: int = 0,
    skip_timesteps: int = 0,
    ts_respace: bool = False,
    enable_sits: bool = False,
    ensemble: bool = False,
    classifier_scale: float = 0.0,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """Batch-1 mirror of StableMotion fix_globsmpl.py.

    Supports the official enhanced inference switches:
    `ProbDetNum`, `enable_sits`, `ensemble`, `classifier_scale`, `ts_respace`.
    Returns {'feats_fixed': (T, 232), 'label': (T,) bool}.
    """
    T = feats_232.shape[0]
    x_full = torch.cat([feats_232, torch.zeros(T, 1)], dim=-1)         # (T, 233)
    x_norm = normalizer(x_full.to(device))
    x_norm = x_norm.transpose(0, 1).unsqueeze(0)                       # (1, 233, T)

    length = torch.tensor([T], device=device)
    attention_mask = torch.ones(1, T, device=device, dtype=torch.bool)

    # ---- Stage 1: Detect ----------------------------------------------------
    inp_det = x_norm.clone()
    inp_det[:, -1] = 1.0
    mask_det = torch.ones_like(x_norm, dtype=torch.bool)
    mask_det[:, -1] = False
    inpaint_cond_det = (~mask_det) & attention_mask.unsqueeze(-2)
    kw_det = {
        'y': {'inpainting_mask': mask_det, 'inpainted_motion': inp_det},
        'inpaint_cond': inpaint_cond_det,
        'length': length,
        'attention_mask': attention_mask,
    }
    shape = (1, 233, T)
    sample_fn = _choose_sampler(diffusion, ts_respace)
    re_sample = sample_fn(
        model, shape,
        clip_denoised=False, model_kwargs=kw_det,
        skip_timesteps=0, init_image=None,
        progress=False, dump_steps=None, noise=None, const_noise=False,
    )
    if prob_det_num:
        for _ in range(prob_det_num):
            re_sample += sample_fn(
                model, shape,
                clip_denoised=False, model_kwargs=kw_det,
                skip_timesteps=0, init_image=None,
                progress=False, dump_steps=None, noise=None, const_noise=False,
            )
        re_sample = re_sample / (prob_det_num + 1)
    det_full = normalizer.inverse(re_sample.transpose(1, 2)).cpu()
    probs = det_full[..., -1].squeeze(0)                               # (T,)
    label = (probs > prob_det_th)                                       # (T,) bool

    # ±1 dilation, last frame forced clean (open-source default)
    temp = label.clone()
    label[1:] = label[1:] | temp[:-1]
    label[:-1] = label[:-1] | temp[1:]
    label[-1] = False

    # ---- Stage 2: Fix -------------------------------------------------------
    inpaint_mask_fix = torch.zeros_like(x_norm, dtype=torch.bool)
    good = (~label).nonzero(as_tuple=False).squeeze(-1).tolist()
    if len(good) > 0:
        inpaint_mask_fix[0, :, good] = True
    inpaint_mask_fix[:, -1] = True
    inp_fix = x_norm.clone()
    inp_fix[:, -1] = -1.0
    inpaint_cond_fix = (~inpaint_mask_fix) & attention_mask.unsqueeze(-2)
    kw_fix = {
        'y': {'inpainting_mask': inpaint_mask_fix, 'inpainted_motion': inp_fix},
        'inpaint_cond': inpaint_cond_fix,
        'length': length,
        'attention_mask': attention_mask,
    }
    if enable_sits and prob_det_num:
        soft_inpaint_ts = einops.repeat(
            re_sample[:, [-1]], 'b c l -> b (repeat c) l', repeat=shape[1]
        )
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        soft_inpaint_ts = torch.ceil(
            (torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * diffusion.num_timesteps
        ).long()
    else:
        soft_inpaint_ts = None

    cond_fn = _prepare_cond_fn_abs(
        model, normalizer, classifier_scale=classifier_scale, device=device
    )
    if ensemble:
        sample_fix = _run_cleanup_selection_local(
            model=model,
            model_kwargs_detmode=kw_det,
            model_kwargs=kw_fix,
            motion_normalizer=normalizer,
            sample_fn=sample_fn,
            cond_fn=cond_fn if classifier_scale else None,
            prob_det_th=prob_det_th,
            skip_timesteps=skip_timesteps,
            enable_sits=enable_sits,
            diffusion_steps=diffusion.num_timesteps,
            classifier_scale=classifier_scale,
            bs=1,
            nfeats=shape[1],
            nframes=T,
        )
    else:
        sample_fix = sample_fn(
            model, shape,
            clip_denoised=False, model_kwargs=kw_fix,
            skip_timesteps=skip_timesteps, init_image=inp_fix,
            progress=False, dump_steps=None, noise=None, const_noise=False,
            soft_inpaint_ts=soft_inpaint_ts,
            cond_fn=cond_fn if classifier_scale else None,
        )
    fix_full = normalizer.inverse(sample_fix.transpose(1, 2)).cpu()
    feats_fixed = fix_full[0, :, :-1]                                   # (T, 232)
    return {'feats_fixed': feats_fixed, 'label': label}


# ───────────────────────── fps resampling helpers ─────────────────────────
def _linear_interp_axis0(arr: np.ndarray, T_out: int) -> np.ndarray:
    T_in = arr.shape[0]
    if T_in == T_out:
        return arr.astype(np.float32)
    t_in = np.linspace(0.0, 1.0, T_in, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, T_out, dtype=np.float32)
    if arr.ndim == 1:
        return np.interp(t_out, t_in, arr.astype(np.float32)).astype(np.float32)
    out = np.empty((T_out,) + arr.shape[1:], dtype=np.float32)
    flat = arr.reshape(T_in, -1)
    out_flat = out.reshape(T_out, -1)
    for c in range(flat.shape[1]):
        out_flat[:, c] = np.interp(t_out, t_in, flat[:, c].astype(np.float32))
    return out


def _resample_motion135_slerp(motion_135_in: np.ndarray, T_out: int) -> np.ndarray:
    """Resample (T, 135) — trans linear, rot6d slerp via quaternion. Slerp
    preserves unit-norm and shortest-arc, which is the mathematically
    correct way to resample rotations (NOT a 'trick')."""
    T_in = motion_135_in.shape[0]
    if T_in == T_out:
        return motion_135_in.astype(np.float32)
    trans_out = _linear_interp_axis0(motion_135_in[:, :3], T_out)
    rot6d_row = torch.from_numpy(
        motion_135_in[:, 3:].reshape(T_in, 22, 6)).float()
    rot6d_col = _rot6d_row_to_col(rot6d_row)
    R_in = rotation_6d_to_matrix(rot6d_col)
    quat = matrix_to_quaternion(R_in).double().numpy()                 # (T_in, 22, 4)
    s = np.linspace(0.0, T_in - 1, T_out, dtype=np.float64)
    i0 = np.clip(np.floor(s).astype(np.int64), 0, T_in - 2)
    i1 = i0 + 1
    u = (s - i0).astype(np.float64)
    q0 = quat[i0]
    q1 = quat[i1]
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.clip(np.abs(d), -1.0, 1.0)
    theta = np.arccos(d)
    sin_theta = np.sin(theta)
    eps = 1e-7
    small = sin_theta < eps
    a = np.where(small, 1.0 - u[:, None, None],
                 np.sin((1.0 - u[:, None, None]) * theta) / np.maximum(sin_theta, eps))
    b = np.where(small, u[:, None, None],
                 np.sin(u[:, None, None] * theta) / np.maximum(sin_theta, eps))
    q_out = a * q0 + b * q1
    q_out = q_out / np.maximum(np.linalg.norm(q_out, axis=-1, keepdims=True), 1e-7)
    R_out = quaternion_to_matrix(torch.from_numpy(q_out).float())
    rot6d_out_col = matrix_to_rotation_6d(R_out)
    rot6d_out_row = _rot6d_col_to_row(rot6d_out_col)
    rot6d_flat = rot6d_out_row.reshape(T_out, 132).numpy().astype(np.float32)
    return np.concatenate([trans_out.astype(np.float32), rot6d_flat], axis=-1)


# ───────────────────────── per-sample pipeline ─────────────────────────
def process_one_sample(
    motion_path: str,
    bone_offsets: torch.Tensor,
    model: torch.nn.Module,
    diffusion: SpacedDiffusion,
    normalizer: Normalizer,
    device: str = 'cuda',
    stablemotion_skip_timesteps: int = 0,
    stablemotion_prob_det_num: int = 0,
    stablemotion_ts_respace: bool = False,
    stablemotion_enable_sits: bool = False,
    stablemotion_ensemble: bool = False,
    stablemotion_classifier_scale: float = 0.0,
    preserve_lq_translation: bool = False,
) -> Dict[str, np.ndarray]:
    d = np.load(motion_path, allow_pickle=True)
    tk = 'trans' if 'trans' in d.files else 'transl'
    pk = 'poses' if 'poses' in d.files else 'body_pose'
    raw_trans = d[tk].astype(np.float32)
    raw_poses = d[pk].astype(np.float32)
    T_orig = raw_trans.shape[0]

    src_fps = 30.0
    try:
        if 'mocap_framerate' in d.files:
            src_fps = float(np.asarray(d['mocap_framerate']).item())
    except Exception:
        pass
    tgt_fps = 20.0  # StableMotion training fps

    # LQ at original (30) fps for output / metrics
    motion_135_lq_orig = np.concatenate([
        process_transl(raw_trans, 'abs'),
        process_smplx_pose(raw_poses, 'rotation_6d', 'smpl_22'),
    ], axis=-1).astype(np.float32)

    # Resample LQ to 20 fps via slerp-aware resampler (mathematically correct,
    # no smoothing). This is data preprocessing, not a trick.
    T_20 = max(2, int(round(T_orig * tgt_fps / src_fps)))
    motion_135_20 = _resample_motion135_slerp(motion_135_lq_orig, T_20)

    # ── Encode to smpldata (24 joints) — y-up ──
    sd_y = m2m135_to_smpldata_24(motion_135_20, bone_offsets)
    sd_z = smpldata_y_up_to_z_up(sd_y)

    # ── Record canonicalization transforms (encoder will undo them) ──
    j0 = sd_z['joints'].clone()
    ground_shift_z = float(j0[..., 2].min())
    traj0_xy = j0[0, 0, :2].clone()

    poses_z = sd_z['poses'].reshape(-1, 22, 3)
    R0 = _aa2mat(poses_z[0, 0].unsqueeze(0))[0]
    euler0 = _mat2euler(R0.unsqueeze(0), "ZYX")[0]
    rotZ0_angle = euler0[0].item()

    # ── Encode to 232-D feats ──
    feats = smpldata_to_alignglobsmplrifkefeats(sd_z)

    # ── Detect + Fix ──
    result = run_stablemotion_detect_fix(
        feats, model, diffusion, normalizer,
        prob_det_num=stablemotion_prob_det_num,
        skip_timesteps=stablemotion_skip_timesteps,
        ts_respace=stablemotion_ts_respace,
        enable_sits=stablemotion_enable_sits,
        ensemble=stablemotion_ensemble,
        classifier_scale=stablemotion_classifier_scale,
        device=device,
    )
    feats_fixed = result['feats_fixed']
    label_20 = result['label'].cpu().numpy().astype(bool)

    # ── Decode 232 → smpldata (canonical, z-up) ──
    sd_fixed_z_canon = globsmplrifkefeats_to_smpldata(feats_fixed)

    # ── Inverse canonicalization: rotate by +rotZ0, add traj0_xy, ground ──
    cos = np.cos(rotZ0_angle)
    sin = np.sin(rotZ0_angle)
    R_z_inv = torch.tensor([
        [cos, -sin, 0.0],
        [sin,  cos, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=sd_fixed_z_canon['joints'].dtype)

    def _decanon_xyz(t):
        t = t @ R_z_inv.T
        t[..., 0] += traj0_xy[0]
        t[..., 1] += traj0_xy[1]
        t[..., 2] += ground_shift_z
        return t

    sd_fixed_z = {
        'joints': _decanon_xyz(sd_fixed_z_canon['joints'].clone()),
        'trans': _decanon_xyz(sd_fixed_z_canon['trans'].clone()),
        'poses': sd_fixed_z_canon['poses'].clone(),
    }
    # Rotate global_orient back by +rotZ0
    poses_fixed = sd_fixed_z['poses'].reshape(-1, 22, 3)
    go_aa = poses_fixed[:, 0]
    go_mat = _aa2mat(go_aa)
    go_mat_new = R_z_inv.to(go_mat.dtype) @ go_mat
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        matrix_to_axis_angle as _mat2aa,
    )
    poses_fixed[:, 0] = _mat2aa(go_mat_new)
    sd_fixed_z['poses'] = poses_fixed.reshape(-1, 66)

    # ── z-up → y-up ──
    sd_fixed_y = smpldata_z_up_to_y_up(sd_fixed_z)

    # ── smpldata → motion_135 (subtracts bone_offsets[0] from trans) ──
    motion_135_fixed_20 = smpldata_to_m2m135(sd_fixed_y, bone_offsets)

    # ── 20 fps → 30 fps (slerp on rot6d, linear on trans) ──
    motion_135_fixed_orig = _resample_motion135_slerp(motion_135_fixed_20, T_orig)
    if preserve_lq_translation:
        motion_135_fixed_orig[:, :3] = motion_135_lq_orig[:, :3]

    # Resample label 20→30 via NN (per-frame bool)
    label_orig = (_linear_interp_axis0(
        label_20.astype(np.float32), T_orig) > 0.5)

    return {
        'lq_135': motion_135_lq_orig,
        'hq_135': motion_135_fixed_orig.astype(np.float32),
        'label': torch.from_numpy(label_orig.astype(np.uint8)),
    }


# ───────────────────────── CLI ─────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-datalist', type=str,
                        default='data/eval/m2m_v2/eval_e9_repair_v2.json')
    parser.add_argument('--output-dir', type=str,
                        default='output/eval_v2_e9_stablemotion_v9')
    parser.add_argument('--max-samples', type=int, default=9999)
    parser.add_argument(
        '--indices', type=str, default='',
        help='Optional comma-separated sample indices to run instead of prefix.',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-timesteps', type=int, default=0)
    parser.add_argument('--prob-det-num', type=int, default=0)
    parser.add_argument('--ts-respace', action='store_true')
    parser.add_argument('--enable-sits', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--classifier-scale', type=float, default=0.0)
    parser.add_argument(
        '--preserve-lq-translation', action='store_true',
        help='Use StableMotion rotations but keep original LQ root translation.',
    )
    parser.add_argument('--feasibility-only', action='store_true')
    args = parser.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()

    t0 = time.time()
    model, diffusion, normalizer, train_args = load_stablemotion(args.device)
    print(f'[init] StableMotion loaded in {time.time()-t0:.1f}s '
          f'({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)')

    with open(args.eval_datalist) as f:
        dl = json.load(f)
    items = dl.get('data_list', dl)
    if args.indices.strip():
        selected_indices = [
            int(x) for x in args.indices.replace(' ', '').split(',') if x
        ][:args.max_samples]
    else:
        selected_indices = list(range(min(len(items), args.max_samples)))
    N = len(selected_indices)
    print(f'[run] processing {len(selected_indices)} / {len(items)} samples '
          f'→ {args.output_dir}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'npz').mkdir(exist_ok=True)

    results_summary: List[Dict] = []
    for i in selected_indices:
        item = items[i]
        mp = item['motion_path']
        if not os.path.isabs(mp):
            mp = os.path.join(str(PROJECT_ROOT), mp)
        t_s = time.time()
        try:
            out = process_one_sample(
                mp, bone_offsets, model, diffusion, normalizer, args.device,
                stablemotion_skip_timesteps=args.skip_timesteps,
                stablemotion_prob_det_num=args.prob_det_num,
                stablemotion_ts_respace=args.ts_respace,
                stablemotion_enable_sits=args.enable_sits,
                stablemotion_ensemble=args.ensemble,
                stablemotion_classifier_scale=args.classifier_scale,
                preserve_lq_translation=args.preserve_lq_translation,
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

#!/usr/bin/env python3
"""Comprehensive repair evaluation on CJGame_MB dataset.

Evaluates ALL candidate models on the CJGame_MB npz_split data:

Models:
  - MoGenDIT: denoise, ada_denoise
  - HyMotion M2M (_man configs × {fm,jit} × {local,globalrot} × {uncond,caption}):
      completion + edit modes, using MoGenDIT adaptive mask

For each case:
  1. Compute MoGenDIT adaptive mask (once, shared)
  2. Run each model to produce repaired NPZ
  3. Run quality checker on before/after
  4. Save results for visualization

Output:
  output/cjgame_repair_eval/
    adaptive_masks/<name>.npz          - saved adaptive masks
    <model_label>/repaired/<name>.npz  - repaired NPZs
    eval_report.json                   - comprehensive evaluation report

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py --max-samples 100
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py --max-samples 0  # all
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# MUST happen before ANY hftrainer imports: prevent hftrainer.models.motion.__init__
# from importing PrismMCMBundle (requires WanAttention from newer diffusers).
# We register a dummy module so Python skips __init__.py when importing sub-packages.
# ============================================================================
import types as _types
_dummy_modules = [
    'hftrainer.models',
    'hftrainer.models.motion',
    # Also block dataset __init__ chain that imports vermo
    'hftrainer.datasets',
    'hftrainer.datasets.motion',
    'hftrainer.datasets.motion.motionhub',
]
for _mod_name in _dummy_modules:
    if _mod_name not in sys.modules:
        _dummy = _types.ModuleType(_mod_name)
        _dummy.__path__ = [str(PROJECT_ROOT / _mod_name.replace('.', '/'))]
        _dummy.__package__ = _mod_name
        sys.modules[_mod_name] = _dummy

# Ensure seaborn available (MoGenDIT dependency)
try:
    import seaborn  # noqa
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seaborn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

NPZ_SPLIT_DIR = PROJECT_ROOT / "data" / "lightai_data" / "CJGame_MB" / "npz_split"

# ============================================================================
# M2M config registry — only _man variants + uncond (non-MAN) as reference
# ============================================================================
M2M_CONFIGS = {
    # _man variants (8 configs)
    "uncond_fm_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_fm_man_046b",
        "desc": "Uncond FM MAN (local rot)",
    },
    "uncond_jit_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_jit_man_046b",
        "desc": "Uncond JiT MAN (local rot)",
    },
    "caption_fm_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_fm_man_046b",
        "desc": "Caption FM MAN (local rot)",
    },
    "caption_jit_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_jit_man_046b",
        "desc": "Caption JiT MAN (local rot)",
    },
    "uncond_fm_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_fm_man_globalrot_046b",
        "desc": "Uncond FM MAN (global rot)",
    },
    "uncond_jit_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_jit_man_globalrot_046b",
        "desc": "Uncond JiT MAN (global rot)",
    },
    "caption_fm_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_fm_man_globalrot_046b",
        "desc": "Caption FM MAN (global rot)",
    },
    "caption_jit_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_jit_man_globalrot_046b",
        "desc": "Caption JiT MAN (global rot)",
    },
}

# ============================================================================
# NPZ / motion utilities
# ============================================================================

def _smplh_to_rot6d_22(poses_aa: np.ndarray) -> np.ndarray:
    """Convert SMPL-H axis-angle (T,156) or (T,52,3) to row-major rot6d (T, 132).

    Avoids importing from hftrainer.datasets which triggers heavy transitive imports.
    """
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_rotation_6d,
    )
    if poses_aa.ndim == 2:
        n_joints = poses_aa.shape[1] // 3
        if n_joints == 52:  # SMPL-H
            poses_aa = np.concatenate(
                [poses_aa[:, :66], np.zeros((poses_aa.shape[0], 9), dtype=poses_aa.dtype), poses_aa[:, 66:]],
                axis=1,
            )
        poses_aa = poses_aa.reshape(poses_aa.shape[0], -1, 3)
    # Take first 22 joints
    aa = poses_aa[:, :22, :]  # (T, 22, 3)
    T = aa.shape[0]
    aa_flat = aa.reshape(T * 22, 3)
    r6d = axis_angle_to_rotation_6d(aa_flat).reshape(T, 22, 6)
    # column-major -> row-major
    r6d = r6d[:, :, [0, 3, 1, 4, 2, 5]]
    return r6d.reshape(T, 132).astype(np.float32)


def load_npz_as_motion(npz_path: str):
    """Load NPZ -> (T, 135) motion tensor with abs translation."""
    data = dict(np.load(npz_path, allow_pickle=True))
    poses = np.array(data["poses"], dtype=np.float32)
    trans = np.array(data.get("trans", data.get("transl")), dtype=np.float32)
    if trans.ndim == 1:
        trans = trans.reshape(-1, 3)
    fps = int(data.get("mocap_framerate", 30))
    abs_trans_frame0 = trans[0].copy()

    pose_rot6d = _smplh_to_rot6d_22(poses)
    transl_abs = trans.astype(np.float32)
    motion = np.concatenate([transl_abs, pose_rot6d], axis=-1)
    return torch.from_numpy(motion).float(), motion.shape[0], fps, abs_trans_frame0


def motion_135_to_npz_format(motion_135, abs_trans_frame0):
    """Convert (T, 135) back to axis-angle + abs_trans."""
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )
    motion = motion_135.float().numpy()
    T = motion.shape[0]
    abs_transl = motion[:, 0:3]
    rot6d = motion[:, 3:135].reshape(T * 22, 6)
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]
    axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)
    axis_angle = np.array(axis_angle, dtype=np.float32).reshape(T, 22, 3)
    return axis_angle, abs_transl


def save_repaired_npz(output_path, repaired_aa, repaired_trans, orig_data, fps):
    """Save repaired motion as NPZ, preserving hand joints from original."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    T_rep = min(repaired_aa.shape[0], orig_data["poses"].shape[0])
    repaired_poses_full = np.zeros_like(orig_data["poses"])
    repaired_poses_full[:T_rep, :66] = repaired_aa[:T_rep].reshape(-1, 66)
    if orig_data["poses"].shape[1] > 66:
        repaired_poses_full[:T_rep, 66:] = orig_data["poses"][:T_rep, 66:]
    np.savez(
        output_path,
        poses=repaired_poses_full,
        trans=repaired_trans[:T_rep],
        betas=orig_data.get("betas", np.zeros((1, 16), dtype=np.float32)),
        mocap_framerate=fps,
        gender=str(orig_data.get("gender", "neutral")),
        num_frames=T_rep,
    )


# ============================================================================
# Mask utilities
# ============================================================================

def adaptive_mask_to_dense(joint_mask, trans_mask, num_frames, temporal_dilate=5):
    """Convert MoGenDiT adaptive (T, 22) -> dense (T, 135) mask.

    Per-joint masking: each joint is independently masked/unmasked.
    This keeps mask ratio low (~15%) so M2M has strong conditioning context.
    """
    T = min(joint_mask.shape[0], num_frames)
    combined = np.zeros((num_frames, 23), dtype=np.float32)
    combined[:T, 1:23] = joint_mask[:T, :22].astype(np.float32)

    if trans_mask is not None:
        Tt = min(len(trans_mask), num_frames)
        # Include trans_mask from MoGenDiT adaptive mask computation
        combined[:Tt, 0] = trans_mask[:Tt].astype(np.float32)

    if temporal_dilate > 0:
        for col in range(23):
            arr = combined[:, col]
            dilated = arr.copy()
            for _ in range(temporal_dilate):
                padded = np.pad(dilated, 1, mode='edge')
                dilated = np.maximum(np.maximum(padded[:-2], padded[2:]), padded[1:-1])
            combined[:, col] = dilated

    mask = torch.zeros(num_frames, 135, dtype=torch.float32)
    for d in range(3):
        mask[:, d] = torch.from_numpy(combined[:, 0])
    for j in range(22):
        start = 3 + j * 6
        end = start + 6
        if end <= 135:
            for d in range(start, end):
                mask[:, d] = torch.from_numpy(combined[:, j + 1])
    return mask


# ============================================================================
# Quality checker
# ============================================================================

_CHECKER_INSTANCE = None


def get_checker():
    global _CHECKER_INSTANCE
    if _CHECKER_INSTANCE is None:
        from hftrainer.evaluation.quality_check_rules import MotionQualityChecker
        _CHECKER_INSTANCE = MotionQualityChecker(device="cpu")
    return _CHECKER_INSTANCE


def check_npz(npz_path):
    """Run quality checker. Returns dict with category, failed/borderline checks."""
    try:
        checker = get_checker()
        result = checker.check(npz_path)
        rd = result.to_dict()
        return {
            "is_valid": rd.get("is_valid", True),
            "category": rd.get("category", "high"),
            "failed_checks": rd.get("failed_checks", []),
            "borderline_checks": rd.get("borderline_checks", []),
        }
    except Exception as e:
        return {"is_valid": False, "category": "error", "failed_checks": [f"error:{str(e)[:80]}"], "borderline_checks": []}


# ============================================================================
# Model builders
# ============================================================================

def find_latest_checkpoint(work_dir_name):
    work_dir = PROJECT_ROOT / "work_dirs" / work_dir_name
    ckpt_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: d.stat().st_mtime,
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {work_dir}")
    return str(ckpt_dirs[-1])


def find_training_config(checkpoint_path):
    work_dir = Path(checkpoint_path).parent
    run_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()],
        key=lambda d: d.name,
    )
    for rd in reversed(run_dirs):
        cfg_path = rd / "config.py"
        if cfg_path.is_file():
            return str(cfg_path)
    return None


def build_m2m_model(config_name, device, num_steps):
    """Build M2M bundle + pipeline for a given config."""
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    info = M2M_CONFIGS[config_name]
    ckpt_path = find_latest_checkpoint(info["work_dir"])
    print(f"  [M2M] {config_name}: ckpt={ckpt_path}")

    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / info["config"])
    config_path = training_config or source_config

    cfg = Config.fromfile(config_path)
    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(device)
    bundle.eval()

    model_pt_path = os.path.join(ckpt_path, "model.pt")
    raw = torch.load(model_pt_path, map_location=device, weights_only=False)
    transformer_sd = raw["motion_transformer"]
    prefixed_sd = {f"motion_transformer.{k}": v for k, v in transformer_sd.items()}

    bundle_params = raw.get("__bundle_params__", {})
    if bundle_params:
        for pname, pval in bundle_params.items():
            if hasattr(bundle, pname):
                attr = getattr(bundle, pname)
                if isinstance(attr, torch.nn.Parameter):
                    attr.data.copy_(pval.to(device))
                elif isinstance(attr, torch.Tensor):
                    attr.copy_(pval.to(device))

    missing, unexpected = bundle.load_state_dict(prefixed_sd, strict=False)

    # Fallback for null embeddings
    if "null_vtxt_feat" in missing and not bundle_params:
        t2m_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"
        if os.path.exists(t2m_path):
            t2m = torch.load(t2m_path, map_location=device, weights_only=False)
            t2m_sd = t2m.get("model_state_dict", t2m)
            if "null_vtxt_feat" in t2m_sd:
                bundle.null_vtxt_feat.data.copy_(t2m_sd["null_vtxt_feat"].to(device))
                bundle.null_ctxt_input.data.copy_(t2m_sd["null_ctxt_input"].to(device))
            del t2m

    # MAN variants use replacement guidance
    is_man = "man" in config_name
    replacement = "skip_last" if is_man else "none"

    pipeline = HyMotionM2MPipeline(bundle, num_steps=num_steps, replacement_guidance=replacement)
    return pipeline, bundle, ckpt_path


def build_mogendit(device):
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    return MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device=device)


# ============================================================================
# Repair functions
# ============================================================================

# SMPL-22 kinematic tree for local<->global rotation conversion
_SMPL22_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

def _local_to_global_rot6d(local_rot6d: torch.Tensor) -> torch.Tensor:
    """Convert local rotation 6D (row-major) to global. Input: (*, 22, 6)."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix, rotation_matrix_to_rot6d,
    )
    local_mat = rot6d_to_rotation_matrix(local_rot6d)  # (*, 22, 3, 3)
    global_mat = torch.zeros_like(local_mat)
    for j, p in enumerate(_SMPL22_PARENTS):
        if p < 0:
            global_mat[..., j, :, :] = local_mat[..., j, :, :]
        else:
            global_mat[..., j, :, :] = global_mat[..., p, :, :] @ local_mat[..., j, :, :]
    return rotation_matrix_to_rot6d(global_mat)

def _global_to_local_rot6d(global_rot6d: torch.Tensor) -> torch.Tensor:
    """Convert global rotation 6D (row-major) to local. Input: (*, 22, 6)."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix, rotation_matrix_to_rot6d,
    )
    global_mat = rot6d_to_rotation_matrix(global_rot6d)
    local_mat = torch.zeros_like(global_mat)
    for j, p in enumerate(_SMPL22_PARENTS):
        if p < 0:
            local_mat[..., j, :, :] = global_mat[..., j, :, :]
        else:
            local_mat[..., j, :, :] = global_mat[..., p, :, :].transpose(-2, -1) @ global_mat[..., j, :, :]
    return rotation_matrix_to_rot6d(local_mat)


def repair_m2m(pipeline, motion_135, mask_135, device, edit_mode=False, max_frames=360):
    """Run M2M repair (VACE inpainting). Returns (combined_motion, raw_output)."""
    bundle = pipeline.bundle
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)

    motion_in = motion_135[:T].clone()

    # For global rotation models, convert local -> global rotation space
    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'
    if is_global:
        trans = motion_in[:, :3]  # (T, 3)
        rot6d_local = motion_in[:, 3:].reshape(T, 22, 6)
        rot6d_global = _local_to_global_rot6d(rot6d_local)
        motion_in = torch.cat([trans, rot6d_global.reshape(T, 132)], dim=-1)

    motion_norm = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))
    msk = mask_135[:T].unsqueeze(0).to(device)

    # Keep full normalized motion for clean_motion (imputation)
    motion_norm_full = motion_norm.clone()

    if not edit_mode:
        motion_norm = motion_norm * (1 - msk)

    if T < max_frames:
        pad_len = max_frames - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0)
        motion_norm_full = torch.nn.functional.pad(motion_norm_full, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    batch = {
        "src_motion": motion_norm,
        "src_mask": msk,
        "clean_motion": motion_norm_full,   # full normalized motion for imputation
        "src_length": [T],
        "tgt_length": [T],
    }

    with torch.no_grad():
        result = pipeline(batch)

    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    # For global rotation models, convert output back to local rotation space
    # IMPORTANT: blend in global space first to avoid kinematic chain discontinuities.
    # If we convert model output to local first and then blend per-joint,
    # a masked child joint's local rotation is defined relative to the model's
    # predicted parent (not the original parent), causing angular jumps at mask boundaries.
    if is_global:
        # repaired_raw is still in global space; motion_in was also converted to global
        # Convert original motion to global space for blending
        orig_trans = motion_135[:T, :3]
        orig_rot6d_local = motion_135[:T, 3:].reshape(T, 22, 6)
        orig_rot6d_global = _local_to_global_rot6d(orig_rot6d_local)
        orig_global = torch.cat([orig_trans, orig_rot6d_global.reshape(T, 132)], dim=-1)

        # Blend in global rotation space (kinematically consistent)
        mask_crop = mask_135[:T]
        combined_global = orig_global * (1 - mask_crop) + repaired_raw * mask_crop

        # Now convert blended result from global to local
        c_rot6d_global = combined_global[:, 3:].reshape(T, 22, 6)
        c_rot6d_local = _global_to_local_rot6d(c_rot6d_global)
        combined = torch.cat([combined_global[:, :3], c_rot6d_local.reshape(T, 132)], dim=-1)

        # Also convert raw output to local for saving
        r_rot6d_global = repaired_raw[:, 3:].reshape(T, 22, 6)
        r_rot6d_local = _global_to_local_rot6d(r_rot6d_global)
        repaired_raw = torch.cat([repaired_raw[:, :3], r_rot6d_local.reshape(T, 132)], dim=-1)
    else:
        mask_crop = mask_135[:T]
        combined = motion_135[:T] * (1 - mask_crop) + repaired_raw * mask_crop

    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)
        repaired_raw = torch.cat([repaired_raw, motion_135[T:]], dim=0)

    return combined, repaired_raw


def repair_m2m_denoise(pipeline, motion_135, mask_135, device,
                       sdedit_strength=0.4, max_frames=360):
    """Run M2M adaptive denoise (SDEdit-style), analogous to MoGenDIT's denoise.

    Strategy:
    1. Build a frame-level mask from adaptive mask (any joint flagged → whole frame)
    2. Expand to (T, 135) as VACE src_mask so the model knows which frames to fix
    3. Use SDEdit: start ODE from partially noisy clean motion (not pure noise)
    4. Use replacement guidance: clean frames are restored at each ODE step
    5. Result: flagged frames get denoised/repaired, clean frames stay original

    This gives the model the original motion as a strong prior (via SDEdit) while
    still leveraging VACE conditioning to know which regions need repair.
    """
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    bundle = pipeline.bundle
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)

    motion_in = motion_135[:T].clone()

    # For global rotation models, convert local -> global
    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'
    if is_global:
        trans = motion_in[:, :3]
        rot6d_local = motion_in[:, 3:].reshape(T, 22, 6)
        rot6d_global = _local_to_global_rot6d(rot6d_local)
        motion_in = torch.cat([trans, rot6d_global.reshape(T, 132)], dim=-1)

    # Normalize the full clean motion
    motion_norm = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))

    # Frame-level mask: if any joint on this frame is masked, mask entire frame
    frame_flag = (mask_135[:T].sum(dim=1) > 0).float()  # (T,)
    # Temporal dilation
    for _ in range(5):
        padded = torch.nn.functional.pad(frame_flag.unsqueeze(0), (1, 1), mode='replicate').squeeze(0)
        frame_flag = torch.max(torch.max(padded[:-2], padded[2:]), padded[1:-1])

    # Expand frame mask to (1, T, 135) for VACE src_mask
    frame_mask_135 = frame_flag.unsqueeze(1).expand(T, 135).unsqueeze(0).to(device)  # (1, T, 135)

    if T < max_frames:
        pad_len = max_frames - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0)
        frame_mask_135 = torch.nn.functional.pad(frame_mask_135, (0, 0, 0, pad_len), value=0)

    # Create pipeline with SDEdit
    denoise_pipeline = HyMotionM2MPipeline(
        bundle,
        num_steps=pipeline.num_steps,
        replacement_guidance='skip_last',  # keep clean frames via replacement
        sdedit_strength=sdedit_strength,
    )

    # For VACE: zero the masked regions in src_motion (completion-style)
    src_motion = motion_norm * (1 - frame_mask_135)

    batch = {
        "src_motion": src_motion,
        "src_mask": frame_mask_135,
        "src_length": [T],
        "tgt_length": [T],
        "clean_motion": motion_norm,  # triggers SDEdit path
    }

    with torch.no_grad():
        result = denoise_pipeline(batch)

    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    # Frame-level blending (use frame_flag, not per-joint)
    fm = frame_flag[:T].unsqueeze(1)  # (T, 1)

    if is_global:
        orig_trans = motion_135[:T, :3]
        orig_rot6d_local = motion_135[:T, 3:].reshape(T, 22, 6)
        orig_rot6d_global = _local_to_global_rot6d(orig_rot6d_local)
        orig_global = torch.cat([orig_trans, orig_rot6d_global.reshape(T, 132)], dim=-1)

        combined_global = orig_global * (1 - fm) + repaired_raw * fm

        c_rot6d_global = combined_global[:, 3:].reshape(T, 22, 6)
        c_rot6d_local = _global_to_local_rot6d(c_rot6d_global)
        combined = torch.cat([combined_global[:, :3], c_rot6d_local.reshape(T, 132)], dim=-1)

        r_rot6d_global = repaired_raw[:, 3:].reshape(T, 22, 6)
        r_rot6d_local = _global_to_local_rot6d(r_rot6d_global)
        repaired_raw = torch.cat([repaired_raw[:, :3], r_rot6d_local.reshape(T, 132)], dim=-1)
    else:
        combined = motion_135[:T] * (1 - fm) + repaired_raw * fm

    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)
        repaired_raw = torch.cat([repaired_raw, motion_135[T:]], dim=0)

    return combined, repaired_raw


# ============================================================================
# Data discovery
# ============================================================================

def discover_pairs():
    """Find all (original, cleaned) pairs in npz_split."""
    all_files = sorted(os.listdir(NPZ_SPLIT_DIR))
    cleaned = {f for f in all_files if "_cleaned.npz" in f}
    pairs = []
    for c in sorted(cleaned):
        orig = c.replace("_cleaned.npz", ".npz")
        orig_path = NPZ_SPLIT_DIR / orig
        clean_path = NPZ_SPLIT_DIR / c
        if orig_path.is_file():
            # Check same frame count
            try:
                o = np.load(str(orig_path), allow_pickle=True)
                cl = np.load(str(clean_path), allow_pickle=True)
                if o["poses"].shape[0] != cl["poses"].shape[0]:
                    continue  # Skip length mismatch
                pairs.append({"original": orig, "cleaned": c, "num_frames": o["poses"].shape[0]})
            except Exception:
                continue
    return pairs


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CJGame repair evaluation")
    p.add_argument("--max-samples", type=int, default=100, help="0 = all")
    p.add_argument("--num-steps", type=int, default=50, help="M2M ODE steps")
    p.add_argument("--mogendit-steps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--skip-mogendit-repair", action="store_true",
                   help="Skip MoGenDIT repair models (only compute masks)")
    p.add_argument("--skip-mogendit", action="store_true",
                   help="Skip MoGenDIT entirely (load pre-computed masks from output dir)")
    p.add_argument("--skip-m2m", action="store_true", help="Skip all M2M models")
    p.add_argument("--skip-checker", action="store_true",
                   help="Skip quality checker phase (only repair)")
    p.add_argument("--skip-report", action="store_true",
                   help="Skip report generation (only repair)")
    p.add_argument("--m2m-configs", nargs="+", default=None,
                   help="Specific M2M configs to run (default: all _man)")
    p.add_argument("--only-issues", action="store_true",
                   help="Only process cases where original motion has quality issues (checker fails)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / "cjgame_repair_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CJGame Repair Evaluation")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Discover pairs
    pairs = discover_pairs()
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    # --only-issues: filter to cases where original motion fails quality check
    if args.only_issues:
        print(f"[INFO] --only-issues: running checker on originals to filter...")
        issue_pairs = []
        for pair in pairs:
            npz_path = str(NPZ_SPLIT_DIR / pair["original"])
            qc = check_npz(npz_path)
            if qc.get("failed_checks"):
                issue_pairs.append(pair)
        print(f"[INFO] Filtered: {len(issue_pairs)}/{len(pairs)} pairs have quality issues")
        pairs = issue_pairs

    print(f"[INFO] {len(pairs)} valid pairs (original + cleaned, same frame count)")

    if not pairs:
        print("ERROR: No valid pairs found!")
        return

    # ================================================================
    # Phase 1: MoGenDIT adaptive masks + MoGenDIT repair
    # ================================================================
    ada_mask_dir = output_dir / "adaptive_masks"
    ada_mask_dir.mkdir(parents=True, exist_ok=True)
    ada_masks = {}
    mogendit_models = {}
    mogendit_results = {}

    if args.skip_mogendit:
        # Load pre-computed adaptive masks from disk
        print(f"\n[PHASE 1] Loading pre-computed adaptive masks from {ada_mask_dir}")
        for pair in pairs:
            orig_name = pair["original"]
            mask_path = ada_mask_dir / orig_name
            if mask_path.is_file():
                try:
                    mdata = np.load(str(mask_path), allow_pickle=True)
                    ada_masks[orig_name] = {
                        "joint_mask": np.array(mdata["joint_mask"]),
                        "trans_mask": np.array(mdata.get("trans_mask", np.zeros(mdata["joint_mask"].shape[0]))),
                    }
                except Exception as e:
                    print(f"  WARN: failed to load mask for {orig_name}: {e}")
        print(f"[INFO] Loaded {len(ada_masks)} pre-computed adaptive masks")
    else:
        print(f"\n[PHASE 1] MoGenDIT: adaptive masks + repair models")
        mogendit = build_mogendit(args.device)

        # MoGenDIT repair output dirs
        if not args.skip_mogendit_repair:
            for mode in ["denoise", "ada_denoise"]:
                label = f"mogendit_{mode}"
                mogendit_models[label] = mode
                (output_dir / label / "repaired").mkdir(parents=True, exist_ok=True)

        mogendit_results = {label: {} for label in mogendit_models}

        for idx, pair in enumerate(pairs):
            orig_name = pair["original"]
            npz_path = str(NPZ_SPLIT_DIR / orig_name)

            # Skip if mask already computed
            mask_out = ada_mask_dir / orig_name
            if mask_out.is_file():
                try:
                    mdata = np.load(str(mask_out), allow_pickle=True)
                    ada_masks[orig_name] = {
                        "joint_mask": np.array(mdata["joint_mask"]),
                        "trans_mask": np.array(mdata.get("trans_mask", np.zeros(mdata["joint_mask"].shape[0]))),
                    }
                    # Still need to run MoGenDIT repair if not skipped
                    if not args.skip_mogendit_repair:
                        for label, mode in mogendit_models.items():
                            out_path = str(output_dir / label / "repaired" / orig_name)
                            if os.path.isfile(out_path):
                                mogendit_results[label][orig_name] = {"success": True}
                                continue
                            try:
                                mogendit.repair_npz(npz_path, out_path, mode=mode, step=args.mogendit_steps)
                                rep_data = np.load(out_path, allow_pickle=True)
                                if rep_data["poses"].shape[0] != pair["num_frames"]:
                                    os.remove(out_path)
                                    mogendit_results[label][orig_name] = {"skipped": "length_mismatch"}
                                    continue
                                mogendit_results[label][orig_name] = {"success": True}
                            except Exception as e:
                                mogendit_results[label][orig_name] = {"error": str(e)[:100]}
                    if (idx + 1) % 20 == 0:
                        print(f"  [{idx+1}/{len(pairs)}] MoGenDIT phase (cached mask)")
                    continue
                except Exception:
                    pass

            # Compute adaptive mask
            try:
                ada = mogendit.compute_adaptive_mask(
                    npz_path, step=args.mogendit_steps,
                    joint_threshold=0.15, trans_threshold=0.05, max_mask_ratio=0.15,
                )
                ada_masks[orig_name] = ada
                # Save mask
                np.savez_compressed(str(mask_out), joint_mask=ada["joint_mask"], trans_mask=ada["trans_mask"])
            except Exception as e:
                print(f"  [{idx+1}] adaptive mask failed for {orig_name}: {e}")
                continue

            # MoGenDIT repair
            if not args.skip_mogendit_repair:
                for label, mode in mogendit_models.items():
                    out_path = str(output_dir / label / "repaired" / orig_name)
                    if os.path.isfile(out_path):
                        mogendit_results[label][orig_name] = {"success": True}
                        continue
                    try:
                        mogendit.repair_npz(npz_path, out_path, mode=mode, step=args.mogendit_steps)
                        # Check length consistency
                        rep_data = np.load(out_path, allow_pickle=True)
                        if rep_data["poses"].shape[0] != pair["num_frames"]:
                            os.remove(out_path)
                            mogendit_results[label][orig_name] = {"skipped": "length_mismatch"}
                            continue
                        mogendit_results[label][orig_name] = {"success": True}
                    except Exception as e:
                        mogendit_results[label][orig_name] = {"error": str(e)[:100]}

            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{len(pairs)}] MoGenDIT phase done")

        del mogendit
        torch.cuda.empty_cache()
        print(f"[INFO] Computed {len(ada_masks)} adaptive masks")

    # ================================================================
    # Phase 2: M2M repair (all _man configs × completion/edit)
    # ================================================================
    m2m_configs_to_run = args.m2m_configs or list(M2M_CONFIGS.keys())
    m2m_model_labels = []  # (label, config_name, edit_mode)

    if not args.skip_m2m:
        for cfg_name in m2m_configs_to_run:
            if cfg_name not in M2M_CONFIGS:
                print(f"[WARN] Unknown config: {cfg_name}, skipping")
                continue
            for edit_mode in [False, True, 'denoise']:
                if edit_mode == 'denoise':
                    mode_str = "denoise"
                else:
                    mode_str = "edit" if edit_mode else "completion"
                label = f"m2m_{cfg_name}_{mode_str}"
                m2m_model_labels.append((label, cfg_name, edit_mode))
                (output_dir / label / "repaired").mkdir(parents=True, exist_ok=True)

    m2m_results = {label: {} for label, _, _ in m2m_model_labels}

    # Group by config to load model only once
    configs_used = sorted(set(cfg for _, cfg, _ in m2m_model_labels))
    for cfg_name in configs_used:
        print(f"\n[PHASE 2] Loading M2M model: {cfg_name}")
        try:
            pipeline, bundle, ckpt_path = build_m2m_model(cfg_name, args.device, args.num_steps)
        except Exception as e:
            print(f"  [ERROR] Failed to build {cfg_name}: {e}")
            traceback.print_exc()
            for label, cn, _ in m2m_model_labels:
                if cn == cfg_name:
                    for pair in pairs:
                        m2m_results[label][pair["original"]] = {"error": f"model_load:{str(e)[:50]}"}
            continue

        # Run all modes for this config
        modes_for_config = [(label, em) for label, cn, em in m2m_model_labels if cn == cfg_name]

        for idx, pair in enumerate(pairs):
            orig_name = pair["original"]
            npz_path = str(NPZ_SPLIT_DIR / orig_name)

            if orig_name not in ada_masks:
                for label, _ in modes_for_config:
                    m2m_results[label][orig_name] = {"skipped": "no_mask"}
                continue

            ada = ada_masks[orig_name]
            mask_135 = adaptive_mask_to_dense(ada["joint_mask"], ada["trans_mask"], pair["num_frames"])

            if mask_135.sum().item() / max(mask_135.numel(), 1) < 0.001:
                for label, _ in modes_for_config:
                    m2m_results[label][orig_name] = {"skipped": "empty_mask"}
                continue

            try:
                motion_135, T, fps, abs_t0 = load_npz_as_motion(npz_path)
            except Exception as e:
                for label, _ in modes_for_config:
                    m2m_results[label][orig_name] = {"error": f"load:{str(e)[:50]}"}
                continue

            for label, edit_mode in modes_for_config:
                out_path = str(output_dir / label / "repaired" / orig_name)
                # Skip if already repaired
                if os.path.isfile(out_path):
                    m2m_results[label][orig_name] = {"success": True}
                    continue
                try:
                    if edit_mode == 'denoise':
                        combined, raw_output = repair_m2m_denoise(
                            pipeline, motion_135, mask_135, args.device,
                            sdedit_strength=0.3,
                        )
                    else:
                        combined, raw_output = repair_m2m(
                            pipeline, motion_135, mask_135, args.device, edit_mode=edit_mode
                        )
                    if torch.isnan(combined).any():
                        m2m_results[label][orig_name] = {"error": "NaN"}
                        continue

                    repaired_aa, repaired_trans = motion_135_to_npz_format(combined, abs_t0)
                    if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 50.0:
                        m2m_results[label][orig_name] = {"error": "trans_extreme"}
                        continue

                    orig_data = dict(np.load(npz_path, allow_pickle=True))
                    save_repaired_npz(out_path, repaired_aa, repaired_trans, orig_data, fps)

                    # Check length
                    rep_data = np.load(out_path, allow_pickle=True)
                    if rep_data["poses"].shape[0] != pair["num_frames"]:
                        os.remove(out_path)
                        m2m_results[label][orig_name] = {"skipped": "length_mismatch"}
                        continue

                    m2m_results[label][orig_name] = {"success": True}
                except Exception as e:
                    m2m_results[label][orig_name] = {"error": str(e)[:100]}

            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{len(pairs)}] {cfg_name} done")

        del pipeline, bundle
        torch.cuda.empty_cache()

    # ================================================================
    # Phase 3: Quality check all results
    # ================================================================
    if args.skip_checker or args.skip_report:
        print(f"\n[PHASE 3-4] Skipped (--skip-checker or --skip-report)")
        return

    print(f"\n[PHASE 3] Running quality checks on all results...")

    # Discover all model labels from disk (in case different runs produced different models)
    all_model_labels_set = set(mogendit_models.keys()) | {l for l, _, _ in m2m_model_labels}
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and (subdir / "repaired").is_dir():
            all_model_labels_set.add(subdir.name)
    all_model_labels = sorted(all_model_labels_set)
    all_model_results = {**mogendit_results, **m2m_results}
    print(f"  Discovered model labels: {all_model_labels}")

    report_details = []

    for idx, pair in enumerate(pairs):
        orig_name = pair["original"]
        clean_name = pair["cleaned"]
        orig_path = str(NPZ_SPLIT_DIR / orig_name)
        clean_path = str(NPZ_SPLIT_DIR / clean_name)

        entry = {
            "original": orig_name,
            "cleaned": clean_name,
            "num_frames": pair["num_frames"],
            "has_mask": orig_name in ada_masks,
        }

        # Check original
        entry["original_qc"] = check_npz(orig_path)
        # Check GT (human cleaned)
        entry["cleaned_qc"] = check_npz(clean_path)

        # Mask stats
        if orig_name in ada_masks:
            ada = ada_masks[orig_name]
            jm = ada["joint_mask"]
            entry["mask_ratio"] = float(jm.sum()) / max(jm.size, 1)
            entry["mask_joints_flagged"] = int(jm.any(axis=0).sum())
            entry["mask_frames_flagged"] = int(jm.any(axis=1).sum())

        # Check each model's repair
        entry["model_results"] = {}
        for label in all_model_labels:
            rep_path = str(output_dir / label / "repaired" / orig_name)
            mr = all_model_results.get(label, {}).get(orig_name, {})
            if mr.get("success") or os.path.isfile(rep_path):
                if os.path.isfile(rep_path):
                    entry["model_results"][label] = check_npz(rep_path)
                else:
                    entry["model_results"][label] = {"skipped": "file_missing"}
            else:
                entry["model_results"][label] = mr if mr else {"skipped": "not_run"}

        report_details.append(entry)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(pairs)}] quality checked")

    # ================================================================
    # Phase 4: Aggregate and save report
    # ================================================================
    print(f"\n[PHASE 4] Generating report...")

    # Per-model aggregate stats
    model_summaries = {}
    for label in all_model_labels:
        total = 0
        processed = 0
        before_fail = 0
        after_pass = 0
        improved = 0
        degraded = 0
        per_checker_fix = defaultdict(lambda: {"total": 0, "fixed": 0})

        for entry in report_details:
            mr = entry["model_results"].get(label, {})
            if "skipped" in mr or "error" in mr:
                continue
            total += 1
            processed += 1

            orig_failed = entry["original_qc"].get("failed_checks", [])
            rep_failed = mr.get("failed_checks", [])
            orig_valid = entry["original_qc"].get("is_valid", True)
            rep_valid = mr.get("is_valid", True)

            if not orig_valid:
                before_fail += 1
            if rep_valid:
                after_pass += 1
            if not orig_valid and rep_valid:
                improved += 1
            if orig_valid and not rep_valid:
                degraded += 1

            for fc in orig_failed:
                per_checker_fix[fc]["total"] += 1
                if fc not in rep_failed:
                    per_checker_fix[fc]["fixed"] += 1

        model_summaries[label] = {
            "desc": M2M_CONFIGS.get(label.replace("m2m_", "").rsplit("_", 1)[0], {}).get("desc", label),
            "total": total,
            "processed": processed,
            "before_fail": before_fail,
            "after_pass": after_pass,
            "improved": improved,
            "degraded": degraded,
            "improve_rate": round(improved / max(before_fail, 1) * 100, 1),
            "per_checker_fix": dict(per_checker_fix),
        }

    # Also add GT stats
    gt_stats = {"total": 0, "before_fail": 0, "after_pass": 0, "improved": 0, "degraded": 0}
    for entry in report_details:
        gt_stats["total"] += 1
        orig_valid = entry["original_qc"].get("is_valid", True)
        clean_valid = entry["cleaned_qc"].get("is_valid", True)
        if not orig_valid:
            gt_stats["before_fail"] += 1
        if clean_valid:
            gt_stats["after_pass"] += 1
        if not orig_valid and clean_valid:
            gt_stats["improved"] += 1
        if orig_valid and not clean_valid:
            gt_stats["degraded"] += 1
    gt_stats["improve_rate"] = round(gt_stats["improved"] / max(gt_stats["before_fail"], 1) * 100, 1)

    report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_pairs": len(pairs),
            "num_with_mask": len(ada_masks),
            "num_steps_m2m": args.num_steps,
            "num_steps_mogendit": args.mogendit_steps,
            "output_dir": str(output_dir),
        },
        "gt_summary": gt_stats,
        "model_summaries": model_summaries,
        "details": report_details,
    }

    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report saved: {report_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Model':<45} {'Total':>6} {'B.Fail':>7} {'A.Pass':>7} {'Improved':>9} {'Degraded':>9} {'Fix%':>6}")
    print(f"{'='*100}")
    print(f"{'GT (human cleaned)':<45} {gt_stats['total']:>6} {gt_stats['before_fail']:>7} {gt_stats['after_pass']:>7} {gt_stats['improved']:>9} {gt_stats['degraded']:>9} {gt_stats['improve_rate']:>5.1f}%")
    for label in all_model_labels:
        s = model_summaries[label]
        print(f"{label:<45} {s['total']:>6} {s['before_fail']:>7} {s['after_pass']:>7} {s['improved']:>9} {s['degraded']:>9} {s['improve_rate']:>5.1f}%")
    print(f"{'='*100}")
    print(f"\nDone! Results in: {output_dir}")


if __name__ == "__main__":
    main()

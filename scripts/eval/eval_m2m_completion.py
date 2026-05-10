#!/usr/bin/env python3
"""Comprehensive M2M Completion Evaluation Script.

Evaluates all 8 MAN model variants on 3 completion tasks:
  T1 Transition:  head 30f (keep) | middle (generate) | tail 30f (keep)
  T2 In-between:  first 30f (keep) | middle (generate) | last 30f (keep)
  T6 Joint Edit:  lower body (keep) | upper body (generate)

Uses test data from data/eval/hymotion_m2m/eval_transition.json (game motions).
Output is compatible with motion_annot_web/completion_apps viewer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_completion.py --max-samples 100
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_completion.py --max-samples 20 --num-steps 10 --tasks transition
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_completion.py --m2m-configs uncond_fm_man caption_fm_man
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
# Block heavy transitive imports (same pattern as eval_cjgame_repair.py)
# ============================================================================
import types as _types
_dummy_modules = [
    'hftrainer.models',
    'hftrainer.models.motion',
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

# ============================================================================
# Model config registry — all 8 MAN variants
# ============================================================================
M2M_CONFIGS = {
    "uncond_fm_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_fm_man_046b",
        "desc": "Uncond FM MAN (local rot)",
        "needs_text": False,
    },
    "uncond_jit_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_jit_man_046b",
        "desc": "Uncond JiT MAN (local rot)",
        "needs_text": False,
    },
    "caption_fm_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_fm_man_046b",
        "desc": "Caption FM MAN (local rot)",
        "needs_text": True,
    },
    "caption_jit_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_jit_man_046b",
        "desc": "Caption JiT MAN (local rot)",
        "needs_text": True,
    },
    "uncond_fm_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_fm_man_globalrot_046b",
        "desc": "Uncond FM MAN (global rot)",
        "needs_text": False,
    },
    "uncond_jit_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_jit_man_globalrot_046b",
        "desc": "Uncond JiT MAN (global rot)",
        "needs_text": False,
    },
    "caption_fm_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_fm_man_globalrot_046b",
        "desc": "Caption FM MAN (global rot)",
        "needs_text": True,
    },
    "caption_jit_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_caption_jit_man_globalrot_046b",
        "desc": "Caption JiT MAN (global rot)",
        "needs_text": True,
    },
}

# ============================================================================
# Task definitions (3 tasks, each producing output in completion_apps format)
# ============================================================================
TASK_DEFINITIONS = {
    "transition": {
        "app_name": "eval_transition",
        "description": "T1 Transition: head + tail kept, generate middle",
        "head_f": 30,
        "tail_f": 30,
        "min_frames": 90,  # need at least head + 30 gen + tail
    },
    "inbetween": {
        "app_name": "eval_inbetween",
        "description": "T2 In-between: first/last 30 frames kept, generate middle",
        "head_f": 30,
        "tail_f": 30,
        "min_frames": 90,
    },
    "joint_edit": {
        "app_name": "eval_joint_edit",
        "description": "T6 Joint Edit: lower body kept, regenerate upper body",
        "min_frames": 60,
    },
}

# SMPL-22 upper body joint indices (for joint edit task)
# Joints: 0=pelvis 1=l_hip 2=r_hip 3=spine1 4=l_knee 5=r_knee 6=spine2
# 7=l_ankle 8=r_ankle 9=spine3 10=l_foot 11=r_foot 12=neck
# 13=l_collar 14=r_collar 15=head 16=l_shoulder 17=r_shoulder
# 18=l_elbow 19=r_elbow 20=l_wrist 21=r_wrist
UPPER_BODY_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

MAX_FRAME = 360
D = 135  # 3 abs transl + 22*6 rot6d

# SMPL-22 kinematic tree
_SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


# ============================================================================
# Motion utilities (from eval_cjgame_repair.py)
# ============================================================================

def _smplh_to_rot6d_22(poses_aa: np.ndarray) -> np.ndarray:
    """Convert SMPL-H axis-angle (T,156) to row-major rot6d (T, 132)."""
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_rotation_6d,
    )
    if poses_aa.ndim == 2:
        n_joints = poses_aa.shape[1] // 3
        if n_joints == 52:
            poses_aa = np.concatenate(
                [poses_aa[:, :66], np.zeros((poses_aa.shape[0], 9), dtype=poses_aa.dtype), poses_aa[:, 66:]],
                axis=1,
            )
        poses_aa = poses_aa.reshape(poses_aa.shape[0], -1, 3)
    aa = poses_aa[:, :22, :]
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

    pose_rot6d = _smplh_to_rot6d_22(poses)
    transl_abs = trans.astype(np.float32)
    motion = np.concatenate([transl_abs, pose_rot6d], axis=-1)
    return torch.from_numpy(motion).float(), motion.shape[0], fps, data


def motion_135_to_npz(motion_135, orig_data, output_path, fps=30):
    """Convert (T, 135) back to axis-angle NPZ and save."""
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )
    motion = motion_135.float().numpy()
    T = motion.shape[0]
    abs_transl = motion[:, 0:3]
    rot6d = motion[:, 3:135].reshape(T * 22, 6)
    # row-major -> column-major
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]
    axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)
    axis_angle = np.array(axis_angle, dtype=np.float32).reshape(T, 22, 3)

    # Build full poses array preserving hand joints
    orig_poses = np.array(orig_data.get("poses", np.zeros((T, 156))), dtype=np.float32)
    T_save = min(T, orig_poses.shape[0])
    full_poses = np.zeros((T, orig_poses.shape[1]), dtype=np.float32)
    full_poses[:T_save, :66] = axis_angle[:T_save].reshape(-1, 66)
    if orig_poses.shape[1] > 66:
        full_poses[:T_save, 66:] = orig_poses[:T_save, 66:]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        poses=full_poses[:T],
        trans=abs_transl[:T],
        betas=orig_data.get("betas", np.zeros((1, 16), dtype=np.float32)),
        mocap_framerate=fps,
        gender=str(orig_data.get("gender", "neutral")),
        num_frames=T,
    )


# ============================================================================
# Global <-> Local rotation conversion
# ============================================================================

def _local_to_global_rot6d(local_rot6d: torch.Tensor) -> torch.Tensor:
    """Convert local rotation 6D (row-major) to global. Input: (*, 22, 6)."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix, rotation_matrix_to_rot6d,
    )
    local_mat = rot6d_to_rotation_matrix(local_rot6d)
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


# ============================================================================
# Mask builders for each task
# ============================================================================

def build_transition_mask(num_frames, head_f=30, tail_f=30):
    """T1 Transition mask: keep head and tail, generate middle."""
    mask = torch.zeros(num_frames, D, dtype=torch.float32)
    gen_start = head_f
    gen_end = num_frames - tail_f
    if gen_end <= gen_start:
        gen_end = gen_start + 1
    mask[gen_start:gen_end, :] = 1.0
    return mask


def build_inbetween_mask(num_frames, head_f=30, tail_f=30):
    """T2 In-between mask: same structure as transition."""
    return build_transition_mask(num_frames, head_f, tail_f)


def build_joint_edit_mask(num_frames):
    """T6 Joint Edit mask: upper body joints are masked (generate), lower body kept."""
    mask = torch.zeros(num_frames, D, dtype=torch.float32)
    for j in UPPER_BODY_JOINTS:
        start = 3 + j * 6
        end = start + 6
        mask[:, start:end] = 1.0
    return mask


# ============================================================================
# Metrics
# ============================================================================

def compute_mpjpe(pred_135, gt_135, mask_135=None):
    """Compute MPJPE (mm) between two (T, 135) motions.

    Only computes on the generated region if mask is provided.
    Uses translation dims directly as proxy for joint positions.
    """
    if pred_135.shape[0] != gt_135.shape[0]:
        T = min(pred_135.shape[0], gt_135.shape[0])
        pred_135 = pred_135[:T]
        gt_135 = gt_135[:T]
        if mask_135 is not None:
            mask_135 = mask_135[:T]

    # Use translation (first 3 dims) as root position error
    trans_err = (pred_135[:, :3] - gt_135[:, :3]).norm(dim=-1)  # (T,)

    # Per-joint rotation error (as rotation distance proxy)
    rot_err = (pred_135[:, 3:] - gt_135[:, 3:]).reshape(-1, 22, 6)
    rot_err_per_joint = rot_err.norm(dim=-1)  # (T, 22)

    if mask_135 is not None:
        frame_mask = mask_135.mean(dim=-1) > 0  # (T,) bool
        if frame_mask.sum() > 0:
            trans_err = trans_err[frame_mask].mean().item()
            rot_err_mean = rot_err_per_joint[frame_mask].mean().item()
        else:
            trans_err = trans_err.mean().item()
            rot_err_mean = rot_err_per_joint.mean().item()
    else:
        trans_err = trans_err.mean().item()
        rot_err_mean = rot_err_per_joint.mean().item()

    return {
        "trans_err_mm": round(trans_err * 1000, 2),
        "rot_err": round(rot_err_mean, 6),
    }


def compute_boundary_smoothness(pred_135, mask_135, window=3):
    """Compute boundary smoothness at mask transitions."""
    frame_mask = mask_135.mean(dim=-1)  # (T,)
    T = frame_mask.shape[0]

    # Find boundary frames
    boundaries = []
    for t in range(1, T):
        if (frame_mask[t] > 0.5) != (frame_mask[t - 1] > 0.5):
            boundaries.append(t)

    if not boundaries:
        return {"boundary_jerk": 0.0, "num_boundaries": 0}

    # Compute jerk (2nd order difference) at boundaries
    jerks = []
    for b in boundaries:
        lo = max(0, b - window)
        hi = min(T, b + window)
        if hi - lo < 3:
            continue
        segment = pred_135[lo:hi]
        vel = segment[1:] - segment[:-1]
        acc = vel[1:] - vel[:-1]
        jerk = acc.norm(dim=-1).mean().item()
        jerks.append(jerk)

    return {
        "boundary_jerk": round(np.mean(jerks) if jerks else 0.0, 6),
        "num_boundaries": len(boundaries),
    }


def compute_foot_skating(pred_135, fps=30):
    """Compute foot skating metric (velocity when foot is on ground)."""
    # Foot joints: 7=l_ankle, 8=r_ankle, 10=l_foot, 11=r_foot
    # We use the translation as proxy (motion_135 only has rot6d, not FK positions)
    # So we compute velocity of the root translation as a simple proxy
    trans = pred_135[:, :3]
    if trans.shape[0] < 2:
        return {"foot_skating_proxy": 0.0}
    vel = (trans[1:] - trans[:-1]) * fps
    speed = vel.norm(dim=-1)
    return {"root_speed_mean": round(speed.mean().item(), 4)}


def compute_jitter(pred_135, fps=30):
    """Compute jitter (acceleration magnitude) of the motion."""
    if pred_135.shape[0] < 3:
        return {"jitter": 0.0}
    vel = (pred_135[1:] - pred_135[:-1]) * fps
    acc = (vel[1:] - vel[:-1]) * fps
    jitter = acc.norm(dim=-1).mean().item()
    return {"jitter": round(jitter, 4)}


# ============================================================================
# Model building (from eval_cjgame_repair.py pattern)
# ============================================================================

def find_latest_checkpoint(work_dir_name):
    work_dir = PROJECT_ROOT / "work_dirs" / work_dir_name
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")
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
    replacement = "skip_last"
    pipeline = HyMotionM2MPipeline(bundle, num_steps=num_steps, replacement_guidance=replacement)
    return pipeline, bundle, ckpt_path


# ============================================================================
# Inference core
# ============================================================================

def run_completion(pipeline, bundle, motion_135, mask_135, device, max_frames=MAX_FRAME):
    """Run M2M completion. Returns (combined_motion, raw_output)."""
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)

    motion_in = motion_135[:T].clone()

    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'
    if is_global:
        trans = motion_in[:, :3]
        rot6d_local = motion_in[:, 3:].reshape(T, 22, 6)
        rot6d_global = _local_to_global_rot6d(rot6d_local)
        motion_in = torch.cat([trans, rot6d_global.reshape(T, 132)], dim=-1)

    motion_norm = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))
    msk = mask_135[:T].unsqueeze(0).to(device)

    # Zero out masked regions in src_motion (VACE conditioning)
    motion_norm = motion_norm * (1 - msk)

    if T < max_frames:
        pad_len = max_frames - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    batch = {
        "src_motion": motion_norm,
        "src_mask": msk,
        "src_length": [T],
        "tgt_length": [T],
    }

    with torch.no_grad():
        result = pipeline(batch)

    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    if is_global:
        orig_trans = motion_135[:T, :3]
        orig_rot6d_local = motion_135[:T, 3:].reshape(T, 22, 6)
        orig_rot6d_global = _local_to_global_rot6d(orig_rot6d_local)
        orig_global = torch.cat([orig_trans, orig_rot6d_global.reshape(T, 132)], dim=-1)

        mask_crop = mask_135[:T]
        combined_global = orig_global * (1 - mask_crop) + repaired_raw * mask_crop

        c_rot6d_global = combined_global[:, 3:].reshape(T, 22, 6)
        c_rot6d_local = _global_to_local_rot6d(c_rot6d_global)
        combined = torch.cat([combined_global[:, :3], c_rot6d_local.reshape(T, 132)], dim=-1)

        r_rot6d_global = repaired_raw[:, 3:].reshape(T, 22, 6)
        r_rot6d_local = _global_to_local_rot6d(r_rot6d_global)
        repaired_raw = torch.cat([repaired_raw[:, :3], r_rot6d_local.reshape(T, 132)], dim=-1)
    else:
        mask_crop = mask_135[:T]
        combined = motion_135[:T] * (1 - mask_crop) + repaired_raw * mask_crop

    return combined, repaired_raw


# ============================================================================
# Data loading
# ============================================================================

DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data" / "3D" / "20251111" / "motions"
EVAL_DATALIST = PROJECT_ROOT / "data" / "eval" / "hymotion_m2m" / "eval_transition.json"


def load_eval_data(max_samples=100, min_frames=60, seed=42):
    """Load test samples from eval datalist."""
    with open(str(EVAL_DATALIST)) as f:
        data = json.load(f)

    items = data["data_list"]
    # Filter by min frames
    items = [it for it in items if it["num_frames"] >= min_frames]

    # Shuffle and take first N
    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    items = items[:max_samples]

    # Resolve full paths and verify existence
    valid = []
    for it in items:
        full_path = str(DATA_ROOT / it["motion_path"])
        if os.path.isfile(full_path):
            it["full_path"] = full_path
            valid.append(it)
        else:
            print(f"  WARN: missing {full_path}")

    print(f"[DATA] Loaded {len(valid)}/{max_samples} valid samples (min_frames={min_frames})")
    return valid


# ============================================================================
# Main evaluation loop
# ============================================================================

def run_task(task_name, task_def, samples, config_name, pipeline, bundle,
             device, output_dir, num_steps):
    """Run one task for one config on all samples. Returns metrics list."""
    app_name = task_def["app_name"]
    metrics_list = []
    errors = 0

    for idx, sample in enumerate(samples):
        case_id = f"case_{idx:03d}"
        case_dir = os.path.join(output_dir, app_name, case_id, config_name)
        meta_path = os.path.join(case_dir, "meta.json")

        # Skip if already done
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    existing = json.load(f)
                if existing.get("metrics"):
                    metrics_list.append(existing["metrics"])
                continue
            except Exception:
                pass

        try:
            motion_135, num_frames, fps, orig_data = load_npz_as_motion(sample["full_path"])

            T = min(num_frames, MAX_FRAME)
            motion_135 = motion_135[:T]

            # Build mask based on task
            if task_name == "transition":
                head_f = task_def["head_f"]
                tail_f = task_def["tail_f"]
                mask = build_transition_mask(T, head_f, tail_f)
            elif task_name == "inbetween":
                head_f = task_def["head_f"]
                tail_f = task_def["tail_f"]
                mask = build_inbetween_mask(T, head_f, tail_f)
            elif task_name == "joint_edit":
                mask = build_joint_edit_mask(T)
            else:
                raise ValueError(f"Unknown task: {task_name}")

            # Run completion
            combined, raw_output = run_completion(pipeline, bundle, motion_135, mask, device)

            # Compute metrics
            m = {}
            m.update(compute_mpjpe(combined, motion_135, mask))
            m.update(compute_boundary_smoothness(combined, mask))
            m.update(compute_jitter(combined, fps))
            m.update(compute_foot_skating(combined, fps))

            # Save output NPZ
            os.makedirs(case_dir, exist_ok=True)
            output_npz_path = os.path.join(case_dir, "output.npz")
            motion_135_to_npz(combined, orig_data, output_npz_path, fps)

            # Also save the GT NPZ for reference
            gt_npz_path = os.path.join(case_dir, "gt.npz")
            if not os.path.isfile(gt_npz_path):
                motion_135_to_npz(motion_135, orig_data, gt_npz_path, fps)

            # Save meta
            meta = {
                "task": task_name,
                "config": config_name,
                "config_desc": M2M_CONFIGS[config_name]["desc"],
                "motion_path": sample["motion_path"],
                "num_frames": T,
                "fps": fps,
                "num_steps": num_steps,
                "mask_ratio": float(mask.mean()),
                "metrics": m,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "npz_files": ["output.npz", "gt.npz"],
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

            metrics_list.append(m)

            if (idx + 1) % 10 == 0:
                print(f"    [{config_name}] {task_name}: {idx + 1}/{len(samples)} done")

        except Exception as e:
            errors += 1
            print(f"    ERROR case {idx}: {e}")
            traceback.print_exc()

    return metrics_list, errors


def aggregate_metrics(metrics_list):
    """Compute mean/std of metrics across samples."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            agg[k] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
                "n": len(vals),
            }
    return agg


def main():
    parser = argparse.ArgumentParser(description="M2M Completion Evaluation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=list(TASK_DEFINITIONS.keys()),
                        help="Tasks to run (default: all)")
    parser.add_argument("--m2m-configs", nargs="+", default=None,
                        help="Specific configs to run (default: all 8)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / "test" / "completion_apps")
    os.makedirs(output_dir, exist_ok=True)

    tasks = args.tasks or list(TASK_DEFINITIONS.keys())
    configs = args.m2m_configs or list(M2M_CONFIGS.keys())

    # Validate configs exist
    valid_configs = []
    for c in configs:
        if c not in M2M_CONFIGS:
            print(f"WARN: unknown config '{c}', skipping")
            continue
        try:
            find_latest_checkpoint(M2M_CONFIGS[c]["work_dir"])
            valid_configs.append(c)
        except FileNotFoundError as e:
            print(f"WARN: {c}: {e}, skipping")
    configs = valid_configs

    if not configs:
        print("ERROR: No valid configs found!")
        return

    print(f"\n{'=' * 70}")
    print(f"M2M Completion Evaluation")
    print(f"  Output:   {output_dir}")
    print(f"  Tasks:    {tasks}")
    print(f"  Configs:  {configs}")
    print(f"  Samples:  {args.max_samples}")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Device:   {args.device}")
    print(f"{'=' * 70}")

    # Load test data
    samples = load_eval_data(args.max_samples, min_frames=60, seed=args.seed)
    if not samples:
        print("ERROR: No valid test samples!")
        return

    device = torch.device(args.device)

    # Results summary
    all_results = {}
    total_start = time.time()

    for config_name in configs:
        print(f"\n{'=' * 60}")
        print(f"Loading model: {config_name} ({M2M_CONFIGS[config_name]['desc']})")
        print(f"{'=' * 60}")

        try:
            pipeline, bundle, ckpt_path = build_m2m_model(config_name, device, args.num_steps)
        except Exception as e:
            print(f"ERROR loading {config_name}: {e}")
            traceback.print_exc()
            continue

        config_results = {}
        for task_name in tasks:
            task_def = TASK_DEFINITIONS[task_name]
            min_frames = task_def.get("min_frames", 60)

            # Filter samples for this task
            task_samples = [s for s in samples if s["num_frames"] >= min_frames]
            if not task_samples:
                print(f"  No samples with >= {min_frames} frames for {task_name}")
                continue

            print(f"\n  --- {task_name}: {len(task_samples)} samples ---")
            t0 = time.time()

            metrics_list, errors = run_task(
                task_name, task_def, task_samples, config_name,
                pipeline, bundle, device, output_dir, args.num_steps,
            )

            elapsed = time.time() - t0
            agg = aggregate_metrics(metrics_list)
            config_results[task_name] = {
                "aggregated": agg,
                "num_samples": len(metrics_list),
                "num_errors": errors,
                "elapsed_sec": round(elapsed, 1),
            }

            print(f"    Completed in {elapsed:.1f}s ({len(metrics_list)} ok, {errors} errors)")
            for k, v in agg.items():
                print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

        all_results[config_name] = {
            "desc": M2M_CONFIGS[config_name]["desc"],
            "checkpoint": ckpt_path if 'ckpt_path' in dir() else "",
            "tasks": config_results,
        }

        # Free GPU memory
        del pipeline, bundle
        torch.cuda.empty_cache()

    # Save global report
    total_elapsed = time.time() - total_start
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_elapsed, 1),
        "num_steps": args.num_steps,
        "num_samples": args.max_samples,
        "tasks": tasks,
        "configs": configs,
        "results": all_results,
    }

    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'=' * 70}")
    print(f"Evaluation complete! Total time: {total_elapsed:.1f}s")
    print(f"Report: {report_path}")
    print(f"Results dir: {output_dir}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    for config_name, cr in all_results.items():
        print(f"\n{config_name} ({cr['desc']}):")
        for task_name, tr in cr.get("tasks", {}).items():
            agg = tr.get("aggregated", {})
            te = agg.get("trans_err_mm", {})
            re = agg.get("rot_err", {})
            bj = agg.get("boundary_jerk", {})
            ji = agg.get("jitter", {})
            print(f"  {task_name}: trans_err={te.get('mean', 'N/A')}mm  rot_err={re.get('mean', 'N/A')}  "
                  f"bnd_jerk={bj.get('mean', 'N/A')}  jitter={ji.get('mean', 'N/A')}  "
                  f"({tr['num_samples']} samples, {tr['elapsed_sec']}s)")


if __name__ == "__main__":
    main()

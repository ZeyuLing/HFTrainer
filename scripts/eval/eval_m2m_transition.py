#!/usr/bin/env python3
"""
Evaluate M2M transition: stitch pairs of DIFFERENT motions using HyMotion M2M.

Data source: T2M-generated motion clips from motion_gen_arena evaluation set.
Each pair consists of two different prompts; we take motion_A's tail and motion_B's
head as condition, and let M2M generate the transition in between.

Output structure:
    output/test/completion_apps/eval_transition/
        case_000/
            uncond_fm_man/
                output.npz      # full stitched: A_tail(C) + transition(G) + B_head(C)
                motion_a.npz    # original motion A (full)
                motion_b.npz    # original motion B (full)
                meta.json       # metadata + metrics
            uncond_jit_man/
                ...
        case_001/
            ...

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_transition.py --num-pairs 100 --num-steps 10
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_transition.py --num-pairs 100 --num-steps 10 \
        --configs uncond_fm_man uncond_fm_man_globalrot
"""

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data directory
MOTION_DIR = Path(
    "/apdcephfs_cq10/share_1467498/datasets/motion_gen_arena/evaluation_20251125/"
    "yiran_subset/sft_1210_o6dp1103_04k_qwen3_1B_NB_from3kckpt60_gpus128_e40/"
    "motions_smpl_npz"
)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "test" / "completion_apps"

# Motion representation
D = 135  # 3 transl + 22*6 rot6d
MAX_FRAME = 360

# Transition parameters
HEAD_F = 30   # frames from A's tail used as condition
TAIL_F = 30   # frames from B's head used as condition
TRANS_F = 60  # frames to generate as transition

# Only unconditioned configs (no caption)
UNCOND_CONFIGS = {
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
}

# SMPL-22 parent indices for FK
_SMPL22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]


# ============================================================================
# Rotation helpers (from eval_m2m_completion.py)
# ============================================================================

def _smplh_to_rot6d_22(poses_aa: np.ndarray) -> np.ndarray:
    """Convert SMPL-H axis-angle (T, 156) -> row-major rot6d (T, 132)."""
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_rotation_6d,
    )
    T = poses_aa.shape[0]
    body_aa = poses_aa[:, :66].reshape(T * 22, 3)
    rot6d_colmajor = np.array(axis_angle_to_rotation_6d(
        torch.from_numpy(body_aa.astype(np.float32))
    ), dtype=np.float32)
    # column-major -> row-major
    rot6d_rowmajor = rot6d_colmajor[:, [0, 3, 1, 4, 2, 5]]
    return rot6d_rowmajor.reshape(T, 132)


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
            local_mat[..., j, :, :] = (
                global_mat[..., p, :, :].transpose(-2, -1) @ global_mat[..., j, :, :]
            )
    return rotation_matrix_to_rot6d(local_mat)


# ============================================================================
# Data loading
# ============================================================================

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
    """Convert (T, 135) back to axis-angle NPZ and save.

    Applies temporal continuity correction to axis-angle representation
    to avoid π-discontinuities in the saved output.
    """
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )
    motion = motion_135.float().numpy() if isinstance(motion_135, torch.Tensor) else motion_135
    T = motion.shape[0]
    abs_transl = motion[:, 0:3]
    rot6d = motion[:, 3:135].reshape(T * 22, 6)
    # row-major -> column-major
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]
    axis_angle = rotation_6d_to_axis_angle(
        torch.from_numpy(rot6d_colmajor.astype(np.float32))
    )
    axis_angle = np.array(axis_angle, dtype=np.float32).reshape(T, 22, 3)

    # Fix axis-angle temporal discontinuities near π.
    # When rotation angle is close to π, the axis can flip sign between
    # adjacent frames, causing a ~2π jump in axis-angle representation
    # even though the actual rotation difference is tiny.
    # Fix: for each joint, if flipping the sign reduces the jump, flip it.
    for t in range(1, T):
        for j in range(22):
            aa_prev = axis_angle[t - 1, j]
            aa_curr = axis_angle[t, j]
            diff_normal = np.linalg.norm(aa_curr - aa_prev)
            # Try: negate (equivalent rotation for angle near π: -axis, 2π-angle)
            angle = np.linalg.norm(aa_curr)
            if angle > 1e-6:
                axis = aa_curr / angle
                alt_angle = 2 * np.pi - angle
                alt_aa = -axis * alt_angle
                diff_alt = np.linalg.norm(alt_aa - aa_prev)
                if diff_alt < diff_normal:
                    axis_angle[t, j] = alt_aa

    orig_poses = np.array(orig_data.get("poses", np.zeros((T, 156))), dtype=np.float32)
    pose_dim = orig_poses.shape[1] if orig_poses.ndim > 1 else 156
    full_poses = np.zeros((T, pose_dim), dtype=np.float32)
    full_poses[:, :66] = axis_angle.reshape(-1, 66)
    # Copy hand joints from orig_data for overlapping frames
    T_orig = min(T, orig_poses.shape[0])
    if orig_poses.ndim > 1 and orig_poses.shape[1] > 66:
        full_poses[:T_orig, 66:] = orig_poses[:T_orig, 66:]

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


def load_motion_pairs(num_pairs=100, seed=42, min_frames=60):
    """Sample pairs of motions from different prompts.

    Returns list of dicts:
        {"a_path", "b_path", "a_text", "b_text", "a_prompt_id", "b_prompt_id"}
    """
    npz_files = sorted([f for f in os.listdir(MOTION_DIR) if f.endswith(".npz")])

    # Group by prompt ID
    groups = defaultdict(list)
    for f in npz_files:
        pid = f.split("_")[0]
        groups[pid] = groups.get(pid, [])
        groups[pid].append(f)

    # Filter: only keep prompts with motions having enough frames
    valid_prompts = {}
    for pid, files in groups.items():
        for f in files:
            path = str(MOTION_DIR / f)
            try:
                d = np.load(path, allow_pickle=True)
                fc = int(d["frame_count"])
                if fc >= min_frames:
                    valid_prompts.setdefault(pid, []).append({
                        "file": f,
                        "path": path,
                        "frames": fc,
                        "text": str(d.get("text", "")),
                    })
            except Exception:
                pass

    prompt_ids = sorted(valid_prompts.keys())
    if len(prompt_ids) < 2:
        raise ValueError(f"Need at least 2 valid prompts, got {len(prompt_ids)}")

    rng = np.random.RandomState(seed)
    pairs = []
    attempts = 0
    max_attempts = num_pairs * 10
    used_pairs = set()

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        pid_a, pid_b = rng.choice(prompt_ids, size=2, replace=False)
        pair_key = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        # Pick first valid motion from each prompt (deterministic per seed)
        a_info = valid_prompts[pid_a][0]
        b_info = valid_prompts[pid_b][0]

        pairs.append({
            "a_path": a_info["path"],
            "b_path": b_info["path"],
            "a_text": a_info["text"],
            "b_text": b_info["text"],
            "a_prompt_id": pid_a,
            "b_prompt_id": pid_b,
            "a_frames": a_info["frames"],
            "b_frames": b_info["frames"],
        })

    print(f"[DATA] Sampled {len(pairs)} motion pairs from {len(prompt_ids)} prompts")
    return pairs


# ============================================================================
# Transition construction
# ============================================================================

def align_motion_b_to_a(motion_a, motion_b):
    """Align motion_b's start to motion_a's end in XZ plane.

    Only translates XZ, keeps Y (height) unchanged.
    """
    # Last frame of A
    a_end_xz = motion_a[-1, [0, 2]].clone()
    # First frame of B
    b_start_xz = motion_b[0, [0, 2]].clone()

    offset = a_end_xz - b_start_xz
    aligned_b = motion_b.clone()
    aligned_b[:, 0] += offset[0]
    aligned_b[:, 2] += offset[1]
    return aligned_b


def build_transition_input(motion_a, motion_b, head_f=HEAD_F, tail_f=TAIL_F, trans_f=TRANS_F):
    """Build the input for M2M transition completion.

    Takes last `head_f` frames from A and first `tail_f` frames from B,
    with `trans_f` frames of zeros in between for the model to generate.

    Returns:
        src_motion: (T, 135) tensor with condition baked in, masked region zeroed
        mask: (T, 135) tensor, 1 = generate, 0 = keep
        T: total frames
        a_tail: (head_f, 135) the condition frames from A
        b_head: (tail_f, 135) the condition frames from B
    """
    T_a = motion_a.shape[0]
    T_b = motion_b.shape[0]

    # Take tail of A and head of B
    a_tail = motion_a[max(0, T_a - head_f):T_a].clone()
    b_head = motion_b[:min(tail_f, T_b)].clone()

    actual_head = a_tail.shape[0]
    actual_tail = b_head.shape[0]
    T = actual_head + trans_f + actual_tail

    if T > MAX_FRAME:
        # Reduce trans_f to fit
        trans_f = MAX_FRAME - actual_head - actual_tail
        if trans_f < 10:
            trans_f = 10
            actual_head = min(actual_head, (MAX_FRAME - trans_f) // 2)
            actual_tail = min(actual_tail, MAX_FRAME - trans_f - actual_head)
            a_tail = a_tail[-actual_head:]
            b_head = b_head[:actual_tail]
        T = actual_head + trans_f + actual_tail

    # Align B to A in XZ
    b_head_aligned = b_head.clone()
    offset_xz = a_tail[-1, [0, 2]] - b_head[0, [0, 2]]
    b_head_aligned[:, 0] += offset_xz[0]
    b_head_aligned[:, 2] += offset_xz[1]

    # Build source motion: condition + zeros + condition
    src_motion = torch.zeros(T, D, dtype=torch.float32)
    src_motion[:actual_head] = a_tail
    src_motion[actual_head + trans_f:] = b_head_aligned

    # Build mask: 0 = condition, 1 = generate
    mask = torch.zeros(T, D, dtype=torch.float32)
    mask[actual_head:actual_head + trans_f, :] = 1.0

    return src_motion, mask, T, actual_head, trans_f, actual_tail, a_tail, b_head_aligned


# ============================================================================
# Model building (reused from eval_m2m_completion.py)
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
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    info = UNCOND_CONFIGS[config_name]
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

    replacement = "skip_last"
    pipeline = HyMotionM2MPipeline(bundle, num_steps=num_steps, replacement_guidance=replacement)
    return pipeline, bundle, ckpt_path


# ============================================================================
# Inference
# ============================================================================

def run_transition(pipeline, bundle, src_motion, mask, T, device):
    """Run M2M completion on the transition input."""
    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'
    motion_in = src_motion[:T].clone()

    if is_global:
        # Convert condition frames (non-masked) to global rotation
        trans = motion_in[:, :3]
        rot6d_local = motion_in[:, 3:].reshape(T, 22, 6)
        rot6d_global = _local_to_global_rot6d(rot6d_local)
        motion_in = torch.cat([trans, rot6d_global.reshape(T, 132)], dim=-1)

    motion_norm = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))
    msk = mask[:T].unsqueeze(0).to(device)

    # Zero out masked regions (VACE conditioning)
    motion_norm = motion_norm * (1 - msk)

    if T < MAX_FRAME:
        pad_len = MAX_FRAME - T
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
        # Combine: condition from original (global), generated from model
        mask_crop = mask[:T]
        orig_global = motion_in.cpu()
        combined_global = orig_global * (1 - mask_crop) + repaired_raw * mask_crop

        # Convert back to local rotation
        c_rot6d_global = combined_global[:, 3:].reshape(T, 22, 6)
        c_rot6d_local = _global_to_local_rot6d(c_rot6d_global)
        combined = torch.cat([combined_global[:, :3], c_rot6d_local.reshape(T, 132)], dim=-1)
    else:
        mask_crop = mask[:T]
        combined = src_motion[:T] * (1 - mask_crop) + repaired_raw * mask_crop

    return combined


# ============================================================================
# Metrics
# ============================================================================

def compute_boundary_smoothness(combined, head_f, trans_f, fps=30):
    """Compute boundary jerk at transition boundaries."""
    T = combined.shape[0]
    # Velocity
    vel = combined[1:] - combined[:-1]
    accel = vel[1:] - vel[:-1]
    jerk = accel[1:] - accel[:-1]

    # Boundary frames: where condition meets generated
    boundaries = [head_f, head_f + trans_f]
    window = 3
    jerk_vals = []
    for b in boundaries:
        if b < 3 or b >= T - 3:
            continue
        lo = max(0, b - window - 2)
        hi = min(jerk.shape[0], b + window - 2)
        if hi > lo:
            jerk_vals.append(jerk[lo:hi].norm(dim=-1).mean().item())

    return {
        "boundary_jerk": round(np.mean(jerk_vals) if jerk_vals else 0.0, 6),
        "num_boundaries": len(boundaries),
    }


def compute_jitter(combined, fps=30):
    """Compute motion jitter (mean acceleration magnitude)."""
    vel = (combined[1:] - combined[:-1]) * fps
    accel = (vel[1:] - vel[:-1]) * fps
    jitter = accel.norm(dim=-1).mean().item()
    return {"jitter": round(jitter, 4)}


def compute_foot_skating(combined, fps=30):
    """Approximate foot skating from translation speed."""
    speed = (combined[1:, :3] - combined[:-1, :3]).norm(dim=-1) * fps
    return {"root_speed_mean": round(speed.mean().item(), 3)}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate M2M transition on motion pairs")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-f", type=int, default=HEAD_F)
    parser.add_argument("--tail-f", type=int, default=TAIL_F)
    parser.add_argument("--trans-f", type=int, default=TRANS_F)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config names to evaluate (default: all uncond)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    config_names = args.configs or list(UNCOND_CONFIGS.keys())
    # Filter to only valid configs
    config_names = [c for c in config_names if c in UNCOND_CONFIGS]
    if not config_names:
        print("ERROR: No valid configs specified")
        return

    print(f"=== M2M Transition Evaluation ===")
    print(f"  Pairs: {args.num_pairs}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Configs: {config_names}")
    print(f"  Transition: head={args.head_f}, trans={args.trans_f}, tail={args.tail_f}")
    print(f"  Output: {output_dir}")
    print()

    # Load pairs
    pairs = load_motion_pairs(args.num_pairs, seed=args.seed, min_frames=max(args.head_f, args.tail_f) + 10)

    # Pre-load all motions
    print("[LOAD] Loading motion data...")
    pair_data = []
    for i, pair in enumerate(pairs):
        try:
            ma_135, _, fps_a, data_a = load_npz_as_motion(pair["a_path"])
            mb_135, _, fps_b, data_b = load_npz_as_motion(pair["b_path"])
            pair_data.append({
                **pair,
                "motion_a": ma_135,
                "motion_b": mb_135,
                "fps": fps_a,
                "orig_data_a": data_a,
                "orig_data_b": data_b,
            })
        except Exception as e:
            print(f"  WARN: Failed to load pair {i}: {e}")
    print(f"[LOAD] {len(pair_data)} pairs loaded successfully")

    # Run each config
    for config_name in config_names:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        try:
            pipeline, bundle, ckpt_path = build_m2m_model(config_name, device, args.num_steps)
        except Exception as e:
            print(f"  ERROR building model: {e}")
            traceback.print_exc()
            continue

        all_metrics = []
        errors = 0

        for idx, pd in enumerate(pair_data):
            case_id = f"case_{idx:03d}"
            case_dir = output_dir / "eval_transition" / case_id / config_name
            meta_path = case_dir / "meta.json"

            # Skip if already done
            if meta_path.is_file():
                try:
                    with open(meta_path) as f:
                        existing = json.load(f)
                    if existing.get("metrics"):
                        all_metrics.append(existing["metrics"])
                    continue
                except Exception:
                    pass

            try:
                motion_a = pd["motion_a"]
                motion_b = pd["motion_b"]

                # Build transition input
                src_motion, mask, T, actual_head, actual_trans, actual_tail, a_tail, b_head = \
                    build_transition_input(
                        motion_a, motion_b,
                        head_f=args.head_f, tail_f=args.tail_f, trans_f=args.trans_f,
                    )

                # Run M2M completion
                combined = run_transition(pipeline, bundle, src_motion, mask, T, device)

                # Compute metrics
                m = {}
                m.update(compute_boundary_smoothness(combined, actual_head, actual_trans, pd["fps"]))
                m.update(compute_jitter(combined, pd["fps"]))
                m.update(compute_foot_skating(combined, pd["fps"]))

                # Save outputs
                os.makedirs(str(case_dir), exist_ok=True)

                # Save the full stitched output
                output_path = str(case_dir / "output.npz")
                motion_135_to_npz(combined, pd["orig_data_a"], output_path, pd["fps"])

                # Save original motions for reference
                a_path = str(case_dir / "motion_a.npz")
                b_path = str(case_dir / "motion_b.npz")
                if not os.path.isfile(a_path):
                    motion_135_to_npz(motion_a, pd["orig_data_a"], a_path, pd["fps"])
                if not os.path.isfile(b_path):
                    motion_135_to_npz(motion_b, pd["orig_data_b"], b_path, pd["fps"])

                # Save meta
                meta = {
                    "task": "transition",
                    "config": config_name,
                    "config_desc": UNCOND_CONFIGS[config_name]["desc"],
                    "a_path": pd["a_path"],
                    "b_path": pd["b_path"],
                    "a_text": pd["a_text"],
                    "b_text": pd["b_text"],
                    "a_prompt_id": pd["a_prompt_id"],
                    "b_prompt_id": pd["b_prompt_id"],
                    "head_f": actual_head,
                    "trans_f": actual_trans,
                    "tail_f": actual_tail,
                    "total_frames": T,
                    "fps": pd["fps"],
                    "num_steps": args.num_steps,
                    "mask_ratio": float(mask.mean()),
                    "metrics": m,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "npz_files": ["output.npz", "motion_a.npz", "motion_b.npz"],
                }
                with open(str(meta_path), "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

                all_metrics.append(m)

                if (idx + 1) % 10 == 0:
                    print(f"    [{config_name}] {idx + 1}/{len(pair_data)} done")

            except Exception as e:
                errors += 1
                print(f"    ERROR case {idx}: {e}")
                traceback.print_exc()

        # Summary
        if all_metrics:
            print(f"\n  [{config_name}] Summary ({len(all_metrics)} cases, {errors} errors):")
            keys = all_metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in all_metrics if k in m and isinstance(m[k], (int, float))]
                if vals:
                    print(f"    {k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

        # Save config summary
        summary_path = output_dir / "eval_transition" / f"summary_{config_name}.json"
        summary = {
            "config": config_name,
            "num_cases": len(all_metrics),
            "num_errors": errors,
            "num_steps": args.num_steps,
            "head_f": args.head_f,
            "trans_f": args.trans_f,
            "tail_f": args.tail_f,
        }
        if all_metrics:
            keys = all_metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in all_metrics if k in m and isinstance(m[k], (int, float))]
                if vals:
                    summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                    summary[f"{k}_std"] = round(float(np.std(vals)), 4)
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Unload model to free GPU
        del pipeline, bundle
        torch.cuda.empty_cache()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

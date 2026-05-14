#!/usr/bin/env python3
"""Batch pipeline: Text prompts → HyMotion T2M 1.0-Lite → Embodied G1 Robot (V6: PyRoki).

End-to-end pipeline:
  1. Text → HyMotion T2M inference (201-dim)
  2. Extract motion_135 (first 135 dims) → save NPZ
  3. motion_135 → PyRoki keypoints → PyRoki retarget → ProtoMotions .motion  (pipeline_motion_to_robot.py V6)
  4. .motion → reference render (qpos) + tracked render (ONNX policy)
  5. .motion → JSON for Three.js web visualization
  6. Generate manifest + metadata for comparison website

Usage:
    # Run full pipeline with existing prompts
    python scripts/embodied/batch_t2m_to_embodied.py \
        --prompt-json output/embodied_comparison_v2/motion_text_mapping.json \
        --output-dir output/embodied_comparison_v3/ \
        --max-motions 5

    # Run with custom prompts
    python scripts/embodied/batch_t2m_to_embodied.py \
        --prompts "a person walks forward" "a person jumps" \
        --output-dir output/embodied_test/

    # Skip T2M inference, use existing motion_135 NPZ files
    python scripts/embodied/batch_t2m_to_embodied.py \
        --npz-dir work_dirs/.../npz/ \
        --output-dir output/embodied_comparison_v3/

    # Resume interrupted run (skip existing)
    python scripts/embodied/batch_t2m_to_embodied.py \
        --prompt-json ... --output-dir ... --skip-existing
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import traceback

# Ensure hftrainer is importable (run from any directory)
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Default model config / checkpoint
DEFAULT_T2M_CONFIG = "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"
DEFAULT_T2M_CHECKPOINT = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"

# Embodied pipeline defaults
DEFAULT_MJCF = "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
DEFAULT_ONNX = (
    "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/"
    "g1-bones-deploy/compiled_models/unified_pipeline.onnx"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch: Text → T2M → motion_135 → Embodied G1 Robot pipeline"
    )

    # --- Input modes ---
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt-json", type=str,
        help="JSON file with motion prompts (same format as motion_text_mapping.json)"
    )
    input_group.add_argument(
        "--prompts", nargs="+", type=str,
        help="Inline text prompts"
    )
    input_group.add_argument(
        "--npz-dir", type=str,
        help="Directory of existing motion_135 NPZ files (skip T2M inference)"
    )

    # --- T2M model ---
    p.add_argument("--config", type=str, default=DEFAULT_T2M_CONFIG,
                   help="HyMotion T2M config path")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_T2M_CHECKPOINT,
                   help="HyMotion T2M checkpoint path")
    p.add_argument("--num-steps", type=int, default=50,
                   help="ODE denoising steps (default: 50, matches official HY-Motion)")
    p.add_argument("--guidance-scale", type=float, default=5.0,
                   help="CFG guidance scale (default: 5.0, matching official HY-Motion-1.0 CLI --cfg_scale)")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for T2M inference")

    # --- Output ---
    p.add_argument("--output-dir", required=True,
                   help="Root output directory for comparison website data")
    p.add_argument("--max-motions", type=int, default=None,
                   help="Maximum number of motions to process")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip motions whose JSON already exists")

    # --- Pipeline options ---
    p.add_argument("--no-render", action="store_true",
                   help="Skip video rendering (only generate JSON for web)")
    p.add_argument("--no-tracked", action="store_true",
                   help="Skip tracked mode rendering (only reference)")
    p.add_argument("--no-reference-render", action="store_true",
                   help="Skip reference video rendering (JSON still generated)")
    p.add_argument("--skip-pipeline-on-existing-cache", action="store_true", default=True,
                   help="Skip retarget pipeline if cache .pt already exists")

    # --- Smoothing ---
    p.add_argument("--smooth", action="store_true", default=True,
                   help="Apply Markley quaternion smoothing to motion_135 output (default: True)")
    p.add_argument("--no-smooth", dest="smooth", action="store_false",
                   help="Disable post-generation smoothing on motion_135")

    # --- Rendering ---
    p.add_argument("--render-width", type=int, default=640,
                   help="Video render width")
    p.add_argument("--render-height", type=int, default=480,
                   help="Video render height")

    return p.parse_args()


def load_prompts_from_json(json_path):
    """Load prompts from motion_text_mapping.json format.

    Returns list of dicts: [{id, text, duration_frames}, ...]
    """
    with open(json_path) as f:
        data = json.load(f)

    prompts = []
    for m in data.get("motions", []):
        prompts.append({
            "id": m.get("motion_id", f"motion_{len(prompts):04d}"),
            "text": m["text"],
            "duration_frames": m.get("duration_frames", 120),
        })
    return prompts


def load_prompts_from_list(prompt_list):
    """Create prompt dicts from a list of strings."""
    prompts = []
    for i, text in enumerate(prompt_list):
        prompts.append({
            "id": f"motion_{i:04d}",
            "text": text,
            "duration_frames": 120,  # default 4s @ 30fps
        })
    return prompts


def load_t2m_bundle(args):
    """Load HyMotion T2M bundle once for all prompts (GPU-efficient)."""
    import torch
    from mmengine.config import Config
    import hftrainer  # noqa: trigger auto-imports

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = str(PROJECT_ROOT / ckpt_path)

    cfg = Config.fromfile(config_path)

    # Inject text encoder config if empty (needed for inference with text prompts).
    # The training config has text_encoder=dict() which is falsy — the bundle's __init__
    # treats it as None and later raises RuntimeError when encode_text() is called.
    # Values come from HY-Motion-1.0-Lite/config.yml: llm_type=qwen3, max_length_llm=128.
    if not cfg.model.get('text_encoder'):
        cfg.model.text_encoder = dict(
            type='HYTextModel',
            llm_type='qwen3',
            max_length_llm=128,
        )
        print("[load_t2m_bundle] Injected text_encoder config: HYTextModel/qwen3/128")

    from tools.infer import load_bundle_from_checkpoint
    bundle = load_bundle_from_checkpoint(cfg, ckpt_path, args.device)

    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.guidance_scale,
    )
    return bundle, pipeline


def run_t2m_inference(bundle, pipeline, prompt_text, num_frames, device="cuda"):
    """Run T2M inference for a single prompt, return motion_135 numpy array.

    Returns: (motion_135, motion_201) — both numpy arrays of shape (T, D)
    """
    import torch

    batch = {
        "tgt_length": [num_frames],
        "caption": [prompt_text],
    }

    with torch.no_grad():
        output = pipeline(batch)

    # Extract denormalized motion
    latent_denorm = output.get("latent_denorm")
    if latent_denorm is not None:
        if isinstance(latent_denorm, torch.Tensor):
            latent_denorm = latent_denorm.cpu().float().numpy()
        motion_201 = latent_denorm[0]  # (T, 201)
    else:
        # Manual denormalization
        latent = output["latent"]
        if isinstance(latent, torch.Tensor):
            latent = latent.cpu().float().numpy()
        mean = bundle.mean.cpu().numpy()
        std = bundle.std.cpu().numpy()
        std = np.where(std < 1e-3, 0.0, std)
        motion_201 = latent[0] * std + mean

    # Extract first 135 dims for motion_135 format
    # Layout: [0:3] transl, [3:135] 22x rot6d
    motion_135 = motion_201[:, :135]

    return motion_135, motion_201


def _rot6d_to_rotmat(rot6d):
    """Convert row-major rot6d (N, 6) to rotation matrix (N, 3, 3) via Gram-Schmidt.

    Input rot6d is row-major: [R00, R01, R10, R11, R20, R21].
    First reorder to column-major [R00, R10, R20, R01, R11, R21] = two column vectors,
    then Gram-Schmidt orthogonalize.
    """
    # Reorder row-major → column-major
    col_major = rot6d[:, [0, 2, 4, 1, 3, 5]]  # (N, 6)
    a1 = col_major[:, 0:3]  # first column of R
    a2 = col_major[:, 3:6]  # second column of R

    # Gram-Schmidt
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)


def _rotmat_to_rot6d(rotmat):
    """Convert rotation matrix (N, 3, 3) to row-major rot6d (N, 6).

    Extract first two columns, then reorder column-major → row-major.
    Column-major: [R00, R10, R20, R01, R11, R21]
    Row-major:    [R00, R01, R10, R11, R20, R21] via indices [0, 3, 1, 4, 2, 5]
    """
    col0 = rotmat[:, :, 0]  # (N, 3)
    col1 = rotmat[:, :, 1]  # (N, 3)
    col_major = np.concatenate([col0, col1], axis=-1)  # (N, 6) = [R00,R10,R20,R01,R11,R21]
    row_major = col_major[:, [0, 3, 1, 4, 2, 5]]  # (N, 6) = [R00,R01,R10,R11,R20,R21]
    return row_major


def _rotmat_to_quat(R):
    """Convert rotation matrix (N, 3, 3) to quaternion (N, 4) [w, x, y, z].

    Uses Shepperd's method for numerical stability.
    """
    N = R.shape[0]
    q = np.zeros((N, 4))

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    s = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2  # s = 4*w
    mask = tr > 0
    if mask.any():
        s_m = s[mask]
        q[mask, 0] = 0.25 * s_m
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s_m
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s_m
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s_m

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s2 = np.sqrt(np.maximum(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2], 0.0)) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
        q[mask2, 1] = 0.25 * s2
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s3 = np.sqrt(np.maximum(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2], 0.0)) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
        q[mask3, 2] = 0.25 * s3
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    if mask4.any():
        s4 = np.sqrt(np.maximum(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1], 0.0)) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
        q[mask4, 3] = 0.25 * s4

    # Normalize
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    return q


def _quat_to_rotmat(q):
    """Convert quaternion (N, 4) [w, x, y, z] to rotation matrix (N, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def _fix_quat_continuity(quats):
    """Fix quaternion sign flips for temporal continuity (antipodal alignment).

    quats: (T, 4) [w, x, y, z]
    Returns: (T, 4) with consistent signs across time.
    """
    result = quats.copy()
    for i in range(1, len(result)):
        if np.dot(result[i], result[i-1]) < 0:
            result[i] = -result[i]
    return result


def _wavg_quaternion_markley(quats, weights):
    """Weighted average of quaternions using Markley's eigendecomposition method.

    quats: (K, 4) — quaternions [w, x, y, z]
    weights: (K,) — non-negative weights (will be normalized)

    Returns: (4,) — weighted average quaternion

    Reference: Markley, Cheng, Crassidis, Oshman (2007)
    "Averaging Quaternions", Journal of Guidance, Control, and Dynamics.
    The optimal average is the eigenvector corresponding to the largest
    eigenvalue of M = sum_i w_i * q_i @ q_i^T.
    """
    weights = weights / (weights.sum() + 1e-12)
    # Build 4x4 accumulator matrix M = sum(w_i * q_i * q_i^T)
    M = np.zeros((4, 4))
    for i in range(len(quats)):
        qi = quats[i]  # (4,)
        M += weights[i] * np.outer(qi, qi)
    # Eigenvector of largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # eigh returns eigenvalues in ascending order; take the last
    return eigenvectors[:, -1]


def _gaussian_kernel_weights(sigma, truncate=4.0):
    """Create 1D Gaussian kernel weights.

    sigma: standard deviation in frames
    truncate: truncate at truncate * sigma

    Returns: 1D numpy array of kernel weights (normalized).
    Official HY-Motion-1.0 uses sigma=1.0, truncate=4.0 → radius=4, kernel size=9.
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth_motion_135(motion_135):
    """Apply Gaussian-weighted Markley quaternion smoothing to motion_135.

    Matches official HY-Motion-1.0 post-processing:
      - Rotation: rot6d → rotation matrix → quaternion → Markley weighted average
        with Gaussian kernel (sigma=1.0, truncate=4.0, 9-tap kernel)
      - Translation: Savitzky-Golay filter (window=11, polyorder=5)
      - Applied to all 22 body joints

    This replaces the old approach of applying savgol_filter directly on rot6d,
    which is mathematically invalid since rot6d is not a linear space.

    Args:
        motion_135: (T, 135) array — [0:3] transl + [3:135] 22x rot6d (row-major)

    Returns:
        smoothed motion_135 (T, 135)
    """
    from scipy.signal import savgol_filter
    T = motion_135.shape[0]
    smoothed = motion_135.copy()

    if T < 3:
        return smoothed

    # --- Translation: Savitzky-Golay (matching official: window=11, polyorder=5) ---
    trans_win = min(11, T if T % 2 == 1 else T - 1)
    trans_poly = min(5, trans_win - 1)
    if trans_win >= 3:
        smoothed[:, :3] = savgol_filter(
            smoothed[:, :3], window_length=trans_win, polyorder=trans_poly, axis=0
        )

    # --- Rotation: Markley quaternion smoothing per joint ---
    # Gaussian kernel: sigma=1.0, truncate=4.0 → radius=4, 9-tap kernel
    sigma = 1.0
    kernel = _gaussian_kernel_weights(sigma, truncate=4.0)
    radius = len(kernel) // 2  # = 4

    for j in range(22):
        start = 3 + j * 6
        end = start + 6
        rot6d_j = motion_135[:, start:end]  # (T, 6)

        # rot6d → rotation matrix → quaternion
        rotmats = _rot6d_to_rotmat(rot6d_j)  # (T, 3, 3)
        quats = _rotmat_to_quat(rotmats)      # (T, 4) [w,x,y,z]

        # Fix quaternion sign flips for temporal continuity
        quats = _fix_quat_continuity(quats)

        # Gaussian-weighted Markley averaging per frame
        smoothed_quats = np.zeros_like(quats)
        for t in range(T):
            t_start = max(0, t - radius)
            t_end = min(T, t + radius + 1)
            # Kernel indices corresponding to [t_start, t_end)
            k_start = t_start - (t - radius)  # offset into kernel
            k_end = k_start + (t_end - t_start)

            window_quats = quats[t_start:t_end]  # (W, 4)
            window_weights = kernel[k_start:k_end]  # (W,)

            smoothed_quats[t] = _wavg_quaternion_markley(window_quats, window_weights)

        # Fix sign continuity again after smoothing
        smoothed_quats = _fix_quat_continuity(smoothed_quats)

        # quaternion → rotation matrix → rot6d (row-major)
        smoothed_rotmats = _quat_to_rotmat(smoothed_quats)  # (T, 3, 3)
        smoothed_rot6d = _rotmat_to_rot6d(smoothed_rotmats)  # (T, 6)

        smoothed[:, start:end] = smoothed_rot6d

    return smoothed


def save_motion_135_npz(motion_135, output_path, fps=30):
    """Save motion_135 array as NPZ file compatible with pipeline_motion_to_robot.py.

    The key must be 'motion_135' for motion135_to_smplx.py to load it.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        motion_135=motion_135.astype(np.float32),
        fps=np.array(fps),
    )


def run_retarget_pipeline(npz_path, output_dir, extra_args=None):
    """Run pipeline_motion_to_robot.py (V6 PyRoki): motion_135 NPZ → ProtoMotions .motion.

    The V6 pipeline uses PyRoki trajectory-level retargeting:
      1. motion_135 → PyRoki keypoints (SMPL FK + geometric surgery)
      2. PyRoki retarget (jaxls optimizer, 800 iterations)
      3. Retargeted NPZ → ProtoMotions .motion

    Args:
        npz_path: path to motion_135 NPZ file
        output_dir: directory where .motion file will be written
        extra_args: additional CLI args to pass to pipeline

    Returns True on success.
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "pipeline_motion_to_robot.py"),
        "--input", str(npz_path),
        "--output", str(output_dir),
        "--keep-intermediates",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # PyRoki CPU optimizer takes ~66 min per motion
        )
        if result.returncode != 0:
            print(f"    PIPELINE FAILED (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"      {line}")
            if result.stdout:
                for line in result.stdout.strip().split("\n")[-20:]:
                    print(f"      {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    PIPELINE TIMEOUT (>7200s)")
        return False
    except Exception as e:
        print(f"    PIPELINE ERROR: {e}")
        return False


def run_render(cache_path, output_dir, mode="reference", onnx_path=None,
               width=640, height=480, video=True):
    """Run render_tracker_headless.py on a .motion or cache .pt file.

    Returns True on success.
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "render_tracker_headless.py"),
        "--motion", str(cache_path),
        "--output-dir", str(output_dir),
        "--mode", mode,
        "--width", str(width),
        "--height", str(height),
    ]
    if video:
        cmd.append("--video")
    if mode == "tracked" and onnx_path:
        cmd.extend(["--onnx", str(onnx_path)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min for tracked mode
        )
        if result.returncode != 0:
            print(f"    RENDER ({mode}) FAILED (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"      {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    RENDER ({mode}) TIMEOUT")
        return False
    except Exception as e:
        print(f"    RENDER ({mode}) ERROR: {e}")
        return False


def convert_cache_to_json(cache_path, json_output):
    """Convert ProtoMotions cache .pt → JSON for Three.js visualization."""
    # Import the converter directly
    sys.path.insert(0, str(SCRIPT_DIR))
    from convert_cache_to_json import convert_cache_to_json as _convert
    return _convert(str(cache_path), str(json_output))


def extract_metrics_from_cache(cache_path):
    """Extract basic metrics from a ProtoMotions cache .pt or .motion file.

    Supports two formats:
      - Old .pt cache: keys body_pos, dof_pos, control_dt, num_frames
      - New .motion:   keys rigid_body_pos, dof_pos, motion_dt/fps

    Returns dict with root height stats, joint velocity stats, etc.
    """
    import torch

    cache = torch.load(str(cache_path), map_location="cpu", weights_only=False)

    def to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    dof_pos = to_np(cache["dof_pos"])      # (T, N_dof)

    # Support both old .pt cache and new .motion format
    if "body_pos" in cache:
        body_pos = to_np(cache["body_pos"])    # (T, 33, 3)
        control_dt = float(cache["control_dt"])
        num_frames = int(cache["num_frames"])
    elif "rigid_body_pos" in cache:
        body_pos = to_np(cache["rigid_body_pos"])  # (T, N_bodies, 3)
        if "motion_dt" in cache:
            control_dt = float(cache["motion_dt"])
        elif "fps" in cache:
            control_dt = 1.0 / float(cache["fps"])
        else:
            control_dt = 1.0 / 30.0
        num_frames = body_pos.shape[0]
    else:
        raise KeyError(
            f"Unrecognized cache format in {cache_path}. Keys: {list(cache.keys())}. "
            "Expected 'body_pos' (old .pt) or 'rigid_body_pos' (.motion)."
        )

    # Root (pelvis) height = body_pos[:, 0, 2] (Z-up in MuJoCo)
    root_height = body_pos[:, 0, 2]

    # Joint velocity (finite differences)
    if num_frames > 1:
        dof_vel = np.diff(dof_pos, axis=0) / control_dt
        max_joint_vel = float(np.max(np.abs(dof_vel)))
        mean_joint_vel = float(np.mean(np.abs(dof_vel)))
    else:
        max_joint_vel = 0.0
        mean_joint_vel = 0.0

    # Simple fall detection: root height drops below threshold
    fell = bool(np.any(root_height < 0.3))
    fall_frame = int(np.argmax(root_height < 0.3)) if fell else None

    metrics = {
        "num_frames": num_frames,
        "duration_s": round(num_frames * control_dt, 2),
        "fps": round(1.0 / control_dt),
        "root_height_mean": round(float(np.mean(root_height)), 4),
        "root_height_std": round(float(np.std(root_height)), 4),
        "root_height_min": round(float(np.min(root_height)), 4),
        "root_height_max": round(float(np.max(root_height)), 4),
        "max_joint_velocity": round(max_joint_vel, 2),
        "mean_joint_velocity": round(mean_joint_vel, 2),
        "fell": fell,
        "fall_frame": fall_frame,
    }
    return metrics


def main():
    args = parse_args()
    t_start = time.time()

    # =========================================================================
    # 1. Collect prompts or NPZ files
    # =========================================================================
    prompts = None
    npz_files = None

    if args.prompt_json:
        prompts = load_prompts_from_json(args.prompt_json)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_json}")
    elif args.prompts:
        prompts = load_prompts_from_list(args.prompts)
        print(f"Using {len(prompts)} inline prompts")
    elif args.npz_dir:
        import glob
        npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
        print(f"Found {len(npz_files)} NPZ files in {args.npz_dir}")

    if args.max_motions:
        if prompts:
            prompts = prompts[: args.max_motions]
        if npz_files:
            npz_files = npz_files[: args.max_motions]

    total = len(prompts) if prompts else len(npz_files)
    print(f"\nProcessing {total} motions")

    # =========================================================================
    # 2. Setup output directory structure
    # =========================================================================
    output_root = pathlib.Path(args.output_dir)
    motions_dir = output_root / "data" / "motions"         # reference JSONs
    tracked_dir = output_root / "data" / "tracked_motions"  # tracked JSONs
    cache_dir = output_root / "data" / "caches"             # .motion / .pt caches
    npz_cache_dir = output_root / "data" / "npz"            # motion_135 NPZs
    render_dir = output_root / "data" / "renders"           # render videos
    retarget_dir = output_root / "data" / "retarget"        # PyRoki retarget outputs
    smpl_mesh_dir = output_root / "data" / "smpl_mesh"       # SMPL mesh JSONs for web viz

    for d in [motions_dir, tracked_dir, cache_dir, npz_cache_dir, render_dir, retarget_dir, smpl_mesh_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 3. Load T2M model (only if doing inference)
    # =========================================================================
    bundle = None
    pipeline = None

    if prompts and not args.npz_dir:
        print(f"\nLoading HyMotion T2M 1.0-Lite model...")
        print(f"  Config:     {args.config}")
        print(f"  Checkpoint: {args.checkpoint}")
        bundle, pipeline = load_t2m_bundle(args)
        print(f"  Model loaded on {args.device}")

    # =========================================================================
    # 4. Process each motion
    # =========================================================================
    results = []
    successes = 0
    failures = 0
    skipped = 0

    for idx in range(total):
        if prompts:
            prompt_info = prompts[idx]
            motion_id = prompt_info["id"]
            text = prompt_info["text"]
            duration_frames = prompt_info["duration_frames"]
        else:
            npz_path = npz_files[idx]
            motion_id = f"motion_{pathlib.Path(npz_path).stem}"
            text = None
            duration_frames = None

        # Output paths for this motion
        ref_json_path = motions_dir / f"{motion_id}.json"
        tracked_json_path = tracked_dir / f"{motion_id}.json"
        motion_retarget_dir = retarget_dir / motion_id        # PyRoki output dir
        npz_path_out = npz_cache_dir / f"{motion_id}.npz"
        ref_render_dir = render_dir / motion_id / "reference"
        tracked_render_dir = render_dir / motion_id / "tracked"
        smpl_mesh_json_path = smpl_mesh_dir / f"{motion_id}.json"
        meta_path = output_root / "data" / "meta" / f"{motion_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Find .motion file (V6 PyRoki pipeline output)
        motion_file = None
        if motion_retarget_dir.exists():
            motion_files = list(motion_retarget_dir.glob("*.motion"))
            if motion_files:
                motion_file = motion_files[0]

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] {motion_id}")
        if text:
            print(f"  Prompt: \"{text}\"")
            print(f"  Frames: {duration_frames}")
        print(f"  Retarget: {motion_retarget_dir}")
        print(f"  Ref JSON:     {ref_json_path}")
        print(f"  Tracked JSON: {tracked_json_path}")

        # --- Skip if already done ---
        if args.skip_existing and ref_json_path.exists():
            print(f"  SKIPPED (ref JSON exists)")
            skipped += 1
            continue

        t0 = time.time()
        status = "success"
        error_msg = None

        # ---- Step A: T2M Inference → motion_135 NPZ ----
        if prompts and not npz_path_out.exists():
            try:
                print(f"  [A] Running T2M inference...")
                motion_135, motion_201 = run_t2m_inference(
                    bundle, pipeline, text, duration_frames, args.device
                )
                if args.smooth:
                    motion_135 = smooth_motion_135(motion_135)
                    print(f"      Applied Savitzky-Golay smoothing to motion_135")
                save_motion_135_npz(motion_135, str(npz_path_out))
                print(f"      Generated motion_135: {motion_135.shape}")
            except Exception as e:
                print(f"      T2M INFERENCE FAILED: {e}")
                traceback.print_exc()
                failures += 1
                results.append({"id": motion_id, "status": "t2m_failed", "error": str(e)})
                continue
        elif prompts:
            print(f"  [A] motion_135 NPZ exists, skipping inference")
        else:
            # Using pre-existing NPZ
            npz_path_out = pathlib.Path(npz_files[idx])
            print(f"  [A] Using existing NPZ: {npz_path_out}")

        # ---- Step A2: Generate SMPL mesh JSON for web visualization ----
        if not smpl_mesh_json_path.exists():
            try:
                print(f"  [A2] Generating SMPL mesh JSON...")
                from scripts.embodied.batch_npz_to_smpl_mesh_json import convert_single_npz
                mesh_data = convert_single_npz(
                    str(npz_path_out), smpl_type="smplh", gender="neutral"
                )
                with open(smpl_mesh_json_path, 'w') as f:
                    json.dump(mesh_data, f, separators=(',', ':'))
                n_frames = len(mesh_data["frames"])
                file_size = smpl_mesh_json_path.stat().st_size
                print(f"      SMPL mesh JSON: {n_frames} frames, {file_size/1024:.1f}KB")
            except Exception as e:
                print(f"      SMPL mesh JSON FAILED: {e}")
                traceback.print_exc()
        else:
            print(f"  [A2] SMPL mesh JSON exists, skipping")

        # ---- Step B: Retarget pipeline → ProtoMotions .motion (V6 PyRoki) ----
        if motion_file is None or not args.skip_pipeline_on_existing_cache:
            print(f"  [B] Running PyRoki retarget pipeline (V6)...")
            motion_retarget_dir.mkdir(parents=True, exist_ok=True)
            ok = run_retarget_pipeline(str(npz_path_out), str(motion_retarget_dir))
            if not ok:
                failures += 1
                results.append({"id": motion_id, "status": "pipeline_failed"})
                continue

            # Find the output .motion file
            motion_files_found = list(motion_retarget_dir.glob("*.motion"))
            if motion_files_found:
                motion_file = motion_files_found[0]
                print(f"      Pipeline OK → {motion_file}")
            else:
                print(f"      WARNING: No .motion file found, checking for .pt...")
                pt_files = list(motion_retarget_dir.glob("*.pt"))
                if pt_files:
                    motion_file = pt_files[0]
                    print(f"      Found .pt cache → {motion_file}")
                else:
                    print(f"      ERROR: No output files in {motion_retarget_dir}")
                    failures += 1
                    results.append({"id": motion_id, "status": "pipeline_no_output"})
                    continue
        else:
            print(f"  [B] .motion exists, skipping pipeline: {motion_file}")

        # ---- Step C: Convert .motion → reference JSON (for Three.js) ----
        try:
            print(f"  [C] Converting .motion → reference JSON...")
            ref_info = convert_cache_to_json(str(motion_file), str(ref_json_path))
            print(f"      Ref JSON: {ref_info['num_frames']} frames, {ref_info['fps']} FPS")
        except Exception as e:
            print(f"      REF JSON FAILED: {e}")
            traceback.print_exc()
            status = "ref_json_failed"
            error_msg = str(e)

        # ---- Step D: Run ONNX tracker → tracked cache → tracked JSON ----
        if not args.no_tracked:
            tracked_cache_path = cache_dir / f"{motion_id}_tracked.pt"

            # For tracked mode, we run render_tracker_headless.py in tracked mode,
            # which internally runs the ONNX policy. But the web viewer needs tracked
            # JSON data (dof_pos from simulation, not from reference).
            # The render script doesn't save a separate tracked cache, so we'll
            # use the reference cache for both ref and tracked JSON for now.
            # The difference is shown in the video renders.
            #
            # For Three.js web viewer, both ref and tracked use the same cache format,
            # but the tracked motion JSON could be generated from the tracker's output.
            # For now, copy reference JSON to tracked path (the real difference shows
            # in video rendering).
            if ref_json_path.exists() and not tracked_json_path.exists():
                import shutil
                shutil.copy2(str(ref_json_path), str(tracked_json_path))
                print(f"  [D] Copied ref JSON → tracked JSON (video shows difference)")

        # ---- Step E: Render videos (optional) ----
        if not args.no_render:
            # Reference render
            if not args.no_reference_render:
                print(f"  [E1] Rendering reference video...")
                ref_render_dir.mkdir(parents=True, exist_ok=True)
                ok = run_render(
                    str(motion_file), str(ref_render_dir),
                    mode="reference", width=args.render_width, height=args.render_height,
                )
                if ok:
                    print(f"       Reference render OK")
                else:
                    print(f"       Reference render FAILED (non-fatal)")

            # Tracked render
            if not args.no_tracked:
                onnx_path = str(PROJECT_ROOT / DEFAULT_ONNX)
                if os.path.exists(onnx_path):
                    print(f"  [E2] Rendering tracked video...")
                    tracked_render_dir.mkdir(parents=True, exist_ok=True)
                    ok = run_render(
                        str(motion_file), str(tracked_render_dir),
                        mode="tracked", onnx_path=onnx_path,
                        width=args.render_width, height=args.render_height,
                    )
                    if ok:
                        print(f"       Tracked render OK")
                    else:
                        print(f"       Tracked render FAILED (non-fatal)")
                else:
                    print(f"  [E2] ONNX model not found, skipping tracked render")

        # ---- Step F: Extract metrics ----
        metrics = {}
        try:
            metrics = extract_metrics_from_cache(str(motion_file))
        except Exception as e:
            print(f"  [F] Metrics extraction failed: {e}")

        # ---- Step G: Save per-motion metadata ----
        dt = time.time() - t0
        meta = {
            "id": motion_id,
            "text": text,
            "duration_frames_requested": duration_frames,
            "status": status,
            "error": error_msg,
            "processing_time_s": round(dt, 1),
            "metrics": metrics,
            "paths": {
                "ref_json": str(ref_json_path),
                "tracked_json": str(tracked_json_path),
                "smpl_mesh_json": str(smpl_mesh_json_path),
                "motion_file": str(motion_file) if motion_file else None,
                "npz": str(npz_path_out),
                "retarget_dir": str(motion_retarget_dir),
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if status == "success":
            successes += 1
            print(f"  ✓ OK ({dt:.1f}s)")
        else:
            failures += 1
            print(f"  ✗ {status} ({dt:.1f}s)")

        results.append(meta)

    # =========================================================================
    # 5. Generate manifest for web
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  Generating manifest...")
    print(f"{'='*60}")

    manifest_motions = []
    for r in results:
        if r.get("status") in ("success",):
            m = r.get("metrics", {})
            manifest_motions.append({
                "id": r["id"],
                "text": r.get("text", ""),
                "num_frames": m.get("num_frames", 0),
                "fps": m.get("fps", 50),
                "duration_s": m.get("duration_s", 0),
                "fell": m.get("fell", False),
                "fall_frame": m.get("fall_frame"),
                "root_height_mean": m.get("root_height_mean"),
                "max_joint_velocity": m.get("max_joint_velocity"),
            })

    manifest = {
        "model": "HyMotion T2M 1.0-Lite",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_motions": len(manifest_motions),
        "motions": manifest_motions,
    }

    manifest_path = output_root / "data" / "motions" / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path} ({len(manifest_motions)} motions)")

    # Also save motion_text_mapping.json for website
    if prompts:
        mapping = {
            "model": "HyMotion T2M 1.0-Lite",
            "description": "Motion ID to text prompt mapping",
            "motions": [
                {
                    "motion_id": p["id"],
                    "text": p["text"],
                    "duration_frames": p["duration_frames"],
                }
                for p in prompts
            ],
        }
        mapping_path = output_root / "motion_text_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  Mapping:  {mapping_path}")

    # =========================================================================
    # 6. Summary
    # =========================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Batch Complete!")
    print(f"{'='*60}")
    print(f"  Total:     {total}")
    print(f"  Success:   {successes}")
    print(f"  Failed:    {failures}")
    print(f"  Skipped:   {skipped}")
    print(f"  Time:      {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Output:    {output_root}")
    print(f"")

    # Save batch report
    report_path = output_root / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "success": successes,
            "failed": failures,
            "skipped": skipped,
            "total_time_s": round(total_time, 1),
            "results": results,
        }, f, indent=2)
    print(f"  Report:    {report_path}")


if __name__ == "__main__":
    main()

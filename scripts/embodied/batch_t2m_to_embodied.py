#!/usr/bin/env python3
"""Batch pipeline: Text prompts → HyMotion T2M 1.0-Lite → Embodied G1 Robot (reference + tracked).

End-to-end pipeline:
  1. Text → HyMotion T2M inference (201-dim)
  2. Extract motion_135 (first 135 dims) → save NPZ
  3. motion_135 → SMPL-X → GMR → ProtoMotions cache .pt  (pipeline_motion_to_robot.py)
  4. cache .pt → reference render (qpos) + tracked render (ONNX policy)
  5. cache .pt → JSON for Three.js web visualization
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
    p.add_argument("--num-steps", type=int, default=100,
                   help="ODE denoising steps (default: 100)")
    p.add_argument("--guidance-scale", type=float, default=4.0,
                   help="CFG guidance scale (default: 4.0)")
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
                   help="Apply Savitzky-Golay smoothing to motion_135 output (default: True)")
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
        std = np.where(std < 1e-3, 1.0, std)
        motion_201 = latent[0] * std + mean

    # Extract first 135 dims for motion_135 format
    # Layout: [0:3] transl, [3:135] 22x rot6d
    motion_135 = motion_201[:, :135]

    return motion_135, motion_201


def smooth_motion_135(motion_135):
    """Apply Savitzky-Golay smoothing to motion_135 to reduce T2M output noise.

    Smooths translation (cols 0:3) with a wider window for stable root trajectory,
    and rot6d (cols 3:135) with a narrower window to preserve pose detail.

    Args:
        motion_135: (T, 135) array — [0:3] transl + [3:135] 22x rot6d

    Returns:
        smoothed motion_135 (T, 135)
    """
    from scipy.signal import savgol_filter
    T = motion_135.shape[0]
    smoothed = motion_135.copy()

    # Translation: wider window (~0.23s at 30Hz) for stable root trajectory
    trans_win = min(7, T if T % 2 == 1 else T - 1)
    if trans_win >= 5:
        smoothed[:, :3] = savgol_filter(smoothed[:, :3], window_length=trans_win, polyorder=3, axis=0)

    # Rot6d: narrower window to preserve pose detail but remove frame-to-frame noise
    rot_win = min(5, T if T % 2 == 1 else T - 1)
    if rot_win >= 5:
        smoothed[:, 3:] = savgol_filter(smoothed[:, 3:], window_length=rot_win, polyorder=3, axis=0)

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


def run_retarget_pipeline(npz_path, cache_output, extra_args=None):
    """Run pipeline_motion_to_robot.py: motion_135 NPZ → ProtoMotions cache .pt.

    Returns True on success.
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "pipeline_motion_to_robot.py"),
        "--input", str(npz_path),
        "--output", str(cache_output),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"    PIPELINE FAILED (exit {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"      {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    PIPELINE TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"    PIPELINE ERROR: {e}")
        return False


def run_render(cache_path, output_dir, mode="reference", onnx_path=None,
               width=640, height=480, video=True):
    """Run render_tracker_headless.py on a cache .pt file.

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
    """Extract basic metrics from a ProtoMotions cache .pt file.

    Returns dict with root height stats, joint velocity stats, etc.
    """
    import torch

    cache = torch.load(str(cache_path), map_location="cpu", weights_only=False)

    def to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    body_pos = to_np(cache["body_pos"])    # (T, 33, 3)
    dof_pos = to_np(cache["dof_pos"])      # (T, 29)
    control_dt = float(cache["control_dt"])
    num_frames = int(cache["num_frames"])

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
    cache_dir = output_root / "data" / "caches"             # .pt caches
    npz_cache_dir = output_root / "data" / "npz"            # motion_135 NPZs
    render_dir = output_root / "data" / "renders"           # render videos

    for d in [motions_dir, tracked_dir, cache_dir, npz_cache_dir, render_dir]:
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
        cache_path = cache_dir / f"{motion_id}.pt"
        npz_path_out = npz_cache_dir / f"{motion_id}.npz"
        ref_render_dir = render_dir / motion_id / "reference"
        tracked_render_dir = render_dir / motion_id / "tracked"
        meta_path = output_root / "data" / "meta" / f"{motion_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] {motion_id}")
        if text:
            print(f"  Prompt: \"{text}\"")
            print(f"  Frames: {duration_frames}")
        print(f"  Cache:  {cache_path}")
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

        # ---- Step B: Retarget pipeline → ProtoMotions cache .pt ----
        if not cache_path.exists() or not args.skip_pipeline_on_existing_cache:
            print(f"  [B] Running retarget pipeline...")
            ok = run_retarget_pipeline(str(npz_path_out), str(cache_path))
            if not ok:
                failures += 1
                results.append({"id": motion_id, "status": "pipeline_failed"})
                continue
            print(f"      Pipeline OK → {cache_path}")
        else:
            print(f"  [B] Cache exists, skipping pipeline")

        # ---- Step C: Convert cache → reference JSON (for Three.js) ----
        try:
            print(f"  [C] Converting cache → reference JSON...")
            ref_info = convert_cache_to_json(str(cache_path), str(ref_json_path))
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
                    str(cache_path), str(ref_render_dir),
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
                        str(cache_path), str(tracked_render_dir),
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
            metrics = extract_metrics_from_cache(str(cache_path))
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
                "cache_pt": str(cache_path),
                "npz": str(npz_path_out),
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

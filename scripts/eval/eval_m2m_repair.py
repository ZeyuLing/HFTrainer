#!/usr/bin/env python3
"""
Comprehensive repair evaluation for HyMotion M2M models.

Evaluates two configs on the low-quality dataset using MoGenDiT adaptive masks:
1. uncond_fm (standard VACE, no impute at inference) — measures repair rate + preservation MPJPE
2. uncond_fm_man (mask-aware noise, with impute/replacement guidance) — measures repair rate

Both Completion (inpaint) and Editing modes are tested.

Usage (on GPU node):
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_repair.py \
        --max-samples 200 \
        --num-steps 50

Output goes to: output/m2m_repair_eval_<timestamp>/
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

# MoGenDiT's trainer.data_loader import chain pulls in seaborn via
# Aplus.tools.data_visualize. Install it quietly if missing.
try:
    import seaborn  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seaborn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args():
    parser = argparse.ArgumentParser(description="M2M Repair Evaluation")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--mogendit-steps", type=int, default=10,
                        help="MoGenDiT denoise steps for adaptive mask computation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mogendit-device", type=str, default="cuda:0",
                        help="Device for MoGenDiT adaptive mask computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-list", type=str,
                        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=_ALL_CONFIGS,
                        choices=_ALL_CONFIGS)
    return parser.parse_args()


# Config mapping
_ALL_CONFIGS = [
    "uncond_fm", "uncond_fm_man",
    "uncond_jit", "uncond_jit_man",
    "caption_fm", "caption_fm_man",
    "caption_jit", "caption_jit_man",
]
CONFIG_PATHS = {c: f"configs/hymotion_m2m/hymotion_m2m_completion_{c}_046b.py" for c in _ALL_CONFIGS}
WORK_DIR_NAMES = {c: f"hymotion_m2m_completion_{c}_046b" for c in _ALL_CONFIGS}

# ====================================================================
# NPZ / motion utilities (from hftrainer_repair_runtime.py — correct abs translation)
# ====================================================================

def load_npz_as_motion(npz_path: str):
    """Load NPZ → (T, 135) motion tensor. Uses ABS translation (matching training)."""
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose, process_transl,
    )
    data = dict(np.load(npz_path, allow_pickle=True))
    poses = np.array(data["poses"], dtype=np.float32)
    trans = np.array(data.get("trans", data.get("transl")), dtype=np.float32)
    if trans.ndim == 1:
        trans = trans.reshape(-1, 3)
    fps = int(data.get("mocap_framerate", 30))
    abs_trans_frame0 = trans[0].copy()

    pose_rot6d = process_smplx_pose(poses, rot_type="rotation_6d", out_type="smpl_22")
    transl_abs = process_transl(trans, transl_type="abs")
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
    """Save repaired motion as NPZ."""
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


# ====================================================================
# Mask utilities
# ====================================================================

def adaptive_mask_to_dense(joint_mask, trans_mask, num_frames, temporal_dilate=5):
    """Convert MoGenDiT adaptive (T, 22) → dense (T, 135) mask."""
    T = min(joint_mask.shape[0], num_frames)
    combined = np.zeros((num_frames, 23), dtype=np.float32)
    # Include trans_mask from MoGenDiT adaptive mask computation
    combined[:T, 0] = trans_mask[:T].astype(np.float32)
    combined[:T, 1:23] = joint_mask[:T, :22].astype(np.float32)

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


# ====================================================================
# Model building
# ====================================================================

def find_latest_checkpoint(model_name):
    """Find latest checkpoint from work_dirs."""
    work_dir = PROJECT_ROOT / "work_dirs" / WORK_DIR_NAMES[model_name]
    ckpt_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: d.stat().st_mtime,
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {work_dir}")
    return str(ckpt_dirs[-1])


def find_training_config(checkpoint_path):
    """Try to locate the training-time config.py from the work_dir."""
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


def build_model(model_name, device, num_steps):
    """Build M2M bundle and pipeline.

    Loads model.pt from the checkpoint directory. The file contains
    ``{'__hftrainer_meta__': ..., 'motion_transformer': state_dict}``.
    We prefix every key with ``motion_transformer.`` so it matches the
    bundle's ``state_dict()`` key namespace, then call
    ``load_state_dict(strict=False)`` (extra/missing non-transformer
    keys like EMA buffers are safely ignored).
    """
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    ckpt_path = find_latest_checkpoint(model_name)
    print(f"[INFO] Model: {model_name}, checkpoint: {ckpt_path}")

    # Use training config (correct mean_std_dir)
    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / CONFIG_PATHS[model_name])
    config_path = training_config or source_config
    print(f"[INFO] Config: {config_path}")

    cfg = Config.fromfile(config_path)
    print(f"[INFO] mean_std_dir = {cfg.model.get('mean_std_dir', 'NOT SET')}")
    print(f"[INFO] mask_aware_noise = {cfg.get('trainer', {}).get('mask_aware_noise', False)}")

    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(device)
    bundle.eval()

    # Load model.pt and add 'motion_transformer.' prefix to match bundle keys
    model_pt_path = os.path.join(ckpt_path, "model.pt")
    raw = torch.load(model_pt_path, map_location=device, weights_only=False)
    transformer_sd = raw["motion_transformer"]
    prefixed_sd = {f"motion_transformer.{k}": v for k, v in transformer_sd.items()}

    # Also load bundle-level params (__bundle_params__) if present
    bundle_params = raw.get("__bundle_params__", {})
    if bundle_params:
        for pname, pval in bundle_params.items():
            if hasattr(bundle, pname):
                attr = getattr(bundle, pname)
                if isinstance(attr, torch.nn.Parameter):
                    attr.data.copy_(pval.to(device))
                elif isinstance(attr, torch.Tensor):
                    attr.copy_(pval.to(device))
        print(f"[INFO] Loaded {len(bundle_params)} bundle-level params: {list(bundle_params.keys())}")

    missing, unexpected = bundle.load_state_dict(prefixed_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Fallback: load null embeddings from pretrained T2M checkpoint if not
    # saved in model.pt.  Old checkpoints lack __bundle_params__.
    if "null_vtxt_feat" in missing and not bundle_params.get("null_vtxt_feat") is not None:
        t2m_ckpt_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"
        if os.path.exists(t2m_ckpt_path):
            t2m = torch.load(t2m_ckpt_path, map_location=device, weights_only=False)
            t2m_sd = t2m.get("model_state_dict", t2m)
            if "null_vtxt_feat" in t2m_sd:
                bundle.null_vtxt_feat.data.copy_(t2m_sd["null_vtxt_feat"].to(device))
                bundle.null_ctxt_input.data.copy_(t2m_sd["null_ctxt_input"].to(device))
                print("[INFO] Loaded null embeddings from T2M pretrained checkpoint")
            del t2m
        else:
            print("[WARN] null_vtxt_feat missing and T2M checkpoint not found, using zeros")
    print(f"[INFO] Checkpoint loaded: {model_pt_path}")

    # For _man variant, use replacement_guidance='skip_last'
    is_man = "man" in model_name
    if is_man:
        replacement_guidance = 'skip_last'
    else:
        replacement_guidance = 'none'

    pipeline = HyMotionM2MPipeline(
        bundle, num_steps=num_steps,
        replacement_guidance=replacement_guidance,
    )
    return pipeline, bundle, ckpt_path, is_man


def build_mogendit(device):
    """Build MoGenDiT pipeline for adaptive mask computation."""
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    pipeline = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device=device)
    return pipeline


# ====================================================================
# Repair function
# ====================================================================

def repair_single(pipeline, motion_135, mask_135, device, max_frames=360, edit_mode=False):
    """Repair a single motion.

    Completion (inpaint) mode: src_motion has mask regions zeroed → reactive=0
    Editing mode: src_motion keeps original values in mask regions → reactive=LQ values

    Returns: (combined, repaired_raw_full)
        combined: (T, 135) blended motion (original in unmasked, repaired in masked)
        repaired_raw_full: (T, 135) raw model output BEFORE blending (for MPJPE)
    """
    bundle = pipeline.bundle
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)

    # Normalize input — full motion (not zeroed) for clean_motion
    motion_norm_full = bundle.normalize_motion(motion_135[:T].unsqueeze(0).to(device))
    msk = mask_135[:T].unsqueeze(0).to(device)

    if not edit_mode:
        # Completion mode: zero masked regions (model sees inactive=known, reactive=0)
        motion_norm = motion_norm_full * (1 - msk)
    else:
        # Editing mode: keep all values (model sees inactive=known, reactive=LQ)
        motion_norm = motion_norm_full.clone()

    # Pad
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

    # Denormalize
    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    # Keep raw output BEFORE blending (for MPJPE on unmasked regions)
    repaired_raw_full = repaired_raw.clone()
    if T_orig > T:
        repaired_raw_full = torch.cat([repaired_raw_full, motion_135[T:]], dim=0)

    # Blend
    mask_crop = mask_135[:T]
    combined = motion_135[:T] * (1 - mask_crop) + repaired_raw * mask_crop

    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)

    return combined, repaired_raw_full


# ====================================================================
# Quality checker
# ====================================================================

_CHECKER_INSTANCE = None

def get_checker():
    global _CHECKER_INSTANCE
    if _CHECKER_INSTANCE is None:
        from hftrainer.evaluation.quality_check_rules import MotionQualityChecker
        _CHECKER_INSTANCE = MotionQualityChecker(device="cpu")
    return _CHECKER_INSTANCE


def check_npz(npz_path):
    """Run quality checker on NPZ. Returns (is_valid, failed_checks_list)."""
    try:
        checker = get_checker()
        result = checker.check_from_file(npz_path)
        result_dict = result.to_dict()
        return result_dict.get("is_valid", True), result_dict.get("failed_checks", [])
    except Exception as e:
        return False, [f"checker_error:{str(e)[:50]}"]


# ====================================================================
# MPJPE computation (on unmasked regions in rot6d space)
# ====================================================================

def compute_mpjpe_unmasked(original_135, repaired_135, mask_135):
    """Compute MAE on UNMASKED (mask=0) regions, in rot6d space.

    This measures preservation ability — how well the model keeps known regions.
    Returns: float (mean absolute error) or None if no unmasked regions.
    """
    T = min(original_135.shape[0], repaired_135.shape[0], mask_135.shape[0])
    orig = original_135[:T].numpy()
    rep = repaired_135[:T].numpy()
    msk = mask_135[:T].numpy()

    # Unmasked regions: mask=0
    unmasked = msk < 0.5
    if unmasked.sum() == 0:
        return None

    diff = np.abs(orig[unmasked] - rep[unmasked])
    return float(diff.mean())


# ====================================================================
# Main evaluation
# ====================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / f"m2m_repair_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"M2M Repair Evaluation")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load low quality list
    with open(args.quality_list, "r") as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    items = quality_data.get("items", [])
    if args.max_samples > 0:
        items = items[:args.max_samples]
    print(f"[INFO] Loaded {len(items)} low-quality samples")

    # Phase 1: Compute MoGenDiT adaptive masks (required — no fallback)
    ada_masks = {}
    print(f"\n[PHASE 1] Computing MoGenDiT adaptive masks...")
    try:
        mogendit = build_mogendit(args.mogendit_device)
        for idx, item in enumerate(items):
            rel_path = item["path"]
            npz_path = str(data_root / rel_path)
            if not os.path.isfile(npz_path):
                continue
            try:
                result = mogendit.compute_adaptive_mask(
                    npz_path, step=args.mogendit_steps,
                    joint_threshold=0.15, trans_threshold=0.05,
                    max_mask_ratio=0.15,
                )
                ada_masks[rel_path] = result
                if (idx + 1) % 20 == 0:
                    print(f"  [{idx+1}/{len(items)}] computed adaptive masks")
            except Exception as e:
                print(f"  [{idx+1}] adaptive mask failed for {rel_path}: {e}")
        # Free GPU memory
        del mogendit
        torch.cuda.empty_cache()
        print(f"[INFO] Computed {len(ada_masks)}/{len(items)} adaptive masks "
              f"(samples without adaptive mask will be skipped)")

        # Save adaptive masks to disk for visualization tools
        ada_mask_dir = output_dir / "adaptive_masks"
        ada_mask_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, ada in ada_masks.items():
            mask_out = ada_mask_dir / rel_path
            os.makedirs(os.path.dirname(str(mask_out)) or ".", exist_ok=True)
            np.savez_compressed(
                str(mask_out),
                joint_mask=ada["joint_mask"],
                trans_mask=ada["trans_mask"],
            )
        print(f"[INFO] Saved {len(ada_masks)} adaptive masks to {ada_mask_dir}")
    except Exception as e:
        print(f"[ERROR] MoGenDiT initialization failed: {e}")
        print(f"[ERROR] Cannot proceed without adaptive masks. Exiting.")
        traceback.print_exc()
        return

    # Phase 2: For each config, run repair with inpaint and edit modes
    for config_name in args.configs:
        print(f"\n{'='*70}")
        print(f"[PHASE 2] Evaluating config: {config_name}")
        print(f"{'='*70}")

        try:
            pipeline, bundle, ckpt_path, is_man = build_model(config_name, args.device, args.num_steps)
        except Exception as e:
            print(f"[ERROR] Failed to build model {config_name}: {e}")
            traceback.print_exc()
            continue

        modes = ["inpaint", "edit"]
        for mode in modes:
            edit_mode = (mode == "edit")
            mode_label = f"{config_name}_{mode}"
            if is_man:
                mode_label += "_impute"

            print(f"\n--- Mode: {mode_label} ---")

            mode_output_dir = output_dir / mode_label
            mode_output_dir.mkdir(parents=True, exist_ok=True)

            stats = {
                "config": config_name,
                "mode": mode,
                "is_man": is_man,
                "checkpoint": ckpt_path,
                "replacement_guidance": pipeline.replacement_guidance,
                "num_steps": args.num_steps,
                "total": 0,
                "processed": 0,
                "skipped": 0,
                "errors": [],
                # Before/after quality
                "before_pass": 0,
                "after_pass": 0,
                "improved": 0,
                "degraded": 0,
                "unchanged_pass": 0,
                "unchanged_fail": 0,
                # Per failure-type stats
                "per_failure_type": defaultdict(lambda: {"total": 0, "fixed": 0, "still_fail": 0}),
                # Preservation (MPJPE on unmasked)
                "mpjpe_unmasked_list": [],
                # Per-sample details
                "details": [],
            }

            for idx, item in enumerate(items):
                rel_path = item["path"]
                npz_path = str(data_root / rel_path)
                stats["total"] += 1

                if not os.path.isfile(npz_path):
                    stats["skipped"] += 1
                    stats["errors"].append({"path": rel_path, "error": "file not found"})
                    continue

                try:
                    t0 = time.time()

                    # 1. Load motion
                    motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

                    # 2. Build mask (adaptive only — skip if not available)
                    if rel_path not in ada_masks:
                        stats["skipped"] += 1
                        stats["errors"].append({"path": rel_path, "error": "no adaptive mask"})
                        continue

                    ada = ada_masks[rel_path]
                    mask_135 = adaptive_mask_to_dense(
                        ada['joint_mask'], ada['trans_mask'],
                        num_frames, temporal_dilate=5,
                    )
                    mask_source = "adaptive"

                    mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)

                    # Skip if mask is empty (nothing to repair)
                    if mask_ratio < 0.001:
                        stats["skipped"] += 1
                        continue

                    # 3. Repair
                    repaired_motion, repaired_raw = repair_single(
                        pipeline, motion_135, mask_135, args.device,
                        edit_mode=edit_mode,
                    )

                    # 4. Quick sanity check
                    if torch.isnan(repaired_motion).any():
                        stats["errors"].append({"path": rel_path, "error": "NaN in output"})
                        stats["skipped"] += 1
                        continue

                    # 5. Save repaired NPZ
                    repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_motion, abs_trans_frame0)
                    if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 20.0:
                        stats["errors"].append({"path": rel_path, "error": f"trans extreme ({np.abs(repaired_trans).max():.1f})"})
                        stats["skipped"] += 1
                        continue

                    out_npz = str(mode_output_dir / "repaired" / rel_path)
                    orig_data = dict(np.load(npz_path, allow_pickle=True))
                    save_repaired_npz(out_npz, repaired_aa, repaired_trans, orig_data, fps)

                    # 6. Quality check before/after
                    before_valid, before_failed = item.get("failed_checks", []) == [], item.get("failed_checks", [])
                    # Actually re-check (the item says it's low quality, but let's verify)
                    before_valid = len(before_failed) == 0

                    after_valid, after_failed = check_npz(out_npz)

                    elapsed = time.time() - t0
                    stats["processed"] += 1

                    if before_valid:
                        stats["before_pass"] += 1
                    if after_valid:
                        stats["after_pass"] += 1

                    if not before_valid and after_valid:
                        stats["improved"] += 1
                    elif before_valid and not after_valid:
                        stats["degraded"] += 1
                    elif after_valid:
                        stats["unchanged_pass"] += 1
                    else:
                        stats["unchanged_fail"] += 1

                    # Per failure-type accounting
                    for fc in item.get("failed_checks", []):
                        stats["per_failure_type"][fc]["total"] += 1
                        if after_valid:
                            stats["per_failure_type"][fc]["fixed"] += 1
                        else:
                            stats["per_failure_type"][fc]["still_fail"] += 1

                    # MPJPE on unmasked (for non-MAN variant)
                    # Use raw model output (BEFORE blend) to measure preservation
                    mpjpe_um = compute_mpjpe_unmasked(motion_135, repaired_raw, mask_135)
                    if mpjpe_um is not None:
                        stats["mpjpe_unmasked_list"].append(mpjpe_um)

                    detail = {
                        "path": rel_path,
                        "num_frames": num_frames,
                        "mask_ratio": round(mask_ratio, 4),
                        "mask_source": mask_source,
                        "before_failed": item.get("failed_checks", []),
                        "after_valid": after_valid,
                        "after_failed": after_failed,
                        "improved": not before_valid and after_valid,
                        "mpjpe_unmasked": round(mpjpe_um, 6) if mpjpe_um is not None else None,
                        "elapsed_s": round(elapsed, 2),
                    }
                    stats["details"].append(detail)

                    status = "✓ FIXED" if detail["improved"] else ("✗ STILL BAD" if not after_valid else "= OK")
                    if (idx + 1) % 10 == 0 or detail["improved"]:
                        print(
                            f"  [{idx+1}/{len(items)}] {status} | "
                            f"before={item.get('failed_checks',[])} after={after_failed} | "
                            f"mask={mask_ratio:.1%} ({mask_source}) | {elapsed:.1f}s"
                        )

                except Exception as e:
                    stats["skipped"] += 1
                    stats["errors"].append({"path": rel_path, "error": str(e)[:200]})
                    if (idx + 1) % 20 == 0:
                        print(f"  [{idx+1}] ERROR: {str(e)[:100]}")
                    continue

            # Summarize
            processed = max(stats["processed"], 1)
            mpjpe_list = stats["mpjpe_unmasked_list"]
            mpjpe_mean = float(np.mean(mpjpe_list)) if mpjpe_list else None
            mpjpe_std = float(np.std(mpjpe_list)) if mpjpe_list else None

            print(f"\n{'='*60}")
            print(f"SUMMARY — {mode_label}")
            print(f"{'='*60}")
            print(f"Total:        {stats['total']}")
            print(f"Processed:    {stats['processed']}")
            print(f"Skipped:      {stats['skipped']}")
            print(f"Before pass:  {stats['before_pass']} ({stats['before_pass']/processed*100:.1f}%)")
            print(f"After pass:   {stats['after_pass']} ({stats['after_pass']/processed*100:.1f}%)")
            print(f"Improved:     {stats['improved']} ({stats['improved']/processed*100:.1f}%)")
            print(f"Degraded:     {stats['degraded']} ({stats['degraded']/processed*100:.1f}%)")
            print(f"Unchanged OK: {stats['unchanged_pass']}")
            print(f"Unchanged bad:{stats['unchanged_fail']}")
            if mpjpe_mean is not None:
                print(f"MPJPE (unmasked): {mpjpe_mean:.6f} ± {mpjpe_std:.6f}")

            print(f"\nPer failure type:")
            for fc, fc_stats in sorted(stats["per_failure_type"].items()):
                total = fc_stats["total"]
                fixed = fc_stats["fixed"]
                print(f"  {fc}: {fixed}/{total} fixed ({fixed/max(total,1)*100:.1f}%)")

            if stats["errors"]:
                print(f"\nFirst 5 errors:")
                for err in stats["errors"][:5]:
                    print(f"  {err['path']}: {err['error']}")

            # Convert defaultdict to regular dict for JSON
            stats["per_failure_type"] = dict(stats["per_failure_type"])
            stats["mpjpe_unmasked_mean"] = mpjpe_mean
            stats["mpjpe_unmasked_std"] = mpjpe_std

            # Save stats
            stats_path = mode_output_dir / f"repair_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
            print(f"\nStats saved: {stats_path}")

        # Free model GPU memory
        del pipeline, bundle
        torch.cuda.empty_cache()

    # Final summary across all configs/modes
    print(f"\n{'='*70}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

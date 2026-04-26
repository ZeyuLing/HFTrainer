#!/usr/bin/env python3
"""Single-GPU worker for globalrot M2M repair. Handles local↔global rotation conversion.

Called by eval_globalrot_repair_parallel.py. Each worker processes a slice of the
low-quality dataset on one GPU.

Key differences from _eval_m2m_single.py:
  - Input NPZ is local rotation → convert to global before normalize
  - Model output is in global rotation space → convert back to local before saving
  - Uses MoGenDiT adaptive denoise for mask computation (identical logic)
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

try:
    import seaborn  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seaborn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   choices=["uncond_fm_man_globalrot", "uncond_jit_man_globalrot"])
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--mogendit-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality-list", type=str,
                   default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    p.add_argument("--data-root", type=str, default="data/hymotion_data")
    p.add_argument("--output-dir", type=str, required=True)
    # Slicing: process items[start_idx:end_idx]
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)  # -1 = all
    return p.parse_args()


# ====================================================================
# Config mapping for globalrot variants
# ====================================================================
GLOBALROT_CONFIG_PATHS = {
    "uncond_fm_man_globalrot": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py",
    "uncond_jit_man_globalrot": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_globalrot_046b.py",
}
GLOBALROT_WORK_DIR_NAMES = {
    "uncond_fm_man_globalrot": "hymotion_m2m_completion_uncond_fm_man_globalrot_046b",
    "uncond_jit_man_globalrot": "hymotion_m2m_completion_uncond_jit_man_globalrot_046b",
}


# ====================================================================
# Import shared utilities from eval_m2m_repair
# ====================================================================
from scripts.eval_m2m_repair import (
    load_npz_as_motion, motion_135_to_npz_format, save_repaired_npz,
    adaptive_mask_to_dense,
    build_mogendit,
    get_checker, check_npz,
    compute_mpjpe_unmasked,
)


# ====================================================================
# Global rotation conversion
# ====================================================================
from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
    local_to_global_rot6d_torch,
    global_to_local_rot6d_torch,
)


def local_to_global_motion(motion_135: torch.Tensor) -> torch.Tensor:
    """Convert (T, 135) local rotation motion to global rotation."""
    T = motion_135.shape[0]
    transl = motion_135[:, 0:3]
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)
    rot6d_global = local_to_global_rot6d_torch(rot6d)
    return torch.cat([transl, rot6d_global.reshape(T, 132)], dim=-1)


def global_to_local_motion(motion_135: torch.Tensor) -> torch.Tensor:
    """Convert (T, 135) global rotation motion to local rotation."""
    T = motion_135.shape[0]
    transl = motion_135[:, 0:3]
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)
    rot6d_local = global_to_local_rot6d_torch(rot6d)
    return torch.cat([transl, rot6d_local.reshape(T, 132)], dim=-1)


# ====================================================================
# Model building (globalrot variant)
# ====================================================================

def find_latest_checkpoint(model_name):
    work_dir = PROJECT_ROOT / "work_dirs" / GLOBALROT_WORK_DIR_NAMES[model_name]
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


def build_model(model_name, device, num_steps):
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    ckpt_path = find_latest_checkpoint(model_name)
    print(f"[INFO] Model: {model_name}, checkpoint: {ckpt_path}")

    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / GLOBALROT_CONFIG_PATHS[model_name])
    config_path = training_config or source_config
    print(f"[INFO] Config: {config_path}")

    cfg = Config.fromfile(config_path)
    print(f"[INFO] mean_std_dir = {cfg.model.get('mean_std_dir', 'NOT SET')}")
    print(f"[INFO] rotation_space = {cfg.model.get('rotation_space', 'local')}")
    print(f"[INFO] mask_aware_noise = {cfg.get('trainer', {}).get('mask_aware_noise', False)}")

    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(device)
    bundle.eval()

    # Load checkpoint
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
        print(f"[INFO] Loaded {len(bundle_params)} bundle-level params: {list(bundle_params.keys())}")

    missing, unexpected = bundle.load_state_dict(prefixed_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")

    # Fallback: load null embeddings from T2M
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

    print(f"[INFO] Checkpoint loaded: {model_pt_path}")

    # MAN variant → replacement_guidance='skip_last'
    pipeline = HyMotionM2MPipeline(
        bundle, num_steps=num_steps,
        replacement_guidance='skip_last',
    )
    return pipeline, bundle, ckpt_path


# ====================================================================
# Repair function (globalrot-aware)
# ====================================================================

def repair_single_globalrot(pipeline, motion_135_local, mask_135, device, max_frames=360):
    """Repair using globalrot model. Input/output are local rotation.

    Flow:
    1. local motion → global rotation
    2. normalize with global stats
    3. run M2M pipeline (inpaint mode, skip_last replacement)
    4. denormalize → global rotation
    5. global → local rotation
    6. blend with original (in local space)
    """
    bundle = pipeline.bundle
    T_orig = motion_135_local.shape[0]
    T = min(T_orig, max_frames)

    # 1. Convert local → global
    motion_global = local_to_global_motion(motion_135_local[:T])

    # 2. Normalize (using global rotation stats)
    motion_norm = bundle.normalize_motion(motion_global.unsqueeze(0).to(device))
    msk = mask_135[:T].unsqueeze(0).to(device)

    # Keep full normalized motion for clean_motion (imputation)
    motion_norm_full = motion_norm.clone()

    # Inpaint mode: zero masked regions
    motion_norm = motion_norm * (1 - msk)

    # 3. Pad
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

    # 4. Denormalize → global rotation space
    repaired_latent = result["latent"][0, :T].cpu()
    repaired_global = bundle.denormalize_motion(
        repaired_latent.unsqueeze(0).to(device)
    )[0].cpu()

    # 5. Convert global → local
    repaired_local = global_to_local_motion(repaired_global)

    # Keep raw output BEFORE blending (for MPJPE on unmasked regions)
    repaired_raw_full = repaired_local.clone()
    if T_orig > T:
        repaired_raw_full = torch.cat([repaired_raw_full, motion_135_local[T:]], dim=0)

    # 6. Blend in local space
    mask_crop = mask_135[:T]
    combined = motion_135_local[:T] * (1 - mask_crop) + repaired_local * mask_crop

    if T_orig > T:
        combined = torch.cat([combined, motion_135_local[T:]], dim=0)

    return combined, repaired_raw_full


# ====================================================================
# Main
# ====================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    config_name = args.config
    mode_label = f"{config_name}_inpaint_impute"

    output_dir = Path(args.output_dir)
    mode_output_dir = output_dir / mode_label
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{mode_label}] Starting on {device}")

    # Load quality list
    with open(args.quality_list) as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    all_items = quality_data.get("items", [])

    # Slice
    start = args.start_idx
    end = args.end_idx if args.end_idx > 0 else len(all_items)
    items = all_items[start:end]
    print(f"[{mode_label}] Processing items [{start}:{end}] ({len(items)} samples)")

    # Build MoGenDiT
    print(f"[{mode_label}] Loading MoGenDiT...")
    mogendit = build_mogendit(device)

    # Build globalrot M2M model
    print(f"[{mode_label}] Loading M2M model: {config_name}...")
    pipeline, bundle, ckpt_path = build_model(config_name, device, args.num_steps)

    stats = {
        "config": config_name, "mode": "inpaint", "slice": f"{start}:{end}",
        "checkpoint": ckpt_path,
        "replacement_guidance": pipeline.replacement_guidance,
        "rotation_space": getattr(bundle, 'rotation_space', 'local'),
        "num_steps": args.num_steps,
        "total": 0, "processed": 0, "skipped": 0, "errors": [],
        "before_pass": 0, "after_pass": 0,
        "improved": 0, "degraded": 0, "unchanged_pass": 0, "unchanged_fail": 0,
        "per_failure_type": defaultdict(lambda: {"total": 0, "fixed": 0, "still_fail": 0}),
        "mpjpe_unmasked_list": [], "details": [],
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

            # 1. Load motion (local rotation)
            motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

            # 2. Compute MoGenDiT adaptive mask
            try:
                ada = mogendit.compute_adaptive_mask(
                    npz_path, step=args.mogendit_steps,
                    joint_threshold=0.15, trans_threshold=0.05,
                    max_mask_ratio=0.15,
                )
            except Exception as e:
                stats["skipped"] += 1
                stats["errors"].append({"path": rel_path, "error": f"adaptive mask: {str(e)[:100]}"})
                continue

            mask_135 = adaptive_mask_to_dense(
                ada['joint_mask'], ada['trans_mask'],
                num_frames, temporal_dilate=5,
            )
            mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)

            if mask_ratio < 0.001:
                stats["skipped"] += 1
                continue

            # 3. Repair (globalrot-aware)
            repaired_motion, repaired_raw = repair_single_globalrot(
                pipeline, motion_135, mask_135, device,
            )

            # 4. Sanity check
            if torch.isnan(repaired_motion).any():
                stats["errors"].append({"path": rel_path, "error": "NaN in output"})
                stats["skipped"] += 1
                continue

            # 5. Save repaired NPZ (output is local rotation, standard SMPL format)
            repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_motion, abs_trans_frame0)
            if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 20.0:
                stats["errors"].append({"path": rel_path, "error": f"trans extreme ({np.abs(repaired_trans).max():.1f})"})
                stats["skipped"] += 1
                continue

            out_npz = str(mode_output_dir / "repaired" / rel_path)
            orig_data = dict(np.load(npz_path, allow_pickle=True))
            save_repaired_npz(out_npz, repaired_aa, repaired_trans, orig_data, fps)

            # 6. Quality check
            before_failed = item.get("failed_checks", [])
            before_valid = len(before_failed) == 0
            after_valid, after_failed = check_npz(out_npz)

            elapsed = time.time() - t0
            stats["processed"] += 1

            if before_valid: stats["before_pass"] += 1
            if after_valid: stats["after_pass"] += 1
            if not before_valid and after_valid: stats["improved"] += 1
            elif before_valid and not after_valid: stats["degraded"] += 1
            elif after_valid: stats["unchanged_pass"] += 1
            else: stats["unchanged_fail"] += 1

            for fc in before_failed:
                stats["per_failure_type"][fc]["total"] += 1
                if after_valid: stats["per_failure_type"][fc]["fixed"] += 1
                else: stats["per_failure_type"][fc]["still_fail"] += 1

            mpjpe_um = compute_mpjpe_unmasked(motion_135, repaired_raw, mask_135)
            if mpjpe_um is not None:
                stats["mpjpe_unmasked_list"].append(mpjpe_um)

            detail = {
                "path": rel_path, "num_frames": num_frames,
                "mask_ratio": round(mask_ratio, 4),
                "before_failed": before_failed, "after_valid": after_valid,
                "after_failed": after_failed,
                "improved": not before_valid and after_valid,
                "mpjpe_unmasked": round(mpjpe_um, 6) if mpjpe_um is not None else None,
                "elapsed_s": round(elapsed, 2),
            }
            stats["details"].append(detail)

            # Incremental JSONL
            jsonl_path = mode_output_dir / f"details_live_{start}_{end}.jsonl"
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(detail, ensure_ascii=False) + "\n")

            status = "✓ FIXED" if detail["improved"] else ("✗ STILL BAD" if not after_valid else "= OK")
            if (idx + 1) % 50 == 0 or detail["improved"]:
                print(f"  [{start+idx+1}/{end}] {status} | "
                      f"before={before_failed} after={after_failed} | "
                      f"mask={mask_ratio:.1%} | {elapsed:.1f}s")

        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": str(e)[:200]})
            continue

    # Summary
    processed = max(stats["processed"], 1)
    mpjpe_list = stats["mpjpe_unmasked_list"]
    mpjpe_mean = float(np.mean(mpjpe_list)) if mpjpe_list else None
    mpjpe_std = float(np.std(mpjpe_list)) if mpjpe_list else None

    print(f"\n{'='*60}")
    print(f"SUMMARY — {mode_label} [{start}:{end}]")
    print(f"{'='*60}")
    print(f"Total:        {stats['total']}")
    print(f"Processed:    {stats['processed']}")
    print(f"Skipped:      {stats['skipped']}")
    print(f"Improved:     {stats['improved']} ({stats['improved']/processed*100:.1f}%)")
    print(f"Degraded:     {stats['degraded']}")
    if mpjpe_mean is not None:
        print(f"MPJPE (unmasked): {mpjpe_mean:.6f} ± {mpjpe_std:.6f}")

    stats["per_failure_type"] = dict(stats["per_failure_type"])
    stats["mpjpe_unmasked_mean"] = mpjpe_mean
    stats["mpjpe_unmasked_std"] = mpjpe_std

    stats_path = mode_output_dir / f"repair_stats_{start}_{end}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()

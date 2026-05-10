#!/usr/bin/env python3
"""Quick evaluation of _man M2M repair on low-quality data.

Samples 10 items per failure category from low_quality.json, computes MoGenDiT
adaptive masks, then repairs with 4 uncond _man configs using imputation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_man_repair_quick.py
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

from scripts.eval_m2m_repair import (
    load_npz_as_motion, motion_135_to_npz_format, save_repaired_npz,
    adaptive_mask_to_dense, build_mogendit, get_checker, check_npz,
    compute_mpjpe_unmasked,
)


# ============================================================================
# Checker-derived mask: run checker on NPZ, get combined (T, 22) invalid_mask
# ============================================================================
_CHECKER = None

def get_checker_instance():
    global _CHECKER
    if _CHECKER is None:
        from hftrainer.evaluation.quality_check_rules import MotionQualityChecker
        _CHECKER = MotionQualityChecker(device='cpu')
    return _CHECKER


def compute_checker_mask(npz_path: str) -> np.ndarray:
    """Run quality checker and return combined (T, 22) boolean invalid_mask.

    Merges invalid_mask from all failed checkers via boolean OR.
    """
    checker = get_checker_instance()
    result = checker.check(npz_path)

    combined = None
    for name, cr in result.all_results.items():
        inv = cr.get('invalid_mask')
        if inv is not None and inv.any():
            if combined is None:
                combined = inv.copy()
            else:
                combined = combined | inv

    if combined is None:
        # No checker provided masks — create empty
        data = dict(np.load(npz_path, allow_pickle=True))
        T = data['poses'].shape[0]
        combined = np.zeros((T, 22), dtype=bool)

    return combined


def build_combined_mask(ada_mask_result, npz_path, num_frames, temporal_dilate=5):
    """Build (T, 135) mask combining MoGenDiT adaptive mask + checker invalid_mask.

    Strategy:
    1. Start with MoGenDiT adaptive mask (good for rotation-level anomalies)
    2. OR with checker invalid_mask (catches jitter, foot_sliding, etc.)
    3. Apply temporal dilation to combined result
    """
    # Adaptive mask components
    joint_mask_ada = ada_mask_result['joint_mask']  # (T_ada, 22)
    trans_mask_ada = ada_mask_result['trans_mask']   # (T_ada,)

    # Checker mask
    checker_mask_22 = compute_checker_mask(npz_path)  # (T_checker, 22) bool

    T = min(joint_mask_ada.shape[0], checker_mask_22.shape[0], num_frames)

    # Combine: adaptive OR checker
    combined_joints = np.zeros((num_frames, 22), dtype=np.float32)
    combined_joints[:T] = np.maximum(
        joint_mask_ada[:T, :22].astype(np.float32),
        checker_mask_22[:T, :22].astype(np.float32)
    )

    # Trans mask from adaptive only (checker doesn't flag translation directly)
    combined_trans = np.zeros(num_frames, dtype=np.float32)
    T_trans = min(len(trans_mask_ada), num_frames)
    combined_trans[:T_trans] = trans_mask_ada[:T_trans].astype(np.float32)

    # Now build (T, 135) using the same logic as adaptive_mask_to_dense
    combined_grid = np.zeros((num_frames, 23), dtype=np.float32)
    combined_grid[:, 0] = combined_trans
    combined_grid[:num_frames, 1:23] = combined_joints[:num_frames]

    # Temporal dilation
    if temporal_dilate > 0:
        for col in range(23):
            arr = combined_grid[:, col]
            dilated = arr.copy()
            for _ in range(temporal_dilate):
                padded = np.pad(dilated, 1, mode='edge')
                dilated = np.maximum(np.maximum(padded[:-2], padded[2:]), padded[1:-1])
            combined_grid[:, col] = dilated

    # Expand to 135-dim
    mask = torch.zeros(num_frames, 135, dtype=torch.float32)
    for d in range(3):
        mask[:, d] = torch.from_numpy(combined_grid[:, 0])
    for j in range(22):
        start = 3 + j * 6
        end = start + 6
        if end <= 135:
            for d in range(start, end):
                mask[:, d] = torch.from_numpy(combined_grid[:, j + 1])
    return mask

# ============================================================================
# Config
# ============================================================================
CONFIGS = [
    "uncond_fm_man",
    "uncond_jit_man",
    "uncond_fm_man_globalrot",
    "uncond_jit_man_globalrot",
]
CONFIG_PATHS = {c: f"configs/hymotion_m2m/hymotion_m2m_completion_{c}_046b.py" for c in CONFIGS}
WORK_DIR_NAMES = {c: f"hymotion_m2m_completion_{c}_046b" for c in CONFIGS}

LOW_QUALITY_JSON = PROJECT_ROOT / "data/hymotion_m2m_refine_data/data_quality_list/low_quality.json"
DATA_ROOT = PROJECT_ROOT / "data/hymotion_data"


# ============================================================================
# Sampling
# ============================================================================
def sample_data(per_category=10, seed=42):
    """Sample per_category items per failure type from low_quality.json."""
    rng = np.random.RandomState(seed)
    with open(LOW_QUALITY_JSON) as f:
        data = json.load(f)
    items = data.get("items", data)

    by_reason = defaultdict(list)
    for item in items:
        primary = item["failed_checks"][0] if item.get("failed_checks") else "unknown"
        by_reason[primary].append(item)

    sampled = []
    for reason in sorted(by_reason.keys()):
        pool = by_reason[reason]
        # Only keep items whose files exist
        valid = [it for it in pool if (DATA_ROOT / it["path"]).is_file()]
        rng.shuffle(valid)
        selected = valid[:per_category]
        for item in selected:
            sampled.append({
                "path": item["path"],
                "category": reason,
                "failed_checks": item.get("failed_checks", []),
            })
    return sampled


# ============================================================================
# Global rotation conversion
# ============================================================================
def _has_fk_utils():
    try:
        from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
            local_to_global_rot6d_torch, global_to_local_rot6d_torch)
        return True
    except ImportError:
        return False


def local_to_global_motion(m):
    from hftrainer.datasets.motion.motionhub.transforms.fk_utils import local_to_global_rot6d_torch
    T = m.shape[0]
    return torch.cat([m[:, 0:3], local_to_global_rot6d_torch(
        m[:, 3:135].reshape(T, 22, 6)).reshape(T, 132)], dim=-1)


def global_to_local_motion(m):
    from hftrainer.datasets.motion.motionhub.transforms.fk_utils import global_to_local_rot6d_torch
    T = m.shape[0]
    return torch.cat([m[:, 0:3], global_to_local_rot6d_torch(
        m[:, 3:135].reshape(T, 22, 6)).reshape(T, 132)], dim=-1)


# ============================================================================
# Model building
# ============================================================================
def find_latest_checkpoint(model_name):
    work_dir = PROJECT_ROOT / "work_dirs" / WORK_DIR_NAMES[model_name]
    ckpt_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: d.stat().st_mtime)
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {work_dir}")
    return str(ckpt_dirs[-1])


def find_training_config(checkpoint_path):
    work_dir = Path(checkpoint_path).parent
    for rd in sorted([d for d in work_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()],
                     key=lambda d: d.name, reverse=True):
        cfg_path = rd / "config.py"
        if cfg_path.is_file():
            return str(cfg_path)
    return None


def build_model(model_name, device, num_steps=50):
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    ckpt_path = find_latest_checkpoint(model_name)
    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / CONFIG_PATHS[model_name])
    config_path = training_config or source_config
    cfg = Config.fromfile(config_path)

    bundle = HyMotionM2MBundle.from_config(cfg.model).to(device).eval()

    model_pt_path = os.path.join(ckpt_path, "model.pt")
    raw = torch.load(model_pt_path, map_location=device, weights_only=False)
    prefixed_sd = {f"motion_transformer.{k}": v for k, v in raw["motion_transformer"].items()}

    for pname, pval in raw.get("__bundle_params__", {}).items():
        if hasattr(bundle, pname):
            attr = getattr(bundle, pname)
            if isinstance(attr, torch.nn.Parameter):
                attr.data.copy_(pval.to(device))
            elif isinstance(attr, torch.Tensor):
                attr.copy_(pval.to(device))

    missing, _ = bundle.load_state_dict(prefixed_sd, strict=False)
    if "null_vtxt_feat" in missing:
        t2m_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"
        if os.path.exists(t2m_path):
            t2m = torch.load(t2m_path, map_location=device, weights_only=False)
            t2m_sd = t2m.get("model_state_dict", t2m)
            if "null_vtxt_feat" in t2m_sd:
                bundle.null_vtxt_feat.data.copy_(t2m_sd["null_vtxt_feat"].to(device))
                bundle.null_ctxt_input.data.copy_(t2m_sd["null_ctxt_input"].to(device))
            del t2m

    pipeline = HyMotionM2MPipeline(
        bundle, num_steps=num_steps,
        replacement_guidance='skip_last',
    )
    print(f"[INFO] Loaded {model_name}: {ckpt_path}, "
          f"mean_std={cfg.model.get('mean_std_dir','N/A')}, "
          f"rotation_space={getattr(bundle, 'rotation_space', 'local')}")
    return pipeline, bundle, ckpt_path


# ============================================================================
# Repair
# ============================================================================
def repair_single(pipeline, motion_135, mask_135, device, max_frames=360):
    """Repair using imputation (skip_last replacement guidance).

    For globalrot models: local→global→normalize→pipeline→denormalize→global→local.
    For local models: normalize→pipeline→denormalize.
    """
    bundle = pipeline.bundle
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)
    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'

    motion_in = motion_135[:T].clone()
    if is_global:
        motion_in = local_to_global_motion(motion_in)

    # Normalize
    motion_norm_full = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))
    msk = mask_135[:T].unsqueeze(0).to(device)

    # VACE: zero masked regions
    motion_norm_zeroed = motion_norm_full * (1 - msk)

    # Pad
    if T < max_frames:
        pad_len = max_frames - T
        motion_norm_zeroed = torch.nn.functional.pad(motion_norm_zeroed, (0, 0, 0, pad_len), value=0)
        motion_norm_full = torch.nn.functional.pad(motion_norm_full, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    batch = {
        "src_motion": motion_norm_zeroed,
        "src_mask": msk,
        "clean_motion": motion_norm_full,
        "src_length": [T],
        "tgt_length": [T],
    }

    with torch.no_grad():
        result = pipeline(batch)

    # Denormalize
    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    if is_global:
        repaired_raw = global_to_local_motion(repaired_raw)

    # Raw output before blend (for MPJPE)
    repaired_raw_full = repaired_raw.clone()
    if T_orig > T:
        repaired_raw_full = torch.cat([repaired_raw_full, motion_135[T:]], dim=0)

    # Blend: original in unmasked, repaired in masked
    mask_crop = mask_135[:T]
    combined = motion_135[:T] * (1 - mask_crop) + repaired_raw * mask_crop

    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)

    return combined, repaired_raw_full


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--mogendit-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output" / f"man_repair_quick_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Quick _man Repair Evaluation")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # 1. Sample data
    sampled = sample_data(per_category=args.per_category, seed=args.seed)
    print(f"[INFO] Sampled {len(sampled)} items across categories:")
    cat_counts = defaultdict(int)
    for s in sampled:
        cat_counts[s["category"]] += 1
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")

    # Save sample list
    with open(output_dir / "sample_list.json", "w") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    # 2. Compute MoGenDiT adaptive masks
    print(f"\n[PHASE 1] Computing MoGenDiT adaptive masks...")
    ada_masks = {}
    mogendit = build_mogendit(args.device)

    for idx, item in enumerate(sampled):
        npz_path = str(DATA_ROOT / item["path"])
        try:
            result = mogendit.compute_adaptive_mask(
                npz_path, step=args.mogendit_steps,
                joint_threshold=0.15, trans_threshold=0.05,
                max_mask_ratio=0.15,
            )
            ada_masks[item["path"]] = result
        except Exception as e:
            print(f"  [{idx+1}] mask failed: {item['path']}: {e}")
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(sampled)}] masks computed")

    del mogendit
    torch.cuda.empty_cache()
    print(f"[INFO] Computed {len(ada_masks)}/{len(sampled)} adaptive masks")

    # Save adaptive masks for visualization
    ada_mask_dir = output_dir / "adaptive_masks"
    ada_mask_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, ada in ada_masks.items():
        mask_out = ada_mask_dir / rel_path
        os.makedirs(os.path.dirname(str(mask_out)) or ".", exist_ok=True)
        np.savez_compressed(str(mask_out),
                            joint_mask=ada["joint_mask"],
                            trans_mask=ada["trans_mask"])
    print(f"[INFO] Saved {len(ada_masks)} adaptive masks to {ada_mask_dir}")

    # 3. Evaluate each config
    all_results = {}

    for config_name in CONFIGS:
        print(f"\n{'='*70}")
        print(f"[PHASE 2] Config: {config_name}")
        print(f"{'='*70}")

        try:
            pipeline, bundle, ckpt_path = build_model(config_name, args.device, args.num_steps)
        except Exception as e:
            print(f"[ERROR] Failed to build {config_name}: {e}")
            traceback.print_exc()
            continue

        stats = {
            "config": config_name,
            "checkpoint": ckpt_path,
            "total": 0, "processed": 0, "skipped": 0,
            "improved": 0, "degraded": 0, "unchanged_pass": 0, "unchanged_fail": 0,
            "before_fail": 0, "after_pass": 0,
            "per_category": defaultdict(lambda: {"total": 0, "improved": 0, "after_pass": 0}),
            "mpjpe_unmasked_list": [],
            "errors": [],
        }

        for idx, item in enumerate(sampled):
            rel_path = item["path"]
            npz_path = str(DATA_ROOT / rel_path)
            stats["total"] += 1

            if rel_path not in ada_masks:
                stats["skipped"] += 1
                continue

            try:
                motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

                ada = ada_masks[rel_path]
                mask_135 = build_combined_mask(
                    ada, npz_path, num_frames, temporal_dilate=5)

                mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)
                if mask_ratio < 0.001:
                    stats["skipped"] += 1
                    continue

                # Repair
                repaired_motion, repaired_raw = repair_single(
                    pipeline, motion_135, mask_135, args.device)

                if torch.isnan(repaired_motion).any():
                    stats["errors"].append({"path": rel_path, "error": "NaN"})
                    stats["skipped"] += 1
                    continue

                # Save
                repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_motion, abs_trans_frame0)
                if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 20.0:
                    stats["errors"].append({"path": rel_path, "error": "trans extreme"})
                    stats["skipped"] += 1
                    continue

                out_npz = str(output_dir / config_name / "repaired" / rel_path)
                orig_data = dict(np.load(npz_path, allow_pickle=True))
                save_repaired_npz(out_npz, repaired_aa, repaired_trans, orig_data, fps)

                # Quality check
                before_failed = item.get("failed_checks", [])
                before_valid = len(before_failed) == 0
                after_valid, after_failed = check_npz(out_npz)

                stats["processed"] += 1
                if not before_valid:
                    stats["before_fail"] += 1
                if after_valid:
                    stats["after_pass"] += 1

                cat = item["category"]
                stats["per_category"][cat]["total"] += 1

                if not before_valid and after_valid:
                    stats["improved"] += 1
                    stats["per_category"][cat]["improved"] += 1
                elif before_valid and not after_valid:
                    stats["degraded"] += 1
                elif after_valid:
                    stats["unchanged_pass"] += 1
                else:
                    stats["unchanged_fail"] += 1

                if after_valid:
                    stats["per_category"][cat]["after_pass"] += 1

                mpjpe = compute_mpjpe_unmasked(motion_135, repaired_raw, mask_135)
                if mpjpe is not None:
                    stats["mpjpe_unmasked_list"].append(mpjpe)

                status = "✓" if (not before_valid and after_valid) else ("✗" if not after_valid else "=")
                if (idx + 1) % 20 == 0 or (not before_valid and after_valid):
                    print(f"  [{idx+1}/{len(sampled)}] {status} {cat} mask={mask_ratio:.1%} "
                          f"before={before_failed} after={after_failed}")

            except Exception as e:
                stats["skipped"] += 1
                stats["errors"].append({"path": rel_path, "error": str(e)[:200]})

        # Summary
        processed = max(stats["processed"], 1)
        before_fail = max(stats["before_fail"], 1)
        improve_rate = stats["improved"] / before_fail * 100
        mpjpe_list = stats["mpjpe_unmasked_list"]
        mpjpe_mean = float(np.mean(mpjpe_list)) if mpjpe_list else None

        print(f"\n--- {config_name} ---")
        print(f"Processed: {stats['processed']}, Skipped: {stats['skipped']}")
        print(f"Before fail: {stats['before_fail']}")
        print(f"Improved: {stats['improved']} ({improve_rate:.1f}% of failed)")
        print(f"Degraded: {stats['degraded']}")
        print(f"After pass: {stats['after_pass']}/{stats['processed']} ({stats['after_pass']/processed*100:.1f}%)")
        if mpjpe_mean is not None:
            print(f"MPJPE unmasked: {mpjpe_mean:.6f}")

        print(f"\nPer-category:")
        for cat in sorted(stats["per_category"]):
            cs = stats["per_category"][cat]
            rate = cs["improved"] / max(cs["total"], 1) * 100
            print(f"  {cat}: {cs['improved']}/{cs['total']} fixed ({rate:.0f}%)")

        if stats["errors"]:
            print(f"\nErrors ({len(stats['errors'])}):")
            for err in stats["errors"][:5]:
                print(f"  {err['path']}: {err['error']}")

        stats["improve_rate"] = improve_rate
        stats["per_category"] = dict(stats["per_category"])
        stats["mpjpe_unmasked_mean"] = mpjpe_mean
        all_results[config_name] = stats

        # Save per-config stats
        stats_path = output_dir / config_name / "stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        del pipeline, bundle
        torch.cuda.empty_cache()

    # Final comparison
    print(f"\n\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Improved':>10} {'Rate':>8} {'After Pass':>12} {'MPJPE':>10}")
    print(f"{'-'*75}")
    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        s = all_results[config_name]
        mpjpe_str = f"{s['mpjpe_unmasked_mean']:.6f}" if s['mpjpe_unmasked_mean'] else "N/A"
        print(f"{config_name:<35} {s['improved']:>10} {s['improve_rate']:>7.1f}% "
              f"{s['after_pass']}/{s['processed']:>10} {mpjpe_str:>10}")

    # Save combined report
    report_path = output_dir / "combined_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()

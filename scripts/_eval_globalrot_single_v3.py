#!/usr/bin/env python3
"""Single-GPU worker for globalrot M2M repair v3.

Strategy:
  - Phase 1: MoGenDIT computes adaptive mask (proven quality detector)
  - Phase 2: M2M repairs via denoise-from-near-clean + imputation (no hard blend)
    - Starts from original motion with small noise (SDEdit)
    - Per-step imputation restores known regions (skip_last)
    - Sliding window 360 frames with 20 overlap

Called by eval_globalrot_repair_parallel_v3.py.
"""

import argparse
import json
import os
import sys
import time
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
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--denoise-strength", type=float, default=0.3,
                   help="Noise level for SDEdit. Higher = more repair capacity. "
                        "0.3 means start ODE from t=0.7 (30%% noise + 70%% clean)")
    p.add_argument("--mogendit-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality-list", type=str,
                   default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    p.add_argument("--data-root", type=str, default="data/hymotion_data")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    p.add_argument("--window-size", type=int, default=360)
    p.add_argument("--window-overlap", type=int, default=20)
    return p.parse_args()


# ====================================================================
# Config mapping
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
# Imports
# ====================================================================
from scripts.eval_m2m_repair import (
    load_npz_as_motion, motion_135_to_npz_format, save_repaired_npz,
    adaptive_mask_to_dense, build_mogendit,
    get_checker, check_npz,
)

from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
    local_to_global_rot6d_torch,
    global_to_local_rot6d_torch,
)


def local_to_global_motion(m):
    T = m.shape[0]
    return torch.cat([m[:, 0:3], local_to_global_rot6d_torch(
        m[:, 3:135].reshape(T, 22, 6)).reshape(T, 132)], dim=-1)


def global_to_local_motion(m):
    T = m.shape[0]
    return torch.cat([m[:, 0:3], global_to_local_rot6d_torch(
        m[:, 3:135].reshape(T, 22, 6)).reshape(T, 132)], dim=-1)


# ====================================================================
# Model building
# ====================================================================

def find_latest_checkpoint(model_name):
    work_dir = PROJECT_ROOT / "work_dirs" / GLOBALROT_WORK_DIR_NAMES[model_name]
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


def build_model(model_name, device):
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    ckpt_path = find_latest_checkpoint(model_name)
    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / GLOBALROT_CONFIG_PATHS[model_name])
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

    print(f"[INFO] Loaded {model_name} from {ckpt_path}")
    # Don't set replacement_guidance on pipeline — we handle it manually
    pipeline = HyMotionM2MPipeline(bundle, num_steps=50, replacement_guidance='none')
    return pipeline, bundle, ckpt_path


# ====================================================================
# Core: denoise-from-near-clean with imputation (MoGenDIT-aligned)
# ====================================================================

def denoise_repair_window(bundle, motion_norm_window, keep_mask, device,
                          denoise_strength=0.3, denoise_steps=10):
    """Repair one window via denoise-from-near-clean + imputation.

    Aligned with MoGenDIT denoise():
      MoGenDIT: q_sample(x0, t=step, obs_mask) → DDIM denoise with per-step impute
      Us: x_t = (1-t_start)*noise + t_start*clean, obs regions = clean
          → ODE from t_start to 1.0 with per-step impute (flow_interp mode)

    Args:
        bundle: HyMotionM2MBundle
        motion_norm_window: (1, W, 135) normalized global rotation
        keep_mask: (1, W, 135) bool, True = keep/observed
        device: cuda device
        denoise_strength: noise amount. 0.3 = t_start=0.7
        denoise_steps: ODE steps for the denoise segment
    Returns:
        (1, W, 135) denoised motion in normalized global rotation space
    """
    B, W, D = motion_norm_window.shape
    src_mask_float = (~keep_mask).float()  # 1=generate, 0=keep

    # VACE: src has mask regions zeroed
    src_masked = motion_norm_window * keep_mask.float()
    vace_context = bundle.prepare_vace_input(src_motion=src_masked, src_mask=src_mask_float)

    # Null text
    vtxt = bundle.null_vtxt_feat.expand(B, 1, -1)
    ctxt = bundle.null_ctxt_input.expand(B, 1, -1)
    pad_mask = torch.ones(B, W, dtype=torch.bool, device=device)
    ctxt_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    def velocity_fn(t_val, x):
        x_input = torch.cat([x, vace_context], dim=-1)
        pred = bundle.predict_flow(
            x_input=x_input, ctxt_input=ctxt, vtxt_input=vtxt,
            timesteps=t_val.expand(B),
            x_mask_temporal=pad_mask, ctxt_mask_temporal=ctxt_mask)
        if bundle.pred_type == 'x1':
            pred = (pred - x) / (1.0 - t_val).clamp_min(0.05)
        return pred

    # Step 1: Create noisy starting point (aligned with MoGenDIT q_sample)
    # Flow matching: x_t = (1-t)*noise + t*clean
    t_start = 1.0 - denoise_strength  # e.g., 0.7 for strength=0.3
    z = torch.randn_like(motion_norm_window)
    x = (1.0 - t_start) * z + t_start * motion_norm_window

    # Keep regions stay clean (= MoGenDIT obs_mask in q_sample: obs regions not noised)
    x[keep_mask] = motion_norm_window[keep_mask]

    # Step 2: ODE from t_start → 1.0 with per-step imputation (skip_last)
    t_sched = torch.linspace(t_start, 1.0, denoise_steps + 1,
                             device=device, dtype=motion_norm_window.dtype)

    for i in range(denoise_steps):
        dt = t_sched[i + 1] - t_sched[i]
        is_last = (i == denoise_steps - 1)

        v = velocity_fn(t_sched[i], x)
        x = x + v * dt

        # Imputation: restore keep regions using flow-interp (train-consistent)
        if not is_last:
            t_next = t_sched[i + 1]
            x_interp = (1.0 - t_next) * z + t_next * motion_norm_window
            x[keep_mask] = x_interp[keep_mask]

    return x


# ====================================================================
# Full motion repair with sliding window
# ====================================================================

def repair_full_motion(bundle, motion_local, mask_135, device,
                       denoise_strength=0.3, denoise_steps=10,
                       window_size=360, window_overlap=20):
    """Repair full motion. No hard blend — model output used directly.

    Input: (T, 135) local rotation, (T, 135) mask (1=repair, 0=keep)
    Output: (T, 135) repaired local rotation
    """
    T_orig = motion_local.shape[0]

    # Convert to global, normalize
    motion_global = local_to_global_motion(motion_local)
    motion_norm = bundle.normalize_motion(motion_global.unsqueeze(0).to(device))  # (1, T, 135)
    mask_dev = mask_135.unsqueeze(0).to(device)  # (1, T, 135)
    keep_mask_full = mask_dev < 0.5  # True = keep

    # Sliding window repair
    repaired_norm = motion_norm.clone()
    current_idx = 0
    prev_overlap = 0

    while current_idx < T_orig:
        begin = current_idx
        end = min(begin + window_size, T_orig)

        window_motion = repaired_norm[:, begin:end, :]
        window_keep = keep_mask_full[:, begin:end, :]

        # Denoise this window
        with torch.no_grad():
            window_repaired = denoise_repair_window(
                bundle, window_motion, window_keep, device,
                denoise_strength=denoise_strength,
                denoise_steps=denoise_steps,
            )

        # Stitch: skip the overlap prefix (it was kept from previous window)
        write_start = prev_overlap if prev_overlap > 0 else 0
        repaired_norm[:, begin + write_start:end, :] = window_repaired[:, write_start:, :]

        prev_overlap = window_overlap
        current_idx = end - window_overlap
        if end >= T_orig:
            break

    # Denormalize → global → local
    repaired_global = bundle.denormalize_motion(repaired_norm)[0].cpu()
    repaired_local = global_to_local_motion(repaired_global)

    return repaired_local


# ====================================================================
# Main
# ====================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    config_name = args.config
    mode_label = f"{config_name}_denoise_impute"

    output_dir = Path(args.output_dir)
    mode_output_dir = output_dir / mode_label
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{mode_label}] Starting on {device}")
    print(f"[{mode_label}] denoise_strength={args.denoise_strength}, steps={args.denoise_steps}")

    with open(args.quality_list) as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    all_items = quality_data.get("items", [])
    start = args.start_idx
    end = args.end_idx if args.end_idx > 0 else len(all_items)
    items = all_items[start:end]
    print(f"[{mode_label}] Items [{start}:{end}] ({len(items)} samples)")

    # Build MoGenDIT for mask, then M2M for repair
    print(f"[{mode_label}] Loading MoGenDiT...")
    mogendit = build_mogendit(device)
    print(f"[{mode_label}] Loading M2M: {config_name}...")
    pipeline, bundle, ckpt_path = build_model(config_name, device)

    stats = {
        "config": config_name, "mode": "denoise_impute",
        "denoise_strength": args.denoise_strength,
        "denoise_steps": args.denoise_steps,
        "slice": f"{start}:{end}", "checkpoint": ckpt_path,
        "total": 0, "processed": 0, "skipped": 0, "errors": [],
        "before_pass": 0, "after_pass": 0,
        "improved": 0, "degraded": 0, "unchanged_pass": 0, "unchanged_fail": 0,
        "per_failure_type": defaultdict(lambda: {"total": 0, "fixed": 0, "still_fail": 0}),
        "details": [],
    }

    for idx, item in enumerate(items):
        rel_path = item["path"]
        npz_path = str(data_root / rel_path)
        stats["total"] += 1

        if not os.path.isfile(npz_path):
            stats["skipped"] += 1
            continue

        try:
            t0 = time.time()

            # 1. Load motion (local)
            motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

            # 2. MoGenDIT adaptive mask
            try:
                ada = mogendit.compute_adaptive_mask(
                    npz_path, step=args.mogendit_steps,
                    joint_threshold=0.15, trans_threshold=0.05, max_mask_ratio=0.15)
            except Exception as e:
                stats["skipped"] += 1
                stats["errors"].append({"path": rel_path, "error": f"mask: {str(e)[:80]}"})
                continue

            mask_135 = adaptive_mask_to_dense(
                ada['joint_mask'], ada['trans_mask'], num_frames, temporal_dilate=5)
            mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)
            if mask_ratio < 0.001:
                stats["skipped"] += 1
                continue

            # 3. Repair (denoise from near-clean + imputation, no hard blend)
            with torch.no_grad():
                repaired_local = repair_full_motion(
                    bundle, motion_135, mask_135, device,
                    denoise_strength=args.denoise_strength,
                    denoise_steps=args.denoise_steps,
                    window_size=args.window_size,
                    window_overlap=args.window_overlap,
                )

            # 4. Sanity
            if torch.isnan(repaired_local).any():
                stats["errors"].append({"path": rel_path, "error": "NaN"})
                stats["skipped"] += 1
                continue

            # 5. Save
            repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_local, abs_trans_frame0)
            if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 20.0:
                stats["errors"].append({"path": rel_path, "error": "trans extreme"})
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

            detail = {
                "path": rel_path, "num_frames": num_frames,
                "mask_ratio": round(mask_ratio, 4),
                "before_failed": before_failed, "after_valid": after_valid,
                "after_failed": after_failed,
                "improved": not before_valid and after_valid,
                "elapsed_s": round(elapsed, 2),
            }
            stats["details"].append(detail)

            jsonl_path = mode_output_dir / f"details_live_{start}_{end}.jsonl"
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(detail, ensure_ascii=False) + "\n")

            status = "✓ FIXED" if detail["improved"] else ("✗ STILL BAD" if not after_valid else "= OK")
            if (idx + 1) % 50 == 0 or detail["improved"]:
                print(f"  [{start+idx+1}/{end}] {status} | before={before_failed} after={after_failed} | "
                      f"mask={mask_ratio:.1%} | {elapsed:.1f}s")

        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": str(e)[:200]})
            continue

    processed = max(stats["processed"], 1)
    print(f"\n{'='*60}")
    print(f"SUMMARY — {mode_label} [{start}:{end}]")
    print(f"Processed: {stats['processed']}, Improved: {stats['improved']} ({stats['improved']/processed*100:.1f}%), Degraded: {stats['degraded']}")

    stats["per_failure_type"] = dict(stats["per_failure_type"])
    stats_path = mode_output_dir / f"repair_stats_{start}_{end}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()

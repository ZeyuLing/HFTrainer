#!/usr/bin/env python3
"""Single-GPU worker for globalrot M2M repair, aligned with MoGenDIT ada_denoise.

Key alignment with MoGenDIT:
  1. Adaptive mask: computed in normalized space (not axis-angle)
  2. Light denoise starting from near-clean (SDEdit): x_t = (1-t)*noise + t*clean
     with t_start close to 1.0, so only a small ODE segment is traversed.
  3. Per-step imputation: keep_mask regions restored to clean at every step (skip_last).
  4. Two-phase adaptive: first pass detects high-change dims, second pass
     focuses repair on those dims with stronger imputation on unchanged regions.
  5. Sliding window: 360-frame windows with overlap stitching.
  6. No hard blending: model output is used directly (imputation handles preservation).

Called by eval_globalrot_repair_parallel_v2.py.
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
    p.add_argument("--denoise-steps", type=int, default=10,
                   help="ODE steps for denoise (aligned with MoGenDIT step=10)")
    p.add_argument("--denoise-strength", type=float, default=0.02,
                   help="Noise strength for SDEdit. 0.02 ≈ MoGenDIT t=10/1000. "
                        "Flow matching: t_start = 1 - strength")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality-list", type=str,
                   default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json")
    p.add_argument("--data-root", type=str, default="data/hymotion_data")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    # Adaptive mask thresholds (in normalized M2M space, not axis-angle)
    p.add_argument("--change-threshold", type=float, default=0.1,
                   help="Per-dim change threshold in normalized space (matching MoGenDIT)")
    p.add_argument("--max-mask-ratio", type=float, default=0.15)
    # Window
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
    get_checker, check_npz,
    compute_mpjpe_unmasked,
)

from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
    local_to_global_rot6d_torch,
    global_to_local_rot6d_torch,
)


def local_to_global_motion(motion_135):
    T = motion_135.shape[0]
    transl = motion_135[:, 0:3]
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)
    rot6d_global = local_to_global_rot6d_torch(rot6d)
    return torch.cat([transl, rot6d_global.reshape(T, 132)], dim=-1)


def global_to_local_motion(motion_135):
    T = motion_135.shape[0]
    transl = motion_135[:, 0:3]
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)
    rot6d_local = global_to_local_rot6d_torch(rot6d)
    return torch.cat([transl, rot6d_local.reshape(T, 132)], dim=-1)


# ====================================================================
# Model building
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


def build_model(model_name, device):
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    ckpt_path = find_latest_checkpoint(model_name)
    print(f"[INFO] Model: {model_name}, checkpoint: {ckpt_path}")

    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / GLOBALROT_CONFIG_PATHS[model_name])
    config_path = training_config or source_config
    cfg = Config.fromfile(config_path)
    print(f"[INFO] Config: {config_path}, rotation_space={cfg.model.get('rotation_space','local')}")

    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(device).eval()

    model_pt_path = os.path.join(ckpt_path, "model.pt")
    raw = torch.load(model_pt_path, map_location=device, weights_only=False)
    prefixed_sd = {f"motion_transformer.{k}": v for k, v in raw["motion_transformer"].items()}

    bundle_params = raw.get("__bundle_params__", {})
    if bundle_params:
        for pname, pval in bundle_params.items():
            if hasattr(bundle, pname):
                attr = getattr(bundle, pname)
                if isinstance(attr, torch.nn.Parameter):
                    attr.data.copy_(pval.to(device))
                elif isinstance(attr, torch.Tensor):
                    attr.copy_(pval.to(device))

    missing, _ = bundle.load_state_dict(prefixed_sd, strict=False)

    # Fallback null embeddings
    if "null_vtxt_feat" in missing:
        t2m_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"
        if os.path.exists(t2m_path):
            t2m = torch.load(t2m_path, map_location=device, weights_only=False)
            t2m_sd = t2m.get("model_state_dict", t2m)
            if "null_vtxt_feat" in t2m_sd:
                bundle.null_vtxt_feat.data.copy_(t2m_sd["null_vtxt_feat"].to(device))
                bundle.null_ctxt_input.data.copy_(t2m_sd["null_ctxt_input"].to(device))
            del t2m

    # Pipeline: replacement_guidance='skip_last' for MAN, no sdedit (we handle it manually)
    pipeline = HyMotionM2MPipeline(bundle, num_steps=50, replacement_guidance='skip_last')
    return pipeline, bundle, ckpt_path


# ====================================================================
# Core: MoGenDIT-aligned denoise for flow matching M2M
# ====================================================================

def fm_denoise_impute(pipeline, motion_norm_global, keep_mask, device,
                      denoise_strength=0.02, denoise_steps=10):
    """Flow matching equivalent of MoGenDIT's denoise + imputation.

    MoGenDIT flow:
      1. q_sample(x0, t=step, obs_mask) → add noise, keep obs_mask regions clean
      2. DDIM denoise step-1→0, each step restore keep_mask regions

    Our flow matching equivalent:
      1. x_{t_start} = (1-t_start)*noise + t_start*clean, keep_mask regions = clean
         where t_start = 1 - denoise_strength (close to 1, so close to clean)
      2. ODE integrate from t_start to 1.0, each step restore keep_mask regions

    Args:
        pipeline: HyMotionM2MPipeline (with bundle, predict_flow, etc.)
        motion_norm_global: (1, T, 135) normalized global rotation motion
        keep_mask: (1, T, 135) bool, True = keep (known), False = generate (repair)
        device: torch device
        denoise_strength: amount of noise. 0.02 ≈ MoGenDIT 10/1000 steps
        denoise_steps: number of ODE steps for the denoise segment
    """
    bundle = pipeline.bundle
    B, T, D = motion_norm_global.shape
    src_mask_float = (~keep_mask).float()  # 1 = generate, 0 = keep

    # Build VACE context: src_motion has masked regions zeroed
    src_motion_masked = motion_norm_global * keep_mask.float()
    vace_context = bundle.prepare_vace_input(
        src_motion=src_motion_masked,
        src_mask=src_mask_float,
    )

    # Prepare text (null — unconditioned)
    vtxt_input = bundle.null_vtxt_feat.expand(B, 1, -1)
    ctxt_input = bundle.null_ctxt_input.expand(B, 1, -1)
    tgt_padding_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    ctxt_mask_temporal = torch.ones(B, 1, dtype=torch.bool, device=device)

    def fn(t_val, x):
        """ODE velocity field."""
        x_input = torch.cat([x, vace_context], dim=-1)
        x_pred = bundle.predict_flow(
            x_input=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=t_val.expand(B),
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )
        # Convert x1 prediction to velocity if needed
        if bundle.pred_type == 'x1':
            t_eps = 0.05
            x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)
        return x_pred

    # Step 1: Create noisy starting point
    # Flow matching: x_t = (1-t)*noise + t*clean
    # t_start close to 1 → mostly clean with tiny noise
    t_start = 1.0 - denoise_strength
    z = torch.randn_like(motion_norm_global)
    x = (1.0 - t_start) * z + t_start * motion_norm_global

    # Keep regions stay clean (matching MoGenDIT obs_mask in q_sample)
    x[keep_mask] = motion_norm_global[keep_mask]

    # Step 2: ODE from t_start → 1.0 with per-step imputation
    t_schedule = torch.linspace(t_start, 1.0, denoise_steps + 1,
                                device=device, dtype=motion_norm_global.dtype)

    for i in range(denoise_steps):
        t_curr = t_schedule[i]
        dt = t_schedule[i + 1] - t_schedule[i]
        is_last = (i == denoise_steps - 1)

        v = fn(t_curr, x)
        x = x + v * dt

        # Imputation: restore keep regions (skip_last mode)
        if not is_last:
            # For flow_interp: known regions follow the FM interpolation path
            t_next = t_schedule[i + 1]
            x_interp = (1.0 - t_next) * z + t_next * motion_norm_global
            x[keep_mask] = x_interp[keep_mask]

    return x  # (1, T, 135) in normalized global rotation space


def adaptive_denoise_m2m(pipeline, motion_norm_global, device,
                         denoise_strength=0.02, denoise_steps=10,
                         change_threshold=0.1, max_mask_ratio=0.15):
    """Two-phase adaptive denoise aligned with MoGenDIT ada_denoise.

    Phase 1: Light denoise the entire motion (all keep_mask = first-frame only)
             Compare before/after in normalized space.
    Phase 2: Mark low-change regions as keep, high-change as generate.
             Denoise again with the adaptive mask.

    Returns: (denoised_norm_global, keep_mask)
    """
    B, T, D = motion_norm_global.shape

    # Phase 1: Light denoise (keep_mask = frame 0 only, matching MoGenDIT)
    keep_mask_phase1 = torch.zeros(B, T, D, dtype=torch.bool, device=device)
    keep_mask_phase1[:, :1, :] = True  # keep first frame

    denoised_phase1 = fm_denoise_impute(
        pipeline, motion_norm_global, keep_mask_phase1, device,
        denoise_strength=denoise_strength, denoise_steps=denoise_steps,
    )

    # Phase 2: Compute change map in normalized space
    change = torch.abs(motion_norm_global - denoised_phase1)  # (1, T, D)

    # Mark low-change regions as keep (matching MoGenDIT's ada_denoise logic)
    low_change = change <= change_threshold  # keep these
    high_change = change > change_threshold  # repair these

    # Cap mask ratio
    mask_ratio = high_change.float().mean().item()
    if mask_ratio > max_mask_ratio and change.numel() > 0:
        target_pct = 100.0 * (1.0 - max_mask_ratio)
        adaptive_thresh = float(torch.quantile(change.float().reshape(-1),
                                               target_pct / 100.0).item())
        adaptive_thresh = max(adaptive_thresh, change_threshold)
        low_change = change <= adaptive_thresh

    # Build adaptive keep_mask
    keep_mask_phase2 = keep_mask_phase1.clone()
    keep_mask_phase2[low_change] = True  # also keep low-change regions

    # Phase 3: Denoise again with adaptive mask
    denoised_phase2 = fm_denoise_impute(
        pipeline, motion_norm_global, keep_mask_phase2, device,
        denoise_strength=denoise_strength, denoise_steps=denoise_steps,
    )

    return denoised_phase2, keep_mask_phase2


# ====================================================================
# Windowed repair (360 frames, 20 overlap)
# ====================================================================

def repair_full_motion(pipeline, bundle, motion_local, device,
                       denoise_strength=0.02, denoise_steps=10,
                       change_threshold=0.1, max_mask_ratio=0.15,
                       window_size=360, window_overlap=20):
    """Full repair pipeline with windowing.

    Input: (T, 135) local rotation motion (unnormalized)
    Output: (T, 135) repaired local rotation motion (unnormalized)
    """
    T_orig = motion_local.shape[0]

    # Convert to global rotation
    motion_global = local_to_global_motion(motion_local)

    # Normalize
    motion_norm = bundle.normalize_motion(motion_global.unsqueeze(0).to(device))  # (1, T, 135)

    # Windowed processing
    repaired_norm = motion_norm.clone()
    current_idx = 0
    prev_overlap = 0

    while current_idx < T_orig:
        begin = current_idx
        end = min(begin + window_size, T_orig)
        window_motion = motion_norm[:, begin:end, :]  # (1, W, 135)

        # Run adaptive denoise on this window
        window_repaired, _ = adaptive_denoise_m2m(
            pipeline, window_motion, device,
            denoise_strength=denoise_strength,
            denoise_steps=denoise_steps,
            change_threshold=change_threshold,
            max_mask_ratio=max_mask_ratio,
        )

        # Stitch: for the overlap region, use the new result (it was conditioned
        # on the overlap frames via keep_mask). Skip the overlap prefix from
        # previous window output.
        write_begin = begin + prev_overlap if prev_overlap > 0 else begin
        repaired_norm[:, write_begin:end, :] = window_repaired[:, (write_begin - begin):, :]

        prev_overlap = window_overlap
        current_idx = end - window_overlap
        if end >= T_orig:
            break

    # Denormalize → global rotation
    repaired_global = bundle.denormalize_motion(repaired_norm)[0].cpu()

    # Convert back to local rotation
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
    mode_label = f"{config_name}_ada_denoise"

    output_dir = Path(args.output_dir)
    mode_output_dir = output_dir / mode_label
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{mode_label}] Starting on {device}")
    print(f"[{mode_label}] denoise_strength={args.denoise_strength}, "
          f"denoise_steps={args.denoise_steps}, change_threshold={args.change_threshold}")

    # Load quality list
    with open(args.quality_list) as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    all_items = quality_data.get("items", [])

    start = args.start_idx
    end = args.end_idx if args.end_idx > 0 else len(all_items)
    items = all_items[start:end]
    print(f"[{mode_label}] Processing items [{start}:{end}] ({len(items)} samples)")

    # Build model
    print(f"[{mode_label}] Loading M2M model: {config_name}...")
    pipeline, bundle, ckpt_path = build_model(config_name, device)

    stats = {
        "config": config_name, "mode": "ada_denoise",
        "denoise_strength": args.denoise_strength,
        "denoise_steps": args.denoise_steps,
        "change_threshold": args.change_threshold,
        "slice": f"{start}:{end}",
        "checkpoint": ckpt_path,
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
            continue

        try:
            t0 = time.time()

            # 1. Load motion (local rotation)
            motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

            # 2. Adaptive denoise repair (no separate MoGenDIT needed!)
            with torch.no_grad():
                repaired_local = repair_full_motion(
                    pipeline, bundle, motion_135, device,
                    denoise_strength=args.denoise_strength,
                    denoise_steps=args.denoise_steps,
                    change_threshold=args.change_threshold,
                    max_mask_ratio=args.max_mask_ratio,
                    window_size=args.window_size,
                    window_overlap=args.window_overlap,
                )

            # 3. Sanity check
            if torch.isnan(repaired_local).any():
                stats["errors"].append({"path": rel_path, "error": "NaN"})
                stats["skipped"] += 1
                continue

            # 4. Save repaired NPZ
            repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_local, abs_trans_frame0)
            if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 20.0:
                stats["errors"].append({"path": rel_path, "error": f"trans extreme"})
                stats["skipped"] += 1
                continue

            out_npz = str(mode_output_dir / "repaired" / rel_path)
            orig_data = dict(np.load(npz_path, allow_pickle=True))
            save_repaired_npz(out_npz, repaired_aa, repaired_trans, orig_data, fps)

            # 5. Quality check
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

            # No separate mask for MPJPE — compute full-motion MAE
            mpjpe_um = float(torch.abs(motion_135 - repaired_local).mean().item())
            stats["mpjpe_unmasked_list"].append(mpjpe_um)

            detail = {
                "path": rel_path, "num_frames": num_frames,
                "before_failed": before_failed, "after_valid": after_valid,
                "after_failed": after_failed,
                "improved": not before_valid and after_valid,
                "mae": round(mpjpe_um, 6),
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
                      f"before={before_failed} after={after_failed} | {elapsed:.1f}s")

        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": str(e)[:200]})
            if (idx + 1) % 100 == 0:
                print(f"  [{start+idx+1}] ERROR: {str(e)[:100]}")
            continue

    # Summary
    processed = max(stats["processed"], 1)
    print(f"\n{'='*60}")
    print(f"SUMMARY — {mode_label} [{start}:{end}]")
    print(f"{'='*60}")
    print(f"Total:     {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Improved:  {stats['improved']} ({stats['improved']/processed*100:.1f}%)")
    print(f"Degraded:  {stats['degraded']}")

    stats["per_failure_type"] = dict(stats["per_failure_type"])
    stats_path = mode_output_dir / f"repair_stats_{start}_{end}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()

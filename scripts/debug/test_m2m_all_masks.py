#!/usr/bin/env python3
"""Test M2M model with all 6 mask patterns (M1-M6) on a representative sample.

Outputs results to data/m2m_inference_test/uncond_jit_e51/ with quality report.

Usage:
    python scripts/test_m2m_all_masks.py [--sample NPZ_PATH] [--steps 50] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "work_dirs/hymotion_m2m_completion_uncond_jit_046b/checkpoint-epoch_58",
)
CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "work_dirs/hymotion_m2m_completion_uncond_jit_046b/20260325_212837/config.py",
)
DEFAULT_SAMPLE = os.path.join(
    PROJECT_ROOT,
    "data/motionhub/motionx/motion_data/smplx_55/perform/"
    "Analysis_of_Basic_Calligraphy_3_clip1.npz",
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "output/test/m2m_inference_test/uncond_jit_e58"
)

# Joint group indices (23-group space: 0=transl, 1-22=joints)
UPPER_BODY_JOINTS = [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # Spine3+Neck+Head+Arms
TRANSL_GROUP = [0]
NUM_JOINT_GROUPS = 23
TRANSL_DIM = 3
JOINT_ROT_DIM = 6
TOTAL_DIM = 135


# ---------------------------------------------------------------------------
# Mask builders: each returns (T, 135) torch.Tensor (1=mask, 0=keep)
# ---------------------------------------------------------------------------

def expand_grid_to_mask(grid: np.ndarray) -> torch.Tensor:
    """Expand (T, 23) joint-group grid to (T, 135) mask."""
    mask = torch.from_numpy(grid.astype(np.float32))
    transl_mask = mask[:, 0:1].repeat(1, TRANSL_DIM)
    joint_mask = mask[:, 1:].repeat_interleave(JOINT_ROT_DIM, dim=-1)
    return torch.cat([transl_mask, joint_mask], dim=-1)


def build_m3_inbetween(T: int) -> torch.Tensor:
    """M3 inbetween: keep first 20% + last 20%, mask middle 60%."""
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    t_start = max(1, int(T * 0.2))
    t_end = max(t_start + 1, int(T * 0.8))
    grid[t_start:t_end, :] = 1.0
    return expand_grid_to_mask(grid)


def build_m3_prediction(T: int) -> torch.Tensor:
    """M3 prediction: keep first 30%, mask rest 70%."""
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    t_split = max(1, int(T * 0.3))
    grid[t_split:, :] = 1.0
    return expand_grid_to_mask(grid)


def build_m4_upper_body(T: int) -> torch.Tensor:
    """M4 joint: mask upper body (Spine3+Neck+Head+Arms) all frames."""
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[:, UPPER_BODY_JOINTS] = 1.0
    return expand_grid_to_mask(grid)


def build_m4_translation(T: int) -> torch.Tensor:
    """M4 joint: mask translation (dim 0:3) all frames."""
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[:, TRANSL_GROUP] = 1.0
    return expand_grid_to_mask(grid)


def build_m5_full(T: int) -> torch.Tensor:
    """M5 full: mask everything (unconditional generation)."""
    return torch.ones(T, TOTAL_DIM, dtype=torch.float32)


def build_m6_keyframe(T: int) -> torch.Tensor:
    """M6 keyframe: keep frames 0, T//3, 2T//3, T-1; mask rest."""
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    keyframes = sorted(set([0, T // 3, 2 * T // 3, T - 1]))
    for kf in keyframes:
        grid[kf, :] = 0.0
    return expand_grid_to_mask(grid)


def build_m1_random_cell(T: int) -> torch.Tensor:
    """M1 random cell: 15% random cells (excluding translation)."""
    rng = np.random.RandomState(42)
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    # Random cells for joints only (cols 1-22), skip translation
    p = 0.15
    grid[:, 1:] = (rng.rand(T, 22) < p).astype(np.float32)
    return expand_grid_to_mask(grid)


def build_m2_random_block(T: int) -> torch.Tensor:
    """M2 random block: 2 random temporal x joint blocks."""
    rng = np.random.RandomState(42)
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    for _ in range(2):
        t1 = rng.randint(0, T)
        t2 = rng.randint(t1 + 1, min(t1 + T // 3, T) + 1)
        n_joints = rng.randint(3, 10)
        joints = rng.choice(range(1, NUM_JOINT_GROUPS), size=min(n_joints, 22), replace=False)
        grid[t1:t2, joints] = 1.0
    return expand_grid_to_mask(grid)


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

TEST_CASES = [
    ("m3_inbetween", "M3 inbetween: keep first/last 20%, mask middle 60%", build_m3_inbetween),
    ("m3_prediction", "M3 prediction: keep first 30%, mask rest 70%", build_m3_prediction),
    ("m4_upper_body", "M4 joint: mask upper body all frames", build_m4_upper_body),
    ("m4_translation", "M4 joint: mask translation all frames", build_m4_translation),
    ("m5_full", "M5 full: unconditional generation", build_m5_full),
    ("m6_keyframe", "M6 keyframe: keep 4 keyframes, mask rest", build_m6_keyframe),
    ("m1_random_cell", "M1 random: 15% random cells (no transl)", build_m1_random_cell),
    ("m2_random_block", "M2 block: 2 random blocks", build_m2_random_block),
]


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(
    original: torch.Tensor,
    repaired: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compute quality metrics comparing original vs repaired motion.

    All inputs: (T, 135) tensors.
    Returns dict of scalar metrics.
    """
    T = original.shape[0]

    # Translation velocity (frame-to-frame)
    orig_trans = original[:, :3].numpy()
    rep_trans = repaired[:, :3].numpy()

    orig_trans_vel = np.linalg.norm(np.diff(orig_trans, axis=0), axis=-1)
    rep_trans_vel = np.linalg.norm(np.diff(rep_trans, axis=0), axis=-1)

    # Pose velocity (rot6d, frame-to-frame L2 per frame)
    orig_pose = original[:, 3:].numpy()
    rep_pose = repaired[:, 3:].numpy()
    orig_pose_vel = np.linalg.norm(np.diff(orig_pose, axis=0), axis=-1)
    rep_pose_vel = np.linalg.norm(np.diff(rep_pose, axis=0), axis=-1)

    # Unmasked diff (should be ~0 for partial masks)
    unmasked_region = mask < 0.5
    if unmasked_region.any():
        unmasked_diff_max = (repaired - original)[unmasked_region].abs().max().item()
        unmasked_diff_mean = (repaired - original)[unmasked_region].abs().mean().item()
    else:
        unmasked_diff_max = -1.0  # N/A for full mask
        unmasked_diff_mean = -1.0

    # Masked diff (should be >0)
    masked_region = mask > 0.5
    if masked_region.any():
        masked_diff_max = (repaired - original)[masked_region].abs().max().item()
        masked_diff_mean = (repaired - original)[masked_region].abs().mean().item()
    else:
        masked_diff_max = 0.0
        masked_diff_mean = 0.0

    # Transition smoothness at mask boundaries (detect jumps)
    # Find frames where mask transitions from 0 to 1 or 1 to 0
    mask_any = mask.any(dim=-1).float().numpy()  # (T,)
    transitions = np.where(np.abs(np.diff(mask_any)) > 0.5)[0]
    boundary_jump_trans = 0.0
    boundary_jump_pose = 0.0
    if len(transitions) > 0:
        for t in transitions:
            if t + 1 < T:
                boundary_jump_trans = max(
                    boundary_jump_trans,
                    np.linalg.norm(rep_trans[t + 1] - rep_trans[t])
                )
                boundary_jump_pose = max(
                    boundary_jump_pose,
                    np.linalg.norm(rep_pose[t + 1] - rep_pose[t])
                )

    # Mask ratio
    mask_ratio = mask.sum().item() / mask.numel()

    return {
        "mask_ratio": mask_ratio,
        "orig_trans_vel_max": float(orig_trans_vel.max()),
        "orig_trans_vel_mean": float(orig_trans_vel.mean()),
        "rep_trans_vel_max": float(rep_trans_vel.max()),
        "rep_trans_vel_mean": float(rep_trans_vel.mean()),
        "orig_pose_vel_max": float(orig_pose_vel.max()),
        "orig_pose_vel_mean": float(orig_pose_vel.mean()),
        "rep_pose_vel_max": float(rep_pose_vel.max()),
        "rep_pose_vel_mean": float(rep_pose_vel.mean()),
        "unmasked_diff_max": unmasked_diff_max,
        "unmasked_diff_mean": unmasked_diff_mean,
        "masked_diff_max": masked_diff_max,
        "masked_diff_mean": masked_diff_mean,
        "boundary_jump_trans": boundary_jump_trans,
        "boundary_jump_pose": boundary_jump_pose,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test M2M with all mask patterns")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE, help="NPZ sample path")
    parser.add_argument("--steps", type=int, default=50, help="ODE solver steps")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Checkpoint path")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=" * 80)
    print(f"M2M All-Masks Test: uncond_jit epoch 58")
    print(f"Sample: {args.sample}")
    print(f"Output: {args.output_dir}")
    print(f"Steps:  {args.steps}")
    print(f"Device: {args.device}")
    print(f"=" * 80)

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print("\n[1/3] Loading model...")
    from motion_annot_web.m2m_database.hftrainer_repair_runtime import (
        CompletionRepairRuntime,
        load_npz_as_motion,
        motion_135_to_npz_format,
        repair_single,
        _save_repaired_npz,
    )

    runtime = CompletionRepairRuntime(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        validation_steps=args.steps,
    )
    print(f"  Model loaded on {args.device}")

    # -----------------------------------------------------------------------
    # 2. Load sample
    # -----------------------------------------------------------------------
    print("\n[2/3] Loading sample...")
    motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(args.sample)
    orig_data = dict(np.load(args.sample, allow_pickle=True))
    print(f"  Frames: {num_frames}, FPS: {fps}, Motion shape: {motion_135.shape}")

    # Save original
    import shutil
    orig_out = os.path.join(args.output_dir, "original.npz")
    shutil.copy2(args.sample, orig_out)
    print(f"  Saved original to {orig_out}")

    # -----------------------------------------------------------------------
    # 3. Run all test cases
    # -----------------------------------------------------------------------
    print(f"\n[3/3] Running {len(TEST_CASES)} test cases...\n")

    results = []
    for name, desc, mask_fn in TEST_CASES:
        print(f"  --- {name}: {desc}")
        t0 = time.time()

        # Build mask
        T = min(num_frames, 360)
        mask_135 = mask_fn(T)

        # Pad to num_frames if needed
        if mask_135.shape[0] < num_frames:
            mask_135 = torch.cat([
                mask_135,
                torch.zeros(num_frames - mask_135.shape[0], TOTAL_DIM)
            ], dim=0)

        mask_ratio = mask_135[:T].sum().item() / (T * TOTAL_DIM)
        print(f"      mask_ratio={mask_ratio:.4f}")

        # Run repair
        try:
            repaired_motion = repair_single(
                runtime.pipeline,
                motion_135,
                mask_135,
                args.device,
                max_frames=360,
            )

            # Compute metrics
            metrics = compute_quality_metrics(
                motion_135[:T], repaired_motion[:T], mask_135[:T]
            )
            elapsed = time.time() - t0
            metrics["elapsed_s"] = round(elapsed, 2)
            metrics["status"] = "OK"

            # Save repaired NPZ
            repaired_aa, repaired_trans = motion_135_to_npz_format(
                repaired_motion, abs_trans_frame0
            )
            out_npz = os.path.join(args.output_dir, f"{name}.npz")
            _save_repaired_npz(out_npz, repaired_aa, repaired_trans, orig_data, fps)
            print(f"      Saved: {out_npz}")

            # Check for NaN
            if np.isnan(repaired_trans).any() or np.isnan(repaired_aa).any():
                metrics["status"] = "NaN_DETECTED"
                print(f"      WARNING: NaN in output!")

        except Exception as e:
            elapsed = time.time() - t0
            metrics = {
                "mask_ratio": mask_ratio,
                "elapsed_s": round(elapsed, 2),
                "status": f"ERROR: {e}",
            }
            print(f"      ERROR: {e}")

        metrics["name"] = name
        metrics["desc"] = desc
        results.append(metrics)

        # Print key metrics
        if "rep_trans_vel_max" in metrics:
            print(
                f"      trans_vel: orig_max={metrics['orig_trans_vel_max']:.4f} "
                f"rep_max={metrics['rep_trans_vel_max']:.4f} | "
                f"pose_vel: orig_max={metrics['orig_pose_vel_max']:.4f} "
                f"rep_max={metrics['rep_pose_vel_max']:.4f}"
            )
            print(
                f"      unmasked_diff_max={metrics['unmasked_diff_max']:.6f} | "
                f"masked_diff_max={metrics['masked_diff_max']:.4f} | "
                f"boundary_jump: trans={metrics['boundary_jump_trans']:.4f} "
                f"pose={metrics['boundary_jump_pose']:.4f}"
            )
        print(f"      elapsed={metrics['elapsed_s']}s")
        print()

    # -----------------------------------------------------------------------
    # 4. Generate report
    # -----------------------------------------------------------------------
    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append("M2M All-Masks Test Report: uncond_jit epoch 58")
    report_lines.append(f"Sample: {args.sample}")
    report_lines.append(f"Frames: {num_frames}, FPS: {fps}")
    report_lines.append(f"Steps: {args.steps}, Device: {args.device}")
    report_lines.append("=" * 120)
    report_lines.append("")

    # Table header
    hdr = (
        f"{'Test':<18} {'Status':<8} {'MaskR':>6} "
        f"{'OTransVMax':>10} {'RTransVMax':>10} "
        f"{'OPoseVMax':>10} {'RPoseVMax':>10} "
        f"{'UnmskDMax':>10} {'MskDMax':>10} "
        f"{'BndJmpTr':>10} {'BndJmpPo':>10} "
        f"{'Time':>6}"
    )
    report_lines.append(hdr)
    report_lines.append("-" * len(hdr))

    for m in results:
        if "rep_trans_vel_max" in m:
            row = (
                f"{m['name']:<18} {m['status']:<8} {m['mask_ratio']:>6.3f} "
                f"{m['orig_trans_vel_max']:>10.4f} {m['rep_trans_vel_max']:>10.4f} "
                f"{m['orig_pose_vel_max']:>10.4f} {m['rep_pose_vel_max']:>10.4f} "
                f"{m['unmasked_diff_max']:>10.6f} {m['masked_diff_max']:>10.4f} "
                f"{m['boundary_jump_trans']:>10.4f} {m['boundary_jump_pose']:>10.4f} "
                f"{m['elapsed_s']:>6.1f}s"
            )
        else:
            row = f"{m['name']:<18} {m['status']:<40}"
        report_lines.append(row)

    report_lines.append("")
    report_lines.append("=" * 120)
    report_lines.append("INTERPRETATION GUIDE:")
    report_lines.append("  - unmasked_diff_max: should be ~0 (unmasked regions preserved). If >0.001 = BUG.")
    report_lines.append("  - boundary_jump_trans/pose: smoothness at mask boundaries. Lower = better.")
    report_lines.append("    Compare with orig_trans_vel_max / orig_pose_vel_max as baseline.")
    report_lines.append("  - rep_trans_vel_max >> orig: potential translation jump/divergence.")
    report_lines.append("  - M5 (full mask): unmasked_diff=-1 is expected (no unmasked region).")
    report_lines.append("=" * 120)
    report_lines.append("")

    # Verdict per test
    report_lines.append("VERDICT:")
    for m in results:
        name = m["name"]
        if m["status"] != "OK":
            verdict = f"  FAIL ({m['status']})"
        elif "rep_trans_vel_max" not in m:
            verdict = "  UNKNOWN (no metrics)"
        else:
            issues = []
            # Check unmasked preservation (skip M5)
            if name != "m5_full" and m["unmasked_diff_max"] > 0.001:
                issues.append(f"unmasked_diff={m['unmasked_diff_max']:.4f}")
            # Check if repaired velocity is too high vs original
            if m["rep_trans_vel_max"] > m["orig_trans_vel_max"] * 3.0:
                issues.append(
                    f"trans_vel_spike={m['rep_trans_vel_max']:.4f} "
                    f"(orig={m['orig_trans_vel_max']:.4f})"
                )
            if m["rep_pose_vel_max"] > m["orig_pose_vel_max"] * 3.0:
                issues.append(
                    f"pose_vel_spike={m['rep_pose_vel_max']:.4f} "
                    f"(orig={m['orig_pose_vel_max']:.4f})"
                )
            # Check boundary jumps
            if m["boundary_jump_trans"] > m["orig_trans_vel_max"] * 2.0 and m["boundary_jump_trans"] > 0.05:
                issues.append(f"boundary_jump_trans={m['boundary_jump_trans']:.4f}")
            if m["boundary_jump_pose"] > m["orig_pose_vel_max"] * 2.0 and m["boundary_jump_pose"] > 0.5:
                issues.append(f"boundary_jump_pose={m['boundary_jump_pose']:.4f}")

            if not issues:
                verdict = "  PASS"
            else:
                verdict = f"  WARN: {'; '.join(issues)}"

        report_lines.append(f"  {name:<18} {verdict}")

    report_lines.append("")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    print(f"NPZ files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

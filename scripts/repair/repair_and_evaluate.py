#!/usr/bin/env python3
"""
Batch repair low-quality motion data using HyMotion M2M models.

Workflow:
1. Load low-quality list
2. For each sample: load NPZ → convert to 135-dim → run quality checker → get sparse mask
3. Feed (motion, mask) to M2M pipeline for repair
4. Re-run quality checker on repaired motion → compute pass rate
5. Save repaired motions + statistics

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/repair_and_evaluate.py \
        --model uncond_flow \
        --max-samples 100 \
        --num-steps 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Add hymotion_1.0_train for quality checker
LEGACY_ROOT = PROJECT_ROOT.parent / "hymotion_1.0_train"
if LEGACY_ROOT.is_dir() and str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Repair low-quality motions with HyMotion M2M")
    parser.add_argument(
        "--model",
        type=str,
        default="uncond_flow",
        choices=["uncond_flow", "uncond_jit", "caption_flow", "caption_jit"],
        help="Which M2M model variant to use",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint path. If None, auto-detect latest from work_dirs.",
    )
    parser.add_argument(
        "--quality-list", type=str,
        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json",
        help="Path to low_quality.json",
    )
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="work_dirs/repair_eval")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to process (0=all)")
    parser.add_argument("--num-steps", type=int, default=50, help="ODE solver steps")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-format",
        type=str,
        default="default",
        choices=["default", "repair_review"],
        help=(
            "Output format. 'repair_review' generates repair_review_current.json "
            "compatible with m2m_database repair workflow manager."
        ),
    )
    parser.add_argument(
        "--repair-review-output",
        type=str,
        default="",
        help=(
            "Path to write repair_review_current.json. Only used when "
            "--output-format=repair_review. Default: <output-dir>/repair_review_current.json"
        ),
    )
    return parser.parse_args()


MODEL_CONFIGS = {
    "uncond_flow": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_046b.py",
    "uncond_jit": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_046b.py",
    "caption_flow": "configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py",
    "caption_jit": "configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_046b.py",
}

WORK_DIR_NAMES = {
    "uncond_flow": "hymotion_m2m_completion_uncond_fm_046b",
    "uncond_jit": "hymotion_m2m_completion_uncond_jit_046b",
    "caption_flow": "hymotion_m2m_completion_caption_fm_046b",
    "caption_jit": "hymotion_m2m_completion_caption_jit_046b",
}


def find_latest_checkpoint(model_name: str) -> str:
    """Auto-detect latest checkpoint from work_dirs."""
    work_dir = PROJECT_ROOT / "work_dirs" / WORK_DIR_NAMES[model_name]
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")
    ckpt_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: d.stat().st_mtime,
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {work_dir}")
    return str(ckpt_dirs[-1])


def load_npz_as_motion(npz_path: str) -> tuple:
    """Load NPZ and convert to (T, 135) motion tensor (SMPL-22, rot6d, rel translation).

    Returns: (motion_135, num_frames, fps)
    """
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose,
        process_transl,
    )

    data = dict(np.load(npz_path, allow_pickle=True))
    poses = np.array(data["poses"], dtype=np.float32)
    trans = np.array(data.get("trans", data.get("transl")), dtype=np.float32)
    if trans.ndim == 1:
        trans = trans.reshape(-1, 3)
    fps = int(data.get("mocap_framerate", 30))

    # Convert poses to SMPL-22 rot6d: (T, 132)
    pose_rot6d = process_smplx_pose(poses, rot_type="rotation_6d", out_type="smpl_22")

    # Convert translation to rel: (T, 3) — relative frame-to-frame displacement
    transl_rel = process_transl(trans, transl_type="rel")

    # Concatenate: [transl_rel(3), rot6d(132)] = (T, 135)
    motion = np.concatenate([transl_rel, pose_rot6d], axis=-1)  # (T, 135)
    motion_tensor = torch.from_numpy(motion).float()
    return motion_tensor, motion_tensor.shape[0], fps


def load_npz_as_axis_angle(npz_path: str) -> tuple:
    """Load NPZ and return (T, 22, 3) axis-angle for quality checker."""
    data = dict(np.load(npz_path, allow_pickle=True))
    poses = np.array(data["poses"], dtype=np.float32)
    if poses.ndim == 3:
        poses = poses.reshape(poses.shape[0], -1)
    # SMPL-22: first 66 dims = 22 joints * 3 axis-angle
    poses_22 = poses[:, :66].reshape(-1, 22, 3)
    trans = np.array(data.get("trans", data.get("transl")), dtype=np.float32)
    if trans.ndim == 1:
        trans = trans.reshape(-1, 3)
    fps = int(data.get("mocap_framerate", 30))
    return poses_22, trans, fps


_CHECKER_INSTANCE = None


def _get_checker():
    global _CHECKER_INSTANCE
    if _CHECKER_INSTANCE is None:
        from hymotion.utils.quality_check_rules.motion_quality_checker import MotionQualityChecker
        _CHECKER_INSTANCE = MotionQualityChecker(device="cpu")
    return _CHECKER_INSTANCE


def run_quality_check(npz_path: str) -> tuple:
    """Run quality checker on an NPZ file.

    Returns: (result_dict, sparse_mask_dict)
        result_dict: {is_valid, failed_checks, ...}
        sparse_mask_dict: {num_frames, num_joints, frames: {frame_idx: [joint_ids]}} or {}
    """
    from hymotion.utils.quality_check_rules.mask_utils import (
        merge_invalid_masks,
        mask_to_sparse_dict,
    )

    checker = _get_checker()
    result = checker.check_from_file(npz_path)
    result_dict = result.to_dict()

    # Extract union of all per-checker invalid masks
    masks = []
    num_frames = 0
    for checker_name, checker_result in result.all_results.items():
        mask = checker_result.get("invalid_mask")
        if mask is not None:
            try:
                num_frames = max(num_frames, int(mask.shape[0]))
            except Exception:
                pass
            masks.append(mask)

    if masks and num_frames > 0:
        union_mask = merge_invalid_masks(masks, num_frames=num_frames)
        sparse_mask = mask_to_sparse_dict(union_mask)
    else:
        sparse_mask = {}

    return result_dict, sparse_mask


def sparse_mask_to_dense(invalid_mask: dict, num_frames: int, motion_dim: int = 135, expand_frames: int = 5) -> torch.Tensor:
    """Convert sparse {frames: {frame_idx: [joint_ids]}} to dense (T, D) mask.

    Layout for 135-dim: [rel_transl(3), rot6d_joint0(6), ..., rot6d_joint21(6)]
    mask=1 means needs repair, mask=0 means keep.

    expand_frames: expand each bad frame by ±N frames for smoother repair context.
    """
    mask = torch.zeros(num_frames, motion_dim, dtype=torch.float32)
    frames = invalid_mask.get("frames", {})
    for frame_str, joint_ids in frames.items():
        frame_idx = int(frame_str)
        # Expand to neighboring frames for smoother repair
        for f in range(max(0, frame_idx - expand_frames), min(num_frames, frame_idx + expand_frames + 1)):
            for j in joint_ids:
                j = int(j)
                if j < 0 or j >= 22:
                    continue
                # Joint 0 (pelvis) includes translation dims 0:3 + rotation dims 3:9
                if j == 0:
                    mask[f, 0:9] = 1.0  # transl(3) + pelvis_rot6d(6)
                else:
                    # Joint j: rotation starts at 3 + j*6
                    start = 3 + j * 6
                    end = start + 6
                    if end <= motion_dim:
                        mask[f, start:end] = 1.0
    return mask


def motion_135_to_npz_format(motion_135: torch.Tensor, abs_trans_frame0: np.ndarray) -> tuple:
    """Convert (T, 135) back to axis-angle poses (T, 22, 3) + abs trans (T, 3).

    IMPORTANT: Must use the SAME rotation_convert module as the forward conversion
    in load_smplx.py, NOT the geometry module (different rot6d conventions!).

    Args:
        motion_135: (T, 135) tensor [rel_transl(3), rot6d(132)]
        abs_trans_frame0: (3,) the original absolute translation of frame 0

    Returns: (axis_angle (T, 22, 3) np, abs_trans (T, 3) np)
    """
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )

    motion = motion_135.float().numpy()
    T = motion.shape[0]

    # Extract rel translation (first 3 dims) → reconstruct abs with original frame0
    rel_transl = motion[:, 0:3]
    abs_transl = np.zeros_like(rel_transl)
    abs_transl[0] = abs_trans_frame0
    for i in range(1, T):
        abs_transl[i] = abs_transl[i - 1] + rel_transl[i]

    # Extract rot6d: dims 3:135 → (T*22, 6)
    # CRITICAL: data was stored in HyMotion row-major convention [R00,R01,R10,R11,R20,R21]
    # but rotation_6d_to_axis_angle expects column-major [R00,R10,R20,R01,R11,R21]
    # The forward path in load_smplx.py applies: out[:, :, [0,3,1,4,2,5]] (col→row)
    # We reverse it: [0,2,4,1,3,5] (row→col)
    rot6d = motion[:, 3:135].reshape(T * 22, 6)
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]  # row-major → column-major
    axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)  # (T*22, 3) numpy
    axis_angle = np.array(axis_angle, dtype=np.float32).reshape(T, 22, 3)

    return axis_angle, abs_transl


def build_model(args):
    """Build M2M bundle and pipeline."""
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    config_path = str(PROJECT_ROOT / MODEL_CONFIGS[args.model])
    cfg = Config.fromfile(config_path)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.model)
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    # Build bundle — use the TRAINING config (dumped in work_dir), not the current source config,
    # because mean_std_dir may have changed since training.
    training_config = None
    if not args.checkpoint:
        ckpt_path = find_latest_checkpoint(args.model)
    else:
        ckpt_path = args.checkpoint
    work_dir = Path(ckpt_path).parent  # checkpoint-epoch_N -> work_dir
    # Find the latest run_dir's config.py (the config used for actual training)
    run_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("2026")],
        key=lambda d: d.name,
    )
    for rd in reversed(run_dirs):
        training_cfg_path = rd / "config.py"
        if training_cfg_path.is_file():
            training_config = str(training_cfg_path)
            break

    if training_config:
        print(f"[INFO] Using TRAINING config: {training_config}")
        cfg = Config.fromfile(training_config)
    else:
        print(f"[WARN] No training config found, using source config (mean/std may mismatch!)")
        cfg = Config.fromfile(config_path)

    print(f"[INFO] mean_std_dir = {cfg.model.get('mean_std_dir', 'NOT SET')}")
    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(args.device)
    bundle.eval()

    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_path, map_location=args.device)
    bundle.load_state_dict_selective(state_dict)
    print(f"[INFO] Checkpoint loaded successfully")

    # Build pipeline
    pipeline = HyMotionM2MPipeline(bundle, num_steps=args.num_steps)
    return pipeline, bundle


def repair_single(
    pipeline,
    motion_135: torch.Tensor,
    mask_135: torch.Tensor,
    device: str,
    max_frames: int = 360,
) -> torch.Tensor:
    """Repair a single motion using the M2M pipeline.

    Args:
        motion_135: (T, 135) source motion
        mask_135: (T, 135) binary mask, 1=needs repair
        max_frames: pad/crop to this length

    Returns:
        (T, 135) repaired motion (original length, unpadded)
    """
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)

    # Crop if too long
    src = motion_135[:T].unsqueeze(0).to(device)  # (1, T, 135)
    msk = mask_135[:T].unsqueeze(0).to(device)  # (1, T, 135)

    # Pad if shorter than max_frames
    if T < max_frames:
        pad_len = max_frames - T
        src = torch.nn.functional.pad(src, (0, 0, 0, pad_len), mode='constant', value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), mode='constant', value=0)

    batch = {
        "src_motion": src,
        "src_mask": msk,
        "src_length": [T],
        "tgt_length": [T],
    }

    with torch.no_grad():
        result = pipeline(batch)

    # Get repaired latent and combine with original
    repaired_latent = result["latent"][0, :T].cpu()  # (T, 135)

    # Denormalize if needed (pipeline already does this in decode_motion_from_latent)
    # We need the raw latent to combine with src
    bundle = pipeline.bundle
    std = torch.where(bundle.std.cpu() < 1e-3, torch.zeros_like(bundle.std.cpu()), bundle.std.cpu())
    repaired_denorm = repaired_latent * std + bundle.mean.cpu()

    # Combine: keep original where mask=0, use repaired where mask=1
    mask_crop = mask_135[:T]
    combined = motion_135[:T] * (1 - mask_crop) + repaired_denorm * mask_crop

    # If original was longer, append the tail
    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)

    return combined


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load low quality list
    with open(args.quality_list, "r") as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    items = quality_data.get("items", [])
    if args.max_samples > 0:
        items = items[:args.max_samples]
    print(f"[INFO] Processing {len(items)} low-quality samples")

    # Build model
    print("[INFO] Building model...")
    pipeline, bundle = build_model(args)
    print("[INFO] Model ready")

    # Process each sample
    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "before_pass": 0,
        "after_pass": 0,
        "improved": 0,
        "degraded": 0,
        "unchanged": 0,
        "errors": [],
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

            # 1. Load motion as 135-dim
            motion_135, num_frames, fps = load_npz_as_motion(npz_path)

            # 2. Run quality checker BEFORE repair
            before_result, before_sparse = run_quality_check(npz_path)
            before_valid = before_result.get("is_valid", True)
            before_failed = before_result.get("failed_checks", [])

            # 3. Create repair mask from checker's sparse invalid mask
            mask_135 = sparse_mask_to_dense(before_sparse, num_frames)
            mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)

            # If sparse mask is too empty but sample is invalid, expand mask more
            if mask_ratio < 0.01 and not before_valid:
                # Expand more aggressively: use 15-frame radius
                mask_135 = sparse_mask_to_dense(before_sparse, num_frames, expand_frames=15)
                mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)

            # 4. Run repair
            repaired_motion = repair_single(pipeline, motion_135, mask_135, args.device)

            # 5. Save repaired motion as temp NPZ for re-checking
            orig_data = dict(np.load(npz_path, allow_pickle=True))
            abs_trans_frame0 = np.array(orig_data.get("trans", orig_data.get("transl")), dtype=np.float32)
            if abs_trans_frame0.ndim == 1:
                abs_trans_frame0 = abs_trans_frame0.reshape(-1, 3)
            abs_trans_frame0 = abs_trans_frame0[0]  # (3,)
            repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_motion, abs_trans_frame0)
            temp_npz = str(output_dir / "temp_repaired.npz")
            repaired_poses_full = np.zeros_like(orig_data["poses"])
            T_rep = min(repaired_aa.shape[0], repaired_poses_full.shape[0])
            repaired_poses_full[:T_rep, :66] = repaired_aa[:T_rep].reshape(-1, 66)
            if orig_data["poses"].shape[1] > 66:
                repaired_poses_full[:T_rep, 66:] = orig_data["poses"][:T_rep, 66:]
            np.savez(
                temp_npz,
                poses=repaired_poses_full,
                trans=repaired_trans[:T_rep],
                betas=orig_data.get("betas", np.zeros((1, 16), dtype=np.float32)),
                mocap_framerate=fps,
                gender=str(orig_data.get("gender", "neutral")),
                num_frames=T_rep,
            )

            # 6. Run quality checker AFTER repair
            after_result, _ = run_quality_check(temp_npz)
            after_valid = after_result.get("is_valid", True)
            after_failed = after_result.get("failed_checks", [])

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
            else:
                stats["unchanged"] += 1

            detail = {
                "path": rel_path,
                "num_frames": num_frames,
                "fps": fps,
                "mask_ratio": round(mask_ratio, 4),
                "before_valid": before_valid,
                "before_failed": before_failed,
                "after_valid": after_valid,
                "after_failed": after_failed,
                "improved": not before_valid and after_valid,
                "elapsed_s": round(elapsed, 2),
            }
            stats["details"].append(detail)

            status = "✓ FIXED" if detail["improved"] else ("✗ STILL BAD" if not after_valid else "= OK")
            print(
                f"[{idx+1}/{len(items)}] {status} | "
                f"before={before_failed} after={after_failed} | "
                f"mask={mask_ratio:.1%} | {elapsed:.1f}s | {rel_path}"
            )

            # Save repaired motion
            if detail["improved"]:
                out_path = output_dir / "repaired" / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                orig_data = dict(np.load(npz_path, allow_pickle=True))
                # Update poses and trans
                repaired_poses_full = np.zeros_like(orig_data["poses"])
                repaired_poses_full[:, :66] = repaired_aa.reshape(-1, 66)
                if orig_data["poses"].shape[1] > 66:
                    repaired_poses_full[:, 66:] = orig_data["poses"][:, 66:]
                np.savez(
                    str(out_path),
                    poses=repaired_poses_full,
                    trans=repaired_trans,
                    betas=orig_data.get("betas", np.zeros((1, 16))),
                    mocap_framerate=fps,
                    gender=orig_data.get("gender", "neutral"),
                    num_frames=num_frames,
                )

        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": str(e)})
            print(f"[{idx+1}/{len(items)}] ERROR: {e} | {rel_path}")
            continue

    # Print summary
    print("\n" + "=" * 70)
    print(f"REPAIR SUMMARY — Model: {args.model}")
    print(f"=" * 70)
    print(f"Total:        {stats['total']}")
    print(f"Processed:    {stats['processed']}")
    print(f"Skipped:      {stats['skipped']}")
    print(f"Before pass:  {stats['before_pass']} ({stats['before_pass']/max(stats['processed'],1)*100:.1f}%)")
    print(f"After pass:   {stats['after_pass']} ({stats['after_pass']/max(stats['processed'],1)*100:.1f}%)")
    print(f"Improved:     {stats['improved']} ({stats['improved']/max(stats['processed'],1)*100:.1f}%)")
    print(f"Degraded:     {stats['degraded']} ({stats['degraded']/max(stats['processed'],1)*100:.1f}%)")
    print(f"Unchanged:    {stats['unchanged']}")
    print(f"=" * 70)

    # Save stats
    stats_path = output_dir / f"repair_stats_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to: {stats_path}")

    # Optionally generate repair_review_current.json
    if args.output_format == "repair_review":
        _generate_repair_review(args, stats, output_dir)


def _generate_repair_review(args, stats: dict, output_dir: Path) -> None:
    """Generate repair_review_current.json compatible with m2m_database workflow."""
    from datetime import datetime

    review_path = args.repair_review_output or str(output_dir / "repair_review_current.json")
    os.makedirs(os.path.dirname(review_path), exist_ok=True)

    items = []
    for detail in stats.get("details", []):
        rel_path = detail["path"]
        repaired_path = str(output_dir / "repaired" / rel_path) if detail.get("improved") else ""
        source_abs = str(Path(args.data_root) / rel_path)

        candidate = None
        if repaired_path and os.path.isfile(repaired_path):
            candidate = {
                "candidate_id": f"{args.model}::{rel_path}",
                "recipe_name": args.model,
                "recipe_display_name": args.model,
                "model_id": args.model,
                "model_display": args.model,
                "candidate_path": repaired_path,
                "is_valid": detail.get("after_valid", False),
                "improved": detail.get("improved", False),
                "after_failed_checks": detail.get("after_failed", []),
            }

        item = {
            "source_path": rel_path,
            "source_abs_path": source_abs,
            "source_quality": {
                "is_valid": detail.get("before_valid", True),
                "failed_checks": detail.get("before_failed", []),
                "mask_ratio": detail.get("mask_ratio", 0),
            },
            "candidate_count": 1 if candidate else 0,
            "status": "pending" if candidate else "not_selected",
            "manual_confirmed": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": f"batch_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}",
            "selected_candidate": candidate,
        }
        items.append(item)

    review_data = {
        "schema_version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "items": items,
        "stats": {
            "total": stats.get("total", 0),
            "processed": stats.get("processed", 0),
            "improved": stats.get("improved", 0),
            "before_pass_rate": round(stats["before_pass"] / max(stats["processed"], 1), 4),
            "after_pass_rate": round(stats["after_pass"] / max(stats["processed"], 1), 4),
            "model": args.model,
            "num_steps": args.num_steps,
        },
    }

    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] repair_review_current.json saved to: {review_path}")


if __name__ == "__main__":
    main()

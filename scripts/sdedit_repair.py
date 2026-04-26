#!/usr/bin/env python3
"""
SDEdit-style motion repair using pretrained HyMotion T2M model.

Instead of training a specialized M2M completion model, this script uses the
pretrained T2M (text-to-motion) model's learned distribution prior to "clean"
corrupted motion via the SDEdit approach:

  1. Convert source motion: NPZ → (T, 135) → pad to (T, 201) for T2M
  2. Normalize with T2M statistics (201-dim mean/std from checkpoint)
  3. SDEdit: add noise at timestep t_start, then denoise via ODE to t=1
  4. Extract first 135 dims, denormalize, merge with original via mask
  5. Convert back to NPZ

Flow Matching SDEdit:
  x_t = (1-t)*x₀ + t*x₁  where x₀=noise, x₁=clean motion
  For SDEdit: start from x_{t_start} = (1-t_start)*noise + t_start*motion_norm
  Then integrate ODE from t_start to 1.0

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/sdedit_repair.py \\
        --t-start 0.5 --num-steps 50 --max-samples 20 \\
        --quality-list data/hymotion_m2m_refine_data/data_quality_list/low_quality.json \\
        --output-dir work_dirs/sdedit_repair_eval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Add hymotion_1.0_train for quality checker
LEGACY_ROOT = PROJECT_ROOT.parent / "hymotion_1.0_train"
if LEGACY_ROOT.is_dir() and str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))


# ---------------------------------------------------------------------------
# Reuse utilities from hftrainer_repair_runtime
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT / "motion_annot_web" / "m2m_database"))

from hftrainer_repair_runtime import (
    load_npz_as_motion,
    motion_135_to_npz_format,
    sparse_mask_to_dense,
    _save_repaired_npz,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SDEdit motion repair using pretrained T2M model"
    )
    parser.add_argument(
        "--t-start", type=float, default=0.5,
        help="SDEdit noise level [0,1]. Lower=stronger repair, higher=preserve more.",
    )
    parser.add_argument("--num-steps", type=int, default=50, help="ODE solver steps")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt",
        help="Path to T2M pretrained checkpoint",
    )
    parser.add_argument(
        "--quality-list", type=str,
        default="data/hymotion_m2m_refine_data/data_quality_list/low_quality.json",
        help="Path to low_quality.json",
    )
    parser.add_argument("--data-root", type=str, default="data/hymotion_data")
    parser.add_argument("--output-dir", type=str, default="work_dirs/sdedit_repair_eval")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-mask", action="store_true", default=True,
        help="Use quality checker mask to blend repaired with original",
    )
    parser.add_argument(
        "--no-mask", action="store_true", default=False,
        help="Apply SDEdit to entire motion (no mask-based blending)",
    )
    parser.add_argument(
        "--output-format", type=str, default="default",
        choices=["default", "repair_review"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# T2M model dimensions (from checkpoint config.yml)
# ---------------------------------------------------------------------------

T2M_MOTION_DIM = 201
M2M_MOTION_DIM = 135
T2M_FEAT_DIM = 1024
T2M_NUM_LAYERS = 18
T2M_NUM_HEADS = 16
T2M_VTXT_DIM = 768
T2M_CTXT_DIM = 4096


# ---------------------------------------------------------------------------
# SDEdit Pipeline
# ---------------------------------------------------------------------------

def _length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


class SDEditRepairPipeline:
    """SDEdit-style repair using pretrained T2M (flow matching) model.

    Key difference from HyMotionT2MPipeline:
    - Initial condition = noisy version of input motion (not pure noise)
    - ODE integration starts from t_start (not 0)
    """

    def __init__(self, bundle, num_steps: int = 50, t_start: float = 0.5):
        self.bundle = bundle
        self.num_steps = num_steps
        self.t_start = t_start

    @torch.no_grad()
    def __call__(
        self,
        motion_201: torch.Tensor,
        lengths: list,
        mask_201: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run SDEdit on normalized 201-dim motion.

        Args:
            motion_201: (B, L, 201) normalized motion (T2M space)
            lengths: List[int] of actual sequence lengths
            mask_201: (B, L, 201) optional mask, 1=repair, 0=keep

        Returns:
            (B, L, 201) repaired normalized motion
        """
        device = next(self.bundle.motion_transformer.parameters()).device
        dtype = next(self.bundle.motion_transformer.parameters()).dtype

        motion_201 = motion_201.to(device=device, dtype=dtype)
        B, L, D = motion_201.shape

        tgt_padding_mask = _length_to_mask(
            torch.tensor(lengths, dtype=torch.long, device=device), L
        )

        # Null text embeddings (unconditional SDEdit — no text guidance)
        vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1).to(dtype=dtype)
        ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1).to(dtype=dtype)
        ctxt_length = torch.tensor([1], device=device).expand(B)
        ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)

        # ODE velocity function (same as HyMotionT2MPipeline but no CFG)
        def fn(t_val: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            x_pred = self.bundle.predict_flow(
                x_input=x,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_val.expand(B),
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            # velocity prediction: model directly outputs v = x₁ - x₀
            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)
            return x_pred

        # SDEdit: create noisy initial condition
        # flow matching: x_t = (1-t)*x₀ + t*x₁ where x₀=noise, x₁=clean
        # At t_start: x_{t_start} = (1-t_start)*noise + t_start*motion
        noise = torch.randn(B, L, D, device=device, dtype=dtype)
        x_t = (1.0 - self.t_start) * noise + self.t_start * motion_201

        # Euler ODE integration from t_start to 1.0
        num_denoise_steps = max(1, int(self.num_steps * (1.0 - self.t_start)))
        dt = (1.0 - self.t_start) / num_denoise_steps

        x = x_t
        for i in range(num_denoise_steps):
            t_val = torch.tensor(
                self.t_start + i * dt, device=device, dtype=dtype
            )
            v = fn(t_val, x)
            x = x + v * dt

        sampled = x

        # Optional mask-based blending in normalized space
        if mask_201 is not None:
            mask_201 = mask_201.to(device=device, dtype=dtype)
            sampled = motion_201 * (1.0 - mask_201) + sampled * mask_201

        return sampled


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_t2m_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load pretrained T2M model into HyMotionT2MBundle."""
    from hftrainer.models.motion.hymotion_t2m.bundle import HyMotionT2MBundle

    print(f"[INFO] Loading T2M checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Extract mean/std from checkpoint (they're top-level keys)
    mean_201 = sd.pop("mean")  # (201,)
    std_201 = sd.pop("std")    # (201,)

    # Build bundle with matching architecture
    # Note: HyMotionT2MBundle expects motion_transformer config dict
    motion_transformer_cfg = dict(
        type="HunyuanMotionMMDiT",
        input_dim=T2M_MOTION_DIM,
        output_dim=T2M_MOTION_DIM,
        feat_dim=T2M_FEAT_DIM,
        num_layers=T2M_NUM_LAYERS,
        num_heads=T2M_NUM_HEADS,
        vtxt_input_dim=T2M_VTXT_DIM,
        ctxt_input_dim=T2M_CTXT_DIM,
        mlp_ratio=4.0,
        dropout=0.0,
        mask_mode="narrowband",
        time_factor=1000.0,
        apply_rope_to_single_branch=False,
        trainable=True,
        save_ckpt=True,
    )

    bundle = HyMotionT2MBundle(
        motion_transformer=motion_transformer_cfg,
        mean_std_dir=None,  # We'll load mean/std from checkpoint directly
        motion_type="smpl_22",
        pred_type="velocity",
        uncondition_mode=False,
        noise_scheduler_cfg={"method": "euler"},
        infer_noise_scheduler_cfg={"validation_steps": 50},
        cond_mask_prob=0.1,
        vtxt_input_dim=T2M_VTXT_DIM,
        ctxt_input_dim=T2M_CTXT_DIM,
    )

    # Load mean/std into buffers (overwrite the default ones)
    std_clamped = torch.where(std_201 < 1e-3, torch.ones_like(std_201), std_201)
    bundle.mean = mean_201
    bundle.std = std_clamped
    bundle.register_buffer("mean", mean_201)
    bundle.register_buffer("std", std_clamped)

    # Load model weights — handle the key format
    # Checkpoint has keys like "motion_transformer.double_blocks.0..."
    # and also "null_vtxt_feat", "null_ctxt_input"
    missing, unexpected = bundle.load_state_dict(sd, strict=False)
    if missing:
        # Filter out expected missing keys (mean, std already loaded manually)
        real_missing = [k for k in missing if k not in ("mean", "std")]
        if real_missing:
            print(f"[WARN] Missing keys: {real_missing[:10]}...")
    if unexpected:
        print(f"[INFO] Unexpected keys (ignored): {unexpected[:5]}...")

    bundle = bundle.to(device)
    bundle.eval()
    print(f"[INFO] T2M model loaded: {sum(p.numel() for p in bundle.parameters())/1e6:.1f}M params")
    print(f"[INFO] mean shape: {bundle.mean.shape}, std shape: {bundle.std.shape}")
    return bundle


# ---------------------------------------------------------------------------
# Single repair flow
# ---------------------------------------------------------------------------

def sdedit_repair_single(
    pipeline: SDEditRepairPipeline,
    motion_135: torch.Tensor,
    mask_135: torch.Tensor,
    bundle,
    device: str,
    max_frames: int = 360,
    use_mask: bool = True,
) -> torch.Tensor:
    """Repair a single 135-dim motion using T2M SDEdit.

    Returns: (T_orig, 135) repaired motion tensor.
    """
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)

    # Crop to max_frames
    src_135 = motion_135[:T].unsqueeze(0)  # (1, T, 135)

    # Pad to max_frames if shorter
    if T < max_frames:
        pad_len = max_frames - T
        src_135 = F.pad(src_135, (0, 0, 0, pad_len), mode="constant", value=0)

    # Expand 135 → 201 (pad with zeros for local joint positions)
    src_201 = F.pad(src_135, (0, T2M_MOTION_DIM - M2M_MOTION_DIM), mode="constant", value=0)

    # Normalize with T2M stats
    mean = bundle.mean.cpu()
    std = bundle.std.cpu()
    src_norm = (src_201 - mean) / std

    # Prepare mask (also expand to 201-dim)
    if use_mask and mask_135 is not None:
        msk_135 = mask_135[:T].unsqueeze(0)
        if T < max_frames:
            msk_135 = F.pad(msk_135, (0, 0, 0, max_frames - T), mode="constant", value=0)
        # Pad mask to 201-dim (extra dims always masked=1 since we don't care about them)
        msk_201 = F.pad(msk_135, (0, T2M_MOTION_DIM - M2M_MOTION_DIM), mode="constant", value=1.0)
    else:
        msk_201 = None

    # Run SDEdit
    repaired_norm = pipeline(src_norm, [T], mask_201=msk_201)

    # Denormalize and extract 135 dims
    repaired_201 = repaired_norm.cpu() * std + mean
    repaired_135 = repaired_201[0, :T, :M2M_MOTION_DIM]  # (T, 135)

    # Final mask-based merge in original space
    if use_mask and mask_135 is not None:
        mask_crop = mask_135[:T]
        combined = motion_135[:T] * (1.0 - mask_crop) + repaired_135 * mask_crop
    else:
        combined = repaired_135

    # Append tail if cropped
    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)

    return combined


# ---------------------------------------------------------------------------
# Quality check (reuse from repair_and_evaluate.py)
# ---------------------------------------------------------------------------

_CHECKER_INSTANCE = None


def _get_checker():
    global _CHECKER_INSTANCE
    if _CHECKER_INSTANCE is None:
        from hymotion.utils.quality_check_rules.motion_quality_checker import MotionQualityChecker
        _CHECKER_INSTANCE = MotionQualityChecker(device="cpu")
    return _CHECKER_INSTANCE


def run_quality_check(npz_path: str) -> tuple:
    from hymotion.utils.quality_check_rules.mask_utils import (
        merge_invalid_masks,
        mask_to_sparse_dict,
    )
    checker = _get_checker()
    result = checker.check_from_file(npz_path)
    result_dict = result.to_dict()
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.no_mask:
        args.use_mask = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load quality list
    quality_list_path = str(PROJECT_ROOT / args.quality_list)
    with open(quality_list_path, "r") as f:
        quality_data = json.load(f)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    items = quality_data.get("items", [])
    if args.max_samples > 0:
        items = items[: args.max_samples]
    print(f"[INFO] Processing {len(items)} low-quality samples")
    print(f"[INFO] SDEdit params: t_start={args.t_start}, num_steps={args.num_steps}")

    # Build T2M model
    ckpt_path = str(PROJECT_ROOT / args.checkpoint)
    bundle = build_t2m_model(ckpt_path, args.device)
    pipeline = SDEditRepairPipeline(
        bundle, num_steps=args.num_steps, t_start=args.t_start
    )

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

            # 1. Load motion
            motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(npz_path)

            # 2. Quality check BEFORE
            before_result, before_sparse = run_quality_check(npz_path)
            before_valid = before_result.get("is_valid", True)
            before_failed = before_result.get("failed_checks", [])

            # 3. Build repair mask
            mask_135 = sparse_mask_to_dense(before_sparse, num_frames, expand_frames=5)
            mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)
            if mask_ratio < 0.01 and not before_valid:
                mask_135 = sparse_mask_to_dense(before_sparse, num_frames, expand_frames=15)
                mask_ratio = mask_135.sum().item() / max(mask_135.numel(), 1)

            # 4. SDEdit repair
            repaired_motion = sdedit_repair_single(
                pipeline, motion_135, mask_135, bundle, args.device,
                max_frames=args.max_frames, use_mask=args.use_mask,
            )

            # 5. Save temp NPZ for quality check
            repaired_aa, repaired_trans = motion_135_to_npz_format(repaired_motion, abs_trans_frame0)
            orig_data = dict(np.load(npz_path, allow_pickle=True))
            temp_npz = str(output_dir / "temp_repaired.npz")
            _save_repaired_npz(temp_npz, repaired_aa, repaired_trans, orig_data, fps)

            # 6. Quality check AFTER
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

            status_str = (
                "✓ FIXED" if detail["improved"]
                else ("✗ STILL BAD" if not after_valid else "= OK")
            )
            print(
                f"[{idx+1}/{len(items)}] {status_str} | "
                f"before={before_failed} after={after_failed} | "
                f"mask={mask_ratio:.1%} | {elapsed:.1f}s | {rel_path}"
            )

            # Save repaired motion permanently
            if detail["improved"]:
                out_path = output_dir / "repaired" / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                _save_repaired_npz(str(out_path), repaired_aa, repaired_trans, orig_data, fps)

        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append({"path": rel_path, "error": str(e)})
            print(f"[{idx+1}/{len(items)}] ERROR: {e} | {rel_path}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 70)
    print(f"SDEDIT REPAIR SUMMARY — T2M pretrained, t_start={args.t_start}")
    print("=" * 70)
    print(f"Total:        {stats['total']}")
    print(f"Processed:    {stats['processed']}")
    print(f"Skipped:      {stats['skipped']}")
    p = max(stats["processed"], 1)
    print(f"Before pass:  {stats['before_pass']} ({stats['before_pass']/p*100:.1f}%)")
    print(f"After pass:   {stats['after_pass']} ({stats['after_pass']/p*100:.1f}%)")
    print(f"Improved:     {stats['improved']} ({stats['improved']/p*100:.1f}%)")
    print(f"Degraded:     {stats['degraded']} ({stats['degraded']/p*100:.1f}%)")
    print(f"Unchanged:    {stats['unchanged']}")
    print("=" * 70)

    # Save stats
    stats_path = output_dir / f"sdedit_stats_t{args.t_start}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to: {stats_path}")

    # Clean up temp file
    temp_npz = output_dir / "temp_repaired.npz"
    if temp_npz.exists():
        temp_npz.unlink()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Base-pose edit sweep: reduce reliance on full-motion blending.

This script evaluates model-centric anchor-window regeneration variants for
the PeacekeeperElite before/after base-pose edit demo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_keyframe_pose_guidance import (  # noqa: E402
    BEFORE_DIR,
    AFTER_DIR,
    D,
    MAN_MODELS,
    MIN_KEYPOSE_DIFF,
    NUM_KEYPOSES,
    compute_metrics,
    find_latest_checkpoint,
    load_before_after_pairs,
    load_m2m_bundle,
    select_keyposes,
)
from scripts.run_hybrid_blend_polish import hybrid_blend_polish  # noqa: E402
from scripts.run_pure_blend_baseline import pure_blend  # noqa: E402


def build_anchor_window_batch(before: np.ndarray, after: np.ndarray, kp_indices: list[int], radius: int) -> dict:
    """Mask a local temporal window around each target keypose.

    The keypose frame itself is observed from the user-provided target pose.
    Everything outside the window remains original motion context.
    """
    T = before.shape[0]
    composite = before.copy()
    src_mask = np.zeros((T, D), dtype=np.float32)
    for ki in kp_indices:
        lo = max(0, int(ki) - radius)
        hi = min(T, int(ki) + radius + 1)
        src_mask[lo:hi] = 1.0
        src_mask[ki] = 0.0
        composite[ki] = after[ki].copy()
    return {
        "composite_motion": composite,
        "src_mask": src_mask,
        "before_motion": before,
        "after_motion": after,
        "keypose_indices": kp_indices,
        "num_frames": T,
    }


@torch.no_grad()
def run_m2m_imputation(bundle, pipeline, batch_info: dict, device: str) -> np.ndarray:
    composite = torch.from_numpy(batch_info["composite_motion"]).float().unsqueeze(0).to(device)
    src_mask = torch.from_numpy(batch_info["src_mask"]).float().unsqueeze(0).to(device)
    before = torch.from_numpy(batch_info["before_motion"]).float().unsqueeze(0).to(device)
    T = int(batch_info["num_frames"])

    norm_full = bundle.normalize_motion(composite)
    infer_batch = {
        "src_motion": norm_full * (1 - src_mask),
        "src_mask": src_mask,
        "src_length": [T],
        "tgt_length": [T],
        "clean_motion": norm_full,
    }
    result = pipeline(infer_batch)
    output = bundle.denormalize_motion(result["latent"])
    final = composite * (1 - src_mask) + output * src_mask
    final[:, :, :3] = before[:, :, :3]
    return final.squeeze(0).cpu().numpy()


def aggregate(rows: list[dict]) -> dict:
    keys = [
        "kf_mpjpe",
        "global_mpjpe",
        "src_mpjpe",
        "boundary_smoothness",
        "overall_smoothness",
        "foot_skating",
    ]
    return {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}


def save_case(path: Path, output: np.ndarray, before: np.ndarray, after: np.ndarray, src_mask: np.ndarray,
              kp_indices: list[int], diffs: np.ndarray, extra: dict | None = None) -> None:
    payload = dict(
        output_motion=output,
        before_motion=before,
        after_motion=after,
        composite_motion=before,
        src_mask=src_mask,
        keypose_indices=np.array(kp_indices, dtype=np.int64),
        correction_diffs=diffs,
    )
    if extra:
        payload.update(extra)
    np.savez_compressed(str(path), **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--radii", default="16,24,40,60")
    parser.add_argument("--model-index", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    before_dir = PROJECT_ROOT / BEFORE_DIR
    after_dir = PROJECT_ROOT / AFTER_DIR
    pairs = load_before_after_pairs(str(before_dir), str(after_dir), max_pairs=args.num_cases)
    if not pairs:
        raise RuntimeError("no before/after pairs loaded")

    model_name, cfg_rel, work_rel, _rot = MAN_MODELS[int(args.model_index)]
    ckpt = find_latest_checkpoint(str(PROJECT_ROOT / work_rel))
    if not ckpt:
        raise RuntimeError(f"checkpoint not found for {model_name}: {work_rel}")
    bundle = load_m2m_bundle(str(PROJECT_ROOT / cfg_rel), ckpt, device=device)
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    flow_pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=args.num_steps, replacement_guidance="flow_interp")
    skip_pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=args.num_steps, replacement_guidance="skip_last")

    base_dir = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot"
    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    variants = [(f"anchor_window_r{r}_flow_interp", r) for r in radii]
    variant_rows: dict[str, list[dict]] = {name: [] for name, _ in variants}
    hybrid_rows: list[dict] = []

    for name, _ in variants:
        (base_dir / name).mkdir(parents=True, exist_ok=True)
    hybrid_out_dir = base_dir / "hybrid_blend_boundary_polish_recheck"
    hybrid_out_dir.mkdir(parents=True, exist_ok=True)

    for case_idx, pair in enumerate(pairs):
        before = pair["before_motion"]
        after = pair["after_motion"]
        kp_indices, diffs = select_keyposes(before, after, k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF)
        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

        try:
            t0 = time.time()
            hybrid, blended, equiv_info, hybrid_mask = hybrid_blend_polish(
                bundle, skip_pipeline, before, after, kp_indices, device
            )
            h_metrics = compute_metrics(hybrid, before, after, kp_indices, hybrid_mask)
            hybrid_rows.append({
                "case_key": case_key,
                "filename": pair["filename"],
                "num_frames": int(pair["num_frames"]),
                "keypose_indices": kp_indices,
                "elapsed_sec": time.time() - t0,
                **h_metrics,
            })
            save_case(
                hybrid_out_dir / f"{case_key}.npz",
                hybrid, before, after, hybrid_mask, kp_indices, diffs,
                {"composite_motion": blended, "equiv_frames": np.array(sorted(set(sum(equiv_info.values(), []))))},
            )
        except Exception as e:
            print(f"[hybrid] {case_key} failed: {e}")
            traceback.print_exc()

        for variant_name, radius in variants:
            try:
                batch = build_anchor_window_batch(before, after, kp_indices, radius=radius)
                t0 = time.time()
                out = run_m2m_imputation(bundle, flow_pipeline, batch, device)
                for ki in kp_indices:
                    out[ki, 3:] = after[ki, 3:]
                metrics = compute_metrics(out, before, after, kp_indices, batch["src_mask"])
                row = {
                    "case_key": case_key,
                    "filename": pair["filename"],
                    "num_frames": int(pair["num_frames"]),
                    "keypose_indices": kp_indices,
                    "radius": radius,
                    "elapsed_sec": time.time() - t0,
                    **metrics,
                }
                variant_rows[variant_name].append(row)
                save_case(
                    base_dir / variant_name / f"{case_key}.npz",
                    out, before, after, batch["src_mask"], kp_indices, diffs,
                    {"radius": np.array(radius, dtype=np.int64)},
                )
                print(
                    f"{variant_name} {case_key}: glob={metrics['global_mpjpe']:.4f} "
                    f"bnd={metrics['boundary_smoothness']:.4f} "
                    f"smooth={metrics['overall_smoothness']:.4f} foot={metrics['foot_skating']:.4f}"
                )
            except Exception as e:
                print(f"[{variant_name}] {case_key} failed: {e}")
                traceback.print_exc()

    summary = {"variants": {}, "baseline_recheck": {}}
    if hybrid_rows:
        h_agg = aggregate(hybrid_rows)
        summary["baseline_recheck"] = {"aggregate": h_agg, "cases": hybrid_rows}
        with open(hybrid_out_dir / "results.json", "w") as f:
            json.dump(summary["baseline_recheck"], f, indent=2)
        print("\n[hybrid recheck]", h_agg)

    for variant_name, rows in variant_rows.items():
        if not rows:
            continue
        agg = aggregate(rows)
        summary["variants"][variant_name] = {"aggregate": agg, "cases": rows}
        with open(base_dir / variant_name / "results.json", "w") as f:
            json.dump({"aggregate": agg, "cases": rows}, f, indent=2)
        print(f"\n[{variant_name}]", agg)

    out_json = base_dir / "base_pose_anchor_regen_sweep_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_json}")


if __name__ == "__main__":
    main()

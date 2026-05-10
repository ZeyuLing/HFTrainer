#!/usr/bin/env python3
"""Keyframe Pose Guidance Evaluation — run imputation inference.

Uses HyMotion M2M (uncond + man variants) and MoGenDIT to perform
keypose-guided imputation on the PeacekeeperElite before/after dataset.

Usage (on GPU node):
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_keypose_guidance.py \
        --models uncond_fm_man uncond_jit_man uncond_fm_man_globalrot uncond_jit_man_globalrot mogendit \
        --max-samples 50 --num-steps 50

Output: output/keypose_eval/results/<model_name>/repaired/<filename>.npz
"""

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
os.chdir(str(PROJECT_ROOT))

# Reuse utilities from eval_m2m_repair
from scripts.eval_m2m_repair import (
    build_model,
    load_npz_as_motion,
    motion_135_to_npz_format,
    save_repaired_npz,
)


ALL_MODELS = [
    "uncond_fm_man",
    "uncond_fm_man_globalrot",
    "uncond_jit_man",
    "uncond_jit_man_globalrot",
    "mogendit",
]


def parse_args():
    p = argparse.ArgumentParser(description="Keypose Guidance Evaluation")
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--mogendit-device", type=str, default=None,
                    help="Device for MoGenDIT. Defaults to --device")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", type=str,
                    default="output/keypose_eval/eval_data.json")
    p.add_argument("--output-dir", type=str,
                    default="output/keypose_eval/results")
    return p.parse_args()


# ====================================================================
# HyMotion M2M keypose imputation
# ====================================================================

def run_m2m_keypose_imputation(pipeline, bundle, case, device, max_frames=360):
    """Run M2M imputation with keypose guidance.

    Strategy:
    - mask = 1 everywhere (regenerate all frames)
    - mask = 0 for first frame, last frame, and all keypose frames
    - src_motion = target values at keypose/anchor frames, zeros elsewhere (completion mode)
    - replacement_guidance handles keeping known regions clean during ODE
    """
    before_path = case["before_path"]
    after_path = case["after_path"]
    keypose_indices = case["keypose_indices"]

    # Load source (before) and target (after) motions
    src_motion, T_src, fps_src, abs_trans_src = load_npz_as_motion(before_path)
    tgt_motion, T_tgt, fps_tgt, abs_trans_tgt = load_npz_as_motion(after_path)

    T = min(T_src, T_tgt, max_frames)
    D = 135

    # Build composite motion: target values at anchor frames, src values elsewhere
    # (in completion mode, masked regions will be zeroed before VACE, so we set anchors from target)
    composite = src_motion[:T].clone()  # Start from src
    composite[0] = tgt_motion[0]  # First frame from target
    composite[T - 1] = tgt_motion[T - 1]  # Last frame from target
    for ki in keypose_indices:
        if ki < T:
            composite[ki] = tgt_motion[ki]  # Keyposes from target

    # Build mask: 1 = generate, 0 = keep
    mask = torch.ones(T, D, dtype=torch.float32)
    mask[0, :] = 0.0  # Keep first frame
    mask[T - 1, :] = 0.0  # Keep last frame
    for ki in keypose_indices:
        if ki < T:
            mask[ki, :] = 0.0  # Keep keypose frames

    # Normalize
    motion_norm = bundle.normalize_motion(composite.unsqueeze(0).to(device))
    msk = mask.unsqueeze(0).to(device)

    # Completion mode: zero masked regions
    motion_norm = motion_norm * (1 - msk)

    # Pad if needed
    if T < max_frames:
        pad_len = max_frames - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    batch = {
        "src_motion": motion_norm,
        "src_mask": msk,
        "src_length": [T],
        "tgt_length": [T],
    }

    with torch.no_grad():
        result = pipeline(batch)

    # Denormalize
    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    # Blend: keep anchor frames exact, use model output for rest
    mask_crop = mask[:T]
    combined = composite[:T] * (1 - mask_crop) + repaired_raw * mask_crop

    return combined, tgt_motion[:T], src_motion[:T], fps_src, abs_trans_tgt


def run_m2m_keypose_imputation_globalrot(pipeline, bundle, case, device, max_frames=360):
    """Run M2M imputation for global rotation models.

    Same as above but converts local rot6d to global rot6d before feeding to model,
    and converts back after.
    """
    from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
        local_to_global_rot6d_torch,
        global_to_local_rot6d_torch,
    )

    def _to_global_135(motion_135):
        T_m = motion_135.shape[0]
        trans = motion_135[:, :3]
        rot6d = motion_135[:, 3:].reshape(T_m, 22, 6)
        g_rot6d = local_to_global_rot6d_torch(rot6d)
        return torch.cat([trans, g_rot6d.reshape(T_m, -1)], dim=-1)

    def _to_local_135(motion_135):
        T_m = motion_135.shape[0]
        trans = motion_135[:, :3]
        rot6d = motion_135[:, 3:].reshape(T_m, 22, 6)
        l_rot6d = global_to_local_rot6d_torch(rot6d)
        return torch.cat([trans, l_rot6d.reshape(T_m, -1)], dim=-1)

    before_path = case["before_path"]
    after_path = case["after_path"]
    keypose_indices = case["keypose_indices"]

    src_motion, T_src, fps_src, abs_trans_src = load_npz_as_motion(before_path)
    tgt_motion, T_tgt, fps_tgt, abs_trans_tgt = load_npz_as_motion(after_path)

    T = min(T_src, T_tgt, max_frames)
    D = 135

    # Convert to global rotation space
    src_global = _to_global_135(src_motion[:T])
    tgt_global = _to_global_135(tgt_motion[:T])

    # Build composite in global space
    composite = src_global.clone()
    composite[0] = tgt_global[0]
    composite[T - 1] = tgt_global[T - 1]
    for ki in keypose_indices:
        if ki < T:
            composite[ki] = tgt_global[ki]

    # Build mask
    mask = torch.ones(T, D, dtype=torch.float32)
    mask[0, :] = 0.0
    mask[T - 1, :] = 0.0
    for ki in keypose_indices:
        if ki < T:
            mask[ki, :] = 0.0

    # Normalize (using global rotation stats)
    motion_norm = bundle.normalize_motion(composite.unsqueeze(0).to(device))
    msk = mask.unsqueeze(0).to(device)
    motion_norm = motion_norm * (1 - msk)

    if T < max_frames:
        pad_len = max_frames - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    batch = {
        "src_motion": motion_norm,
        "src_mask": msk,
        "src_length": [T],
        "tgt_length": [T],
    }

    with torch.no_grad():
        result = pipeline(batch)

    repaired_latent = result["latent"][0, :T].cpu()
    repaired_global = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    mask_crop = mask[:T]
    combined_global = composite[:T] * (1 - mask_crop) + repaired_global * mask_crop

    # Convert back to local rotation
    combined_local = _to_local_135(combined_global)

    return combined_local, tgt_motion[:T], src_motion[:T], fps_src, abs_trans_tgt


# ====================================================================
# MoGenDIT keypose imputation
# ====================================================================

def run_mogendit_keypose_imputation(mogendit_pipeline, case, device):
    """Run MoGenDIT imputation with keypose guidance.

    MoGenDIT uses 201-dim representation. We load the NPZ, construct an observation
    mask for keyposes, and run the imputation-style repair.
    """
    before_path = case["before_path"]
    after_path = case["after_path"]
    keypose_indices = case["keypose_indices"]

    # Load both motions as MoGenDIT 201-dim
    src_motion, src_meta = mogendit_pipeline._load_npz_as_motion(before_path)
    tgt_motion, tgt_meta = mogendit_pipeline._load_npz_as_motion(after_path)

    T = min(src_motion.shape[1], tgt_motion.shape[1])
    D = src_motion.shape[2]  # 201

    # Build composite: keyposes from target, rest from source
    composite = src_motion[:, :T].clone()
    composite[:, 0] = tgt_motion[:, 0]
    composite[:, T - 1] = tgt_motion[:, T - 1]
    for ki in keypose_indices:
        if ki < T:
            composite[:, ki] = tgt_motion[:, ki]

    # Build observation mask (1 = observed/keep, 0 = generate)
    # MoGenDIT convention: obs_mask=1 means keep
    obs_mask = torch.zeros(1, T, D, device=device)
    obs_mask[:, 0, :] = 1.0
    obs_mask[:, T - 1, :] = 1.0
    for ki in keypose_indices:
        if ki < T:
            obs_mask[:, ki, :] = 1.0

    # Run MoGenDIT imputation via refiner
    repaired = mogendit_pipeline.refiner.refine(
        motion=composite.to(device),
        cond=None,
        step=10,
        mode='denoise',
        obs_mask=obs_mask,
        use_windowed=False,
    )

    # Convert back to NPZ format
    repaired_dict = mogendit_pipeline._motion_to_npz_dict(repaired, tgt_meta)

    # Also need src and target in NPZ format for comparison
    src_dict = mogendit_pipeline._motion_to_npz_dict(src_motion[:, :T], src_meta)
    tgt_dict = mogendit_pipeline._motion_to_npz_dict(tgt_motion[:, :T], tgt_meta)

    fps = tgt_meta.get("mocap_framerate", 30.0)
    return repaired_dict, tgt_dict, src_dict, fps


def run_mogendit_keypose_simple(mogendit_pipeline, case, device):
    """Simplified MoGenDIT approach: repair the before motion with denoise,
    then replace keypose frames from target.

    Since MoGenDIT's refiner may not directly support obs_mask argument,
    we use a simpler approach:
    1. Take the before motion
    2. Replace keypose frames with target frames
    3. Run light denoise to smooth transitions
    """
    import tempfile

    before_path = case["before_path"]
    after_path = case["after_path"]
    keypose_indices = case["keypose_indices"]

    # Load raw NPZ data
    before_data = dict(np.load(before_path, allow_pickle=True))
    after_data = dict(np.load(after_path, allow_pickle=True))

    before_poses = np.array(before_data["poses"], dtype=np.float32)
    after_poses = np.array(after_data["poses"], dtype=np.float32)
    before_trans = np.array(before_data.get("trans", before_data.get("transl")), dtype=np.float32)
    after_trans = np.array(after_data.get("trans", after_data.get("transl")), dtype=np.float32)

    T = min(before_poses.shape[0], after_poses.shape[0])

    # Build composite: replace keypose frames
    composite_poses = before_poses[:T].copy()
    composite_trans = before_trans[:T].copy()

    composite_poses[0] = after_poses[0]
    composite_trans[0] = after_trans[0]
    composite_poses[T - 1] = after_poses[T - 1]
    composite_trans[T - 1] = after_trans[T - 1]
    for ki in keypose_indices:
        if ki < T:
            composite_poses[ki] = after_poses[ki]
            composite_trans[ki] = after_trans[ki]

    # Save composite to temp file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
        np.savez(
            tmp_path,
            poses=composite_poses,
            trans=composite_trans,
            betas=before_data.get("betas", np.zeros((1, 16), dtype=np.float32)),
            gender=str(before_data.get("gender", "neutral")),
            mocap_framerate=float(before_data.get("mocap_framerate", 30)),
        )

    # Run MoGenDIT denoise
    repaired_dict = mogendit_pipeline.repair_motion_dict(
        {
            "poses": composite_poses,
            "trans": composite_trans,
            "betas": before_data.get("betas", np.zeros((1, 16), dtype=np.float32)),
            "gender": str(before_data.get("gender", "neutral")),
            "mocap_framerate": float(before_data.get("mocap_framerate", 30)),
        },
        mode="denoise",
        step=10,
    )

    os.unlink(tmp_path)

    fps = float(before_data.get("mocap_framerate", 30))
    return repaired_dict, fps


# ====================================================================
# Metric computation
# ====================================================================

def compute_metrics(repaired_135, target_135, src_135, keypose_indices):
    """Compute evaluation metrics."""
    T = min(repaired_135.shape[0], target_135.shape[0])

    repaired = repaired_135[:T].numpy() if isinstance(repaired_135, torch.Tensor) else repaired_135[:T]
    target = target_135[:T].numpy() if isinstance(target_135, torch.Tensor) else target_135[:T]
    source = src_135[:T].numpy() if isinstance(src_135, torch.Tensor) else src_135[:T]

    # Overall MPJPE (rot6d space)
    overall_err = np.abs(repaired - target).mean()

    # Keypose MPJPE
    kp_errs = []
    for ki in keypose_indices:
        if ki < T:
            kp_errs.append(np.abs(repaired[ki] - target[ki]).mean())
    keypose_err = float(np.mean(kp_errs)) if kp_errs else 0.0

    # First/last frame preservation
    first_err = float(np.abs(repaired[0] - target[0]).mean())
    last_err = float(np.abs(repaired[T-1] - target[T-1]).mean())

    # Smoothness (velocity consistency)
    vel = np.diff(repaired, axis=0)
    smoothness = float(np.std(np.linalg.norm(vel, axis=1)))

    # Source-to-target improvement (lower = better)
    src_target_err = np.abs(source - target).mean()
    improvement_ratio = float(overall_err / (src_target_err + 1e-8))

    return {
        "overall_mpjpe": float(overall_err),
        "keypose_mpjpe": keypose_err,
        "first_frame_err": first_err,
        "last_frame_err": last_err,
        "smoothness": smoothness,
        "src_target_mpjpe": float(src_target_err),
        "improvement_ratio": improvement_ratio,
    }


# ====================================================================
# Main evaluation
# ====================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mogendit_device is None:
        args.mogendit_device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    cases = manifest["cases"]
    if args.max_samples > 0:
        cases = cases[:args.max_samples]
    print(f"Loaded {len(cases)} eval cases")

    # Run each model
    all_results = {}

    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        model_dir = output_dir / model_name / "repaired"
        model_dir.mkdir(parents=True, exist_ok=True)
        details_path = output_dir / model_name / "details.jsonl"
        is_globalrot = "globalrot" in model_name

        if model_name == "mogendit":
            # Build MoGenDIT
            try:
                from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
                mogendit = MoGenDITRepairPipeline(
                    model_name='MoreDiff-0.1B',
                    device=args.mogendit_device,
                )
            except Exception as e:
                print(f"[ERROR] Failed to build MoGenDIT: {e}")
                traceback.print_exc()
                continue

            model_metrics = []
            with open(details_path, "w") as df:
                for idx, case in enumerate(cases):
                    fname = case["filename"]
                    out_path = model_dir / fname
                    if out_path.exists():
                        print(f"  [{idx+1}/{len(cases)}] {fname} — SKIP (exists)")
                        continue

                    try:
                        t0 = time.time()
                        repaired_dict, fps = run_mogendit_keypose_simple(
                            mogendit, case, args.mogendit_device,
                        )
                        elapsed = time.time() - t0

                        # Save repaired NPZ
                        np.savez(str(out_path), **repaired_dict)

                        # Compute metrics using 135-dim representation
                        # Load back for consistent comparison
                        try:
                            rep_motion, _, _, _ = load_npz_as_motion(str(out_path))
                            tgt_motion, _, _, _ = load_npz_as_motion(case["after_path"])
                            src_motion, _, _, _ = load_npz_as_motion(case["before_path"])
                            metrics = compute_metrics(
                                rep_motion, tgt_motion, src_motion,
                                case["keypose_indices"],
                            )
                        except Exception:
                            metrics = {}

                        detail = {
                            "filename": fname,
                            "elapsed_s": round(elapsed, 2),
                            **metrics,
                        }
                        df.write(json.dumps(detail) + "\n")
                        df.flush()
                        model_metrics.append(metrics)

                        print(f"  [{idx+1}/{len(cases)}] {fname} — "
                              f"overall={metrics.get('overall_mpjpe', '?'):.4f}, "
                              f"kp={metrics.get('keypose_mpjpe', '?'):.4f}, "
                              f"{elapsed:.1f}s")

                    except Exception as e:
                        print(f"  [{idx+1}/{len(cases)}] {fname} — ERROR: {e}")
                        traceback.print_exc()

            del mogendit
            torch.cuda.empty_cache()
            all_results[model_name] = model_metrics

        else:
            # Build HyMotion M2M model
            try:
                pipeline, bundle, ckpt_path, is_man = build_model(
                    model_name, args.device, args.num_steps,
                )
            except Exception as e:
                print(f"[ERROR] Failed to build model {model_name}: {e}")
                traceback.print_exc()
                continue

            model_metrics = []
            with open(details_path, "w") as df:
                for idx, case in enumerate(cases):
                    fname = case["filename"]
                    out_path = model_dir / fname

                    if out_path.exists():
                        print(f"  [{idx+1}/{len(cases)}] {fname} — SKIP (exists)")
                        continue

                    try:
                        t0 = time.time()

                        if is_globalrot:
                            combined, tgt_motion, src_motion, fps, abs_trans = \
                                run_m2m_keypose_imputation_globalrot(
                                    pipeline, bundle, case, args.device,
                                )
                        else:
                            combined, tgt_motion, src_motion, fps, abs_trans = \
                                run_m2m_keypose_imputation(
                                    pipeline, bundle, case, args.device,
                                )

                        elapsed = time.time() - t0

                        # Compute metrics
                        metrics = compute_metrics(
                            combined, tgt_motion, src_motion,
                            case["keypose_indices"],
                        )

                        # Save repaired NPZ
                        orig_data = dict(np.load(case["after_path"], allow_pickle=True))
                        repaired_aa, repaired_trans = motion_135_to_npz_format(combined, abs_trans)
                        save_repaired_npz(str(out_path), repaired_aa, repaired_trans, orig_data, fps)

                        detail = {
                            "filename": fname,
                            "elapsed_s": round(elapsed, 2),
                            **metrics,
                        }
                        df.write(json.dumps(detail) + "\n")
                        df.flush()
                        model_metrics.append(metrics)

                        print(f"  [{idx+1}/{len(cases)}] {fname} — "
                              f"overall={metrics['overall_mpjpe']:.4f}, "
                              f"kp={metrics['keypose_mpjpe']:.4f}, "
                              f"{elapsed:.1f}s")

                    except Exception as e:
                        print(f"  [{idx+1}/{len(cases)}] {fname} — ERROR: {e}")
                        traceback.print_exc()

            # Free GPU
            del pipeline, bundle
            torch.cuda.empty_cache()
            all_results[model_name] = model_metrics

    # ====================================================================
    # Summary report
    # ====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")

    summary = {}
    for model_name, metrics_list in all_results.items():
        if not metrics_list:
            continue
        agg = {}
        for key in metrics_list[0].keys():
            vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        summary[model_name] = agg
        print(f"\n{model_name}:")
        print(f"  Samples: {len(metrics_list)}")
        for k, v in agg.items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()

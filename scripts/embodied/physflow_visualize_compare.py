#!/usr/bin/env python3
"""PhysFlow Visualization Comparison: Original vs Fine-tuned T2M output.

Generates motions from both original pretrained model and PhysFlow fine-tuned
model, then runs RL physics simulation on each. Produces:
  1. NPZ files for 3D viewer comparison
  2. Physics metrics comparison (completion, tracking error, root height)
  3. Summary report

Usage:
    python3 scripts/embodied/physflow_visualize_compare.py \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --pretrained-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
        --finetuned-ckpt output/physflow_v2_train_blend50/model_iter500.pt \
        --text-cache output/physflow_v2_test/text_embeddings.pt \
        --output-dir output/physflow_v2_compare \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def load_bundle_and_generate(
    config_path: str,
    checkpoint_path: str,
    text_cache_path: str,
    prompts: List[str],
    num_frames: int,
    device: str,
    finetuned_ckpt: Optional[str] = None,
) -> List[np.ndarray]:
    """Load T2M bundle and generate motions for given prompts.

    Args:
        config_path: Path to T2M config .py file
        checkpoint_path: Path to pretrained checkpoint
        text_cache_path: Path to pre-computed text embeddings
        prompts: List of text prompts to generate
        num_frames: Number of frames per motion
        device: CUDA device string
        finetuned_ckpt: Optional path to fine-tuned checkpoint (loads on top)

    Returns:
        List of motion_135 arrays, each shape (num_frames, 135)
    """
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    # Load config and build bundle
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()

    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load pretrained weights
    print(f"Loading pretrained checkpoint: {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)

    # Load fine-tuned weights on top (partial state dict — only transformer blocks)
    if finetuned_ckpt:
        print(f"Loading fine-tuned checkpoint: {finetuned_ckpt}")
        ft_data = torch.load(finetuned_ckpt, map_location='cpu')
        ft_state = ft_data.get('model_state_dict', ft_data)

        # Apply to motion_transformer
        mt = bundle.motion_transformer
        mt_state = mt.state_dict()
        loaded_count = 0
        for key, value in ft_state.items():
            if key in mt_state:
                mt_state[key] = value
                loaded_count += 1
        mt.load_state_dict(mt_state)
        print(f"  Applied {loaded_count}/{len(ft_state)} fine-tuned params to motion_transformer")

        # Report checkpoint metadata
        if 'iteration' in ft_data:
            print(f"  Checkpoint iteration: {ft_data['iteration']}")
        if 'curriculum_state' in ft_data:
            cs = ft_data['curriculum_state']
            print(f"  Curriculum: level={cs.get('current_level')}, "
                  f"total_iters={cs.get('total_iterations')}")

    bundle = bundle.to(device)
    bundle.eval()

    # Load text cache
    print(f"Loading text embeddings from: {text_cache_path}")
    text_cache = torch.load(text_cache_path, map_location='cpu')

    # Helper: length to mask
    def _length_to_mask(lengths, max_len):
        mask = torch.arange(max_len, device=lengths.device).expand(
            lengths.shape[0], max_len) < lengths.unsqueeze(1)
        return mask

    # Generate motions
    TRAIN_FRAMES = 360
    motion_dim = bundle.motion_transformer.output_dim
    num_ode_steps = 50
    cfg_scale = 5.0

    results = []

    for i, prompt in enumerate(prompts):
        print(f"  Generating [{i+1}/{len(prompts)}]: \"{prompt}\"")
        t0 = time.time()

        L = num_frames
        L_padded = max(L, TRAIN_FRAMES)
        B = 1

        # Get text features
        if prompt not in text_cache:
            print(f"    WARNING: prompt not in text cache, skipping: {prompt}")
            results.append(None)
            continue

        feats = text_cache[prompt]
        vtxt_input = feats['text_vec_raw'].to(device)
        ctxt_input = feats['text_ctxt_raw'].to(device)
        ctxt_length = feats['text_ctxt_raw_length'].to(device)
        ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])

        # Target padding mask
        tgt_padding_mask = _length_to_mask(
            torch.tensor([L], dtype=torch.long, device=device), L_padded
        )

        # CFG setup
        null_vtxt = bundle.null_vtxt_feat.expand_as(vtxt_input)
        vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
        ctxt_cfg = torch.cat([ctxt_input, ctxt_input], dim=0)
        ctxt_mask_cfg = torch.cat([ctxt_mask_temporal, ctxt_mask_temporal], dim=0)

        def fn(t_val, x):
            x_double = torch.cat([x, x], dim=0)
            x_pred = bundle.predict_flow(
                x_input=x_double,
                ctxt_input=ctxt_cfg,
                vtxt_input=vtxt_cfg,
                timesteps=t_val.expand(2 * B),
                x_mask_temporal=tgt_padding_mask.repeat(2, 1),
                ctxt_mask_temporal=ctxt_mask_cfg,
            )
            if bundle.pred_type == 'x1':
                t_eps = 0.05
                x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
            pred_uncond, pred_text = x_pred.chunk(2, dim=0)
            x_pred = pred_uncond + cfg_scale * (pred_text - pred_uncond)
            return x_pred

        # Euler ODE integration
        with torch.no_grad():
            y0 = torch.randn(B, L_padded, motion_dim, device=device, dtype=torch.float32)
            dt = 1.0 / num_ode_steps
            x = y0
            for step in range(num_ode_steps):
                t_val = torch.tensor(step * dt, device=device, dtype=torch.float32)
                v = fn(t_val, x)
                x = x + v * dt

            sampled = x[:, :L, :]
            latent_denorm = bundle.denormalize_motion(sampled)
            motion_201 = latent_denorm[0].cpu().numpy()
            motion_135 = motion_201[:, :135].astype(np.float32)

        elapsed = time.time() - t0
        print(f"    Generated in {elapsed:.1f}s, shape={motion_135.shape}")
        results.append(motion_135)

    return results


def run_rl_physics_evaluation(
    motions_135: List[np.ndarray],
    prompts: List[str],
    output_dir: str,
    label: str,
) -> List[dict]:
    """Run RL physics simulation on generated motions.

    Returns list of stats dicts with physics metrics.
    """
    from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle

    oracle = RLPhysicsOracle()
    results = []

    os.makedirs(output_dir, exist_ok=True)

    for i, (motion_135, prompt) in enumerate(zip(motions_135, prompts)):
        if motion_135 is None:
            results.append({'status': 'skipped', 'prompt': prompt})
            continue

        print(f"  RL sim [{i+1}/{len(motions_135)}] ({label}): \"{prompt}\"")
        t0 = time.time()

        try:
            motion_135_rl, stats = oracle.correct(motion_135)
            elapsed = time.time() - t0

            stats['prompt'] = prompt
            stats['label'] = label
            stats['elapsed'] = elapsed

            # Save NPZ files
            safe_name = prompt.replace(' ', '_')[:40]
            npz_raw = os.path.join(output_dir, f"{label}_{i:02d}_{safe_name}_raw.npz")
            npz_rl = os.path.join(output_dir, f"{label}_{i:02d}_{safe_name}_rl.npz")

            np.savez(npz_raw, motion_135=motion_135, fps=30, prompt=prompt)
            np.savez(npz_rl, motion_135=motion_135_rl, fps=30, prompt=prompt)

            stats['npz_raw'] = npz_raw
            stats['npz_rl'] = npz_rl

            status = stats.get('status', 'unknown')
            completion = stats.get('completion_ratio', 0)
            root_h = stats.get('root_height_min', 0)
            print(f"    Status={status}, completion={completion:.2f}, "
                  f"root_h_min={root_h:.3f}, time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            stats = {
                'status': 'error',
                'error': str(e),
                'prompt': prompt,
                'label': label,
                'elapsed': elapsed,
            }
            print(f"    ERROR: {e}")

        results.append(stats)

    return results


def print_comparison_report(
    pretrained_stats: List[dict],
    finetuned_stats: List[dict],
    prompts: List[str],
    output_path: str,
):
    """Print and save a comparison report."""

    lines = []
    lines.append("=" * 80)
    lines.append("PhysFlow Visualization Comparison Report")
    lines.append("=" * 80)
    lines.append("")

    # Per-prompt comparison
    lines.append(f"{'Prompt':<40} | {'Pretrained':^20} | {'Fine-tuned':^20} | {'Δ':<10}")
    lines.append("-" * 95)

    pre_completions = []
    ft_completions = []
    pre_successes = 0
    ft_successes = 0

    for i, prompt in enumerate(prompts):
        pre = pretrained_stats[i] if i < len(pretrained_stats) else {}
        ft = finetuned_stats[i] if i < len(finetuned_stats) else {}

        pre_comp = pre.get('completion_ratio', 0)
        ft_comp = ft.get('completion_ratio', 0)
        pre_status = pre.get('status', 'N/A')
        ft_status = ft.get('status', 'N/A')
        pre_root = pre.get('root_height_min', 0)
        ft_root = ft.get('root_height_min', 0)

        pre_completions.append(pre_comp)
        ft_completions.append(ft_comp)
        if pre_status == 'success' or pre_comp >= 0.8:
            pre_successes += 1
        if ft_status == 'success' or ft_comp >= 0.8:
            ft_successes += 1

        delta = ft_comp - pre_comp
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"

        short_prompt = prompt[:38] + ".." if len(prompt) > 40 else prompt
        pre_info = f"{pre_status[:4]} c={pre_comp:.2f} h={pre_root:.2f}"
        ft_info = f"{ft_status[:4]} c={ft_comp:.2f} h={ft_root:.2f}"

        lines.append(f"{short_prompt:<40} | {pre_info:^20} | {ft_info:^20} | {delta_str:<10}")

    lines.append("-" * 95)
    lines.append("")

    # Summary statistics
    n = len(prompts)
    lines.append("SUMMARY STATISTICS")
    lines.append(f"  Total prompts evaluated: {n}")
    lines.append(f"  Pretrained:")
    lines.append(f"    Avg completion:   {np.mean(pre_completions):.3f}")
    lines.append(f"    Success rate:     {pre_successes}/{n} ({100*pre_successes/n:.1f}%)")
    lines.append(f"  Fine-tuned (PhysFlow blend50, iter 500):")
    lines.append(f"    Avg completion:   {np.mean(ft_completions):.3f}")
    lines.append(f"    Success rate:     {ft_successes}/{n} ({100*ft_successes/n:.1f}%)")
    lines.append(f"  Improvement:")
    lines.append(f"    Avg completion:   {np.mean(ft_completions) - np.mean(pre_completions):+.3f}")
    lines.append(f"    Success rate:     {ft_successes - pre_successes:+d} "
                 f"({100*(ft_successes - pre_successes)/n:+.1f}%)")
    lines.append("")

    # Per-category breakdown
    categories = {
        'standing': [],
        'walking': [],
        'upper_body': [],
        'transitions': [],
        'dynamic': [],
    }
    # Categorize prompts based on keywords
    for i, prompt in enumerate(prompts):
        pl = prompt.lower()
        if any(w in pl for w in ['stand', 'shift weight', 'idle']):
            categories['standing'].append(i)
        elif any(w in pl for w in ['walk', 'step', 'pace']):
            categories['walking'].append(i)
        elif any(w in pl for w in ['wave', 'raise', 'clap', 'point', 'gesture']):
            categories['upper_body'].append(i)
        elif any(w in pl for w in ['walk then', 'turn', 'stop', 'transition']):
            categories['transitions'].append(i)
        elif any(w in pl for w in ['kick', 'jump', 'squat', 'dance', 'punch', 'run']):
            categories['dynamic'].append(i)
        else:
            categories['dynamic'].append(i)  # default to dynamic

    lines.append("PER-CATEGORY BREAKDOWN")
    for cat, indices in categories.items():
        if not indices:
            continue
        pre_cat = [pre_completions[j] for j in indices]
        ft_cat = [ft_completions[j] for j in indices]
        lines.append(f"  {cat} (n={len(indices)}): "
                     f"pretrained={np.mean(pre_cat):.3f} → finetuned={np.mean(ft_cat):.3f} "
                     f"(Δ={np.mean(ft_cat)-np.mean(pre_cat):+.3f})")

    lines.append("")
    lines.append("=" * 80)
    lines.append("NPZ files saved in output directory for 3D visualization.")
    lines.append("Use motion_annot_web/embodied_viz to view side-by-side.")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    # Save report
    report_path = os.path.join(output_path, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save detailed JSON
    json_path = os.path.join(output_path, "comparison_results.json")
    results_json = {
        'prompts': prompts,
        'pretrained_stats': [{k: v for k, v in s.items()
                              if not isinstance(v, np.ndarray)}
                             for s in pretrained_stats],
        'finetuned_stats': [{k: v for k, v in s.items()
                             if not isinstance(v, np.ndarray)}
                            for s in finetuned_stats],
        'summary': {
            'pretrained_avg_completion': float(np.mean(pre_completions)),
            'finetuned_avg_completion': float(np.mean(ft_completions)),
            'pretrained_success_rate': pre_successes / n,
            'finetuned_success_rate': ft_successes / n,
            'improvement_completion': float(np.mean(ft_completions) - np.mean(pre_completions)),
            'improvement_success_rate': (ft_successes - pre_successes) / n,
        }
    }
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"JSON results saved to: {json_path}")


# ======================================================================
# Test prompts covering all curriculum levels
# ======================================================================

TEST_PROMPTS = [
    # Standing (easy - level 0)
    "a person stands still",
    "a person stands in a relaxed pose",
    "a person shifts weight from left to right foot",
    # Walking (medium - level 1)
    "a person walks forward at a normal pace",
    "a person walks in a small circle",
    "a person walks forward slowly",
    "a person walks with long strides",
    # Upper body (medium - level 2)
    "a person waves with their right hand",
    "a person raises both arms above their head",
    "a person claps their hands together",
    "a person stretches arms to the sides",
    # Transitions (hard - level 3)
    "a person walks and then stops",
    "a person walks forward then turns around",
    "a person jogs slowly then walks",
    # Dynamic (hardest - level 4)
    "a person kicks with their right leg",
    "a person squats down and stands back up",
    "a person jumps in place",
    "a person does a jumping jack",
    "a person does a high kick",
]


def main():
    parser = argparse.ArgumentParser(description="PhysFlow Visual Comparison")
    parser.add_argument("--t2m-config", type=str, required=True,
                        help="Path to T2M config .py")
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="Path to original pretrained checkpoint")
    parser.add_argument("--finetuned-ckpt", type=str, required=True,
                        help="Path to PhysFlow fine-tuned checkpoint")
    parser.add_argument("--text-cache", type=str, required=True,
                        help="Path to pre-computed text embeddings .pt")
    parser.add_argument("--output-dir", type=str, default="output/physflow_v2_compare",
                        help="Output directory for NPZ and report")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--num-frames", type=int, default=120,
                        help="Number of frames per motion (default: 120 = 4s @ 30fps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible comparison")
    parser.add_argument("--skip-rl", action="store_true",
                        help="Skip RL physics evaluation (only generate)")
    parser.add_argument("--prompts", type=str, nargs='+', default=None,
                        help="Custom prompts (overrides default test set)")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = args.prompts if args.prompts else TEST_PROMPTS
    print(f"Will compare {len(prompts)} prompts")
    print(f"Output directory: {args.output_dir}")
    print()

    # ============================================================
    # Phase 1: Generate motions with PRETRAINED model
    # ============================================================
    print("=" * 60)
    print("PHASE 1: Generating motions with PRETRAINED model")
    print("=" * 60)

    # Reset seed before each generation for fair comparison
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pretrained_motions = load_bundle_and_generate(
        config_path=args.t2m_config,
        checkpoint_path=args.pretrained_ckpt,
        text_cache_path=args.text_cache,
        prompts=prompts,
        num_frames=args.num_frames,
        device=args.device,
        finetuned_ckpt=None,  # No fine-tuning
    )

    # Free GPU memory
    torch.cuda.empty_cache()

    # ============================================================
    # Phase 2: Generate motions with FINE-TUNED model
    # ============================================================
    print()
    print("=" * 60)
    print("PHASE 2: Generating motions with FINE-TUNED model")
    print("=" * 60)

    # Reset seed to same value for fair comparison (same noise init)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    finetuned_motions = load_bundle_and_generate(
        config_path=args.t2m_config,
        checkpoint_path=args.pretrained_ckpt,
        text_cache_path=args.text_cache,
        prompts=prompts,
        num_frames=args.num_frames,
        device=args.device,
        finetuned_ckpt=args.finetuned_ckpt,
    )

    # Free GPU memory
    torch.cuda.empty_cache()

    # ============================================================
    # Phase 3: RL Physics Evaluation
    # ============================================================
    if not args.skip_rl:
        print()
        print("=" * 60)
        print("PHASE 3: RL Physics Evaluation")
        print("=" * 60)

        npz_dir = os.path.join(args.output_dir, "npz")
        os.makedirs(npz_dir, exist_ok=True)

        print("\n--- Pretrained model outputs ---")
        pretrained_stats = run_rl_physics_evaluation(
            pretrained_motions, prompts, npz_dir, label="pretrained")

        print("\n--- Fine-tuned model outputs ---")
        finetuned_stats = run_rl_physics_evaluation(
            finetuned_motions, prompts, npz_dir, label="finetuned")

        # ============================================================
        # Phase 4: Comparison Report
        # ============================================================
        print()
        print("=" * 60)
        print("PHASE 4: Comparison Report")
        print("=" * 60)
        print()

        print_comparison_report(
            pretrained_stats, finetuned_stats, prompts, args.output_dir)

    else:
        # Just save NPZ without RL
        npz_dir = os.path.join(args.output_dir, "npz")
        os.makedirs(npz_dir, exist_ok=True)
        for i, (motion, prompt) in enumerate(zip(pretrained_motions, prompts)):
            if motion is not None:
                safe_name = prompt.replace(' ', '_')[:40]
                np.savez(os.path.join(npz_dir, f"pretrained_{i:02d}_{safe_name}.npz"),
                         motion_135=motion, fps=30, prompt=prompt)
        for i, (motion, prompt) in enumerate(zip(finetuned_motions, prompts)):
            if motion is not None:
                safe_name = prompt.replace(' ', '_')[:40]
                np.savez(os.path.join(npz_dir, f"finetuned_{i:02d}_{safe_name}.npz"),
                         motion_135=motion, fps=30, prompt=prompt)
        print(f"\nNPZ files saved to: {npz_dir}")
        print("Skipped RL evaluation (use without --skip-rl for full comparison)")


if __name__ == "__main__":
    main()

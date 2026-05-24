"""PhysFlow Evaluation: Compare T2M quality before/after PhysFlow training.

Generates motions from a set of test prompts using:
  1. Original T2M model (before PhysFlow)
  2. PhysFlow-trained T2M model (after)

For each motion, runs RL physics correction and computes metrics:
  - Physics pass rate (RL tracker completion ratio >= threshold)
  - Tracking error (RL closeness to reference)
  - Root height stability (no collapse/floating)
  - Foot sliding (estimated from ground contact)
  - Curriculum level success rates

Also exports demo NPZ files for website visualization.

Usage:
    # Compare original vs trained model
    python3 scripts/embodied/physflow_evaluate.py evaluate \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
        --trained-ckpt output/physflow_v2/model_final.pt \
        --output-dir output/physflow_v2/eval

    # Evaluate single model (quick physics pass rate check)
    python3 scripts/embodied/physflow_evaluate.py evaluate \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
        --output-dir output/physflow_v2/eval_baseline --quick

    # Analyze training log
    python3 scripts/embodied/physflow_evaluate.py analyze \
        --log-file output/physflow_v2/training_log.jsonl \
        --output-dir output/physflow_v2/analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.embodied.physflow_curriculum import PHYSFLOW_LEVELS, PhysFlowCurriculum
from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle
from scripts.embodied.physflow_trainer import (
    PhysFlowTrainer,
    load_bundle,
    motion_135_to_201,
)
from scripts.embodied.physflow_motion_converter import MotionFormatConverter


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Prompts (held-out from curriculum training set)
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_PROMPTS_BY_LEVEL = {
    "standing": [
        "a person stands and tilts their head",
        "a person stands with weight on left leg",
        "a person stands and takes a deep breath",
    ],
    "walking": [
        "a person walks forward three steps",
        "a person walks slowly to the right",
        "a person walks and pauses midstep",
    ],
    "upper_body": [
        "a person reaches up with both hands",
        "a person claps above their head",
        "a person gestures with their left hand",
    ],
    "transitions": [
        "a person walks then turns left",
        "a person jogs briefly and waves",
        "a person steps forward and bends down",
    ],
    "dynamic": [
        "a person does a deep squat",
        "a person kicks forward with right leg",
        "a person hops on their left foot",
    ],
}

# Flat evaluation set
ALL_EVAL_PROMPTS = []
for level_name, prompts in EVAL_PROMPTS_BY_LEVEL.items():
    for prompt in prompts:
        ALL_EVAL_PROMPTS.append({"prompt": prompt, "level": level_name})


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_physics_metrics(
    oracle: RLPhysicsOracle,
    motion_135: np.ndarray,
    min_completion: float = 0.8,
    min_root_height: float = 0.3,
) -> Dict:
    """Run RL physics correction and compute quality metrics.

    Returns:
        Dictionary with:
          - pass: bool (overall physics quality gate)
          - status: str (RL tracker outcome)
          - completion_ratio: float
          - tracking_error_mean: float
          - root_height_min: float
          - corrected_motion: np.ndarray (RL-corrected motion_135)
    """
    try:
        motion_corrected, stats = oracle.correct(motion_135)
    except Exception as e:
        return {
            "pass": False,
            "status": f"error: {str(e)[:80]}",
            "completion_ratio": 0.0,
            "tracking_error_mean": float("inf"),
            "root_height_min": 0.0,
            "corrected_motion": None,
            "actual_sim_steps": 0,
            "total_sim_steps": 0,
        }

    passed = oracle.is_good_quality(
        stats,
        min_completion=min_completion,
        min_root_height=min_root_height,
    )

    return {
        "pass": passed,
        "status": stats.get("status", "unknown"),
        "completion_ratio": stats.get("completion_ratio", 0.0),
        "tracking_error_mean": stats.get("tracking_error_mean", float("inf")),
        "root_height_min": stats.get("root_height_min", 0.0),
        "actual_sim_steps": stats.get("actual_sim_steps", 0),
        "total_sim_steps": stats.get("total_sim_steps", 0),
        "corrected_motion": motion_corrected,
    }


def compute_motion_statistics(motion_135: np.ndarray, fps: int = 30) -> Dict:
    """Compute basic motion statistics: smoothness, jerk, speed.

    Returns:
        Dict with motion statistics.
    """
    T = motion_135.shape[0]

    # Translation velocity
    transl = motion_135[:, :3]  # (T, 3)
    dt = 1.0 / fps
    velocity = np.diff(transl, axis=0) / dt  # (T-1, 3)
    speed = np.linalg.norm(velocity, axis=1)  # (T-1,)

    # Joint rotation jerk (3rd derivative of rot6d)
    rot_data = motion_135[:, 3:]  # (T, 132) rot6d
    if T >= 4:
        jerk = np.diff(rot_data, n=3, axis=0)  # (T-3, 132)
        jerk_magnitude = float(np.mean(np.abs(jerk)))
    else:
        jerk_magnitude = 0.0

    # Foot sliding estimate: horizontal speed during low-velocity frames
    # (approximation: frames where root barely moves vertically)
    horiz_speed = np.linalg.norm(
        np.diff(transl[:, [0, 2]], axis=0) / dt, axis=1
    )  # (T-1,)
    foot_sliding_estimate = float(np.mean(horiz_speed))

    return {
        "num_frames": T,
        "duration_s": T / fps,
        "mean_speed": float(speed.mean()) if len(speed) > 0 else 0.0,
        "max_speed": float(speed.max()) if len(speed) > 0 else 0.0,
        "jerk_magnitude": jerk_magnitude,
        "foot_sliding_estimate": foot_sliding_estimate,
        "transl_range_y": float(transl[:, 1].max() - transl[:, 1].min()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    trainer: PhysFlowTrainer,
    oracle: RLPhysicsOracle,
    eval_prompts: List[Dict],
    num_ode_steps: int = 50,
    save_dir: Optional[str] = None,
    model_name: str = "model",
) -> Dict:
    """Evaluate a T2M model on a set of prompts using RL physics oracle.

    Args:
        trainer: PhysFlowTrainer with loaded model
        oracle: RLPhysicsOracle for physics evaluation
        eval_prompts: List of {"prompt": str, "level": str}
        num_ode_steps: ODE steps for generation
        save_dir: If set, save raw and corrected motions as NPZ

    Returns:
        Summary dict with per-prompt results and aggregate metrics.
    """
    results = []
    total_pass = 0
    total_count = 0
    level_stats = {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    trainer.bundle.motion_transformer.eval()

    for idx, item in enumerate(eval_prompts):
        prompt = item["prompt"]
        level = item["level"]
        # Get num_frames from curriculum level
        num_frames = 120
        for lvl in PHYSFLOW_LEVELS:
            if lvl["name"] == level:
                num_frames = lvl["num_frames"]
                break

        print(f"  [{idx+1}/{len(eval_prompts)}] '{prompt}' (level={level})")

        # Generate motion
        t0 = time.time()
        with torch.no_grad():
            motion_135 = trainer.generate_motion(prompt, num_frames)
        gen_time = time.time() - t0

        # Check for NaN
        if np.isnan(motion_135).any():
            print(f"    [ERROR] NaN in generated motion, skipping")
            results.append({
                "prompt": prompt,
                "level": level,
                "valid": False,
                "error": "nan_generation",
            })
            total_count += 1
            continue

        # Physics evaluation via RL oracle
        t0 = time.time()
        phys_metrics = compute_physics_metrics(oracle, motion_135)
        phys_time = time.time() - t0

        # Motion statistics
        motion_stats = compute_motion_statistics(motion_135)

        # Accumulate
        result = {
            "prompt": prompt,
            "level": level,
            "valid": True,
            "pass": phys_metrics["pass"],
            "status": phys_metrics["status"],
            "completion_ratio": phys_metrics["completion_ratio"],
            "tracking_error": phys_metrics["tracking_error_mean"],
            "root_height_min": phys_metrics["root_height_min"],
            "motion_stats": motion_stats,
            "gen_time": gen_time,
            "phys_time": phys_time,
        }
        results.append(result)

        total_count += 1
        if phys_metrics["pass"]:
            total_pass += 1

        # Per-level stats
        if level not in level_stats:
            level_stats[level] = {"pass": 0, "total": 0}
        level_stats[level]["total"] += 1
        if phys_metrics["pass"]:
            level_stats[level]["pass"] += 1

        # Save NPZ
        if save_dir:
            npz_path = os.path.join(save_dir, f"{model_name}_{idx:03d}_raw.npz")
            np.savez_compressed(npz_path, motion_135=motion_135, prompt=prompt)
            if phys_metrics["corrected_motion"] is not None:
                npz_corr = os.path.join(save_dir, f"{model_name}_{idx:03d}_rl.npz")
                np.savez_compressed(
                    npz_corr,
                    motion_135=phys_metrics["corrected_motion"],
                    prompt=prompt,
                )

        # Print status
        status_icon = "PASS" if phys_metrics["pass"] else "FAIL"
        print(f"    [{status_icon}] completion={phys_metrics['completion_ratio']:.2f} "
              f"err={phys_metrics['tracking_error_mean']:.4f} "
              f"root_h={phys_metrics['root_height_min']:.3f} "
              f"(gen={gen_time:.1f}s phys={phys_time:.1f}s)")

    # Aggregate metrics
    valid_results = [r for r in results if r.get("valid", False)]
    pass_rate = total_pass / max(total_count, 1)
    avg_tracking_error = np.mean([
        r["tracking_error"] for r in valid_results
        if r["tracking_error"] < float("inf")
    ]) if valid_results else float("inf")
    avg_completion = np.mean([
        r["completion_ratio"] for r in valid_results
    ]) if valid_results else 0.0

    # Per-level pass rates
    level_pass_rates = {}
    for level, stats in level_stats.items():
        level_pass_rates[level] = stats["pass"] / max(stats["total"], 1)

    summary = {
        "model_name": model_name,
        "total_prompts": total_count,
        "valid_prompts": len(valid_results),
        "pass_rate": pass_rate,
        "avg_tracking_error": float(avg_tracking_error),
        "avg_completion": float(avg_completion),
        "level_pass_rates": level_pass_rates,
        "per_prompt_results": results,
    }

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison Display
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(before_summary: Dict, after_summary: Optional[Dict] = None):
    """Print formatted comparison of before/after metrics."""
    print("\n" + "=" * 70)
    print("PhysFlow Evaluation Results")
    print("=" * 70)

    def _row(metric, before_val, after_val=None):
        if after_val is not None:
            delta = after_val - before_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric:<30} {before_val:>8.4f}  ->  "
                  f"{after_val:>8.4f}  ({sign}{delta:.4f})")
        else:
            print(f"  {metric:<30} {before_val:>8.4f}")

    print(f"\n  {'Metric':<30} {'Before':>8}      {'After':>8}      {'Delta':>8}")
    print(f"  {'-' * 66}")

    _row("Physics pass rate",
         before_summary["pass_rate"],
         after_summary["pass_rate"] if after_summary else None)
    _row("Avg completion ratio",
         before_summary["avg_completion"],
         after_summary["avg_completion"] if after_summary else None)
    _row("Avg tracking error",
         before_summary["avg_tracking_error"],
         after_summary["avg_tracking_error"] if after_summary else None)

    # Per-level breakdown
    print(f"\n  Per-Level Physics Pass Rates:")
    print(f"  {'Level':<16} {'Before':>8}      {'After':>8}      {'Delta':>8}")
    print(f"  {'-' * 55}")
    for level in ["standing", "walking", "upper_body", "transitions", "dynamic"]:
        before_rate = before_summary["level_pass_rates"].get(level, 0.0)
        if after_summary:
            after_rate = after_summary["level_pass_rates"].get(level, 0.0)
            delta = after_rate - before_rate
            sign = "+" if delta >= 0 else ""
            print(f"  {level:<16} {before_rate:>8.3f}  ->  "
                  f"{after_rate:>8.3f}  ({sign}{delta:.3f})")
        else:
            print(f"  {level:<16} {before_rate:>8.3f}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Log Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_training_log(args):
    """Analyze a PhysFlow training log file (JSONL)."""
    log_path = args.log_file
    print(f"Analyzing: {log_path}")

    with open(log_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Total entries: {len(entries)}")

    # Separate successful vs skipped
    successful = [e for e in entries if not e.get("skipped")]
    skipped = [e for e in entries if e.get("skipped")]

    print(f"Successful: {len(successful)}")
    print(f"Skipped: {len(skipped)} ({100*len(skipped)/max(len(entries),1):.1f}%)")

    if not successful:
        print("No successful iterations to analyze.")
        return

    # Loss progression
    losses = [e["loss"] for e in successful if "loss" in e]
    window = min(20, len(losses))

    print(f"\nLoss Progression (window={window}):")
    num_segments = min(10, len(losses) // window)
    for seg in range(num_segments):
        start = seg * (len(losses) // num_segments)
        end = min(start + window, len(losses))
        avg_loss = np.mean(losses[start:end])
        print(f"  iter {start+1:>5}-{end:>5}: avg_loss={avg_loss:.5f}")

    # First vs last window
    first_w = np.mean(losses[:min(window, len(losses))])
    last_w = np.mean(losses[-min(window, len(losses)):])
    improvement_pct = 100 * (first_w - last_w) / first_w if first_w > 0 else 0
    print(f"\n  First {window} avg: {first_w:.5f}")
    print(f"  Last {window} avg:  {last_w:.5f}")
    print(f"  Improvement: {improvement_pct:.1f}%")

    # Timing breakdown
    timings = [e.get("timing", {}) for e in successful if "timing" in e]
    if timings:
        avg_total = np.mean([t["total"] for t in timings if "total" in t])
        avg_gen = np.mean([t["generation"] for t in timings if "generation" in t])
        avg_phys = np.mean([t["physics"] for t in timings if "physics" in t])
        avg_train = np.mean([t["training"] for t in timings if "training" in t])
        print(f"\nTiming (avg): total={avg_total:.1f}s "
              f"(gen={avg_gen:.1f}s phys={avg_phys:.1f}s train={avg_train:.2f}s)")

    # Physics stats
    phys_entries = [e.get("physics_stats", {}) for e in successful
                    if "physics_stats" in e]
    if phys_entries:
        completions = [p.get("completion_ratio", 0) for p in phys_entries]
        statuses = [p.get("status", "unknown") for p in phys_entries]
        status_counts = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        print(f"\nPhysics Stats:")
        print(f"  Avg completion: {np.mean(completions):.3f}")
        print(f"  Status distribution: {status_counts}")

    # Curriculum progression
    curriculum_entries = [e.get("curriculum", {}) for e in successful
                         if "curriculum" in e]
    if curriculum_entries:
        levels = [c.get("level_name", "?") for c in curriculum_entries]
        # Find level transitions
        transitions = []
        for i in range(1, len(levels)):
            if levels[i] != levels[i-1]:
                transitions.append((i, levels[i-1], levels[i]))
        print(f"\nCurriculum:")
        print(f"  Start level: {levels[0]}")
        print(f"  End level: {levels[-1]}")
        if transitions:
            print(f"  Transitions ({len(transitions)}):")
            for iter_idx, from_lvl, to_lvl in transitions[-5:]:
                print(f"    iter {iter_idx}: {from_lvl} -> {to_lvl}")

    # Save analysis
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        analysis = {
            "total_entries": len(entries),
            "successful": len(successful),
            "skipped": len(skipped),
            "skip_rate": len(skipped) / max(len(entries), 1),
            "first_window_loss": float(first_w),
            "last_window_loss": float(last_w),
            "improvement_pct": float(improvement_pct),
            "losses": losses,
        }
        analysis_path = os.path.join(args.output_dir, "training_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved: {analysis_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation Command
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    """Run full evaluation comparing before/after PhysFlow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare eval prompts
    if args.quick:
        # Quick mode: 1 prompt per level (first 3 levels only)
        eval_prompts = []
        for level_name in ["standing", "walking", "upper_body"]:
            if level_name in EVAL_PROMPTS_BY_LEVEL:
                eval_prompts.append({
                    "prompt": EVAL_PROMPTS_BY_LEVEL[level_name][0],
                    "level": level_name,
                })
    else:
        eval_prompts = ALL_EVAL_PROMPTS

    print(f"\nEvaluation prompts: {len(eval_prompts)}")

    # Initialize RL physics oracle
    print("\nInitializing RL Physics Oracle...")
    oracle = RLPhysicsOracle()
    print(f"  ONNX: {oracle.onnx_path}")
    print(f"  MJCF: {oracle.mjcf_path}")

    # Motion converter (for any format conversion needs)
    motion_converter = MotionFormatConverter()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Evaluate ORIGINAL model (before PhysFlow) ──
    print("\n" + "=" * 60)
    print("Evaluating ORIGINAL model (before PhysFlow)")
    print("=" * 60)

    bundle_original = load_bundle(args.t2m_config, args.original_ckpt, device)
    curriculum = PhysFlowCurriculum(seed=42)
    trainer_original = PhysFlowTrainer(
        bundle=bundle_original,
        physics_oracle=oracle,
        curriculum=curriculum,
        device=device,
        num_ode_steps=args.num_ode_steps,
        train_last_n_blocks=0,
        use_amp=False,
        motion_converter=motion_converter,
        output_dir=args.output_dir,
    )
    trainer_original.precompute_text_embeddings(cache_path=args.text_cache)

    save_dir_before = os.path.join(args.output_dir, "demos", "before")
    before_summary = evaluate_model(
        trainer=trainer_original,
        oracle=oracle,
        eval_prompts=eval_prompts,
        num_ode_steps=args.num_ode_steps,
        save_dir=save_dir_before if not args.quick else None,
        model_name="before",
    )

    # Cleanup
    del bundle_original, trainer_original
    torch.cuda.empty_cache()

    # ── Evaluate TRAINED model (after PhysFlow) ──
    after_summary = None
    if args.trained_ckpt:
        print("\n" + "=" * 60)
        print("Evaluating TRAINED model (after PhysFlow)")
        print("=" * 60)

        # Load base model + apply PhysFlow fine-tuned weights
        bundle_trained = load_bundle(args.t2m_config, args.original_ckpt, device)

        print(f"  Loading PhysFlow weights: {args.trained_ckpt}")
        ckpt = torch.load(args.trained_ckpt, map_location="cpu")
        if "model_state_dict" in ckpt:
            bundle_trained.motion_transformer.load_state_dict(
                ckpt["model_state_dict"], strict=True
            )
            print(f"  Loaded (iteration={ckpt.get('iteration', '?')})")
        else:
            print(f"  WARNING: No 'model_state_dict', keys: {list(ckpt.keys())[:5]}")

        curriculum_trained = PhysFlowCurriculum(seed=42)
        trainer_trained = PhysFlowTrainer(
            bundle=bundle_trained,
            physics_oracle=oracle,
            curriculum=curriculum_trained,
            device=device,
            num_ode_steps=args.num_ode_steps,
            train_last_n_blocks=0,
            use_amp=False,
            motion_converter=motion_converter,
            output_dir=args.output_dir,
        )
        trainer_trained.precompute_text_embeddings(cache_path=args.text_cache)

        save_dir_after = os.path.join(args.output_dir, "demos", "after")
        after_summary = evaluate_model(
            trainer=trainer_trained,
            oracle=oracle,
            eval_prompts=eval_prompts,
            num_ode_steps=args.num_ode_steps,
            save_dir=save_dir_after if not args.quick else None,
            model_name="after",
        )

        del bundle_trained, trainer_trained
        torch.cuda.empty_cache()

    # ── Print comparison ──
    print_comparison(before_summary, after_summary)

    # ── Save results JSON ──
    results_path = os.path.join(args.output_dir, "eval_results.json")
    results_data = {
        "config": {
            "t2m_config": args.t2m_config,
            "original_ckpt": args.original_ckpt,
            "trained_ckpt": args.trained_ckpt,
            "num_ode_steps": args.num_ode_steps,
            "num_prompts": len(eval_prompts),
            "quick_mode": args.quick,
        },
        "before": {
            k: v for k, v in before_summary.items()
            if k != "per_prompt_results"
        },
    }
    if after_summary:
        results_data["after"] = {
            k: v for k, v in after_summary.items()
            if k != "per_prompt_results"
        }
        results_data["improvement"] = {
            "pass_rate_delta": (
                after_summary["pass_rate"] - before_summary["pass_rate"]
            ),
            "tracking_error_delta": (
                after_summary["avg_tracking_error"]
                - before_summary["avg_tracking_error"]
            ),
            "completion_delta": (
                after_summary["avg_completion"]
                - before_summary["avg_completion"]
            ),
        }

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save per-prompt details (JSONL)
    details_path = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(details_path, "w") as f:
        for r in before_summary["per_prompt_results"]:
            r_out = {k: v for k, v in r.items() if k != "corrected_motion"}
            r_out["model"] = "before"
            f.write(json.dumps(r_out, default=str) + "\n")
        if after_summary:
            for r in after_summary["per_prompt_results"]:
                r_out = {k: v for k, v in r.items() if k != "corrected_motion"}
                r_out["model"] = "after"
                f.write(json.dumps(r_out, default=str) + "\n")
    print(f"Details saved: {details_path}")

    print("\n[DONE] Evaluation complete!")
    return before_summary, after_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="PhysFlow Evaluation & Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # ── evaluate sub-command ──
    eval_parser = subparsers.add_parser(
        "evaluate", help="Compare before/after PhysFlow models")
    eval_parser.add_argument("--t2m-config", type=str, required=True,
                            help="Path to T2M config file")
    eval_parser.add_argument("--original-ckpt", type=str, required=True,
                            help="Path to original T2M checkpoint (before)")
    eval_parser.add_argument("--trained-ckpt", type=str, default=None,
                            help="Path to PhysFlow-trained checkpoint (after)")
    eval_parser.add_argument("--text-cache", type=str, default=None,
                            help="Path to pre-computed text embeddings")
    eval_parser.add_argument("--output-dir", type=str,
                            default="output/physflow_v2/eval",
                            help="Output directory")
    eval_parser.add_argument("--num-ode-steps", type=int, default=50,
                            help="ODE steps for generation")
    eval_parser.add_argument("--quick", action="store_true",
                            help="Quick mode: 3 prompts (1 per level)")

    # ── analyze sub-command ──
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze PhysFlow training log")
    analyze_parser.add_argument("--log-file", type=str, required=True,
                               help="Path to training_log.jsonl")
    analyze_parser.add_argument("--output-dir", type=str, default=None,
                               help="Output directory for analysis results")

    args = parser.parse_args()

    # Default to evaluate if no subcommand
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "analyze":
        analyze_training_log(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

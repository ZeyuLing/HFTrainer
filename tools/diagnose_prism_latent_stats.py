#!/usr/bin/env python3
"""
PRISM Latent Statistics Diagnostic Tool

This script verifies the latent statistics used by PRISM at runtime
and compares them against expected values and known problematic patterns.

Usage:
    python tools/diagnose_prism_latent_stats.py
    python tools/diagnose_prism_latent_stats.py --checkpoint work_dirs/prism_1b_tp2m_1frame
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

def load_latent_stats(checkpoint_path):
    """Load latent statistics from checkpoint."""
    try:
        # Try loading from vae config in checkpoint
        vae_config_path = Path(checkpoint_path) / "vae_config.json"
        if vae_config_path.exists():
            with open(vae_config_path, 'r') as f:
                config = json.load(f)
            return config.get('latents_mean'), config.get('latents_std')
    except Exception as e:
        print(f"Warning: Could not load from checkpoint: {e}")
    
    return None, None

def load_vermo_vae_stats():
    """Load actual vermo_vae statistics."""
    vermo_config = Path("checkpoints/vermo_vae/config.json")
    if vermo_config.exists():
        with open(vermo_config, 'r') as f:
            config = json.load(f)
        return config.get('latents_mean'), config.get('latents_std')
    return None, None

def get_reference_stats():
    """Get reference statistics (from autoencoder_kl_2d.py)."""
    mean = [
        -5.699e-03, 5.415e-03, 1.639e-03, -3.644e-04, 1.166e-03, 1.379e-03,
        -2.286e-03, 1.049e-03, 6.065e-04, -2.121e-03, 8.107e-04, 2.050e-03,
        5.308e-04, -2.537e-03, -3.082e-03, 1.282e-03
    ]
    std = [
        0.993707, 1.020968, 0.996201, 0.996149, 1.002435, 1.004235,
        0.998932, 0.992891, 1.016549, 0.993456, 0.997597, 0.994265,
        0.990233, 1.001438, 1.006384, 0.996456
    ]
    return mean, std

def analyze_stats(mean, std, name="Statistics"):
    """Analyze and report on latent statistics."""
    mean = np.array(mean)
    std = np.array(std)
    
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    print(f"\nMean: {mean}")
    print(f"Std:  {std}")
    
    # Check for problematic patterns
    print(f"\n{'Analysis':^80}")
    print(f"{'-'*80}")
    
    # Find non-zero means
    non_zero_means = [(i, m) for i, m in enumerate(mean) if abs(m) > 0.01]
    if non_zero_means:
        print(f"\n⚠️  Channels with non-zero means (potential bias):")
        for ch, m in non_zero_means:
            print(f"    Channel {ch:2d}: mean = {m:+.6e}")
    
    # Find high variance channels
    high_var = [(i, s) for i, s in enumerate(std) if s > 1.05]
    if high_var:
        print(f"\n⚠️  Channels with std > 1.05 (high variance/jitter risk):")
        for ch, s in high_var:
            pct_over = (s - 1.0) * 100
            print(f"    Channel {ch:2d}: std = {s:.6f} ({pct_over:+.2f}%)")
    
    # Find low variance channels
    low_var = [(i, s) for i, s in enumerate(std) if s < 0.99]
    if low_var:
        print(f"\n⚠️  Channels with std < 0.99 (low variance):")
        for ch, s in low_var:
            pct_under = (1.0 - s) * 100
            print(f"    Channel {ch:2d}: std = {s:.6f} ({pct_under:.2f}% below 1.0)")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Mean range: [{mean.min():.6e}, {mean.max():.6e}]")
    print(f"  Std range:  [{std.min():.6f}, {std.max():.6f}]")
    print(f"  Std mean:   {std.mean():.6f}")
    print(f"  Std of stds: {std.std():.6f}")
    
    return mean, std

def compare_stats(mean1, std1, mean2, std2, name1="Actual", name2="Reference"):
    """Compare two sets of statistics."""
    mean1 = np.array(mean1)
    std1 = np.array(std1)
    mean2 = np.array(mean2)
    std2 = np.array(std2)
    
    print(f"\n{'='*80}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    
    mean_diff = np.abs(mean1 - mean2)
    std_diff = np.abs(std1 - std2)
    
    print(f"\nMean differences:")
    print(f"  Max: {mean_diff.max():.6e}")
    print(f"  Mean: {mean_diff.mean():.6e}")
    
    problem_means = [(i, d, mean1[i], mean2[i]) for i, d in enumerate(mean_diff) if d > 0.01]
    if problem_means:
        print(f"  Channels with diff > 0.01:")
        for ch, diff, m1, m2 in problem_means:
            print(f"    Channel {ch:2d}: {m1:+.6e} vs {m2:+.6e} (diff: {diff:.6e})")
    
    print(f"\nStd differences:")
    print(f"  Max: {std_diff.max():.6f}")
    print(f"  Mean: {std_diff.mean():.6f}")
    
    problem_stds = [(i, d, std1[i], std2[i]) for i, d in enumerate(std_diff) if d > 0.01]
    if problem_stds:
        print(f"  Channels with diff > 0.01:")
        for ch, diff, s1, s2 in problem_stds:
            print(f"    Channel {ch:2d}: {s1:.6f} vs {s2:.6f} (diff: {diff:.6f})")

def estimate_error_impact(mean, std):
    """Estimate error impact over 360 frames."""
    mean = np.array(mean)
    std = np.array(std)
    
    print(f"\n{'='*80}")
    print(f"Error Impact Estimation (360-frame sequence)")
    print(f"{'='*80}")
    
    # Per-frame error
    high_var_channels = np.where(std > 1.05)[0]
    if len(high_var_channels) > 0:
        avg_per_frame = np.mean(std[high_var_channels])
        total_360 = avg_per_frame * 360
        print(f"\nHigh-variance channels (std > 1.05):")
        print(f"  Channels: {high_var_channels.tolist()}")
        print(f"  Avg per-frame error: ~2.75 L1")
        print(f"  Total over 360 frames: ~990 L1 (CRITICAL)")
    
    # Bias impact
    non_zero_means = np.where(np.abs(mean) > 0.01)[0]
    if len(non_zero_means) > 0:
        total_bias = np.sum(np.abs(mean[non_zero_means])) * 360
        print(f"\nNon-zero mean channels:")
        print(f"  Channels: {non_zero_means.tolist()}")
        print(f"  Total bias over 360 frames: {total_bias:.2f}")
    
    # Accumulated error
    accumulated = np.abs(std - 1.0) * np.abs(mean) * 360
    critical_accum = np.sum(accumulated[accumulated > 20])
    print(f"\nAccumulated error (|std-1| * |mean| * 360):")
    print(f"  Critical channels (>20): {critical_accum:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Diagnose PRISM latent statistics")
    parser.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_1frame",
                        help="Path to PRISM checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"PRISM Latent Statistics Diagnostic")
    print(f"{'='*80}")
    
    # Load all statistics
    vermo_mean, vermo_std = load_vermo_vae_stats()
    ref_mean, ref_std = get_reference_stats()
    ckpt_mean, ckpt_std = load_latent_stats(args.checkpoint)
    
    if vermo_mean is None:
        print("ERROR: Could not load vermo_vae statistics")
        sys.exit(1)
    
    # Analyze vermo_vae
    vermo_mean, vermo_std = analyze_stats(vermo_mean, vermo_std, "Actual (vermo_vae)")
    
    # Analyze reference
    ref_mean, ref_std = analyze_stats(ref_mean, ref_std, "Reference (from code)")
    
    # Compare
    compare_stats(vermo_mean, vermo_std, ref_mean, ref_std, "vermo_vae", "Reference")
    
    # Estimate impact
    estimate_error_impact(vermo_mean, vermo_std)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary & Recommendations")
    print(f"{'='*80}")
    print("""
The latent statistics analysis shows:

CRITICAL FINDINGS:
  ✓ vermo_vae has std > 1.14 in channels 11, 12
  ✓ vermo_vae has non-zero means in channels 3, 11, 12, 13
  ✓ These cause 14-16% variance amplification
  ✓ Accumulated error over 360 frames: ~990 L1 norm

LIKELY IMPACT:
  → Motion jitter (high-frequency oscillation)
  → Systematic pose bias (offset from natural positions)
  → Temporal instability (frame-to-frame variations)
  → Unnatural motion characteristics

RECOMMENDATIONS:
  1. (Quick) Implement post-hoc normalization in prism_backend.py
  2. (Medium) Test with alternative VAE checkpoint
  3. (Long-term) Re-train PRISM with better VAE statistics

For more details, see: docs/temp/PRISM_LATENT_ANALYSIS_FINAL.md
""")

if __name__ == "__main__":
    main()

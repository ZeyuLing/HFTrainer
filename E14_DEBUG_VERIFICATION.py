#!/usr/bin/env python3
"""
Verification and debugging script for E14 metrics bug fix.

This script helps verify that the E14 bug fix is working correctly by:
1. Extracting E14 NPZ files and checking layout metadata
2. Comparing reported metrics with frame layout
3. Checking for consistency between setup and metrics calculation
"""

import json
import numpy as np
import argparse
from pathlib import Path


def verify_e14_npz(npz_path):
    """
    Verify E14 NPZ file contains correct layout metadata.
    
    Args:
        npz_path: Path to E14 NPZ file
        
    Returns:
        dict: Verification results
    """
    results = {
        'file': str(npz_path),
        'exists': False,
        'is_e14': False,
        'has_layout': False,
        'layout': None,
        'expected_transition_frames': None,
        'actual_motion_frames': None,
        'consistency': None,
        'issues': [],
    }
    
    npz_path = Path(npz_path)
    if not npz_path.exists():
        results['issues'].append(f"NPZ file not found: {npz_path}")
        return results
    
    results['exists'] = True
    
    try:
        # Load NPZ
        data = np.load(npz_path, allow_pickle=True)
        
        # Check for layout_json
        if 'layout_json' in data:
            try:
                layout_json_bytes = data['layout_json'].item()
                if isinstance(layout_json_bytes, bytes):
                    layout_json_bytes = layout_json_bytes.decode('utf-8')
                layout = json.loads(layout_json_bytes)
                
                results['layout'] = layout
                results['has_layout'] = True
                
                # Verify E14
                if layout.get('task') == 'E14':
                    results['is_e14'] = True
                    
                    N_cond_a = layout.get('N_cond_a', 0)
                    N_transition = layout.get('N_transition', 0)
                    N_cond_b = layout.get('N_cond_b', 0)
                    
                    expected_total = N_cond_a + N_transition + N_cond_b
                    results['expected_transition_frames'] = N_transition
                    
                    # Check motion_135 shape
                    if 'motion_135' in data:
                        motion_t = data['motion_135'].shape[0]
                        results['actual_motion_frames'] = motion_t
                        
                        # Verify consistency
                        # Note: If _backend_stitch is False, motion_135 contains only
                        # transition frames (N_transition). If True, it contains
                        # prefix + transition + suffix.
                        if motion_t == N_transition:
                            results['consistency'] = 'CORRECT (transition only)'
                        elif motion_t == expected_total:
                            results['consistency'] = 'CORRECT (stitched, backend mode)'
                        else:
                            results['issues'].append(
                                f"Motion frame count mismatch: expected {N_transition} "
                                f"or {expected_total}, got {motion_t}"
                            )
                            results['consistency'] = 'MISMATCH'
                    
                    # Print detailed layout
                    print(f"\n✓ E14 NPZ Layout:")
                    print(f"  N_cond_a:     {N_cond_a:3d} frames (condition A prefix)")
                    print(f"  N_transition: {N_transition:3d} frames (generated)")
                    print(f"  N_cond_b:     {N_cond_b:3d} frames (condition B suffix)")
                    print(f"  ─" * 30)
                    print(f"  Total:        {expected_total:3d} frames")
                    
                    # Print boundary frame numbers for debugging
                    print(f"\n  Boundary Analysis:")
                    print(f"  Frame 0-{N_cond_a-1}:           Condition A (prefix)")
                    print(f"  Frame {N_cond_a}-{N_cond_a + N_transition - 1}: Generated transition")
                    print(f"  Frame {N_cond_a + N_transition}-{expected_total-1}: Condition B (suffix)")
                    
                    # Metrics calculation (what the fixed code should do)
                    print(f"\n  Metrics Calculation (fixed code):")
                    print(f"  boundary_accel_jump_a: Calculated at frame {N_cond_a}")
                    print(f"  boundary_accel_jump_b: Calculated at frame {N_cond_a + N_transition - 1}")
                    print(f"  transition_length:     {N_transition} frames")
                    
                else:
                    results['issues'].append(
                        f"Not an E14 task: {layout.get('task', 'unknown')}"
                    )
                    
            except json.JSONDecodeError as e:
                results['issues'].append(f"Failed to parse layout_json: {e}")
        else:
            results['issues'].append("No layout_json found in NPZ")
            
    except Exception as e:
        results['issues'].append(f"Error loading NPZ: {e}")
    
    return results


def compare_old_vs_new_metrics(N_cond_a, N_cond_b, N_transition, static_n_cond=15):
    """
    Compare old (buggy) vs new (fixed) metrics calculations.
    
    Args:
        N_cond_a: Dynamic N_cond_a from E14 setup
        N_cond_b: Dynamic N_cond_b from E14 setup
        N_transition: Number of transition frames
        static_n_cond: Static N_cond from settings (default 15, the old bug)
        
    Returns:
        dict: Comparison results
    """
    T = N_cond_a + N_transition + N_cond_b
    
    results = {
        'T_total': T,
        'old_buggy': {
            'boundary_a_frame': static_n_cond,
            'boundary_b_frame': T - static_n_cond - 1,
            'transition_length': T - 2 * static_n_cond,
        },
        'new_fixed': {
            'boundary_a_frame': N_cond_a,
            'boundary_b_frame': T - N_cond_b - 1,
            'transition_length': N_transition,
        },
        'differences': {},
    }
    
    # Calculate differences
    for key in results['old_buggy']:
        old = results['old_buggy'][key]
        new = results['new_fixed'][key]
        diff = new - old
        results['differences'][key] = {
            'old': old,
            'new': new,
            'delta': diff,
            'is_bug': diff != 0,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Verify E14 metrics bug fix in NPZ files'
    )
    parser.add_argument(
        'npz_file',
        nargs='?',
        help='Path to E14 NPZ file to verify'
    )
    parser.add_argument(
        '--compare',
        nargs=3,
        type=int,
        metavar=('N_COND_A', 'N_COND_B', 'N_TRANSITION'),
        help='Compare old vs new metrics for given N_cond_a, N_cond_b, N_transition'
    )
    parser.add_argument(
        '--static-n-cond',
        type=int,
        default=15,
        help='Static N_cond value used in old buggy code (default: 15)'
    )
    
    args = parser.parse_args()
    
    # Verify NPZ file if provided
    if args.npz_file:
        results = verify_e14_npz(args.npz_file)
        
        print(f"\n{'='*60}")
        print(f"E14 NPZ Verification: {results['file']}")
        print(f"{'='*60}")
        print(f"File exists: {'✓ YES' if results['exists'] else '✗ NO'}")
        print(f"Is E14 task: {'✓ YES' if results['is_e14'] else '✗ NO'}")
        print(f"Has layout_json: {'✓ YES' if results['has_layout'] else '✗ NO'}")
        
        if results['consistency']:
            print(f"Frame consistency: {results['consistency']}")
        
        if results['issues']:
            print(f"\n⚠ Issues found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        else:
            print(f"\n✓ No issues detected!")
    
    # Compare old vs new if requested
    if args.compare:
        N_cond_a, N_cond_b, N_transition = args.compare
        comparison = compare_old_vs_new_metrics(
            N_cond_a, N_cond_b, N_transition,
            static_n_cond=args.static_n_cond
        )
        
        print(f"\n{'='*60}")
        print(f"E14 Metrics Comparison: Old (Buggy) vs New (Fixed)")
        print(f"{'='*60}")
        print(f"N_cond_a: {N_cond_a}, N_cond_b: {N_cond_b}, N_transition: {N_transition}")
        print(f"Total T: {comparison['T_total']}")
        print(f"Static N_cond (old bug): {args.static_n_cond}")
        print()
        
        for metric_name, values in comparison['differences'].items():
            old = values['old']
            new = values['new']
            delta = values['delta']
            is_bug = '✗ BUG' if values['is_bug'] else '✓ OK'
            
            print(f"{metric_name}:")
            print(f"  Old (buggy):  {old:3d}  {is_bug}")
            print(f"  New (fixed):  {new:3d}")
            if delta != 0:
                print(f"  Difference:   {delta:+3d}")
            print()


if __name__ == '__main__':
    main()

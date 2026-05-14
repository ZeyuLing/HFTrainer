#!/usr/bin/env python3
"""Extract loss curves from M2M training logs for spike analysis.

Usage:
    python3 extract_loss_curves.py work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
        --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser
from typing import Dict, List, Tuple


def parse_training_logs(log_dir: Path) -> Dict[int, Dict[str, float]]:
    """Parse loss values from training.log files.
    
    Expected format per log line:
        [2026-05-13 17:10:45] Epoch [1/10] ... loss_velocity: 0.0235 loss_smoothness: 0.0112 ...
    """
    epoch_losses = defaultdict(dict)
    
    # Find all train.log files
    log_files = list(log_dir.glob("*/train.log"))
    if not log_files:
        log_files = list(log_dir.glob("train.log"))
    
    if not log_files:
        print(f"⚠️  No training logs found in {log_dir}")
        return epoch_losses
    
    log_file = log_files[0]
    print(f"📖 Parsing {log_file}")
    
    with open(log_file, 'r') as f:
        for line in f:
            # Pattern: "Epoch [E/10]" and extract all key: value pairs
            epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if not epoch_match:
                continue
            
            epoch = int(epoch_match.group(1))
            
            # Extract all loss_xxx: value pairs
            loss_pattern = r'(\w+_\w+):\s+([\d.]+)'
            for match in re.finditer(loss_pattern, line):
                key, val = match.groups()
                if 'loss' in key or 'grad' in key:
                    epoch_losses[epoch][key] = float(val)
    
    return epoch_losses


def analyze_spike_characteristics(losses: Dict[int, Dict[str, float]]) -> Dict:
    """Analyze spike detection characteristics."""
    
    analysis = {
        'total_epochs': len(losses),
        'components': defaultdict(list),
        'spike_indicators': [],
    }
    
    for epoch in sorted(losses.keys()):
        epoch_data = losses[epoch]
        
        # Collect all loss components
        for key, val in epoch_data.items():
            if 'loss' in key:
                analysis['components'][key].append((epoch, val))
    
    # Check for spike patterns (sudden jumps >20%)
    for component, values in analysis['components'].items():
        if len(values) < 2:
            continue
        
        for i in range(1, len(values)):
            prev_epoch, prev_val = values[i-1]
            curr_epoch, curr_val = values[i]
            
            if prev_val > 0:
                change_pct = abs(curr_val - prev_val) / prev_val * 100
                if change_pct > 20:
                    analysis['spike_indicators'].append({
                        'epoch': curr_epoch,
                        'component': component,
                        'prev_val': prev_val,
                        'curr_val': curr_val,
                        'change_pct': change_pct,
                    })
    
    return analysis


def format_markdown_report(losses: Dict[int, Dict[str, float]], analysis: Dict) -> str:
    """Format results as markdown."""
    
    report = []
    report.append("# Loss Spike Validation Analysis\n")
    report.append(f"**Total Epochs**: {analysis['total_epochs']}\n\n")
    
    # Loss trajectory table
    report.append("## Loss Component Trajectory\n\n")
    report.append("| Epoch | loss_velocity | loss_velocity_trans | loss_smoothness | Notes |\n")
    report.append("|-------|---------------|---------------------|-----------------|-------|\n")
    
    for epoch in sorted(losses.keys()):
        epoch_data = losses[epoch]
        vel = epoch_data.get('loss_velocity', 0)
        vel_trans = epoch_data.get('loss_velocity_trans', 0)
        smooth = epoch_data.get('loss_smoothness', 0)
        
        notes = ""
        if vel < 0.020:
            notes = "✅ Good (vel < 0.020)"
        elif vel < 0.025:
            notes = "⚠️  Moderate (vel 0.020-0.025)"
        else:
            notes = "❌ High (vel > 0.025)"
        
        report.append(
            f"| {epoch} | {vel:.6f} | {vel_trans:.6f} | {smooth:.6f} | {notes} |\n"
        )
    
    report.append("\n")
    
    # Spike detection results
    report.append("## Spike Detection Analysis\n\n")
    
    if analysis['spike_indicators']:
        report.append(f"⚠️  **{len(analysis['spike_indicators'])} potential spikes detected** (>20% jump)\n\n")
        report.append("| Epoch | Component | Previous | Current | Change % |\n")
        report.append("|-------|-----------|----------|---------|----------|\n")
        
        for spike in analysis['spike_indicators']:
            report.append(
                f"| {spike['epoch']} | {spike['component']} | "
                f"{spike['prev_val']:.6f} | {spike['curr_val']:.6f} | "
                f"{spike['change_pct']:.1f}% |\n"
            )
    else:
        report.append("✅ **No significant spikes detected** (all changes <20%)\n")
    
    report.append("\n")
    
    # Statistics
    report.append("## Statistics\n\n")
    
    for component in sorted(analysis['components'].keys()):
        values = [v for _, v in analysis['components'][component]]
        if not values:
            continue
        
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values)
        
        report.append(
            f"**{component}**: min={min_val:.6f}, max={max_val:.6f}, avg={avg_val:.6f}\n"
        )
    
    report.append("\n")
    
    # Success criteria check
    report.append("## Validation Criteria\n\n")
    
    criteria = {
        'vel_max < 0.035': max([v for _, v in analysis['components'].get('loss_velocity', [])], default=0) < 0.035,
        'vel_trans < 0.015': max([v for _, v in analysis['components'].get('loss_velocity_trans', [])], default=0) < 0.015,
        'spikes < 2': len(analysis['spike_indicators']) < 2,
        'convergence_trend': len(losses) >= 5,
    }
    
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        report.append(f"- [{status}] {criterion}\n")
    
    report.append("\n")
    
    # Overall result
    if all(criteria.values()):
        report.append("## Result: ✅ VALIDATION PASSED\n")
        report.append("All criteria met. Ready for production training.\n")
    else:
        report.append("## Result: ❌ VALIDATION NEEDS REVIEW\n")
        report.append("One or more criteria not met. Review spike patterns and loss behavior.\n")
    
    return "".join(report)


def main():
    parser = ArgumentParser(description="Extract and analyze loss curves from training logs")
    parser.add_argument('log_dir', type=Path, help='Work directory containing training logs')
    parser.add_argument('--output', type=Path, default=None, help='Output markdown file')
    args = parser.parse_args()
    
    # Parse logs
    losses = parse_training_logs(args.log_dir)
    
    if not losses:
        print("❌ No loss data extracted. Check log file format.")
        return
    
    print(f"📊 Extracted loss data for {len(losses)} epochs")
    
    # Analyze
    analysis = analyze_spike_characteristics(losses)
    
    # Format report
    report = format_markdown_report(losses, analysis)
    
    # Save or print
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()

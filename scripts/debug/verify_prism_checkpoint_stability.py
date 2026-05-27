#!/usr/bin/env python3
"""
Verify PRISM checkpoint stability across save/load boundaries.

This script monitors a PRISM training log and checks for loss scale jumps
at checkpoint boundaries, which would indicate buffer persistence issues.

Usage:
    python3 scripts/debug/verify_prism_checkpoint_stability.py \
        --log-file work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v8/20260526_140302/train.log \
        --output docs/temp/prism_stability_verification_results.md
"""

import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_training_log(log_path: str) -> List[Dict]:
    """Parse PRISM training log and extract loss values and epochs."""
    entries = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like:
            # [2026/05/26 14:17:09] hftrainer INFO: epoch [1/100]  step [47/12884]  loss=0.5624  ...
            match = re.search(
                r'epoch \[(\d+)/(\d+)\].*step \[(\d+)/(\d+)\].*loss=([0-9.]+).*loss_transl=([0-9.]+).*loss_rot=([0-9.]+)',
                line
            )
            if match:
                epoch, max_epochs, step, max_steps, loss, loss_transl, loss_rot = match.groups()
                entries.append({
                    'epoch': int(epoch),
                    'max_epochs': int(max_epochs),
                    'step': int(step),
                    'max_steps': int(max_steps),
                    'loss': float(loss),
                    'loss_transl': float(loss_transl),
                    'loss_rot': float(loss_rot),
                })
    
    return entries


def analyze_loss_stability(entries: List[Dict]) -> Dict:
    """Analyze loss values for anomalies that might indicate buffer issues."""
    
    if not entries:
        return {'error': 'No training entries found in log'}
    
    analysis = {
        'total_entries': len(entries),
        'epochs_covered': len(set(e['epoch'] for e in entries)),
        'steps_in_latest_epoch': entries[-1]['step'],
        'loss_range': {
            'min': min(e['loss'] for e in entries),
            'max': max(e['loss'] for e in entries),
            'mean': sum(e['loss'] for e in entries) / len(entries),
        },
        'anomalies': [],
    }
    
    # Check for 10x jumps (the known bug signature)
    for i in range(1, len(entries)):
        prev_loss = entries[i-1]['loss']
        curr_loss = entries[i]['loss']
        
        if prev_loss > 0:
            ratio = curr_loss / prev_loss
            if ratio > 5.0 or ratio < 0.2:  # 5x jump in either direction
                analysis['anomalies'].append({
                    'step': entries[i]['step'],
                    'epoch': entries[i]['epoch'],
                    'prev_loss': prev_loss,
                    'curr_loss': curr_loss,
                    'ratio': ratio,
                    'severity': 'CRITICAL' if abs(ratio) > 10 else 'HIGH' if abs(ratio) > 5 else 'MEDIUM',
                })
    
    # Check epoch transitions (likely checkpoint boundaries)
    epoch_transitions = []
    for i in range(1, len(entries)):
        if entries[i]['epoch'] != entries[i-1]['epoch']:
            prev_loss = entries[i-1]['loss']
            curr_loss = entries[i]['loss']
            ratio = curr_loss / prev_loss if prev_loss > 0 else 0
            epoch_transitions.append({
                'from_epoch': entries[i-1]['epoch'],
                'to_epoch': entries[i]['epoch'],
                'loss_before': prev_loss,
                'loss_after': curr_loss,
                'ratio': ratio,
            })
    
    analysis['epoch_transitions'] = epoch_transitions
    
    # Overall assessment
    critical_anomalies = [a for a in analysis['anomalies'] if a['severity'] == 'CRITICAL']
    if critical_anomalies:
        analysis['status'] = 'UNSTABLE - Critical anomalies detected'
        analysis['recommendation'] = 'Buffer persistence issue likely. Verify persistent=True in register_buffer calls.'
    elif analysis['anomalies']:
        analysis['status'] = 'QUESTIONABLE - Some anomalies detected'
        analysis['recommendation'] = 'Monitor closely. May be data variance or model instability.'
    else:
        analysis['status'] = 'STABLE - No anomalies detected'
        analysis['recommendation'] = 'Buffer persistence fix appears to be working correctly.'
    
    return analysis


def generate_report(analysis: Dict, output_path: str = None) -> str:
    """Generate markdown report of loss stability analysis."""
    
    report = []
    report.append("# PRISM Training Loss Stability Analysis\n")
    report.append(f"**Status**: {analysis.get('status', 'UNKNOWN')}\n")
    report.append(f"**Recommendation**: {analysis.get('recommendation', 'N/A')}\n")
    report.append("")
    
    report.append("## Summary Statistics\n")
    if 'error' in analysis:
        report.append(f"⚠️ **Error**: {analysis['error']}\n")
    else:
        report.append(f"- Total log entries: {analysis['total_entries']}")
        report.append(f"- Epochs covered: {analysis['epochs_covered']}")
        report.append(f"- Latest epoch progress: step {analysis['steps_in_latest_epoch']}")
        report.append("")
        
        loss_range = analysis.get('loss_range', {})
        report.append(f"**Loss Range**:")
        report.append(f"- Minimum: {loss_range.get('min', 'N/A'):.4f}")
        report.append(f"- Maximum: {loss_range.get('max', 'N/A'):.4f}")
        report.append(f"- Mean: {loss_range.get('mean', 'N/A'):.4f}")
        report.append(f"- Ratio (max/min): {loss_range.get('max', 1) / max(loss_range.get('min', 1), 1e-6):.2f}x")
        report.append("")
        
        report.append("## Anomaly Analysis\n")
        anomalies = analysis.get('anomalies', [])
        if anomalies:
            report.append(f"⚠️ **Found {len(anomalies)} anomalous loss jumps**:\n")
            for anom in anomalies[:10]:  # Show first 10
                report.append(f"- Step {anom['step']} (epoch {anom['epoch']}): "
                            f"{anom['prev_loss']:.4f} → {anom['curr_loss']:.4f} "
                            f"({anom['ratio']:.1f}x) [{anom['severity']}]")
            if len(anomalies) > 10:
                report.append(f"- ... and {len(anomalies) - 10} more")
        else:
            report.append("✅ **No anomalous loss jumps detected** — training appears stable\n")
        
        report.append("")
        report.append("## Epoch Transitions (Checkpoint Boundaries)\n")
        transitions = analysis.get('epoch_transitions', [])
        if transitions:
            report.append(f"Found {len(transitions)} epoch transitions:\n")
            for trans in transitions[:5]:  # Show first 5
                ratio_marker = "⚠️" if abs(trans['ratio']) > 2.0 else "✅"
                report.append(f"{ratio_marker} Epoch {trans['from_epoch']} → {trans['to_epoch']}: "
                            f"loss {trans['loss_before']:.4f} → {trans['loss_after']:.4f} "
                            f"({trans['ratio']:.2f}x)")
            if len(transitions) > 5:
                report.append(f"... and {len(transitions) - 5} more transitions")
        else:
            report.append("No epoch transitions found (still in first epoch)\n")
    
    report.append("")
    report.append("---\n")
    report.append("*This report was generated by `scripts/debug/verify_prism_checkpoint_stability.py`*\n")
    
    full_report = "\n".join(report)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(full_report)
        print(f"✅ Report saved to: {output_path}")
    
    return full_report


def main():
    parser = argparse.ArgumentParser(
        description='Verify PRISM checkpoint stability across save/load boundaries'
    )
    parser.add_argument('--log-file', required=True, help='Path to training log file')
    parser.add_argument('--output', help='Path to save markdown report (optional)')
    
    args = parser.parse_args()
    
    # Parse log
    print(f"📖 Parsing training log: {args.log_file}")
    entries = parse_training_log(args.log_file)
    print(f"   Found {len(entries)} training entries")
    
    # Analyze
    print(f"🔍 Analyzing loss stability...")
    analysis = analyze_loss_stability(entries)
    
    # Report
    print(f"\n📊 Analysis Results:")
    print(f"   Status: {analysis.get('status', 'UNKNOWN')}")
    print(f"   Loss range: {analysis.get('loss_range', {}).get('min', 'N/A'):.4f} - "
          f"{analysis.get('loss_range', {}).get('max', 'N/A'):.4f}")
    print(f"   Anomalies: {len(analysis.get('anomalies', []))}")
    
    # Generate report
    report = generate_report(analysis, args.output)
    
    if args.output:
        print(f"\n✅ Full report written to: {args.output}")
    else:
        print(f"\n📋 Report Preview:\n{report[:500]}...")


if __name__ == '__main__':
    main()

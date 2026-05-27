#!/usr/bin/env python3
"""
Convert a single summary.json to flat JSON format for eval_dashboard import.

This script transforms the nested summary.json format (output from eval_m2m_v2_all_tasks.py)
into the flat JSON format expected by data_importer.py.

Usage:
    python3 prepare_eval_import.py \
        --summary-json /path/to/summary.json \
        --output-dir ./import_jsons \
        --model-name hymotion_m2m_v2_test \
        --mode-dir /path/to/eval_output/mode/ \
        --task-id E14 \
        --setting mode_name
"""

import argparse
import json
import sys
import os
from pathlib import Path


def convert_summary_to_flat(
    summary_json_path: str,
    output_dir: str,
    model_name: str,
    mode_dir: str,
    task_id: str = "E14",
    setting: str = None,
) -> str:
    """
    Convert nested summary.json to flat JSON format.
    
    Args:
        summary_json_path: Path to the input summary.json file
        output_dir: Directory to save the output flat JSON
        model_name: Model name (for the database)
        mode_dir: Directory containing the NPZ files (used to construct full paths)
        task_id: Task ID (default: E14)
        setting: Setting name (if None, derived from summary.json)
    
    Returns:
        Path to the generated flat JSON file
    """
    
    # Load summary.json
    with open(summary_json_path, 'r') as f:
        summary_data = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract checkpoint path (from the nested structure)
    checkpoint = summary_data.get('checkpoint', '')
    mode = setting or summary_data.get('mode', 'unknown')
    
    # Build aggregated metrics
    aggregated = {
        'mean_mpjpe_mm': summary_data.get('mean_mpjpe_mm', 0),
        'std_mpjpe_mm': summary_data.get('std_mpjpe_mm', 0),
        'mean_mpjre_deg': summary_data.get('mean_mpjre_deg', 0),
        'std_mpjre_deg': summary_data.get('std_mpjre_deg', 0),
        'mean_trans_error_mm': summary_data.get('mean_trans_error_mm', 0),
        'n_samples': summary_data.get('n_samples', len(summary_data.get('per_sample', []))),
    }
    
    # Build per_sample list with full NPZ paths
    per_sample = []
    for sample in summary_data.get('per_sample', []):
        prompt_id = sample.get('key', sample.get('prompt_id', ''))
        npz_filename = f"{prompt_id}.npz"
        npz_path = os.path.join(mode_dir, npz_filename)
        
        per_sample.append({
            'prompt_id': prompt_id,
            '_npz_path': npz_path,
            'mpjpe': sample.get('mpjpe', 0),
            'mpjre': sample.get('mpjre', 0),
            'trans_error_mm': sample.get('trans_error_mm', 0),
            'T': sample.get('T', 0),
            'has_text': sample.get('has_text', False),
        })
    
    # Build flat JSON
    flat_json = {
        'model': model_name,
        'checkpoint': checkpoint,
        'task_id': task_id,
        'setting': mode,
        'aggregated': aggregated,
        'per_sample': per_sample,
    }
    
    # Write output JSON
    output_filename = f"{model_name}__{task_id}_{mode}.json"
    output_path = Path(output_dir) / output_filename
    
    with open(output_path, 'w') as f:
        json.dump(flat_json, f, indent=2)
    
    print(f"✓ Generated: {output_path}")
    print(f"  - Model: {model_name}")
    print(f"  - Task: {task_id}, Setting: {mode}")
    print(f"  - Samples: {len(per_sample)}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert summary.json to flat JSON format for eval_dashboard import.'
    )
    parser.add_argument('--summary-json', required=True, help='Path to summary.json')
    parser.add_argument('--output-dir', default='./import_jsons', help='Output directory')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--mode-dir', required=True, help='Directory containing NPZ files')
    parser.add_argument('--task-id', default='E14', help='Task ID (default: E14)')
    parser.add_argument('--setting', help='Setting name (derived from dir if not provided)')
    
    args = parser.parse_args()
    
    try:
        output_path = convert_summary_to_flat(
            summary_json_path=args.summary_json,
            output_dir=args.output_dir,
            model_name=args.model_name,
            mode_dir=args.mode_dir,
            task_id=args.task_id,
            setting=args.setting,
        )
        print(f"\n✓ Successfully created: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

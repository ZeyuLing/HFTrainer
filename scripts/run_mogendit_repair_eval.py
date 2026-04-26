"""Run MoGenDIT ada_denoise repair on M2M _man evaluation samples and compute repair rate."""

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Must run from hf_trainer root
os.chdir('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

SAMPLE_LIST = 'output/m2m_repair_eval_man_v2/sample_list.json'
DATA_ROOT = 'data/hymotion_data/'
OUTPUT_DIR = 'output/m2m_repair_eval_man_v2/mogendit_ada_denoise'
REPAIRED_DIR = os.path.join(OUTPUT_DIR, 'repaired')
STATS_PATH = os.path.join(OUTPUT_DIR, 'repair_stats.json')

MODE = 'ada_denoise'
STEP = 10


def main():
    # Load sample list
    with open(SAMPLE_LIST, 'r') as f:
        samples = json.load(f)
    print(f'Loaded {len(samples)} samples from {SAMPLE_LIST}')

    # Build MoGenDIT pipeline
    print('Building MoGenDIT pipeline...')
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    pipeline = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device='cuda:0')
    print('Pipeline ready.')

    # Build quality checker
    print('Building quality checker...')
    from hftrainer.evaluation.quality_check_rules import MotionQualityChecker
    checker = MotionQualityChecker(device='cpu')
    print('Checker ready.')

    os.makedirs(REPAIRED_DIR, exist_ok=True)

    results = []
    per_category = defaultdict(lambda: {'total': 0, 'fixed': 0, 'error': 0, 'still_failed': 0})
    overall = {'total': 0, 'fixed': 0, 'error': 0, 'still_failed': 0}

    for i, sample in enumerate(samples):
        rel_path = sample['path']
        category = sample.get('category', 'unknown')
        orig_failed = sample.get('failed_checks', [])

        input_path = os.path.join(DATA_ROOT, rel_path)
        # Output: flatten path to avoid deep dirs
        safe_name = rel_path.replace('/', '__')
        output_path = os.path.join(REPAIRED_DIR, safe_name)

        record = {
            'path': rel_path,
            'category': category,
            'original_failed_checks': orig_failed,
            'repaired': False,
            'repair_error': None,
            'after_is_valid': None,
            'after_failed_checks': None,
            'fixed': False,
        }

        overall['total'] += 1
        per_category[category]['total'] += 1

        # Step 1: repair
        t0 = time.time()
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f'Input not found: {input_path}')
            pipeline.repair_npz(input_path, output_path, mode=MODE, step=STEP)
            record['repaired'] = True
            repair_time = time.time() - t0
        except Exception as e:
            record['repair_error'] = str(e)
            overall['error'] += 1
            per_category[category]['error'] += 1
            results.append(record)
            print(f'[{i+1}/{len(samples)}] ERROR repair {rel_path}: {e}')
            continue

        # Step 2: quality check on repaired
        try:
            result = checker.check(output_path)
            rd = result.to_dict()
            is_valid = rd.get('is_valid', True)
            failed_checks = rd.get('failed_checks', [])
            record['after_is_valid'] = is_valid
            record['after_failed_checks'] = failed_checks

            if is_valid or len(failed_checks) == 0:
                record['fixed'] = True
                overall['fixed'] += 1
                per_category[category]['fixed'] += 1
            else:
                overall['still_failed'] += 1
                per_category[category]['still_failed'] += 1
        except Exception as e:
            record['repair_error'] = f'checker error: {e}'
            overall['error'] += 1
            per_category[category]['error'] += 1
            print(f'[{i+1}/{len(samples)}] ERROR check {rel_path}: {e}')

        results.append(record)

        status = 'FIXED' if record['fixed'] else ('ERROR' if record['repair_error'] else 'STILL_FAILED')
        after_info = f" after={record['after_failed_checks']}" if record['after_failed_checks'] else ""
        print(f'[{i+1}/{len(samples)}] {status} ({category}) {rel_path} [{repair_time:.1f}s]{after_info}')

    # Summary
    print('\n' + '=' * 70)
    print('OVERALL REPAIR RESULTS')
    print('=' * 70)
    print(f"Total samples:  {overall['total']}")
    print(f"Fixed:          {overall['fixed']} ({100*overall['fixed']/max(overall['total'],1):.1f}%)")
    print(f"Still failed:   {overall['still_failed']} ({100*overall['still_failed']/max(overall['total'],1):.1f}%)")
    print(f"Errors:         {overall['error']} ({100*overall['error']/max(overall['total'],1):.1f}%)")

    print(f"\n{'Category':<25} {'Total':>6} {'Fixed':>6} {'Rate':>8} {'StillFail':>10} {'Error':>6}")
    print('-' * 70)
    for cat in sorted(per_category.keys()):
        s = per_category[cat]
        rate = 100 * s['fixed'] / max(s['total'], 1)
        print(f"{cat:<25} {s['total']:>6} {s['fixed']:>6} {rate:>7.1f}% {s['still_failed']:>10} {s['error']:>6}")

    # Save stats
    stats = {
        'mode': MODE,
        'step': STEP,
        'overall': overall,
        'per_category': dict(per_category),
        'details': results,
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f'\nStats saved to {STATS_PATH}')


if __name__ == '__main__':
    main()

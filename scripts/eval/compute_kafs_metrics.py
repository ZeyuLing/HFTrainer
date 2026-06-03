#!/usr/bin/env python3
"""Convert KAFS SMPLX NPZ outputs to 135-dim .npy and run MotionCLIP evaluator.

Usage:
    python3 scripts/eval/compute_kafs_metrics.py \
        --kafs-dir work_dirs/prism_kafs_ablation \
        --modes none depth_driven uniform random \
        --anno-file data/annotation/test_motionhub_t2m.json \
        --data-dir data/motionhub \
        --evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq

This script:
1. For each mode, reads all .npz files in <kafs_dir>/<mode>/
2. Converts SMPLX axis-angle to 135-dim (transl[3] + 22j rot6d[132])
3. Saves as .npy in <kafs_dir>/<mode>_135d/
4. Runs eval_with_motionclip_evaluator.py for each mode
5. Prints a consolidated comparison table
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def convert_smplx_npz_to_135d(npz_path: Path) -> np.ndarray:
    """Convert SMPLX NPZ (axis-angle) to 135-dim motion array.

    135-dim = transl[3] + global_orient_rot6d[6] + body_pose_rot6d[21*6=126]
    """
    import torch
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
        matrix_to_rotation_6d,
    )

    npz = np.load(str(npz_path), allow_pickle=True)
    transl = np.asarray(npz['transl'], dtype=np.float32)  # (T, 3)
    T = transl.shape[0]

    go_aa = torch.from_numpy(
        np.asarray(npz['global_orient'], dtype=np.float32)
    ).reshape(T, 3)
    bp_aa = torch.from_numpy(
        np.asarray(npz['body_pose'], dtype=np.float32)
    ).reshape(T, 21, 3)

    go_rotmat = axis_angle_to_matrix(go_aa)
    bp_rotmat = axis_angle_to_matrix(bp_aa)
    go = matrix_to_rotation_6d(go_rotmat).numpy().reshape(T, -1)  # (T, 6)
    bp = matrix_to_rotation_6d(bp_rotmat).numpy().reshape(T, -1)  # (T, 126)

    motion135 = np.concatenate([transl, go, bp], axis=-1)
    assert motion135.shape[-1] == 135, f'Expected 135, got {motion135.shape[-1]}'
    return motion135.astype(np.float32)


def _conv_worker_init():
    """Pool initializer: hide GPU so conversion stays CPU-only.

    rotation_convert math is CPU torch; importing hftrainer pulls deepspeed
    which otherwise initializes a CUDA context per worker. With many forked
    workers that contends/OOMs the single GPU and deadlocks imap. Forcing
    CUDA invisible keeps conversion purely on CPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _convert_one(task):
    """Worker for parallel conversion. Returns (status, name, msg).

    status: 'ok' | 'skip' | 'fail'. Skips when target .npy already exists
    so re-runs only process newly generated samples (incremental).
    """
    npz_path, npy_path = task
    npz_path, npy_path = Path(npz_path), Path(npy_path)
    if npy_path.exists():
        return ('skip', npz_path.stem, '')
    try:
        motion135 = convert_smplx_npz_to_135d(npz_path)
        np.save(str(npy_path), motion135)
        return ('ok', npz_path.stem, '')
    except Exception as e:  # noqa: BLE001
        return ('fail', npz_path.stem, str(e))


def main():
    parser = argparse.ArgumentParser(description='KAFS metric computation')
    parser.add_argument('--kafs-dir', type=str, required=True,
                        help='Root directory of KAFS ablation outputs')
    parser.add_argument('--modes', nargs='+',
                        default=['none', 'depth_driven', 'uniform', 'random'],
                        help='KAFS modes to evaluate')
    parser.add_argument('--anno-file', type=str,
                        default='data/annotation/test_motionhub_t2m.json')
    parser.add_argument('--rewritten-caption-file', type=str, default=None,
                        help='Standalone {motion_id: caption} JSON. When set, metrics '
                             'are computed against rewritten captions (consistent '
                             'generate-rewritten + evaluate-rewritten protocol).')
    parser.add_argument('--data-dir', type=str,
                        default='data/motionhub')
    parser.add_argument('--evaluator-ckpt', type=str,
                        default='checkpoints/motion_clip/motionclip_base_1p_aug_hq')
    parser.add_argument('--clip-pretrained', type=str,
                        default='checkpoints/clip-vit-base-patch32')
    parser.add_argument('--stats-file', type=str,
                        default='data/statistic/smplx55_stats_hymotion_aug.json')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Limit number of test pairs (for fast debug)')
    parser.add_argument('--n-repeats', type=int, default=20,
                        help='Number of repeats for metric averaging')
    parser.add_argument('--chunk-size', type=int, default=256,
                        help='R-Precision/MM-Dist pool size. Paper main-table '
                             'protocol uses 64; TMRMetric default is 256.')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip NPZ->135d conversion (use existing .npy)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Parallel processes for NPZ->135d conversion. '
                             'cephfs reads benefit from high concurrency.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    kafs_dir = Path(args.kafs_dir)
    results = {}

    for mode in args.modes:
        npz_dir = kafs_dir / mode
        npy_dir = kafs_dir / f'{mode}_135d'

        if not npz_dir.exists():
            print(f'[!] Mode {mode}: directory {npz_dir} not found, skipping')
            continue

        npz_files = sorted(npz_dir.glob('*.npz'))
        print(f'\n{"="*60}')
        print(f'Mode: {mode}  ({len(npz_files)} NPZ files)')
        print(f'{"="*60}')

        # Step 1: Convert NPZ to 135-dim .npy (parallel, incremental)
        if not args.skip_convert:
            npy_dir.mkdir(parents=True, exist_ok=True)
            tasks = [
                (str(f), str(npy_dir / f'{f.stem}.npy')) for f in npz_files
            ]
            converted = skipped = failed = 0
            n_workers = max(1, args.workers)
            print(f'  Converting with {n_workers} workers '
                  f'({len(tasks)} files, skip-existing)...')
            t0 = time.time()
            if n_workers == 1:
                results_iter = (_convert_one(t) for t in tasks)
            else:
                import multiprocessing as mp
                pool = mp.Pool(n_workers, initializer=_conv_worker_init)
                results_iter = pool.imap_unordered(_convert_one, tasks, chunksize=8)
            done = 0
            for status, name, msg in results_iter:
                done += 1
                if status == 'ok':
                    converted += 1
                elif status == 'skip':
                    skipped += 1
                else:
                    failed += 1
                    if failed <= 10:
                        print(f'  [WARN] Failed to convert {name}: {msg}')
                if done % 500 == 0:
                    rate = done / max(1e-6, time.time() - t0)
                    print(f'    progress {done}/{len(tasks)} '
                          f'({rate:.0f}/s, new={converted} skip={skipped} fail={failed})')
            if n_workers > 1:
                pool.close()
                pool.join()
            print(f'  Converted(new)={converted}, Skipped(existing)={skipped}, '
                  f'Failed={failed} -> {npy_dir} '
                  f'({time.time()-t0:.0f}s)')
        else:
            npy_files = list(npy_dir.glob('*.npy'))
            print(f'  Skipping conversion, found {len(npy_files)} .npy files')

        # Step 2: Run evaluator
        out_json_path = kafs_dir / f'metrics_{mode}.json'
        eval_cmd = [
            sys.executable, 'scripts/eval/eval_with_motionclip_evaluator.py',
            '--anno_file', args.anno_file,
            '--data_dir', args.data_dir,
            '--evaluator_ckpt', args.evaluator_ckpt,
            '--clip_pretrained', args.clip_pretrained,
            '--stats_file', args.stats_file,
            '--pred_dir', str(npy_dir),
            '--out_json', str(out_json_path),
            '--n_repeats', str(args.n_repeats),
            '--seed', '42',
        ]
        if args.rewritten_caption_file:
            eval_cmd.extend(['--rewritten_caption_file', args.rewritten_caption_file])
        eval_cmd.extend(['--chunk_size', str(args.chunk_size)])
        if args.max_pairs:
            eval_cmd.extend(['--max_pairs', str(args.max_pairs)])

        print(f'  Running evaluator...')
        print(f'  cmd: {" ".join(eval_cmd)}')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        result = subprocess.run(
            eval_cmd, capture_output=True, text=True, env=env
        )
        if result.stdout:
            # Print last 30 lines of stdout for visibility
            lines = result.stdout.strip().split('\n')
            for line in lines[-30:]:
                print(f'  {line}')
        if result.returncode != 0:
            print(f'  [ERROR] Evaluator failed for mode {mode}:')
            stderr = result.stderr
            print(stderr[-2000:] if len(stderr) > 2000 else stderr)
            continue

        # Step 3: Load metrics from JSON output
        if out_json_path.exists():
            with open(str(out_json_path)) as f:
                metrics = json.load(f)
            results[mode] = metrics
            print(f'  FID={metrics.get("fid_mean", "?"):.4f}, '
                  f'R-P T3={metrics.get("r_precision_pred_top3_mean", "?"):.4f}, '
                  f'MM-D={metrics.get("mm_dist_pred_mean", "?"):.4f}')
        else:
            print(f'  [ERROR] Output JSON not found: {out_json_path}')

    # Step 4: Print consolidated table
    print(f'\n{"="*80}')
    print('KAFS Ablation Results — Consolidated Table')
    print(f'{"="*80}')
    header = (f'{"Mode":<15} {"FID↓":>10} {"R-P T1↑":>10} {"R-P T3↑":>10} '
              f'{"MM-D↓":>10} {"Div-pred":>10} {"Samples":>8}')
    print(header)
    print('-' * 80)
    for mode in args.modes:
        if mode not in results:
            print(f'{mode:<15} {"N/A":>10} {"N/A":>10} {"N/A":>10} '
                  f'{"N/A":>10} {"N/A":>10} {"N/A":>8}')
            continue
        m = results[mode]
        print(f'{mode:<15} '
              f'{m.get("fid_mean", float("nan")):>10.3f} '
              f'{m.get("r_precision_pred_top1_mean", float("nan")):>10.4f} '
              f'{m.get("r_precision_pred_top3_mean", float("nan")):>10.4f} '
              f'{m.get("mm_dist_pred_mean", float("nan")):>10.4f} '
              f'{m.get("diversity_pred_mean", float("nan")):>10.3f} '
              f'{m.get("samples", "?"):>8}')
    print(f'{"="*80}')

    # Save consolidated results JSON
    consolidated = kafs_dir / 'kafs_metrics_all.json'
    with open(str(consolidated), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved consolidated metrics to {consolidated}')

    # Print LaTeX table snippet
    print('\n--- LaTeX Table Snippet ---')
    mode_labels = {
        'none': 'None (baseline)',
        'uniform': 'Uniform',
        'random': 'Random',
        'depth_driven': 'Depth-driven (ours)',
    }
    for mode in args.modes:
        if mode not in results:
            continue
        m = results[mode]
        label = mode_labels.get(mode, mode)
        fid = f'{m.get("fid_mean", 0):.2f}'
        rpt3 = f'{m.get("r_precision_pred_top3_mean", 0):.3f}'
        mmd = f'{m.get("mm_dist_pred_mean", 0):.3f}'
        bold = mode == 'depth_driven'
        if bold:
            print(f'{label} & \\textbf{{{fid}}} & \\textbf{{{rpt3}}} & \\textbf{{{mmd}}} \\\\')
        else:
            print(f'{label} & {fid} & {rpt3} & {mmd} \\\\')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MoGenDIT motion repair CLI.

Supports single-file and batch-directory modes.

Single file:
    python scripts/mogendit_repair.py \\
        --input-npz data/motion.npz \\
        --output-npz work_dirs/repaired.npz \\
        --model-name MoreDiff-0.1B \\
        --mode denoise --denoise-step 10

Batch directory:
    python scripts/mogendit_repair.py \\
        --input-dir data/hymotion_data/ \\
        --output-dir work_dirs/mogendit_repair/ \\
        --max-samples 20 \\
        --mode denoise --denoise-step 10

Available modes:
    denoise      - Light denoising for jitter/artifact cleanup
    ada_denoise  - Adaptive denoising (auto-detects high-change regions)
    trans_regen  - Regenerate translation while keeping pose rotations
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MoGenDIT motion repair',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output (single file)
    parser.add_argument('--input-npz', type=str, default=None,
                        help='Path to a single input .npz file.')
    parser.add_argument('--output-npz', type=str, default=None,
                        help='Path to save repaired .npz file.')

    # Input/output (batch)
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing input .npz files (recursive).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save repaired .npz files.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max number of files to process in batch mode.')

    # Model config
    parser.add_argument('--model-name', type=str, default='MoreDiff-0.1B',
                        help='MoGenDIT model variant name.')
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Override checkpoint directory.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device string.')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='Use EMA checkpoint (default: True).')
    parser.add_argument('--no-ema', action='store_true',
                        help='Disable EMA checkpoint.')

    # Repair config
    parser.add_argument('--mode', type=str, default='denoise',
                        choices=['denoise', 'ada_denoise', 'trans_regen'],
                        help='Repair mode.')
    parser.add_argument('--denoise-step', type=int, default=10,
                        help='Number of denoising steps.')
    parser.add_argument('--use-windowed', action='store_true', default=True,
                        help='Enable windowed processing (default: True).')
    parser.add_argument('--no-windowed', action='store_true',
                        help='Disable windowed processing.')
    parser.add_argument('--window-size', type=int, default=224,
                        help='Window size in frames.')
    parser.add_argument('--prev-padding', type=int, default=20,
                        help='Overlap frames between windows.')

    args = parser.parse_args()

    # Validate
    if args.input_npz is None and args.input_dir is None:
        parser.error('Must specify either --input-npz or --input-dir.')
    if args.input_npz is not None and args.input_dir is not None:
        parser.error('Cannot specify both --input-npz and --input-dir.')
    if args.input_npz is not None and args.output_npz is None:
        parser.error('--output-npz is required when using --input-npz.')
    if args.input_dir is not None and args.output_dir is None:
        parser.error('--output-dir is required when using --input-dir.')

    # Handle ema flag
    if args.no_ema:
        args.use_ema = False
    if args.no_windowed:
        args.use_windowed = False

    return args


def collect_npz_files(input_dir: str, max_samples: int = None):
    """Recursively collect .npz files from input_dir."""
    input_path = Path(input_dir)
    npz_files = sorted(input_path.rglob('*.npz'))
    if max_samples is not None:
        npz_files = npz_files[:max_samples]
    return npz_files


def main():
    args = parse_args()

    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

    # Initialize pipeline
    logger.info(f'Initializing MoGenDIT pipeline: model={args.model_name}, '
                f'device={args.device}, ema={args.use_ema}')
    t0 = time.time()
    pipeline = MoGenDITRepairPipeline(
        model_name=args.model_name,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        use_ema=args.use_ema,
    )
    logger.info(f'Pipeline initialized in {time.time() - t0:.1f}s')

    if args.input_npz:
        # Single file mode
        logger.info(f'Repairing: {args.input_npz} -> {args.output_npz}')
        t0 = time.time()
        pipeline.repair_npz(
            input_path=args.input_npz,
            output_path=args.output_npz,
            mode=args.mode,
            step=args.denoise_step,
            use_windowed=args.use_windowed,
            window_size=args.window_size,
            prev_padding=args.prev_padding,
        )
        logger.info(f'Done in {time.time() - t0:.1f}s')

    else:
        # Batch mode
        npz_files = collect_npz_files(args.input_dir, args.max_samples)
        logger.info(f'Found {len(npz_files)} .npz files in {args.input_dir}')

        if len(npz_files) == 0:
            logger.warning('No .npz files found. Exiting.')
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_root = Path(args.input_dir)
        success = 0
        failed = 0

        for i, npz_file in enumerate(npz_files):
            # Preserve relative directory structure
            rel_path = npz_file.relative_to(input_root)
            out_path = output_dir / rel_path

            logger.info(f'[{i+1}/{len(npz_files)}] {rel_path}')
            t0 = time.time()

            try:
                pipeline.repair_npz(
                    input_path=str(npz_file),
                    output_path=str(out_path),
                    mode=args.mode,
                    step=args.denoise_step,
                    use_windowed=args.use_windowed,
                    window_size=args.window_size,
                    prev_padding=args.prev_padding,
                )
                elapsed = time.time() - t0
                logger.info(f'  OK ({elapsed:.1f}s)')
                success += 1
            except Exception as e:
                elapsed = time.time() - t0
                logger.error(f'  FAILED ({elapsed:.1f}s): {e}')
                failed += 1

        logger.info(f'Batch complete: {success} succeeded, {failed} failed '
                    f'out of {len(npz_files)} total.')


if __name__ == '__main__':
    main()

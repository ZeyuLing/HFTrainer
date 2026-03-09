"""
tools/train.py — Main training entry point for hftrainer.

Usage:
    python tools/train.py configs/classification/vit_base_demo.py [--work-dir WORK_DIR]
    # or with accelerate:
    accelerate launch tools/train.py configs/classification/vit_base_demo.py
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with hftrainer')
    parser.add_argument('config', help='Path to config file (.py)')
    parser.add_argument('--work-dir', '--work_dir', dest='work_dir',
                        help='Override work_dir in config')
    parser.add_argument('--auto-resume', '--auto_resume', dest='auto_resume',
                        action='store_true',
                        help='Auto-resume from latest checkpoint in work_dir')
    parser.add_argument('--load-from', '--load_from', dest='load_from',
                        help='Path to checkpoint to load from')
    parser.add_argument('--load-scope', '--load_scope', dest='load_scope',
                        default='model', choices=['model', 'full'],
                        help='Checkpoint load scope (model: weights only, full: full resume)')
    parser.add_argument('--cfg-options', '--cfg_options', dest='cfg_options', nargs='+',
                        help='Override config options, e.g. optimizer.lr=1e-4')
    # Accelerate passes this; ignore it
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Load config
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)

    # Apply CLI overrides
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.auto_resume:
        cfg.auto_resume = True
    if args.load_from:
        cfg.load_from = dict(path=args.load_from, load_scope=args.load_scope)
    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))

    # Set default work_dir if not specified
    if not getattr(cfg, 'work_dir', None):
        cfg.work_dir = os.path.join(
            'work_dirs', os.path.splitext(os.path.basename(args.config))[0]
        )

    os.makedirs(cfg.work_dir, exist_ok=True)

    from hftrainer.utils.logger import get_logger
    logger = get_logger()
    logger.info(f"Config: {args.config}")
    logger.info(f"Work dir: {cfg.work_dir}")

    # Build runner and train
    from hftrainer import AccelerateRunner
    runner = AccelerateRunner.from_cfg(cfg)
    runner.train()


def _parse_cfg_options(options):
    """Parse cfg_options like ['optimizer.lr=1e-4', 'train_cfg.max_iters=1000']."""
    result = {}
    for opt in options:
        key, val = opt.split('=', 1)
        try:
            import ast
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return result


if __name__ == '__main__':
    main()

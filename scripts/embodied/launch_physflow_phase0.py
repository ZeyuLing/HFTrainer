#!/usr/bin/env python3
"""
PhysFlow Phase 0 Launcher

Launches Phase 0 baseline experiments (C0, C1) with proper environment setup,
logging, and metrics collection.

Usage:
    python3 scripts/embodied/launch_physflow_phase0.py --config c0 [--dry-run]
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from mmengine import Config

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def setup_logging(output_dir):
    """Configure logging for Phase 0 experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"phase0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    
    return log_file


def verify_environment():
    """Verify all dependencies are available."""
    logger.info("Verifying environment...")
    
    checks = {
        'CUDA Available': torch.cuda.is_available(),
        'CUDA Devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'ProtoMotions': check_protomotions(),
        'HyMotion Checkpoint': os.path.exists('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'),
        'MuJoCo': check_mujoco(),
    }
    
    logger.info("Environment check results:")
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        logger.info(f"  {status} {check_name}: {result}")
    
    # Critical checks
    critical = ['CUDA Available', 'ProtoMotions', 'HyMotion Checkpoint']
    for check in critical:
        if not checks[check]:
            logger.error(f"Critical requirement failed: {check}")
            return False
    
    return True


def check_protomotions():
    """Check ProtoMotions availability."""
    try:
        sys.path.insert(0, 'ref_repo/ProtoMotions')
        from protomotions.components.motion_lib import MotionLib
        return True
    except:
        return False


def check_mujoco():
    """Check MuJoCo availability."""
    try:
        import mujoco
        return True
    except:
        return False


def load_experiment_config(config_id):
    """Load Phase 0 experiment configuration."""
    config_map = {
        'c0': 'configs/experiments/physflow_phase0/phase0_baseline_c0.py',
    }
    
    if config_id.lower() not in config_map:
        raise ValueError(f"Unknown config: {config_id}. Available: {list(config_map.keys())}")
    
    config_path = config_map[config_id.lower()]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    cfg = Config.fromfile(config_path)
    logger.info(f"Loaded config: {config_id} from {config_path}")
    
    return cfg


def save_experiment_metadata(cfg, output_dir):
    """Save experiment metadata to results directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'phase': cfg.experiment_metadata.phase,
        'config_id': cfg.experiment_metadata.config_id,
        'description': cfg.experiment_metadata.description,
        'direction_a': cfg.experiment_metadata.direction_a_enabled,
        'direction_b': cfg.experiment_metadata.direction_b_enabled,
        'expected_metrics': {
            'ppr': cfg.experiment_metadata.expected_ppr_range,
            'fid': cfg.experiment_metadata.expected_fid_range,
            'diversity': cfg.experiment_metadata.expected_diversity,
        },
        'environment': {
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count(),
            'pytorch_version': torch.__version__,
        }
    }
    
    metadata_file = os.path.join(output_dir, 'experiment_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved experiment metadata to {metadata_file}")
    return metadata


def print_experiment_summary(cfg):
    """Print human-readable experiment summary."""
    logger.info("=" * 70)
    logger.info("PhysFlow Phase 0 Experiment")
    logger.info("=" * 70)
    logger.info(f"Configuration: {cfg.experiment_metadata.config_id}")
    logger.info(f"Experiment: {cfg.experiment_metadata.experiment_name}")
    logger.info(f"Description: {cfg.experiment_metadata.description}")
    logger.info("")
    logger.info("Evaluation Metrics:")
    logger.info(f"  - FID: {cfg.evaluation.compute_fid}")
    logger.info(f"  - R-Precision: {cfg.evaluation.compute_r_precision}")
    logger.info(f"  - Diversity: {cfg.evaluation.compute_diversity}")
    logger.info(f"  - PPR (Physics): {cfg.evaluation.compute_ppr}")
    logger.info("")
    logger.info("Expected Results:")
    logger.info(f"  - PPR Range: {cfg.experiment_metadata.expected_ppr_range}")
    logger.info(f"  - FID Range: {cfg.experiment_metadata.expected_fid_range}")
    logger.info(f"  - Diversity Range: {cfg.experiment_metadata.expected_diversity}")
    logger.info("")
    logger.info(f"Output Directory: {cfg.output_dir}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Launch PhysFlow Phase 0 experiments')
    parser.add_argument('--config', type=str, default='c0', choices=['c0'],
                        help='Configuration to run (c0=baseline T2M)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only verify setup, do not run experiment')
    parser.add_argument('--output-base', type=str, default='results/physflow_phase0',
                        help='Base output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.output_base)
    logger.info(f"PhysFlow Phase 0 Launcher started (log: {log_file})")
    
    # Verify environment
    if not verify_environment():
        logger.error("Environment verification failed")
        return 1
    
    # Load config
    try:
        cfg = load_experiment_config(args.config)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Save metadata
    output_dir = cfg.output_dir
    experiment_metadata = save_experiment_metadata(cfg, output_dir)
    
    # Print summary
    print_experiment_summary(cfg)
    
    if args.dry_run:
        logger.info("Dry-run mode: Environment verified, config loaded, ready to run")
        return 0
    
    # ===== PHASE 0 LAUNCH WOULD GO HERE =====
    # The actual experiment runner would be called here
    # For now, just verify everything is ready
    logger.info(f"Ready to launch Phase 0 experiment: {args.config}")
    logger.info("Next steps:")
    logger.info(f"  1. python3 scripts/embodied/physflow_evaluate.py --config {args.config}")
    logger.info(f"  2. Results will be saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

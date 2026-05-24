#!/usr/bin/env python3
"""Batch evaluation script for PRISM KAFS ablation experiments.

Generates motions from a PRISM model under different KAFS (Kinematic-Adaptive
Flow Scheduling) modes and saves them as per-sample NPZ files for downstream
metric computation.

Usage:
    python scripts/eval/eval_prism_kafs_ablation.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
        --kafs-mode depth_driven \
        --anno-file data/annotation/test_hml3d.json \
        --data-dir data/motionhub \
        --output-dir work_dirs/prism_kafs_ablation \
        --num-inference-steps 50 \
        --max-samples 100

KAFS modes:
    none          — Standard baseline (no per-joint timestep scaling)
    depth_driven  — Per-joint scaling based on kinematic tree depth
    uniform       — All joints get alpha=1.0 (equivalent to baseline, sanity check)
    random        — Random alphas in [0.85, 1.15] for ablation control
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure hftrainer is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent  # scripts/eval/.. -> scripts/.. -> hf_trainer/
sys.path.insert(0, str(HF_ROOT))


# ---------------------------------------------------------------------------
# Caption loader (mirrors eval_with_motionclip_evaluator.py)
# ---------------------------------------------------------------------------

def _load_caption(caption_path: Path) -> Optional[str]:
    """Load a single randomly chosen caption from hierarchical or hymotion format."""
    if not caption_path.exists():
        return None
    try:
        data = json.loads(caption_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    # Hierarchical format: {macro: [...], meso: [...], micro: [...]}
    if all(k in data and isinstance(data[k], list) for k in ('macro', 'meso', 'micro')):
        pool = []
        for g in ('macro', 'meso', 'micro'):
            for c in data[g]:
                if isinstance(c, str) and c.strip():
                    pool.append(c.strip())
        return random.choice(pool) if pool else None

    # HYMotion format: {result: [{short_caption_rewritten: [...], ...}]}
    if 'result' in data and isinstance(data['result'], list):
        pool = []
        for item in data['result']:
            if not isinstance(item, dict):
                continue
            for rk in ('short_caption_rewritten', 'short caption_rewritten'):
                if isinstance(item.get(rk), list):
                    for v in item[rk]:
                        if isinstance(v, str) and v.strip():
                            pool.append(v.strip())
                    break
            else:
                for ck in ('short_caption', 'short caption'):
                    if isinstance(item.get(ck), str) and item[ck].strip():
                        pool.append(item[ck].strip())
                        break
        return random.choice(pool) if pool else None

    return None


# ---------------------------------------------------------------------------
# Annotation / test sample loading
# ---------------------------------------------------------------------------

def load_test_samples(
    anno_file: Path,
    data_dir: Path,
    motion_key: str = 'smplx',
    caption_key: str = 'hierarchical_caption',
    min_frames: int = 24,
    max_frames: int = 360,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load test samples from a motionhub-format annotation JSON.

    Returns list of dicts with keys:
        name, caption, num_frames, motion_path, caption_path
    """
    raw = json.loads(anno_file.read_text())

    # Parse annotation: dict with data_list or flat list
    if isinstance(raw, dict) and 'data_list' in raw:
        dl = raw['data_list']
        if isinstance(dl, dict):
            entries = list(dl.items())
        else:
            entries = [
                (e.get('motion_id') or e.get('id') or str(i), e)
                for i, e in enumerate(dl)
            ]
    elif isinstance(raw, list):
        entries = [
            (e.get('motion_id') or e.get('id') or str(i), e)
            for i, e in enumerate(raw)
        ]
    else:
        raise ValueError(f'Unrecognized annotation format in {anno_file}')

    samples = []
    for name, entry in entries:
        m_rel = entry.get(f'{motion_key}_path')
        c_rel = entry.get(f'{caption_key}_path')
        num_frames = entry.get('num_frames')
        if not (m_rel and c_rel and num_frames):
            continue

        num_frames = int(num_frames)
        if num_frames < min_frames:
            continue
        num_frames = min(num_frames, max_frames)

        m_path = Path(data_dir) / m_rel
        c_path = Path(data_dir) / c_rel

        caption = _load_caption(c_path)
        if caption is None:
            continue

        if not m_path.exists():
            continue

        samples.append({
            'name': name,
            'caption': caption,
            'num_frames': num_frames,
            'motion_path': str(m_path),
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_prism_bundle(config_path: str, checkpoint_dir: str, device: torch.device):
    """Load PrismBundle from config + checkpoint directory.

    The checkpoint directory should contain model.pt (the standard
    AccelerateRunner checkpoint format with per-module state dicts).
    """
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    # Load checkpoint (model.pt is a dict keyed by module name)
    ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        print(f'[+] Loading checkpoint from {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    else:
        # Fallback: try model.safetensors (only has transformer weights)
        st_path = os.path.join(checkpoint_dir, 'model.safetensors')
        if os.path.isfile(st_path):
            print(f'[+] Loading safetensors from {st_path}')
            from safetensors.torch import load_file
            st_dict = load_file(st_path)
            bundle.transformer.load_state_dict(st_dict, strict=False)
        else:
            raise FileNotFoundError(
                f'No model.pt or model.safetensors found in {checkpoint_dir}'
            )

    bundle = bundle.eval().to(device)
    return bundle


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_motion(
    pipeline,
    caption: str,
    num_frames: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
) -> Dict:
    """Generate a single motion sample via the PRISM pipeline.

    Returns the smplx_dict from pipeline output.
    """
    output = pipeline(
        prompts=caption,
        num_frames_per_segment=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    return output


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------

def save_smplx_npz(out_path: str, smplx_dict: Dict):
    """Save smplx_dict to compressed NPZ."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pack = {}
    for k, v in smplx_dict.items():
        if isinstance(v, np.ndarray):
            pack[k] = v.astype(np.float32, copy=False)
        elif isinstance(v, torch.Tensor):
            pack[k] = v.detach().cpu().numpy().astype(np.float32)
        else:
            pack[k] = v
    np.savez_compressed(out_path, **pack)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='PRISM KAFS ablation evaluation: generate motions under different KAFS modes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/prism/prism_1b_tp2m_multiframe.py',
        help='Path to PRISM config file.',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000',
        help='Path to checkpoint directory (containing model.pt or model.safetensors).',
    )
    parser.add_argument(
        '--kafs-mode', type=str, default='none',
        choices=['none', 'depth_driven', 'uniform', 'random'],
        help='KAFS mode for per-joint adaptive timestep scaling.',
    )
    parser.add_argument(
        '--anno-file', type=str,
        default='data/annotation/test_hml3d.json',
        help='Annotation JSON file (motionhub format).',
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='data/motionhub',
        help='Root data directory for resolving relative paths in annotation.',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='work_dirs/prism_kafs_ablation',
        help='Root output directory. Results saved under <output-dir>/<kafs-mode>/.',
    )
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='Total number of samples to generate. None = all available.',
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Limit number of samples loaded from annotation (for debugging).',
    )
    parser.add_argument(
        '--num-inference-steps', type=int, default=50,
        help='Number of denoising steps per segment.',
    )
    parser.add_argument(
        '--guidance-scale', type=float, default=5.0,
        help='Classifier-free guidance scale.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size for generation (currently only batch_size=1 is supported).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--motion-key', type=str, default='smplx',
        help='Key prefix for motion path in annotation (e.g. smplx -> smplx_path).',
    )
    parser.add_argument(
        '--caption-key', type=str, default='hierarchical_caption',
        help='Key prefix for caption path in annotation.',
    )
    parser.add_argument(
        '--min-frames', type=int, default=24,
        help='Minimum number of frames to include a sample.',
    )
    parser.add_argument(
        '--max-frames', type=int, default=360,
        help='Maximum number of frames per sample.',
    )
    args = parser.parse_args()

    # ---- Seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[+] Device: {device}')
    print(f'[+] KAFS mode: {args.kafs_mode}')

    # ---- Load test samples ----
    anno_path = Path(args.anno_file)
    data_path = Path(args.data_dir)
    print(f'[+] Loading test samples from {anno_path} ...')

    samples = load_test_samples(
        anno_file=anno_path,
        data_dir=data_path,
        motion_key=args.motion_key,
        caption_key=args.caption_key,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
    )
    print(f'    Loaded {len(samples)} test samples')
    if not samples:
        raise RuntimeError(
            f'No valid test samples found. Check --anno-file ({args.anno_file}) '
            f'and --data-dir ({args.data_dir}).'
        )

    # Optionally limit number of samples to generate
    if args.num_samples is not None and args.num_samples < len(samples):
        samples = samples[:args.num_samples]
        print(f'    Limited to {len(samples)} samples (--num-samples)')

    # ---- Load model ----
    print(f'[+] Loading PRISM bundle from config: {args.config}')
    print(f'    Checkpoint: {args.checkpoint}')
    t_load = time.time()
    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    print(f'    Bundle loaded in {time.time() - t_load:.1f}s')

    # ---- Build pipeline ----
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
    pipeline = PrismPipeline(bundle=bundle)

    # ---- Set KAFS mode ----
    pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)

    # ---- Output directory ----
    out_dir = Path(args.output_dir) / args.kafs_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[+] Output directory: {out_dir}')

    # Save run metadata
    meta = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'kafs_mode': args.kafs_mode,
        'anno_file': args.anno_file,
        'data_dir': args.data_dir,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
        'num_samples': len(samples),
    }
    meta_path = out_dir / 'run_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'    Saved run metadata to {meta_path}')

    # ---- Generate motions ----
    print(f'[+] Generating {len(samples)} motions (steps={args.num_inference_steps}, '
          f'guidance={args.guidance_scale}) ...')
    t_start = time.time()
    n_success = 0
    n_fail = 0
    results_manifest = []

    for i, sample in enumerate(samples):
        sample_name = sample['name']
        caption = sample['caption']
        num_frames = sample['num_frames']

        try:
            smplx_dict = generate_motion(
                pipeline=pipeline,
                caption=caption,
                num_frames=num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )

            # Save output NPZ
            out_path = out_dir / f'{sample_name}.npz'
            save_smplx_npz(str(out_path), smplx_dict)

            # Determine output frame count
            out_frames = -1
            if 'transl' in smplx_dict:
                v = smplx_dict['transl']
                if isinstance(v, np.ndarray):
                    out_frames = v.shape[0]
                elif isinstance(v, torch.Tensor):
                    out_frames = v.shape[0]

            results_manifest.append({
                'name': sample_name,
                'caption': caption,
                'gt_num_frames': num_frames,
                'gen_num_frames': out_frames,
                'npz_path': str(out_path),
                'status': 'success',
            })
            n_success += 1

        except Exception as e:
            print(f'  [!] Failed on sample {sample_name}: {e}')
            results_manifest.append({
                'name': sample_name,
                'caption': caption,
                'gt_num_frames': num_frames,
                'gen_num_frames': -1,
                'npz_path': '',
                'status': f'error: {e}',
            })
            n_fail += 1

        # Progress report every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t_start
            avg_time = elapsed / (i + 1)
            eta = avg_time * (len(samples) - i - 1)
            print(f'  [{i + 1}/{len(samples)}] '
                  f'success={n_success} fail={n_fail} '
                  f'elapsed={elapsed:.1f}s avg={avg_time:.2f}s/sample '
                  f'ETA={eta:.0f}s')

    # ---- Summary ----
    total_time = time.time() - t_start
    print()
    print('=' * 60)
    print(f'PRISM KAFS Ablation — {args.kafs_mode}')
    print(f'  Total samples:  {len(samples)}')
    print(f'  Successful:     {n_success}')
    print(f'  Failed:         {n_fail}')
    print(f'  Total time:     {total_time:.1f}s')
    if n_success > 0:
        print(f'  Avg time/sample: {total_time / n_success:.2f}s')
    print(f'  Output dir:     {out_dir}')
    print('=' * 60)

    # Save manifest
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(results_manifest, indent=2))
    print(f'[+] Saved manifest ({len(results_manifest)} entries) to {manifest_path}')


if __name__ == '__main__':
    main()

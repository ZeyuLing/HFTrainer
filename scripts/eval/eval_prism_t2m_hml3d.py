#!/usr/bin/env python3
"""PRISM T2M evaluation on HumanML3D test set — Multi-GPU parallel.

Generates motions from a PRISM model on test_hml3d_rewritten.json
(simple {motion_id: caption} format) with frame counts from test_hml3d.json.

Each GPU loads the model independently and processes a shard of the test set.
Saves per-sample NPZ files for downstream metric computation.

Usage:
    # 8-GPU parallel (default: use all available GPUs)
    python scripts/eval/eval_prism_t2m_hml3d.py \
        --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
        --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten \
        --num-inference-steps 50 \
        --guidance-scale 5.0 \
        --gpus 0 1 2 3 4 5 6 7

    # Single GPU (fallback)
    python scripts/eval/eval_prism_t2m_hml3d.py --gpus 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))


def load_test_samples(
    rewritten_file: Path,
    meta_file: Path,
    max_samples: Optional[int] = None,
    min_frames: int = 24,
    max_frames: int = 300,
) -> List[Dict]:
    """Load test samples by combining rewritten captions with metadata."""
    captions = json.loads(rewritten_file.read_text())
    meta = json.loads(meta_file.read_text())
    data_list = meta['data_list']

    samples = []
    for motion_id, caption in captions.items():
        if motion_id not in data_list:
            continue
        entry = data_list[motion_id]
        num_frames = int(entry.get('num_frames', 0))
        if num_frames < min_frames:
            continue
        num_frames = min(num_frames, max_frames)

        samples.append({
            'name': motion_id,
            'caption': caption,
            'num_frames': num_frames,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def load_prism_bundle(config_path: str, checkpoint_dir: str, device: torch.device):
    """Load PrismBundle from config + checkpoint directory."""
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    else:
        st_path = os.path.join(checkpoint_dir, 'model.safetensors')
        if os.path.isfile(st_path):
            from safetensors.torch import load_file
            st_dict = load_file(st_path)
            bundle.transformer.load_state_dict(st_dict, strict=False)
        else:
            raise FileNotFoundError(
                f'No model.pt or model.safetensors found in {checkpoint_dir}'
            )

    bundle = bundle.eval().to(device)
    return bundle


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


def worker_fn(
    gpu_id: int,
    samples: List[Dict],
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    result_queue: Queue,
):
    """Worker process: load model on one GPU, generate assigned samples."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0')

        # Seed per worker (different but deterministic)
        worker_seed = seed + gpu_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)

        print(f'[GPU {gpu_id}] Loading model... ({len(samples)} samples assigned)')
        t0 = time.time()
        bundle = load_prism_bundle(config_path, checkpoint_dir, device)
        print(f'[GPU {gpu_id}] Model loaded in {time.time() - t0:.1f}s')

        from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
        pipeline = PrismPipeline(bundle=bundle)

        out_dir = Path(output_dir)
        n_success = 0
        n_fail = 0
        t_start = time.time()

        for i, sample in enumerate(samples):
            name = sample['name']
            caption = sample['caption']
            num_frames = sample['num_frames']

            # Skip if already exists (resume support)
            out_path = out_dir / f'{name}.npz'
            if out_path.exists():
                n_success += 1
                continue

            try:
                with torch.no_grad():
                    smplx_dict = pipeline(
                        prompts=caption,
                        num_frames_per_segment=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    )

                save_smplx_npz(str(out_path), smplx_dict)
                n_success += 1

            except Exception as e:
                print(f'[GPU {gpu_id}] Failed: {name}: {e}')
                n_fail += 1

            # Progress every 20 samples
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_start
                avg_t = elapsed / (i + 1)
                eta = avg_t * (len(samples) - i - 1)
                print(f'[GPU {gpu_id}] [{i + 1}/{len(samples)}] '
                      f'ok={n_success} fail={n_fail} '
                      f'avg={avg_t:.1f}s/sample ETA={eta/60:.0f}min')

        elapsed = time.time() - t_start
        print(f'[GPU {gpu_id}] Done: {n_success} ok, {n_fail} fail, {elapsed:.0f}s total')
        result_queue.put({
            'gpu_id': gpu_id,
            'success': n_success,
            'fail': n_fail,
            'time': elapsed,
        })

    except Exception as e:
        print(f'[GPU {gpu_id}] FATAL: {e}')
        traceback.print_exc()
        result_queue.put({
            'gpu_id': gpu_id,
            'success': 0,
            'fail': len(samples),
            'time': 0,
            'error': str(e),
        })


def main():
    parser = argparse.ArgumentParser(
        description='PRISM T2M eval on HumanML3D — Multi-GPU parallel.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str,
                        default='configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0')
    parser.add_argument('--anno-file', type=str,
                        default='data/annotation/test_hml3d_rewritten.json')
    parser.add_argument('--meta-file', type=str,
                        default='data/annotation/test_hml3d.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Default: <checkpoint_parent>/eval_hml3d_rewritten/')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-frames', type=int, default=24)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='GPU IDs to use. Default: all available.')
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent / 'eval_hml3d_rewritten')

    # Determine GPUs
    if args.gpus is None:
        n_gpus = torch.cuda.device_count()
        args.gpus = list(range(n_gpus))
    num_workers = len(args.gpus)
    print(f'[+] Using {num_workers} GPUs: {args.gpus}')

    # Load test samples
    print(f'[+] Loading test samples...')
    samples = load_test_samples(
        rewritten_file=Path(args.anno_file),
        meta_file=Path(args.meta_file),
        max_samples=args.max_samples,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    print(f'    Total: {len(samples)} samples')

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter already-done samples for progress display
    existing = set(p.stem for p in out_dir.glob('*.npz'))
    remaining = [s for s in samples if s['name'] not in existing]
    print(f'    Already done: {len(existing)}, remaining: {len(remaining)}')

    if not remaining:
        print('[+] All samples already generated!')
        return

    # Save run metadata
    meta_info = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'anno_file': args.anno_file,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
        'total_samples': len(samples),
        'gpus': args.gpus,
        'num_workers': num_workers,
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(meta_info, indent=2))

    # Shard samples across GPUs
    shards = [[] for _ in range(num_workers)]
    for i, s in enumerate(remaining):
        shards[i % num_workers].append(s)

    for i, shard in enumerate(shards):
        print(f'    GPU {args.gpus[i]}: {len(shard)} samples')

    # Launch workers
    print(f'[+] Launching {num_workers} worker processes...')
    t_start = time.time()
    result_queue = Queue()
    processes = []

    for i, gpu_id in enumerate(args.gpus):
        p = Process(
            target=worker_fn,
            args=(
                gpu_id,
                shards[i],
                args.config,
                args.checkpoint,
                args.output_dir,
                args.num_inference_steps,
                args.guidance_scale,
                args.seed,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all
    for p in processes:
        p.join()

    # Collect results
    total_success = 0
    total_fail = 0
    while not result_queue.empty():
        r = result_queue.get()
        total_success += r['success']
        total_fail += r['fail']

    total_time = time.time() - t_start
    print()
    print('=' * 60)
    print(f'PRISM T2M Eval — HumanML3D (rewritten) — {num_workers} GPUs')
    print(f'  Total remaining: {len(remaining)}')
    print(f'  Successful:      {total_success}')
    print(f'  Failed:          {total_fail}')
    print(f'  Wall time:       {total_time:.0f}s ({total_time/60:.1f}min)')
    if total_success > 0:
        print(f'  Throughput:      {total_success / total_time:.2f} samples/s')
    print(f'  Output dir:      {out_dir}')
    print('=' * 60)

    # Save final manifest by scanning all NPZ files
    all_npz = sorted(out_dir.glob('*.npz'))
    manifest = []
    caption_map = json.loads(Path(args.anno_file).read_text())
    for npz_path in all_npz:
        name = npz_path.stem
        manifest.append({
            'name': name,
            'caption': caption_map.get(name, ''),
            'npz_path': str(npz_path),
            'status': 'success',
        })
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[+] Saved manifest ({len(manifest)} entries)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Batch inference script for VerMo: 2 models × 9 tasks × N samples.

Usage:
    # Infer all models × all tasks (default 100 samples each)
    python tools/batch_infer_vermo.py

    # Infer specific model and task
    python tools/batch_infer_vermo.py --models qwen1.7b --tasks t2m_1p m2t_1p --num-samples 50

    # Specify custom checkpoint paths
    python tools/batch_infer_vermo.py \
        --models qwen1.7b \
        --qwen-ckpt work_dirs/vermo_pretrain_4k_qwen1.7b/checkpoint-iter_35000 \
        --num-samples 10

Output structure:
    work_dirs/vermo_eval/{model}/{task}/sample_{i:04d}.npz   (for motion tasks)
    work_dirs/vermo_eval/{model}/{task}/sample_{i:04d}.txt   (for caption tasks)
    work_dirs/vermo_eval/{model}/{task}/sample_{i:04d}.wav   (for music tasks)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -----------------------------------------------------------------------
# Model registry: name → (config, default checkpoint)
# -----------------------------------------------------------------------

MODELS = {
    'qwen1.7b': {
        'config': 'configs/vermo/vermo_sft_16k_qwen1.7b_wavtokenizer.py',
        'checkpoint': 'work_dirs/vermo_sft_16k_qwen1.7b_wavtokenizer/checkpoint-epoch_5',
    },
    'llama1b': {
        'config': 'configs/vermo/vermo_sft_16k_llama1b_wavtokenizer.py',
        'checkpoint': 'work_dirs/vermo_sft_16k_llama1b_wavtokenizer/checkpoint-epoch_9',
    },
}

# -----------------------------------------------------------------------
# Tasks: name → required input fields and how to generate test data
# -----------------------------------------------------------------------

TASK_INFO = {
    't2m_1p': {
        'desc': 'Text → single-person motion',
        'requires': ['caption'],
        'output_type': 'motion',
        'num_person': 1,
    },
    't2m_2p': {
        'desc': 'Text → two-person motion',
        'requires': ['caption'],
        'output_type': 'motion',
        'num_person': 2,
    },
    'm2t_1p': {
        'desc': 'Single-person motion → text',
        'requires': ['motion'],
        'output_type': 'caption',
    },
    'm2t_2p': {
        'desc': 'Two-person motion → text',
        'requires': ['motion'],
        'output_type': 'caption',
    },
    'm2d': {
        'desc': 'Motion → dance music',
        'requires': ['motion'],
        'output_type': 'audio',
    },
    'd2m': {
        'desc': 'Music → dance motion',
        'requires': ['music'],
        'output_type': 'motion',
    },
    's2g': {
        'desc': 'Speech → gesture motion',
        'requires': ['audio'],
        'output_type': 'motion',
    },
    'pred': {
        'desc': 'Past motion → future motion prediction',
        'requires': ['past_motion'],
        'output_type': 'motion',
    },
    'inbetween': {
        'desc': 'Past + future motion → interpolation',
        'requires': ['past_motion', 'future_motion'],
        'output_type': 'motion',
    },
}

# Sample captions for T2M tasks
SAMPLE_CAPTIONS = [
    'a person walks forward slowly',
    'a person raises both hands above their head',
    'a person sits down on a chair',
    'a person kicks with the right leg',
    'a person waves goodbye with the right hand',
    'a person turns around in a circle',
    'a person jumps up and lands',
    'a person stretches their arms out to the sides',
    'a person bends down to pick something up',
    'a person does a squat',
    'a person walks in a zigzag pattern',
    'a person claps their hands together',
    'a person shrugs their shoulders',
    'a person points forward with the right hand',
    'a person takes a step back',
    'a person leans to the left',
    'a person puts hands on hips',
    'a person walks forward and then stops',
    'a person punches the air with the left fist',
    'a person stands on one leg',
    'two people shake hands',
    'two people walk side by side',
    'two people face each other and bow',
    'two people high five each other',
    'two people dance together in a circle',
]

# -----------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------


# Task → annotation file mapping (test splits preferred for evaluation)
TASK_ANNO_FILES = {
    't2m_1p': 'data/annotation/test_motionhub_t2m.json',
    't2m_2p': 'data/annotation/test_motionhub_2p.json',
    'm2t_1p': 'data/annotation/test_motionhub_t2m.json',
    'm2t_2p': 'data/annotation/test_motionhub_2p.json',
    'm2d': 'data/annotation/test_motionhub_m2d.json',
    'd2m': 'data/annotation/test_motionhub_m2d.json',
    's2g': 'data/annotation/test_motionhub_s2g.json',
    'pred': 'data/annotation/test_motionhub_pred.json',
    'inbetween': 'data/annotation/test_motionhub_pred.json',
}


def _load_caption_from_file(caption_path: str) -> Optional[str]:
    """Load a short caption from a hierarchical caption JSON file."""
    if not os.path.exists(caption_path):
        return None
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            cap_data = json.load(f)
        # Format: {"macro": ["cap1", ...], ...} or {"result": [{"short_caption": ...}]}
        if isinstance(cap_data, dict):
            if 'macro' in cap_data and cap_data['macro']:
                return cap_data['macro'][0]
            if 'result' in cap_data and cap_data['result']:
                first = cap_data['result'][0]
                if isinstance(first, dict) and 'short_caption' in first:
                    return first['short_caption']
        return None
    except Exception:
        return None


def load_test_data_list(
    task: str = 't2m_1p',
    data_dir: str = 'data/motionhub',
    max_samples: int = 100,
) -> List[Dict[str, Any]]:
    """Load annotation entries for test data.

    Uses task-specific annotation files. Handles the dict-based annotation
    format: {'meta_info': ..., 'data_list': {key: entry_dict}}.

    Returns list of dicts with keys: motion_path, caption, audio_path, etc.
    """
    anno_file = TASK_ANNO_FILES.get(task, 'data/annotation/test_motionhub_t2m.json')

    if not os.path.exists(anno_file):
        print(f'Warning: annotation file {anno_file} not found, using synthetic data')
        return []

    with open(anno_file, 'r') as f:
        raw = json.load(f)

    # Handle both dict-based and list-based formats
    if isinstance(raw, dict) and 'data_list' in raw:
        data_list = raw['data_list']
        if isinstance(data_list, dict):
            annotations = list(data_list.values())
        else:
            annotations = data_list
    elif isinstance(raw, list):
        annotations = raw
    else:
        print(f'Warning: unrecognized annotation format in {anno_file}')
        return []

    # Take a deterministic subset
    if len(annotations) > max_samples:
        step = max(1, len(annotations) // max_samples)
        annotations = annotations[::step][:max_samples]

    entries = []
    for ann in annotations:
        entry = {}
        # Motion path (smplx_path can be str or list for 2-person)
        if 'smplx_path' in ann:
            paths = ann['smplx_path']
            if isinstance(paths, list):
                entry['motion_path'] = os.path.join(data_dir, paths[0])
                if len(paths) > 1:
                    entry['motion_path_2p'] = [
                        os.path.join(data_dir, p) for p in paths[:2]
                    ]
            elif isinstance(paths, str):
                entry['motion_path'] = os.path.join(data_dir, paths)
        # Caption from hierarchical caption file
        if 'hierarchical_caption_path' in ann:
            cap_path = os.path.join(data_dir, ann['hierarchical_caption_path'])
            caption = _load_caption_from_file(cap_path)
            if caption:
                entry['caption'] = caption
        # Audio / music paths
        if 'music_path' in ann and ann['music_path']:
            entry['audio_path'] = os.path.join(data_dir, ann['music_path'])
            entry['genre'] = ann.get('genre')
        if 'audio_path' in ann and ann['audio_path']:
            entry['audio_path'] = os.path.join(data_dir, ann['audio_path'])
        # Speech script
        if 'speech_script_path' in ann and ann['speech_script_path']:
            script_path = os.path.join(data_dir, ann['speech_script_path'])
            if os.path.exists(script_path):
                try:
                    with open(script_path, 'r') as f:
                        entry['speech_script'] = f.read().strip()
                except Exception:
                    pass
        # Duration
        if 'duration' in ann:
            entry['duration'] = float(ann['duration'])
        # Num person
        if 'num_person' in ann:
            entry['num_person'] = int(ann['num_person'])
        entries.append(entry)

    return entries


def get_task_inputs(
    task: str,
    sample_idx: int,
    entries: List[Dict[str, Any]],
    num_samples: int,
) -> Optional[Dict[str, Any]]:
    """Build input kwargs for a single inference call."""
    info = TASK_INFO[task]
    kwargs: Dict[str, Any] = {}

    entry = entries[sample_idx % len(entries)] if entries else {}

    if 'caption' in info['requires']:
        if entry.get('caption'):
            kwargs['caption'] = entry['caption']
        else:
            captions = SAMPLE_CAPTIONS
            kwargs['caption'] = captions[sample_idx % len(captions)]
        # Duration for T2M — use annotation duration if available, else default
        kwargs['duration'] = entry.get('duration', 5.0)

    if 'motion' in info['requires']:
        # For 2-person M2T, use the 2p motion paths
        if task == 'm2t_2p' and entry.get('motion_path_2p'):
            paths = entry['motion_path_2p']
            if all(os.path.exists(p) for p in paths):
                kwargs['motion'] = paths
            else:
                return None
        elif entry.get('motion_path') and os.path.exists(entry['motion_path']):
            kwargs['motion'] = entry['motion_path']
        else:
            return None  # Skip if no motion data

    if 'past_motion' in info['requires']:
        if entry.get('motion_path') and os.path.exists(entry['motion_path']):
            kwargs['past_motion'] = entry['motion_path']
        else:
            return None

    if 'future_motion' in info['requires']:
        # For inbetween: past_motion and future_motion must be different segments
        # of the same motion.  The training pipeline (SplitInbetween) uses
        # past_ratio=0.2, future_ratio=0.2, so we replicate that here.
        motion_path = entry.get('motion_path', '')
        if motion_path and os.path.exists(motion_path):
            kwargs['future_motion'] = motion_path
            # Mark for splitting before inference
            kwargs['_needs_inbetween_split'] = True
        else:
            return None

    if 'music' in info['requires']:
        if entry.get('audio_path') and os.path.exists(entry.get('audio_path', '')):
            kwargs['music'] = entry['audio_path']
            if entry.get('genre'):
                kwargs['genre'] = entry['genre']
        else:
            return None

    if 'audio' in info['requires']:
        if entry.get('audio_path') and os.path.exists(entry.get('audio_path', '')):
            kwargs['audio'] = entry['audio_path']
            if entry.get('speech_script'):
                kwargs['speech_script'] = entry['speech_script']
        else:
            return None

    if 'num_person' in info:
        kwargs['num_person'] = info['num_person']

    return kwargs


# -----------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------


def run_batch_inference(
    model_name: str,
    config_path: str,
    checkpoint_path: str,
    tasks: List[str],
    num_samples: int,
    output_base: str,
    max_new_tokens: int = 8192,
    device: str = 'cuda',
):
    """Run inference for one model across multiple tasks."""
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.pipelines.motion.vermo_pipeline import VermoPipeline

    print(f'\n{"="*60}')
    print(f'Loading model: {model_name}')
    print(f'  Config:     {config_path}')
    print(f'  Checkpoint: {checkpoint_path}')
    print(f'{"="*60}')

    # Build bundle
    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    bundle.eval()

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = load_checkpoint(checkpoint_path, map_location='cpu')
        bundle.load_state_dict_selective(state_dict)
        del state_dict
        print(f'  Checkpoint loaded.')
    else:
        print(f'  Warning: checkpoint not found at {checkpoint_path}, using random weights')

    bundle = bundle.to(device)
    pipeline = VermoPipeline(bundle=bundle)

    for task in tasks:
        # Load task-specific test data
        entries = load_test_data_list(task=task, max_samples=num_samples)
        task_dir = os.path.join(output_base, model_name, task)
        os.makedirs(task_dir, exist_ok=True)

        print(f'\n  Task: {task} ({TASK_INFO[task]["desc"]})')
        print(f'  Output: {task_dir}')

        success = 0
        errors = 0
        start_time = time.time()

        for i in range(num_samples):
            kwargs = get_task_inputs(task, i, entries, num_samples)
            if kwargs is None:
                continue

            try:
                # For inbetween, split the full motion into past/future segments
                # Training uses SplitInbetween with past_ratio=0.2, future_ratio=0.2
                tmp_files = []
                if kwargs.pop('_needs_inbetween_split', False):
                    src_path = kwargs['past_motion']  # same as future_motion
                    npz_data = dict(np.load(src_path, allow_pickle=True))
                    T = npz_data['poses'].shape[0]
                    past_end = max(4, int(T * 0.2))
                    future_start = min(T - 4, int(T * 0.8))

                    past_npz = {}
                    future_npz = {}
                    for k, v in npz_data.items():
                        if hasattr(v, 'shape') and v.ndim >= 1 and v.shape[0] == T:
                            past_npz[k] = v[:past_end]
                            future_npz[k] = v[future_start:]
                        else:
                            past_npz[k] = v
                            future_npz[k] = v

                    past_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
                    future_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
                    np.savez(past_tmp.name, **past_npz)
                    np.savez(future_tmp.name, **future_npz)
                    tmp_files = [past_tmp.name, future_tmp.name]
                    kwargs['past_motion'] = past_tmp.name
                    kwargs['future_motion'] = future_tmp.name

                output = pipeline(
                    task=task,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    **kwargs,
                )

                # Save output based on type
                saved = False
                for key, value in output.items():
                    modal_name = getattr(key, 'name', None)
                    if modal_name is None:
                        continue

                    if modal_name in {'motion', 'middle_motion', 'future_motion'} and isinstance(value, dict):
                        out_path = os.path.join(task_dir, f'sample_{i:04d}.npz')
                        bundle.processor.smpl_pose_processor.save_smplx_npz(out_path, value)
                        saved = True
                        break
                    elif modal_name == 'caption' and isinstance(value, str):
                        out_path = os.path.join(task_dir, f'sample_{i:04d}.txt')
                        with open(out_path, 'w', encoding='utf-8') as f:
                            f.write(value)
                        saved = True
                        break
                    elif modal_name in {'audio', 'music'} and isinstance(value, torch.Tensor):
                        out_path = os.path.join(task_dir, f'sample_{i:04d}.wav')
                        wav = value.detach().cpu().float()
                        if wav.ndim == 1:
                            wav = wav.unsqueeze(0)  # (C, T)
                        # Use soundfile backend to avoid torchaudio FFmpeg
                        # channel_layout=0x0 crash
                        try:
                            import soundfile as sf
                            # soundfile expects (T, C)
                            sf.write(out_path, wav.T.numpy(), 24000)
                        except ImportError:
                            import torchaudio
                            torchaudio.save(
                                out_path, wav, 24000,
                                backend='soundfile',
                            )
                        saved = True
                        break

                if not saved:
                    # Save raw response
                    response = output.get('response', str(output))
                    out_path = os.path.join(task_dir, f'sample_{i:04d}.txt')
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(str(response))

                success += 1
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f'    [{i+1}/{num_samples}] {rate:.1f} samples/s, {success} ok, {errors} err')

            except Exception as e:
                errors += 1
                err_path = os.path.join(task_dir, f'sample_{i:04d}_error.txt')
                with open(err_path, 'w') as f:
                    f.write(f'{type(e).__name__}: {e}')
                if errors <= 3:
                    print(f'    Warning: sample {i} failed: {e}')
            finally:
                # Clean up inbetween temp files
                for tmp in tmp_files:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

        elapsed = time.time() - start_time
        print(f'  Done: {success}/{num_samples} ok, {errors} errors, {elapsed:.1f}s')

    # Cleanup
    del pipeline, bundle
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='Batch inference for VerMo')
    parser.add_argument(
        '--models', nargs='+', default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help='Which models to evaluate',
    )
    parser.add_argument(
        '--tasks', nargs='+', default=list(TASK_INFO.keys()),
        choices=list(TASK_INFO.keys()),
        help='Which tasks to evaluate',
    )
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='Number of samples per task',
    )
    parser.add_argument(
        '--output-dir', type=str, default='work_dirs/vermo_eval',
        help='Output directory',
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=8192,
        help='Max new tokens for generation',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use',
    )
    # Custom checkpoint overrides
    parser.add_argument('--qwen-config', type=str, default=None)
    parser.add_argument('--qwen-ckpt', type=str, default=None)
    parser.add_argument('--llama-config', type=str, default=None)
    parser.add_argument('--llama-ckpt', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply checkpoint overrides
    if args.qwen_config:
        MODELS['qwen1.7b']['config'] = args.qwen_config
    if args.qwen_ckpt:
        MODELS['qwen1.7b']['checkpoint'] = args.qwen_ckpt
    if args.llama_config:
        MODELS['llama1b']['config'] = args.llama_config
    if args.llama_ckpt:
        MODELS['llama1b']['checkpoint'] = args.llama_ckpt

    print(f'VerMo Batch Inference')
    print(f'  Models:  {args.models}')
    print(f'  Tasks:   {args.tasks}')
    print(f'  Samples: {args.num_samples} per task')
    print(f'  Output:  {args.output_dir}')

    for model_name in args.models:
        model_info = MODELS[model_name]
        run_batch_inference(
            model_name=model_name,
            config_path=model_info['config'],
            checkpoint_path=model_info['checkpoint'],
            tasks=args.tasks,
            num_samples=args.num_samples,
            output_base=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

    print(f'\nAll done! Results in {args.output_dir}/')


if __name__ == '__main__':
    main()

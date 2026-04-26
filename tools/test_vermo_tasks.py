#!/usr/bin/env python3
"""Test all VerMo tasks with a given config + checkpoint.

Usage:
    python tools/test_vermo_tasks.py \
        --config configs/vermo/vermo_sft_16k_llama1b_wavtokenizer.py \
        --checkpoint work_dirs/vermo_sft_16k_llama1b_wavtokenizer/iter_44000.pth
"""

import argparse
import os
import sys
import time
import traceback

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-import datasets to avoid shadowing
try:
    import datasets as _hf_datasets  # noqa: F401
except ImportError:
    pass


def load_bundle(config_path, checkpoint_path, device='cuda'):
    """Build bundle from config and load checkpoint weights."""
    import torch
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()

    print(f'Building bundle from {config_path}...')
    t0 = time.time()
    bundle = MODEL_BUNDLES.build(model_cfg)
    print(f'  Built in {time.time() - t0:.1f}s')

    print(f'Loading checkpoint from {checkpoint_path}...')
    t0 = time.time()
    state_dict = load_checkpoint(checkpoint_path)
    bundle.load_state_dict_selective(state_dict, strict=False)
    print(f'  Loaded in {time.time() - t0:.1f}s')

    # Materialize any remaining meta tensors before moving to device
    for name, param in bundle.named_parameters():
        if param.device == torch.device('meta'):
            materialized = torch.zeros(param.shape, dtype=param.dtype)
            parts = name.rsplit('.', 1)
            parent = bundle
            for attr in parts[0].split('.'):
                parent = getattr(parent, attr)
            setattr(parent, parts[1], torch.nn.Parameter(materialized, requires_grad=param.requires_grad))
    for name, buf in bundle.named_buffers():
        if buf.device == torch.device('meta'):
            materialized = torch.zeros(buf.shape, dtype=buf.dtype)
            parts = name.rsplit('.', 1)
            parent = bundle
            for attr in parts[0].split('.'):
                parent = getattr(parent, attr)
            setattr(parent, parts[1], materialized)

    bundle = bundle.to(device)
    bundle.eval()
    return bundle


def find_test_data():
    """Find test motion and audio files from the dataset."""
    import json

    data_dir = 'data/motionhub'
    anno_path = 'data/annotation/train_audio_motionhub_hymotion.json'

    test_data = {
        'motion_npz': None,  # single-person motion
        'music_wav': None,   # music for dance
        'audio_wav': None,   # speech audio
    }

    if os.path.exists(anno_path):
        with open(anno_path) as f:
            anno = json.load(f)
        data_list = anno.get('data_list', anno)

        for key, item in (data_list.items() if isinstance(data_list, dict) else enumerate(data_list)):
            if isinstance(item, dict):
                # Find a music+motion sample (AIST)
                if test_data['music_wav'] is None and 'music_path' in item:
                    music_path = os.path.join(data_dir, item['music_path'])
                    motion_path = os.path.join(data_dir, item.get('smplx_path', ''))
                    if os.path.exists(music_path) and os.path.exists(motion_path):
                        test_data['music_wav'] = music_path
                        test_data['motion_npz'] = motion_path

                # Find a speech sample (beat/ted)
                if test_data['audio_wav'] is None and 'audio_path' in item:
                    audio_path = os.path.join(data_dir, item['audio_path'])
                    if os.path.exists(audio_path):
                        test_data['audio_wav'] = audio_path

                if all(v is not None for v in test_data.values()):
                    break

    return test_data


def test_task(pipeline, task, test_data, output_dir, max_new_tokens=200, **extra_kwargs):
    """Run a single task and report the result."""
    import torch

    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, **extra_kwargs)
    print(f'\n{"="*60}')
    print(f'Task: {task}')
    print(f'{"="*60}')

    try:
        t0 = time.time()
        if task == 't2m_1p':
            output = pipeline(
                task=task,
                caption='A person walks forward and then turns around.',
                duration=4.0,
                **kwargs,
            )
        elif task == 't2m_2p':
            output = pipeline(
                task=task,
                caption='Two people shake hands with each other.',
                num_person=2,
                duration=4.0,
                **kwargs,
            )
        elif task == 'm2t_1p':
            if test_data['motion_npz'] is None:
                print('  SKIP: No motion data available')
                return 'skip'
            output = pipeline(
                task=task,
                motion=test_data['motion_npz'],
                **kwargs,
            )
        elif task == 'm2t_2p':
            print('  SKIP: Need 2-person motion data (not available in test set)')
            return 'skip'
        elif task == 'm2d':
            if test_data['music_wav'] is None:
                print('  SKIP: No music data available')
                return 'skip'
            output = pipeline(
                task=task,
                music=test_data['music_wav'],
                duration=4.0,
                **kwargs,
            )
        elif task == 'd2m':
            if test_data['motion_npz'] is None:
                print('  SKIP: No motion data available')
                return 'skip'
            output = pipeline(
                task=task,
                motion=test_data['motion_npz'],
                **kwargs,
            )
        elif task == 's2g':
            if test_data['audio_wav'] is None:
                print('  SKIP: No speech audio available')
                return 'skip'
            output = pipeline(
                task=task,
                audio=test_data['audio_wav'],
                speech_script='Hello, how are you today?',
                duration=4.0,
                **kwargs,
            )
        elif task == 'pred':
            if test_data['motion_npz'] is None:
                print('  SKIP: No motion data available')
                return 'skip'
            output = pipeline(
                task=task,
                past_motion=test_data['motion_npz'],
                duration=4.0,
                **kwargs,
            )
        elif task == 'inbetween':
            if test_data['motion_npz'] is None:
                print('  SKIP: No motion data available')
                return 'skip'
            output = pipeline(
                task=task,
                past_motion=test_data['motion_npz'],
                future_motion=test_data['motion_npz'],
                duration=4.0,
                **kwargs,
            )
        else:
            print(f'  SKIP: Unknown task {task}')
            return 'skip'

        elapsed = time.time() - t0
        print(f'  Time: {elapsed:.1f}s')

        # Analyze output
        if isinstance(output, dict):
            for k, v in output.items():
                modal_name = getattr(k, 'name', str(k))
                if isinstance(v, dict):
                    v_info = {kk: (vv.shape if hasattr(vv, 'shape') else type(vv).__name__) for kk, vv in v.items()}
                    print(f'  Output[{modal_name}]: dict with {v_info}')
                elif isinstance(v, str):
                    print(f'  Output[{modal_name}]: "{v[:200]}"')
                elif hasattr(v, 'shape'):
                    print(f'  Output[{modal_name}]: tensor {v.shape}')
                else:
                    print(f'  Output[{modal_name}]: {type(v).__name__}')
        else:
            print(f'  Output: {type(output).__name__}')
            if isinstance(output, str):
                print(f'  Content: "{output[:200]}"')

        # Save output
        saved = False
        if isinstance(output, dict):
            for k, v in output.items():
                modal_name = getattr(k, 'name', None)
                if modal_name in {'motion', 'middle_motion', 'future_motion'} and isinstance(v, dict):
                    out_path = os.path.join(output_dir, f'{task}_motion.npz')
                    try:
                        pipeline.bundle.processor.smpl_pose_processor.save_smplx_npz(out_path, v)
                        print(f'  Saved motion to: {out_path}')
                        saved = True
                    except Exception as e:
                        print(f'  Save failed: {e}')
                    break
                elif modal_name == 'caption' and isinstance(v, str):
                    out_path = os.path.join(output_dir, f'{task}_caption.txt')
                    with open(out_path, 'w') as f:
                        f.write(v)
                    print(f'  Saved caption to: {out_path}')
                    saved = True
                    break

        if not saved and isinstance(output, dict):
            resp = output.get('response', str(output))
            out_path = os.path.join(output_dir, f'{task}_response.txt')
            with open(out_path, 'w') as f:
                f.write(str(resp))
            print(f'  Saved response to: {out_path}')

        print(f'  PASS')
        return 'pass'

    except Exception as e:
        print(f'  FAIL: {e}')
        traceback.print_exc()
        return 'fail'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific tasks to test (default: all)')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--output-dir', default='work_dirs/vermo_test_outputs')
    args = parser.parse_args()

    import torch

    all_tasks = ['t2m_1p', 't2m_2p', 'm2t_1p', 'm2d', 'd2m', 's2g', 'pred', 'inbetween']
    tasks = args.tasks or all_tasks

    # Find test data
    print('Finding test data...')
    test_data = find_test_data()
    for k, v in test_data.items():
        print(f'  {k}: {v}')

    # Determine model name from config path
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = os.path.join(args.output_dir, config_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load bundle
    bundle = load_bundle(args.config, args.checkpoint, device=args.device)

    # Create pipeline
    from hftrainer.pipelines.motion.vermo_pipeline import VermoPipeline
    pipeline = VermoPipeline(bundle=bundle)
    pipeline.bundle = bundle  # for save_smplx_npz access

    # Run tasks
    results = {}
    for task in tasks:
        with torch.no_grad():
            results[task] = test_task(
                pipeline, task, test_data, output_dir,
                max_new_tokens=args.max_new_tokens,
            )

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY: {config_name}')
    print(f'{"="*60}')
    for task, result in results.items():
        emoji = {'pass': 'PASS', 'fail': 'FAIL', 'skip': 'SKIP'}[result]
        print(f'  {task:15s}  {emoji}')

    passed = sum(1 for r in results.values() if r == 'pass')
    failed = sum(1 for r in results.values() if r == 'fail')
    skipped = sum(1 for r in results.values() if r == 'skip')
    print(f'\nTotal: {passed} passed, {failed} failed, {skipped} skipped')


if __name__ == '__main__':
    main()

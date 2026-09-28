"""One-command real-weight execution checks; visual acceptance remains manual.

Run from any directory with the prepared environment. Existing feature caches
may only be reused explicitly; training always uses a fresh output directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', choices=['all', 'h3', 'ltx'], default='all')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data/video_validation_20260928')
    p.add_argument('--reuse-features', action='store_true')
    p.add_argument('--resume', action='store_true',
                   help='Reuse successful stages in this output directory; rerun failed/pending stages')
    args = p.parse_args()
    output, data = args.output.resolve(), args.data.resolve()
    inputs = dict(models=args.models, data=str(data), reuse_features=args.reuse_features)
    if args.resume:
        report = json.loads((output / 'status.json').read_text())
        if report.get('inputs') != inputs:
            raise ValueError('Resume requires identical models/data/cache options to the original run')
        report['execution'] = 'RUNNING'
        report.pop('error', None)
    else:
        output.mkdir(parents=True, exist_ok=False)
        report = dict(execution='RUNNING', visual_quality='NOT_ACCEPTED', stages=[], inputs=inputs)
    os.chdir(ROOT)
    env = dict(os.environ, OMP_NUM_THREADS='8')
    (output / 'status.json').write_text(json.dumps(report, indent=2))

    def run(name, command, extra=None):
        previous = [s for s in report['stages'] if s['name'] == name]
        if args.resume and previous and previous[-1]['returncode'] == 0:
            print(f'Reusing successful stage {name}', flush=True)
            return
        log_name = name + (f'_attempt{len(previous) + 1}' if previous else '') + '.log'
        print(f'Running {name}: {output / log_name}', flush=True)
        with (output / log_name).open('w') as log:
            result = subprocess.run([sys.executable, '-u', *map(str, command)],
                                    env=dict(env, **(extra or {})), stdout=log, stderr=subprocess.STDOUT)
        report['stages'].append(dict(name=name, returncode=result.returncode, log=log_name))
        (output / 'status.json').write_text(json.dumps(report, indent=2))
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {output / log_name}')

    try:
        if not args.reuse_features:
            run('prepare', ['-m', 'tools.validation.prepare_video_subset', '--output', data])
        if args.models in ('all', 'h3'):
            h3root = env.get('MINIMAX_H3_ROOT', str(ROOT / 'checkpoints/MiniMax-H3'))
            if not args.reuse_features:
                run('h3_preprocess', ['-m', 'tools.validation.preprocess_h3', '--model-root', h3root,
                                     '--manifest', data / 'all.jsonl', '--output', data / 'h3'])
            for name in ('train.jsonl', 'validation.jsonl', '02.safetensors'):
                if not (data / 'h3' / name).is_file():
                    raise FileNotFoundError(data / 'h3' / name)
            h3env = dict(MINIMAX_H3_ROOT=h3root, MINIMAX_H3_CACHE_MANIFEST=str(data / 'h3/train.jsonl'),
                         HFTRAINER_WORK_DIR=str(output / 'h3_train'))
            cmd = ['tools/train.py', 'configs/minimax_h3/validate_h3_local.py']
            val = f'val_dataloader.dataset.manifest={data / "h3/validation.jsonl"}'
            run('h3_train', [*cmd, '--cfg-options', val], h3env)
            run('h3_resume', [*cmd, '--auto-resume', '--cfg-options', val, 'train_cfg.max_iters=3'], h3env)
            for variant in ('base', 'adapter'):
                cmd = ['-m', 'tools.validation.infer_h3', '--feature', data / 'h3/02.safetensors',
                       '--output', output / f'h3_{variant}.mp4', '--size', '768', '--steps', '50']
                if variant == 'adapter':
                    cmd += ['--checkpoint', output / 'h3_train/checkpoint-iter_3']
                run(f'h3_{variant}', cmd, h3env)
        if args.models in ('all', 'ltx'):
            ltxroot = Path(env.get('LTX25_CHECKPOINT_ROOT', str(ROOT / 'checkpoints/LTX-2.5')))
            ltxenv = dict(LTX25_CHECKPOINT_ROOT=str(ltxroot), LTX25_PREPROCESSED_DATA=str(data / 'ltx'),
                          HFTRAINER_WORK_DIR=str(output / 'ltx_train'))
            if not args.reuse_features:
                run('ltx_preprocess', ['tools/preprocess_ltx_video.py', data / 'train.jsonl',
                    '--resolution-buckets', '256x256x49',
                    '--model-path', ltxroot / 'diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors',
                    '--text-encoder-path', ltxroot / 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
                    '--video-vae-path', ltxroot / 'vae/ltx-2.5-video-vae-bf16.safetensors',
                    '--audio-vae-path', ltxroot / 'vae/ltx-2.5-audio-vae-bf16.safetensors',
                    '--output-dir', data / 'ltx', '--overwrite'], ltxenv)
            for name in ('conditions', 'latents', 'audio_latents'):
                if not any((data / 'ltx' / name).rglob('*.pt')):
                    raise FileNotFoundError(data / 'ltx' / name)
            cmd = ['tools/train.py', 'configs/ltx_video/validate_ltx_2_5_local.py']
            run('ltx_train', cmd, ltxenv)
            run('ltx_resume', [*cmd, '--auto-resume', '--cfg-options', 'trainer.native_config.optimization.steps=3'], ltxenv)
            ltxenv['LTX25_USER_LORA'] = str(output / 'ltx_train/checkpoints/lora_weights_step_00003.safetensors')
            run('ltx_infer', ['tools/infer_ltx_video.py', 'configs/ltx_video/infer_ltx_video_2_5_dev_lora.py',
                '--prompt', 'A dancer performs a dance routine with music.', '--output', output / 'ltx_adapter.mp4',
                '--height', '512', '--width', '768', '--num-frames', '49', '--num-inference-steps', '30',
                '--cfg-options', 'model.duration_head_path=None'], ltxenv)
        report['execution'] = 'PASSED'
    except Exception as error:
        report['execution'] = 'FAILED'
        report['error'] = str(error)
        raise
    finally:
        (output / 'status.json').write_text(json.dumps(report, indent=2))
    print('Execution checks completed. Visual quality is NOT accepted automatically.', flush=True)


if __name__ == '__main__':
    main()

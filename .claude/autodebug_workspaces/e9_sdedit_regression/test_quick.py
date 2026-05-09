"""Fast smoke test: run evaluate_sample on a single short-T E9 C_full_inpaint sample
for a given checkpoint, return jitter_pos.

Hypothesis to check: E9 agg jitter jumped from 353 (checkpoint ~epoch_657) to 5835
(checkpoint epoch_846) for uncond_global C_full_inpaint. User ruled out
sliding-window. Per-sample breakdown shows short samples also degraded.
If jitter varies wildly across different checkpoints of the SAME model with the
SAME inference code, training itself regressed (not an inference bug).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def run_one(model_name, checkpoint_path, task_id='E9', setting='C_full_inpaint',
            sample_idx=0, device='cuda'):
    from tools.eval_m2m_v2_all_tasks import (
        load_eval_samples, evaluate_sample,
        ALL_MODELS,
    )
    from hftrainer.evaluation.motion.m2m_eval_tasks import get_task
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    task = get_task(task_id)
    model_info = dict(ALL_MODELS[model_name])

    # Build bundle + load the specific checkpoint (not latest)
    cfg = Config.fromfile(model_info['config'])
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    print(f'  Loading checkpoint: {checkpoint_path}')
    sd = load_checkpoint(checkpoint_path, map_location='cpu')
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval()
    bundle = bundle.to(device)
    pipeline = HyMotionM2MPipeline(
        bundle=bundle, num_steps=50, replacement_guidance='none'
    )

    bone_offsets = torch.load(
        'data/hymotion_m2m_data/bone_offsets_22.pt',
        map_location='cpu').numpy()

    data_file = str(PROJECT_ROOT / 'data' / 'eval' / 'm2m_v2' / task.data_file)
    convert_198 = (model_info.get('motion_dim', 198) == 198)
    samples = load_eval_samples(
        data_file, 'data/hymotion_data',
        max_samples=sample_idx + 1,
        bone_offsets=bone_offsets,
        convert_to_198=convert_198,
        task_id='E9',
    )
    if sample_idx >= len(samples):
        raise RuntimeError(f'Not enough samples: asked {sample_idx}, got {len(samples)}')
    sample = samples[sample_idx]

    # Fix RNG so different ckpts get the same init noise
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    metrics, output_135 = evaluate_sample(
        bundle, pipeline, sample, task, setting,
        model_info, bone_offsets, device,
        replacement_guidance='skip_last',
        text_guidance_scale=1.0,
        num_steps=50,
    )
    T = sample['T']
    j = metrics.get('jitter_pos', float('nan'))
    print(f'  [{Path(checkpoint_path).name}] T={T}, jitter_pos={j:.2f}')
    # Free
    del bundle, pipeline
    torch.cuda.empty_cache()
    return {
        'checkpoint': Path(checkpoint_path).name,
        'T': T,
        'metrics': {k: float(v) for k, v in metrics.items()
                    if not k.startswith('_') and isinstance(v, (int, float))},
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='uncond_global')
    p.add_argument('--setting', default='C_full_inpaint')
    p.add_argument('--sample-idx', type=int, default=0)
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='Full checkpoint paths (will skip missing)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default='/tmp/e9_debug_jitter_vs_ckpt.json')
    args = p.parse_args()
    results = []
    for ckpt in args.checkpoints:
        if not Path(ckpt).is_dir():
            print(f'SKIP (not found): {ckpt}')
            continue
        r = run_one(args.model, ckpt, task_id='E9',
                    setting=args.setting, sample_idx=args.sample_idx,
                    device=args.device)
        results.append(r)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {args.output}')
    print('\n=== Jitter vs Checkpoint ===')
    for r in results:
        j = r['metrics'].get('jitter_pos', 'N/A')
        print(f'  {r["checkpoint"]}: T={r["T"]}, jitter_pos={j}')

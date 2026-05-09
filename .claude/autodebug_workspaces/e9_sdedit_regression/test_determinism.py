"""Deterministic ablation: run E9/C_full_inpaint single sample with fixed seed,
do it 3 times WITHIN SAME process, and also fresh processes, to separate
randomness vs. stateful leakage.
"""
import sys
import os
from pathlib import Path
HF = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(HF))
os.chdir(HF)

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from tools.eval_m2m_v2_all_tasks import (
    load_eval_samples, evaluate_sample, ALL_MODELS,
)
from hftrainer.evaluation.motion.m2m_eval_tasks import get_task
from mmengine.config import Config
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

MODEL = 'uncond_global'
SETTING = 'C_full_inpaint'
TASK_ID = 'E9'
SAMPLE_IDX = 0

device = 'cuda'

model_info = dict(ALL_MODELS[MODEL])
cfg = Config.fromfile(model_info['config'])
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

# find latest ckpt
ckpt_root = Path(model_info['work_dir'])
ckpts = sorted([p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith('checkpoint-epoch_')],
               key=lambda p: int(p.name.split('_')[-1]))
latest = ckpts[-1]
print(f'Using ckpt: {latest}')
sd = load_checkpoint(str(latest), map_location='cpu')
bundle.load_state_dict_selective(sd)
del sd
bundle.eval().to(device)

pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=50, replacement_guidance='skip_last')

bone_offsets = torch.load(HF / 'data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu').numpy()

task = get_task(TASK_ID)
data_file = str(HF / 'data/eval/m2m_v2' / task.data_file)
samples = load_eval_samples(
    data_file, 'data/hymotion_data',
    max_samples=SAMPLE_IDX + 1,
    bone_offsets=bone_offsets,
    convert_to_198=True, task_id='E9',
)
sample = samples[SAMPLE_IDX]
print(f'Sample T={sample["T"]}')

print()
print('=== Run same sample 3 times WITHOUT seeding ===')
for i in range(3):
    metrics, _ = evaluate_sample(
        bundle, pipeline, sample, task, SETTING,
        model_info, bone_offsets, device,
        replacement_guidance='skip_last',
        text_guidance_scale=1.0, num_steps=50,
    )
    print(f'  run {i}: jitter={metrics["jitter_pos"]:.1f}')

print()
print('=== Run same sample 3 times WITH seed=42 before each ===')
for i in range(3):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    metrics, _ = evaluate_sample(
        bundle, pipeline, sample, task, SETTING,
        model_info, bone_offsets, device,
        replacement_guidance='skip_last',
        text_guidance_scale=1.0, num_steps=50,
    )
    print(f'  run {i}: jitter={metrics["jitter_pos"]:.1f}')

print()
print('=== seeded runs with 3 different seeds ===')
for s in [42, 123, 456]:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    metrics, _ = evaluate_sample(
        bundle, pipeline, sample, task, SETTING,
        model_info, bone_offsets, device,
        replacement_guidance='skip_last',
        text_guidance_scale=1.0, num_steps=50,
    )
    print(f'  seed={s}: jitter={metrics["jitter_pos"]:.1f}')

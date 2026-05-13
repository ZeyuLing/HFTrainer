#!/usr/bin/env python3
"""Quick T2M inference verification: check output quality after 360-frame padding + ground alignment fixes."""
import sys
import os
import time
import numpy as np
import torch

# Setup path
sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
os.chdir('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

from mmengine.config import Config

print("="*60)
print("T2M Inference Verification")
print("="*60)

# Step 1: Load config
config_path = 'configs/hymotion_t2m/hymotion_t2m_201dim_046b.py'
print(f"\n[1] Loading config: {config_path}")
cfg = Config.fromfile(config_path)
print(f"    motion_dim = {cfg.model.motion_transformer.output_dim}")
print(f"    pred_type = {cfg.model.pred_type}")
print(f"    mean_std_dir = {cfg.model.mean_std_dir}")
print(f"    noise_scheduler = {cfg.model.noise_scheduler_cfg}")

# Step 1.5: Inject text encoder config (training config has text_encoder=dict() which is falsy)
if not cfg.model.get('text_encoder'):
    cfg.model.text_encoder = dict(
        type='HYTextModel',
        llm_type='qwen3',
        max_length_llm=128,
    )
    print(f"    Injected text_encoder config: {cfg.model.text_encoder}")

# Step 2+3: Build bundle and load checkpoint using proper loading pipeline
ckpt_path = cfg.load_from
if isinstance(ckpt_path, dict):
    ckpt_path = ckpt_path['path']
print(f"\n[2] Building bundle + loading checkpoint: {ckpt_path}")

device = 'cuda:0'
from tools.infer import load_bundle_from_checkpoint
bundle = load_bundle_from_checkpoint(cfg, ckpt_path, device)

# Check mean/std loaded
print(f"\n    mean shape: {bundle.mean.shape}, range: [{bundle.mean.min():.4f}, {bundle.mean.max():.4f}]")
print(f"    std shape: {bundle.std.shape}, range: [{bundle.std.min():.4f}, {bundle.std.max():.4f}]")
print(f"    null_vtxt_feat norm: {bundle.null_vtxt_feat.norm():.6f}")
print(f"    null_ctxt_input norm: {bundle.null_ctxt_input.norm():.6f}")
print(f"    Model on {device}, dtype={next(bundle.motion_transformer.parameters()).dtype}")

# Step 3: Run inference with pipeline
print(f"\n[3] Running T2M inference...")
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,
    text_guidance_scale=5.0,
)

# Test prompts with different lengths
test_cases = [
    ("a person walks forward slowly", 120),
    ("a person jumps in place", 90),
    ("a person waves their right hand", 90),
    ("a person stands still", 60),
]

for prompt, num_frames in test_cases:
    print(f"\n  Prompt: '{prompt}' ({num_frames} frames)")
    t0 = time.time()
    
    batch = {
        'caption': [prompt],
        'tgt_length': [num_frames],
    }
    
    with torch.no_grad():
        result = pipeline(batch)
    
    dt = time.time() - t0
    
    # Check output
    latent = result['latent']
    rot6d = result.get('rot6d')
    transl = result.get('transl')
    k3d = result.get('keypoints3d')
    
    print(f"    Time: {dt:.1f}s")
    print(f"    latent shape: {latent.shape}, range: [{latent.min():.4f}, {latent.max():.4f}]")
    
    if rot6d is not None:
        print(f"    rot6d shape: {rot6d.shape}")
        # Check rot6d norm (should be ~1.0 for each 3D column)
        r6d = rot6d[0]  # (L, 22, 6)
        col1_norm = r6d[:, :, :3].norm(dim=-1).mean()
        col2_norm = r6d[:, :, 3:].norm(dim=-1).mean()
        print(f"    rot6d col norms: col1={col1_norm:.4f}, col2={col2_norm:.4f} (should be ~1.0)")
    
    if transl is not None:
        print(f"    transl shape: {transl.shape}")
        # Check root height (Y in SMPL = up direction)
        root_y = transl[0, :, 1]  # Y coordinate
        print(f"    root Y (height): mean={root_y.mean():.4f}, min={root_y.min():.4f}, max={root_y.max():.4f}")
        print(f"    root XZ range: X=[{transl[0,:,0].min():.3f}, {transl[0,:,0].max():.3f}], Z=[{transl[0,:,2].min():.3f}, {transl[0,:,2].max():.3f}]")
    
    if k3d is not None:
        print(f"    keypoints3d shape: {k3d.shape}")
        # Check heights of key joints
        pelvis_y = k3d[0, :, 0, 1].mean()
        head_y = k3d[0, :, 15, 1].mean()
        lfoot_y = k3d[0, :, 10, 1].mean()
        rfoot_y = k3d[0, :, 11, 1].mean()
        print(f"    Joint heights (Y mean): pelvis={pelvis_y:.3f}m, head={head_y:.3f}m, lfoot={lfoot_y:.3f}m, rfoot={rfoot_y:.3f}m")
        
        # Min Y should be ~0 (ground alignment)
        min_y = k3d[0, :, :, 1].min()
        print(f"    Ground alignment: min_y = {min_y:.4f}m (should be ~0)")
    else:
        print(f"    keypoints3d: None (SMPL body model not available)")
    
    # Save motion_135 for further testing
    latent_denorm = result.get('latent_denorm')
    if latent_denorm is not None:
        motion_135 = latent_denorm[0, :, :135].cpu().numpy()
        save_path = f'/tmp/t2m_verify_{prompt.replace(" ", "_")[:20]}_{num_frames}f.npy'
        np.save(save_path, motion_135)
        print(f"    Saved motion_135 ({motion_135.shape}) to {save_path}")

print(f"\n{'='*60}")
print("Verification complete!")
print("="*60)

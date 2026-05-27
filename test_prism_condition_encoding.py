"""Diagnostic script to compare training vs inference condition encoding."""

import torch
import json
import numpy as np
from pathlib import Path

# Load config and checkpoint
config_path = "work_dirs/prism_overfit_100/20260526_212303/config.py"
checkpoint_path = "work_dirs/prism_overfit_100/checkpoint-epoch_1174"
anno_path = "data/annotation/train_overfit_prism_100.json"

print("="*80)
print("DIAGNOSTIC: PRISM TRAINING vs INFERENCE DATA FLOW")
print("="*80)

# Load annotation
with open(anno_path) as f:
    anno = json.load(f)

print(f"\nLoaded {len(anno)} samples from {anno_path}")

# Use first sample
sample_key = anno[0]["motion_key"]
motion_path = Path("data/motionhub") / anno[0]["motion_path"]

print(f"Using sample: {sample_key}")
print(f"Motion path: {motion_path}")

if not motion_path.exists():
    print(f"ERROR: Motion path does not exist: {motion_path}")
    exit(1)

# Load raw motion data
motion_data = np.load(motion_path)
print(f"Loaded motion: {motion_data['motion_data'].shape}, dtype={motion_data['motion_data'].dtype}")

# Now simulate what happens during training vs inference
print("\n" + "="*80)
print("TRAINING FLOW:")
print("="*80)

# Training loads motion from dataset (LoadSmplx55 transform)
# The motion should be in axis-angle format from the dataset
raw_motion = motion_data['motion_data']  # [T, 132] or similar
print(f"1. Raw motion from dataset: {raw_motion.shape}")

# Then goes through LoadSmplx55, which outputs [T, J*6] in 6D rotation format
# This is what arrives at encode_motion in bundle.py
print(f"2. After LoadSmplx55 transform (expected to be rotation_6d format)")

# In bundle.encode_motion:
# - Input: [T, J*6] unnormalized motion
# - Step 1: normalize via smpl_pose_processor.normalize()
# - Step 2: encode via VAE
# - Step 3: normalize latents via (z - mean) / std

print("\n" + "="*80)
print("INFERENCE FLOW:")
print("="*80)

# In inference, load_condition_pose:
# - Load motion from NPZ
# - Convert to [B, T, J, 6]
# - normalize via smpl_processor.normalize() <- FIRST NORMALIZATION
# - Then pass to encode_motion

# In encode_motion:
# - Input: [B, T, J, 6] ALREADY NORMALIZED
# - Step 1: encode via VAE (expects unnormalized!)
# - Step 2: normalize latents via (z - mean) / std

print("1. Raw motion loaded from NPZ")
print("2. First normalization in load_condition_pose()")
print("3. Pass to encode_motion() which expects UNNORMALIZED motion!")
print("4. VAE encoder gets NORMALIZED motion (WRONG!)")
print("5. Normalize latents")

print("\n" + "="*80)
print("KEY DIFFERENCE:")
print("="*80)
print("TRAINING:   motion (unnormalized) -> encode_motion -> normalize in encode_motion -> VAE")
print("INFERENCE:  motion -> normalize (load_condition_pose) -> encode_motion -> VAE with NORMALIZED motion")
print("\nThe VAE is being fed NORMALIZED motion in inference but UNNORMALIZED in training!")


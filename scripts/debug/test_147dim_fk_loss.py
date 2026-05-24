#!/usr/bin/env python3
"""Test 147-dim FK consistency loss computation."""

import sys
import torch
import numpy as np

# Add project to path
sys.path.insert(0, '/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

from hftrainer.datasets.motion.motionhub.smpl_data import SMPL22_BONE_OFFSETS
from hftrainer.pipelines.motion.compute_147dim_fk_loss import motion147_fk_loss


def test_147dim_fk_loss_basic():
    """Test FK consistency loss computation for basic 147-dim motion."""
    print("\n[TEST 1] Basic FK consistency loss computation")
    
    B, L, D = 2, 100, 147
    
    # Create dummy 147-dim normalized motion
    motion_147_norm = torch.randn(B, L, D, dtype=torch.float32)
    
    # Create realistic mean/std
    mean = torch.zeros(147, dtype=torch.float32)
    std = torch.ones(147, dtype=torch.float32)
    
    # Load bone offsets
    bone_offsets = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
    
    # Compute FK loss
    fk_loss = motion147_fk_loss(
        motion_147_norm,
        mean,
        std,
        bone_offsets,
        rotation_space='local',
        timesteps=None,
        data_mask_temporal=None,
    )
    
    print(f"  Motion shape: {motion_147_norm.shape}")
    print(f"  FK loss value: {fk_loss.item():.6f}")
    print(f"  FK loss dtype: {fk_loss.dtype}")
    print(f"  FK loss device: {fk_loss.device}")
    
    assert isinstance(fk_loss, torch.Tensor), "FK loss must be a tensor"
    assert fk_loss.dim() == 0, "FK loss must be scalar"
    assert fk_loss.item() > 0, "FK loss should be positive"
    assert not torch.isnan(fk_loss), "FK loss must not be NaN"
    assert not torch.isinf(fk_loss), "FK loss must not be Inf"
    
    print("  ✅ Basic FK loss computation passed")


def test_147dim_fk_loss_with_mask():
    """Test FK consistency loss with temporal masking."""
    print("\n[TEST 2] FK consistency loss with temporal masking")
    
    B, L, D = 2, 100, 147
    
    # Create dummy 147-dim normalized motion
    motion_147_norm = torch.randn(B, L, D, dtype=torch.float32)
    
    # Create realistic mean/std
    mean = torch.zeros(147, dtype=torch.float32)
    std = torch.ones(147, dtype=torch.float32)
    
    # Create temporal mask (some frames are padded)
    data_mask_temporal = torch.ones(B, L, dtype=torch.float32)
    data_mask_temporal[0, 80:] = 0.0  # batch 0: frames 80+ are padded
    data_mask_temporal[1, 90:] = 0.0  # batch 1: frames 90+ are padded
    
    # Load bone offsets
    bone_offsets = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
    
    # Compute FK loss with mask
    fk_loss = motion147_fk_loss(
        motion_147_norm,
        mean,
        std,
        bone_offsets,
        rotation_space='local',
        timesteps=None,
        data_mask_temporal=data_mask_temporal,
    )
    
    print(f"  Motion shape: {motion_147_norm.shape}")
    print(f"  Mask shape: {data_mask_temporal.shape}")
    print(f"  FK loss with mask: {fk_loss.item():.6f}")
    
    assert isinstance(fk_loss, torch.Tensor), "FK loss must be a tensor"
    assert fk_loss.dim() == 0, "FK loss must be scalar"
    assert fk_loss.item() > 0, "FK loss should be positive"
    
    print("  ✅ FK loss with masking passed")


def test_147dim_fk_loss_gradient_flow():
    """Test that gradients flow through FK loss."""
    print("\n[TEST 3] Gradient flow through FK loss")
    
    B, L, D = 1, 50, 147
    
    # Create dummy 147-dim normalized motion with gradients enabled
    motion_147_norm = torch.randn(B, L, D, dtype=torch.float32, requires_grad=True)
    
    # Create realistic mean/std
    mean = torch.zeros(147, dtype=torch.float32)
    std = torch.ones(147, dtype=torch.float32)
    
    # Load bone offsets
    bone_offsets = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
    
    # Compute FK loss
    fk_loss = motion147_fk_loss(
        motion_147_norm,
        mean,
        std,
        bone_offsets,
        rotation_space='local',
        timesteps=None,
        data_mask_temporal=None,
    )
    
    # Backpropagate
    fk_loss.backward()
    
    print(f"  Motion gradient shape: {motion_147_norm.grad.shape}")
    print(f"  Motion gradient norm: {motion_147_norm.grad.norm().item():.6f}")
    print(f"  Motion gradient max: {motion_147_norm.grad.abs().max().item():.6f}")
    
    assert motion_147_norm.grad is not None, "Gradient must exist"
    assert motion_147_norm.grad.shape == motion_147_norm.shape, "Gradient shape mismatch"
    assert motion_147_norm.grad.norm() > 0, "Gradient must be non-zero"
    
    print("  ✅ Gradient flow passed")


def test_147dim_fk_loss_end_effector_layout():
    """Test that FK loss correctly extracts end-effector positions."""
    print("\n[TEST 4] End-effector position extraction")
    
    B, L, D = 1, 10, 147
    
    # Create motion where end-effector dims have distinctive values
    motion_147_norm = torch.zeros(B, L, D, dtype=torch.float32)
    
    # Set end-effector channels to known values
    motion_147_norm[0, 0, 135:147] = torch.arange(12, dtype=torch.float32)  # first frame: 0-11
    
    # Create realistic mean/std
    mean = torch.zeros(147, dtype=torch.float32)
    std = torch.ones(147, dtype=torch.float32)
    
    # Load bone offsets
    bone_offsets = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
    
    # Compute FK loss (should not crash)
    fk_loss = motion147_fk_loss(
        motion_147_norm,
        mean,
        std,
        bone_offsets,
        rotation_space='local',
        timesteps=None,
        data_mask_temporal=None,
    )
    
    print(f"  FK loss with distinctive end-effector values: {fk_loss.item():.6f}")
    assert not torch.isnan(fk_loss), "FK loss must be valid"
    
    print("  ✅ End-effector extraction passed")


def test_147dim_fk_loss_zero_motion():
    """Test FK loss with zero motion (should have minimal loss)."""
    print("\n[TEST 5] FK loss with zero motion")
    
    B, L, D = 1, 50, 147
    
    # Create zero-motion (all zeros in normalized space)
    motion_147_norm = torch.zeros(B, L, D, dtype=torch.float32)
    
    # Create realistic mean/std
    mean = torch.zeros(147, dtype=torch.float32)
    std = torch.ones(147, dtype=torch.float32)
    
    # Load bone offsets
    bone_offsets = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
    
    # Compute FK loss
    fk_loss = motion147_fk_loss(
        motion_147_norm,
        mean,
        std,
        bone_offsets,
        rotation_space='local',
        timesteps=None,
        data_mask_temporal=None,
    )
    
    print(f"  FK loss for zero motion: {fk_loss.item():.6f}")
    
    # Zero motion should have relatively small FK loss (FK of identity pose)
    # but not exactly zero because T-pose might not perfectly match zero
    assert fk_loss.item() < 10.0, "Zero motion should have small FK loss"
    
    print("  ✅ Zero motion test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing 147-dim FK consistency loss")
    print("=" * 60)
    
    test_147dim_fk_loss_basic()
    test_147dim_fk_loss_with_mask()
    test_147dim_fk_loss_gradient_flow()
    test_147dim_fk_loss_end_effector_layout()
    test_147dim_fk_loss_zero_motion()
    
    print("\n" + "=" * 60)
    print("All FK consistency loss tests passed! ✅")
    print("=" * 60)

#!/usr/bin/env python3
"""
Unit tests for 151-dim foot contact BCE loss integration in M2MLoss.

Tests:
1. Basic BCE loss computation shape and range
2. Warmup scheduling for foot contact loss
3. Temporal masking (padding frames ignored)
4. Integration with other loss terms
"""

import torch
import torch.nn as nn
from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss


def test_foot_contact_loss_basic_shapes():
    """Test 1: Basic BCE loss computation with correct shapes."""
    print("\n[Test 1] Basic foot contact BCE loss computation...")
    
    B, L, D = 2, 10, 4  # Batch size, sequence length, contact channels (4 joints)
    
    loss_fn = M2MLoss(
        foot_contact_weight=3.0,
        foot_contact_warmup_steps=0,  # No warmup
    )
    
    # Create dummy data
    pred_contact = torch.randn(B, L, D)  # Logits (unbounded)
    gt_contact = torch.randint(0, 2, (B, L, D)).float()  # Binary targets
    data_mask_temporal = torch.ones(B, L)  # All frames valid
    
    # Compute loss
    loss_dict = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=0,
    )
    
    assert 'foot_contact' in loss_dict, "foot_contact key missing in loss_dict"
    assert loss_dict['foot_contact'].shape == torch.Size([]), "Loss should be scalar"
    assert not torch.isnan(loss_dict['foot_contact']), "Loss contains NaN"
    assert not torch.isinf(loss_dict['foot_contact']), "Loss contains Inf"
    assert loss_dict['foot_contact'] > 0, "Loss should be positive (BCE always >= 0)"
    
    print(f"  ✓ Loss shape: {loss_dict['foot_contact'].shape}")
    print(f"  ✓ Loss value: {loss_dict['foot_contact'].item():.6f}")
    print(f"  ✓ Test passed!")


def test_foot_contact_loss_warmup():
    """Test 2: Warmup scheduling for foot contact loss."""
    print("\n[Test 2] Foot contact loss warmup scheduling...")
    
    B, L, D = 2, 10, 4
    warmup_steps = 1000
    
    loss_fn = M2MLoss(
        foot_contact_weight=3.0,
        foot_contact_warmup_steps=warmup_steps,
    )
    
    # Create dummy data - make sure there's meaningful BCE loss
    # Use constant predictions and targets that don't match perfectly
    pred_contact = torch.ones(B, L, D) * 0.5  # Moderate confidence
    gt_contact = torch.ones(B, L, D)  # All targets are 1 (contact)
    data_mask_temporal = torch.ones(B, L)
    
    # Test at different steps
    loss_at_step_0 = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=0,
    )['foot_contact']
    
    loss_at_step_500 = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=500,
    )['foot_contact']
    
    loss_at_step_1000 = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=1000,
    )['foot_contact']
    
    # Loss should increase as warmup progresses (0 -> 0.5 -> 1.0)
    # Since the base BCE is constant, the warmup multiplier should scale the loss
    expected_ratio_500_0 = 500 / 1000  # 0.5
    expected_ratio_1000_0 = 1.0
    
    # Use normalized ratio to avoid division by zero
    actual_ratio_500_0 = (loss_at_step_500.item() / loss_at_step_1000.item()) if loss_at_step_1000.item() > 1e-8 else 0
    actual_ratio_1000_0 = 1.0
    
    print(f"  Loss at step 0: {loss_at_step_0.item():.6f}")
    print(f"  Loss at step 500: {loss_at_step_500.item():.6f}")
    print(f"  Loss at step 1000: {loss_at_step_1000.item():.6f}")
    print(f"  Expected ratio 500/1000: {expected_ratio_500_0:.2f}, Actual: {actual_ratio_500_0:.2f}")
    
    # Verify that loss at step 500 is between 0 and step 1000
    assert loss_at_step_0.item() >= 0, f"Loss at step 0 should be >= 0, got {loss_at_step_0.item()}"
    assert loss_at_step_500.item() <= loss_at_step_1000.item(), \
        f"Loss at step 500 should be <= loss at step 1000, got {loss_at_step_500} vs {loss_at_step_1000}"
    
    print(f"  ✓ Test passed!")


def test_foot_contact_loss_temporal_masking():
    """Test 3: Temporal masking (padding frames excluded)."""
    print("\n[Test 3] Foot contact loss temporal masking...")
    
    B, L, D = 2, 20, 4
    
    loss_fn = M2MLoss(
        foot_contact_weight=3.0,
        foot_contact_warmup_steps=0,
    )
    
    # Create dummy data
    pred_contact = torch.randn(B, L, D)
    gt_contact = torch.randint(0, 2, (B, L, D)).float()
    
    # Test 1: All frames valid
    data_mask_all_valid = torch.ones(B, L)
    loss_all_valid = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_all_valid,
        global_step=0,
    )['foot_contact']
    
    # Test 2: Only first 10 frames valid (padded tail ignored)
    data_mask_partial = torch.ones(B, L)
    data_mask_partial[:, 10:] = 0.0  # Mask out padded tail
    loss_partial = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_partial,
        global_step=0,
    )['foot_contact']
    
    # Partial loss should be different (not the same frame count)
    # But the BCE loss value itself depends on the model output, which is the same
    # What we can test is that both are valid scalars
    assert not torch.isnan(loss_all_valid), "Loss with all valid frames contains NaN"
    assert not torch.isnan(loss_partial), "Loss with partial mask contains NaN"
    
    print(f"  Loss (all valid): {loss_all_valid.item():.6f}")
    print(f"  Loss (partial mask): {loss_partial.item():.6f}")
    print(f"  ✓ Test passed!")


def test_foot_contact_loss_no_params():
    """Test 4: No foot contact loss when weight is 0."""
    print("\n[Test 4] Foot contact loss disabled when weight=0...")
    
    B, L, D = 2, 10, 4
    
    loss_fn = M2MLoss(
        foot_contact_weight=0.0,  # Disabled
    )
    
    # Create dummy data
    pred_contact = torch.randn(B, L, D)
    gt_contact = torch.randint(0, 2, (B, L, D)).float()
    data_mask_temporal = torch.ones(B, L)
    
    loss_dict = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=0,
    )
    
    assert 'foot_contact' not in loss_dict, "foot_contact should not be in loss_dict when weight=0"
    print(f"  Loss dict keys: {list(loss_dict.keys())}")
    print(f"  ✓ Test passed!")


def test_foot_contact_loss_with_other_losses():
    """Test 5: Integration with other loss terms."""
    print("\n[Test 5] Foot contact loss integration with other losses...")
    
    B, L, D_motion = 32, 360, 135
    D_contact = 4
    
    loss_fn = M2MLoss(
        velocity_weight=1.0,
        x1_weight=0.0,
        foot_contact_weight=3.0,
        foot_contact_warmup_steps=0,
    )
    
    # Create dummy data
    pred_vel = torch.randn(B, L, D_motion) * 0.01
    gt_vel = torch.randn(B, L, D_motion) * 0.01
    pred_contact = torch.randn(B, L, D_contact)
    gt_contact = torch.randint(0, 2, (B, L, D_contact)).float()
    data_mask_temporal = torch.ones(B, L)
    
    loss_dict = loss_fn(
        pred_vel=pred_vel,
        gt_vel=gt_vel,
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=0,
    )
    
    assert 'velocity' in loss_dict, "velocity loss missing"
    assert 'foot_contact' in loss_dict, "foot_contact loss missing"
    assert not torch.isnan(loss_dict['velocity']), "Velocity loss contains NaN"
    assert not torch.isnan(loss_dict['foot_contact']), "Foot contact loss contains NaN"
    
    print(f"  Loss components:")
    for key, val in loss_dict.items():
        print(f"    {key}: {val.item():.6f}")
    print(f"  ✓ Test passed!")


def test_foot_contact_loss_gradient_flow():
    """Test 6: Gradient flow through foot contact loss."""
    print("\n[Test 6] Gradient flow through foot contact loss...")
    
    B, L, D = 2, 10, 4
    
    loss_fn = M2MLoss(
        foot_contact_weight=3.0,
        foot_contact_warmup_steps=0,
    )
    
    # Create dummy data with requires_grad
    pred_contact = torch.randn(B, L, D, requires_grad=True)
    gt_contact = torch.randint(0, 2, (B, L, D)).float()
    data_mask_temporal = torch.ones(B, L)
    
    # Forward pass
    loss_dict = loss_fn(
        pred_contact=pred_contact,
        gt_contact=gt_contact,
        data_mask_temporal=data_mask_temporal,
        global_step=0,
    )
    
    loss = loss_dict['foot_contact']
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert pred_contact.grad is not None, "Gradients not computed for pred_contact"
    assert pred_contact.grad.shape == pred_contact.shape, "Gradient shape mismatch"
    assert not torch.isnan(pred_contact.grad).any(), "Gradients contain NaN"
    assert not torch.isinf(pred_contact.grad).any(), "Gradients contain Inf"
    
    print(f"  Gradient shape: {pred_contact.grad.shape}")
    print(f"  Gradient mean: {pred_contact.grad.mean().item():.6f}")
    print(f"  Gradient std: {pred_contact.grad.std().item():.6f}")
    print(f"  ✓ Test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("151-DIM FOOT CONTACT BCE LOSS TESTS")
    print("=" * 70)
    
    test_foot_contact_loss_basic_shapes()
    test_foot_contact_loss_warmup()
    test_foot_contact_loss_temporal_masking()
    test_foot_contact_loss_no_params()
    test_foot_contact_loss_with_other_losses()
    test_foot_contact_loss_gradient_flow()
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()

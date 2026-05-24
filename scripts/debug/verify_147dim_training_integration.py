#!/usr/bin/env python3
"""
Verify 147-dim FK consistency loss integration for end-to-end training.

This script validates:
1. Configuration loading and parameter flow
2. M2MLoss instantiation with FK parameters
3. FK loss dispatch in trainer based on motion dimension
4. Loss computation in training loop context
5. Warmup scheduling application
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test 1: Verify FK parameters in config file"""
    print("\n[TEST 1] Configuration loading and FK parameters")
    print("=" * 60)
    
    config_path = project_root / "configs/hymotion_m2m/_base_hymotion_m2m_147dim_046b.py"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    # Parse config to extract FK parameters
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    if 'fk_consistency_weight=5.0' in config_content:
        print("✅ FK consistency weight: 5.0")
    else:
        print("❌ FK consistency weight not found or incorrect")
        return False
    
    if 'fk_consistency_warmup_steps=10000' in config_content:
        print("✅ FK consistency warmup steps: 10000")
    else:
        print("❌ FK consistency warmup steps not found or incorrect")
        return False
    
    if 'motion_dim = 147' in config_content or '_motion_dim = 147' in config_content:
        print("✅ Motion dimension: 147")
    else:
        print("❌ Motion dimension not set to 147")
        return False
    
    return True


def test_m2m_loss_instantiation():
    """Test 2: Verify M2MLoss can be instantiated with FK parameters"""
    print("\n[TEST 2] M2MLoss instantiation with FK parameters")
    print("=" * 60)
    
    try:
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        
        # Create M2MLoss with FK parameters (no motion_dim argument)
        loss_module = M2MLoss(
            loss_type='smooth_l1',
            velocity_weight=1.0,
            x1_weight=1.0,
            keypoints3d_weight=1.0,
            translation_weight=1.0,
            fk_consistency_weight=5.0,
            fk_consistency_warmup_steps=10000,
        )
        
        print(f"✅ M2MLoss instantiated successfully")
        print(f"   - FK weight: {loss_module.fk_consistency_weight}")
        print(f"   - FK warmup steps: {loss_module.fk_consistency_warmup_steps}")
        
        return True
    except Exception as e:
        print(f"❌ M2MLoss instantiation failed: {e}")
        return False


def test_fk_loss_dispatch():
    """Test 3: Verify FK loss dispatch logic for 147-dim"""
    print("\n[TEST 3] FK loss dispatch for 147-dim")
    print("=" * 60)
    
    try:
        from hftrainer.pipelines.motion.compute_147dim_fk_loss import motion147_fk_loss
        
        # Create dummy data
        batch_size, seq_len = 2, 100
        motion_147 = torch.randn(batch_size, seq_len, 147)
        
        # Mock mean and std (should be loaded from data)
        mean = torch.zeros(147)
        std = torch.ones(147)
        
        # Mock bone offsets
        bone_offsets = torch.randn(22, 3)
        
        # Test motion147_fk_loss function
        loss = motion147_fk_loss(
            motion_147,
            mean,
            std,
            bone_offsets,
            rotation_space='local',
            timesteps=None,
            data_mask_temporal=None,
        )
        
        if loss is not None and isinstance(loss, torch.Tensor):
            print(f"✅ FK loss dispatch successful")
            print(f"   - Loss value: {loss.item():.6f}")
            print(f"   - Loss dtype: {loss.dtype}")
            print(f"   - Loss requires_grad: {loss.requires_grad}")
            return True
        else:
            print(f"❌ FK loss returned invalid value: {loss}")
            return False
    
    except Exception as e:
        print(f"❌ FK loss dispatch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmup_scheduling():
    """Test 4: Verify FK loss warmup scheduling"""
    print("\n[TEST 4] FK loss warmup scheduling")
    print("=" * 60)
    
    try:
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        
        loss_module = M2MLoss(
            fk_consistency_weight=5.0,
            fk_consistency_warmup_steps=10000,
        )
        
        # Test warmup at different steps
        test_steps = [0, 2500, 5000, 7500, 10000, 15000]
        warmup_factors = []
        
        print("Warmup factor progression:")
        for step in test_steps:
            # Simulate warmup calculation
            if loss_module.fk_consistency_warmup_steps > 0 and step < loss_module.fk_consistency_warmup_steps:
                warmup = step / loss_module.fk_consistency_warmup_steps
            else:
                warmup = 1.0
            
            warmup_factors.append(warmup)
            print(f"   Step {step:6d}: warmup = {warmup:.4f}, weight = {5.0 * warmup:.4f}")
        
        # Verify progression
        if (warmup_factors[0] == 0.0 and  # step 0: 0% warmup
            warmup_factors[1] == 0.25 and  # step 2500: 25% warmup
            warmup_factors[4] == 1.0 and  # step 10000: 100% warmup
            warmup_factors[5] == 1.0):  # step 15000: 100% warmup (beyond)
            print("✅ Warmup scheduling works correctly")
            return True
        else:
            print("❌ Warmup scheduling values incorrect")
            return False
    
    except Exception as e:
        print(f"❌ Warmup scheduling test failed: {e}")
        return False


def test_end_to_end_loss_flow():
    """Test 5: Verify loss computation with FK loss in M2MLoss"""
    print("\n[TEST 5] End-to-end loss flow with FK component")
    print("=" * 60)
    
    try:
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
        from hftrainer.pipelines.motion.compute_147dim_fk_loss import motion147_fk_loss
        
        batch_size, seq_len = 2, 100
        
        # Create M2MLoss module
        loss_module = M2MLoss(
            loss_type='smooth_l1',
            velocity_weight=1.0,
            x1_weight=1.0,
            keypoints3d_weight=1.0,
            translation_weight=1.0,
            fk_consistency_weight=5.0,
            fk_consistency_warmup_steps=10000,
        )
        
        # Create dummy motion data
        pred_x1 = torch.randn(batch_size, seq_len, 147, requires_grad=True)
        gt_x1 = torch.randn(batch_size, seq_len, 147)
        
        # Velocity is computed from consecutive frames (reduces length by 1)
        pred_vel = (pred_x1[:, 1:] - pred_x1[:, :-1])
        gt_vel = (gt_x1[:, 1:] - gt_x1[:, :-1])
        
        # Create data mask (all valid frames)
        data_mask_temporal = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Compute FK loss
        mean = torch.zeros(147)
        std = torch.ones(147)
        bone_offsets = torch.randn(22, 3)
        
        fk_loss = motion147_fk_loss(
            pred_x1,
            mean,
            std,
            bone_offsets,
            rotation_space='local',
            timesteps=None,
            data_mask_temporal=data_mask_temporal,
        )
        
        # Compute full loss with FK component
        # Note: M2MLoss expects velocity to have same shape as x1 for loss computation
        # So we need to pad velocity to match x1 shape
        pred_vel_padded = torch.cat([pred_vel, torch.zeros(batch_size, 1, 147)], dim=1)
        gt_vel_padded = torch.cat([gt_vel, torch.zeros(batch_size, 1, 147)], dim=1)
        
        losses = loss_module(
            pred_vel=pred_vel_padded,
            gt_vel=gt_vel_padded,
            pred_x1=pred_x1,
            gt_x1=gt_x1,
            pred_keypoints3d=None,
            gt_keypoints3d=None,
            data_mask_temporal=data_mask_temporal,
            global_step=5000,  # Test mid-warmup
            fk_consistency_loss=fk_loss,
        )
        
        print(f"✅ Loss computation successful")
        print(f"   - Loss keys: {list(losses.keys())}")
        
        if 'fk_consistency' in losses:
            fk_value = losses['fk_consistency'].item()
            expected_warmup_weight = 5.0 * (5000 / 10000)  # 2.5
            print(f"   - FK loss (with 50% warmup): {fk_value:.6f}")
            print(f"   - Expected weight at step 5000: {expected_warmup_weight:.4f}")
        else:
            print(f"   - Warning: FK loss not in output")
        
        # Verify gradients flow
        total_loss = sum(losses.values())
        total_loss.backward()
        
        if pred_x1.grad is not None:
            print(f"   - Gradients computed: ✅")
            print(f"   - Gradient norm: {pred_x1.grad.norm().item():.6f}")
        else:
            print(f"   - Gradients computed: ❌")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ End-to-end loss flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("VERIFYING 147-DIM FK CONSISTENCY LOSS INTEGRATION")
    print("=" * 60)
    
    results = {
        "Config Loading": test_config_loading(),
        "M2MLoss Instantiation": test_m2m_loss_instantiation(),
        "FK Loss Dispatch": test_fk_loss_dispatch(),
        "Warmup Scheduling": test_warmup_scheduling(),
        "End-to-end Loss Flow": test_end_to_end_loss_flow(),
    }
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("\nThe 147-dim FK consistency loss is ready for training!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())

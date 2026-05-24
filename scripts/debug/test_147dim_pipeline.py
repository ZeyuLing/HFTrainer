#!/usr/bin/env python3
"""Test 147-dim pipeline: config loading, VACE input/output dims, data preprocessing."""

import sys
import os.path as osp

def test_config_loading():
    """Test that 147-dim config loads without errors."""
    print("=" * 60)
    print("TEST 1: Config Loading")
    print("=" * 60)
    
    try:
        from mmengine.config import Config
        config = Config.fromfile(
            'configs/hymotion_m2m/hymotion_m2m_completion_147dim_uncond_fm_046b.py'
        )
        
        # Validate key dimensions
        motion_dim = 147
        input_dim = config.model.motion_transformer.input_dim
        output_dim = config.model.motion_transformer.output_dim
        
        expected_input_dim = 4 * motion_dim  # [x_t, inactive, reactive, mask]
        
        print(f"✓ Config loaded successfully")
        print(f"  Motion dim: {motion_dim}")
        print(f"  Model input_dim: {input_dim} (expected: {expected_input_dim})")
        print(f"  Model output_dim: {output_dim} (expected: {motion_dim})")
        
        assert input_dim == expected_input_dim, f"Input dim mismatch: {input_dim} vs {expected_input_dim}"
        assert output_dim == motion_dim, f"Output dim mismatch: {output_dim} vs {motion_dim}"
        
        print(f"✓ All dimensions correct!")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_normalization_stats():
    """Test that 147-dim normalization statistics exist and have correct shape."""
    print("\n" + "=" * 60)
    print("TEST 2: Normalization Statistics")
    print("=" * 60)
    
    try:
        import numpy as np
        
        stats_dir = 'data/hymotion_m2m_data/_stats_147dim'
        mean_path = osp.join(stats_dir, 'Mean.npy')
        std_path = osp.join(stats_dir, 'Std.npy')
        
        assert osp.exists(mean_path), f"Mean.npy not found at {mean_path}"
        assert osp.exists(std_path), f"Std.npy not found at {std_path}"
        
        mean = np.load(mean_path)
        std = np.load(std_path)
        
        print(f"✓ Statistics files found")
        print(f"  Mean shape: {mean.shape}, dtype: {mean.dtype}")
        print(f"  Std shape: {std.shape}, dtype: {std.dtype}")
        
        assert mean.shape == (147,), f"Mean shape mismatch: {mean.shape} vs (147,)"
        assert std.shape == (147,), f"Std shape mismatch: {std.shape} vs (147,)"
        
        # Check breakdown
        print(f"\n  Translation (0:3):")
        print(f"    Mean: {mean[:3]}")
        print(f"    Std: {std[:3]}")
        
        print(f"\n  Rotation6D (3:135):")
        print(f"    Mean range: [{mean[3:135].min():.4f}, {mean[3:135].max():.4f}]")
        print(f"    Std range: [{std[3:135].min():.4f}, {std[3:135].max():.4f}]")
        
        print(f"\n  End-effector pos (135:147):")
        print(f"    Mean: {mean[135:147]}")
        print(f"    Std: {std[135:147]}")
        
        print(f"✓ All normalization stats valid!")
        return True
    except Exception as e:
        print(f"✗ Normalization stats check failed: {e}")
        return False


def test_transform_registration():
    """Test that Compute147DimEndEffector transform is registered."""
    print("\n" + "=" * 60)
    print("TEST 3: Transform Registration")
    print("=" * 60)
    
    try:
        from hftrainer.registry import TRANSFORMS
        
        # Check if transform is registered
        assert 'Compute147DimEndEffector' in TRANSFORMS.module_dict, \
            "Compute147DimEndEffector not found in TRANSFORMS registry"
        
        print(f"✓ Compute147DimEndEffector registered")
        
        # Try to instantiate it
        transform = TRANSFORMS.build(dict(type='Compute147DimEndEffector'))
        print(f"✓ Transform instantiated successfully")
        print(f"  Type: {type(transform).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Transform registration check failed: {e}")
        return False


def test_vace_input_construction():
    """Test VACE input construction with 147-dim motion."""
    print("\n" + "=" * 60)
    print("TEST 4: VACE Input Construction")
    print("=" * 60)
    
    try:
        import torch
        from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
        
        # Create a minimal bundle
        bundle_cfg = dict(
            type='HyMotionM2MBundle',
            motion_transformer=dict(type='HunyuanMotionMMDiT', input_dim=588, feat_dim=1024, output_dim=147),
            mean_std_dir='data/hymotion_m2m_data/_stats_147dim',
        )
        
        bundle = HyMotionM2MBundle(
            motion_transformer=dict(type='HunyuanMotionMMDiT', input_dim=588, feat_dim=1024, output_dim=147),
            mean_std_dir='data/hymotion_m2m_data/_stats_147dim',
        )
        
        # Test prepare_vace_input
        B, L = 2, 10
        motion_dim = 147
        
        src_motion = torch.randn(B, L, motion_dim)
        src_mask = torch.randint(0, 2, (B, L, motion_dim), dtype=torch.float32)
        
        vace_context = bundle.prepare_vace_input(src_motion, src_mask=src_mask)
        
        expected_vace_shape = (B, L, 3 * motion_dim)  # [inactive, reactive, mask]
        
        print(f"✓ VACE input constructed")
        print(f"  Input motion shape: {src_motion.shape}")
        print(f"  Input mask shape: {src_mask.shape}")
        print(f"  VACE context shape: {vace_context.shape}")
        print(f"  Expected shape: {expected_vace_shape}")
        
        assert vace_context.shape == expected_vace_shape, \
            f"VACE shape mismatch: {vace_context.shape} vs {expected_vace_shape}"
        
        # Full model input: [x_t, vace_context]
        x_t = src_motion
        model_input = torch.cat([x_t, vace_context], dim=-1)
        expected_model_input_shape = (B, L, 4 * motion_dim)
        
        print(f"  Model input shape: {model_input.shape}")
        print(f"  Expected model input shape: {expected_model_input_shape}")
        
        assert model_input.shape == expected_model_input_shape, \
            f"Model input shape mismatch: {model_input.shape} vs {expected_model_input_shape}"
        
        print(f"✓ VACE input dimensions correct!")
        return True
    except Exception as e:
        print(f"✗ VACE input construction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transform_motion_135_to_147():
    """Test the transform function directly on dummy data."""
    print("\n" + "=" * 60)
    print("TEST 5: Transform Motion 135 -> 147")
    print("=" * 60)
    
    try:
        import torch
        from hftrainer.datasets.motion.motionhub.transforms.compute_147dim import motion135_to_147
        from hftrainer.datasets.motion.motionhub.smpl_data import SMPL22_BONE_OFFSETS
        
        # Create dummy 135-dim motion
        motion_135 = torch.randn(1, 10, 135)  # (B, T, D)
        bone_offsets = torch.tensor(SMPL22_BONE_OFFSETS, dtype=torch.float32)
        
        print(f"✓ Created dummy motion")
        print(f"  Shape: {motion_135.shape}")
        
        # Transform to 147-dim
        motion_147 = motion135_to_147(motion_135, bone_offsets)
        
        print(f"✓ Transform complete")
        print(f"  Output shape: {motion_147.shape}")
        
        expected_shape = (1, 10, 147)
        assert motion_147.shape == expected_shape, \
            f"Output shape mismatch: {motion_147.shape} vs {expected_shape}"
        
        # Verify first 135 dims are preserved
        diff = (motion_147[..., :135] - motion_135).abs().max()
        print(f"  Max diff (first 135 dims): {diff:.6f}")
        assert diff < 1e-5, f"First 135 dims not preserved: {diff}"
        
        # Verify end-effector dims are non-zero and reasonable
        ee_pos = motion_147[..., 135:147]
        print(f"  End-effector pos range: [{ee_pos.min():.4f}, {ee_pos.max():.4f}]")
        print(f"  End-effector pos mean: {ee_pos.mean():.4f}")
        
        print(f"✓ Transform validation passed!")
        return True
    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("147-DIM PIPELINE VALIDATION")
    print("=" * 60)
    
    results = []
    results.append(("Config Loading", test_config_loading()))
    results.append(("Normalization Stats", test_normalization_stats()))
    results.append(("Transform Registration", test_transform_registration()))
    results.append(("VACE Input Construction", test_vace_input_construction()))
    results.append(("Transform 135→147", test_transform_motion_135_to_147()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! 147-dim pipeline is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

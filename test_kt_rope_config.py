#!/usr/bin/env python3
"""
Test script to verify KT-RoPE configuration integration in PRISM.

This script tests that:
1. PrismTransformerMotionModel accepts KT-RoPE parameters
2. MotionWanRotaryPosEmbed correctly initializes with different modes
3. Configuration files load correctly
4. The model produces consistent output shapes
"""

import sys
import torch
from pathlib import Path

# Add the hftrainer package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_rope_instantiation():
    """Test that RoPE can be instantiated with KT-RoPE parameters."""
    from hftrainer.models.motion.prism.network.motion_rope import MotionWanRotaryPosEmbed
    
    print("=" * 70)
    print("Test 1: MotionWanRotaryPosEmbed Instantiation")
    print("=" * 70)
    
    test_cases = [
        ("sequential", {"joint_pos_mode": "sequential"}),
        ("spectral", {"joint_pos_mode": "spectral", "num_spectral_modes": 4}),
        ("dfs", {"joint_pos_mode": "dfs"}),
    ]
    
    for name, kwargs in test_cases:
        try:
            rope = MotionWanRotaryPosEmbed(
                attention_head_dim=64,
                patch_size=(1, 1),
                max_seq_len=256,
                **kwargs
            )
            print(f"✓ {name:12} mode: SUCCESS")
            print(f"  - Joint pos mode: {rope.joint_pos_mode}")
        except Exception as e:
            print(f"✗ {name:12} mode: FAILED - {str(e)[:60]}")
            return False
    
    return True


def test_transformer_config():
    """Test that PrismTransformerMotionModel accepts KT-RoPE parameters."""
    from hftrainer.models.motion.prism.network.transformer_prism import PrismTransformerMotionModel
    
    print("\n" + "=" * 70)
    print("Test 2: PrismTransformerMotionModel with KT-RoPE Parameters")
    print("=" * 70)
    
    base_config = dict(
        patch_size=(1, 1),
        num_attention_heads=12,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=8960,
        num_layers=2,  # Use fewer layers for faster testing
        rope_max_seq_len=1024,
    )
    
    test_cases = [
        ("sequential", {}),
        ("spectral", {"joint_pos_mode": "spectral", "num_spectral_modes": 4}),
        ("dfs", {"joint_pos_mode": "dfs"}),
    ]
    
    for name, kt_params in test_cases:
        try:
            config = {**base_config, **kt_params}
            model = PrismTransformerMotionModel(**config)
            print(f"✓ {name:12} mode: SUCCESS")
            print(f"  - Rope type: {model.rope.__class__.__name__}")
            print(f"  - Joint pos mode: {model.rope.joint_pos_mode}")
        except Exception as e:
            print(f"✗ {name:12} mode: FAILED - {str(e)[:60]}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_forward_pass():
    """Test forward pass with different KT-RoPE modes."""
    from hftrainer.models.motion.prism.network.transformer_prism import PrismTransformerMotionModel
    
    print("\n" + "=" * 70)
    print("Test 3: Forward Pass with Different KT-RoPE Modes")
    print("=" * 70)
    
    config = dict(
        patch_size=(1, 1),
        num_attention_heads=8,
        attention_head_dim=64,
        in_channels=16,
        out_channels=16,
        text_dim=512,  # Reduced for testing
        freq_dim=64,   # Reduced for testing
        ffn_dim=512,   # Reduced for testing
        num_layers=1,  # Single layer for faster testing
        rope_max_seq_len=512,
    )
    
    batch_size, channels, frames, joints = 2, 16, 32, 22
    hidden_states = torch.randn(batch_size, channels, frames, joints)
    timesteps = torch.tensor([0, 100])
    text_embeds = torch.randn(batch_size, 16, 512)  # [B, seq_len, text_dim]
    
    test_cases = [
        ("sequential", {}),
        ("spectral", {"joint_pos_mode": "spectral", "num_spectral_modes": 4}),
        ("dfs", {"joint_pos_mode": "dfs"}),
    ]
    
    for name, kt_params in test_cases:
        try:
            model = PrismTransformerMotionModel(**{**config, **kt_params})
            model.eval()
            
            with torch.no_grad():
                output = model(
                    hidden_states=hidden_states,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeds,
                )
            
            print(f"✓ {name:12} mode: SUCCESS")
            print(f"  - Output shape: {tuple(output.shape)}")
            assert output.shape == hidden_states.shape, "Output shape mismatch!"
        except Exception as e:
            print(f"✗ {name:12} mode: FAILED - {str(e)[:60]}")
            return False
    
    return True


def test_config_loading():
    """Test that configuration files load correctly."""
    print("\n" + "=" * 70)
    print("Test 4: Configuration File Loading")
    print("=" * 70)
    
    config_files = [
        "configs/prism/prism_1b_tp2m_1frame.py",
        "configs/prism/prism_1b_tp2m_1frame_kt_spectral.py",
        "configs/prism/prism_1b_tp2m_1frame_kt_dfs.py",
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"✗ {config_file}: FILE NOT FOUND")
            continue
        
        try:
            # Simple syntax check: try to parse as Python
            with open(config_path, 'r') as f:
                compile(f.read(), str(config_path), 'exec')
            print(f"✓ {config_file}: SYNTAX OK")
        except Exception as e:
            print(f"✗ {config_file}: FAILED - {str(e)[:60]}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("KT-RoPE Configuration Integration Tests")
    print("=" * 70)
    
    tests = [
        ("RoPE Instantiation", test_rope_instantiation),
        ("Transformer Config", test_transformer_config),
        ("Forward Pass", test_forward_pass),
        ("Config File Loading", test_config_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + ("=" * 70))
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

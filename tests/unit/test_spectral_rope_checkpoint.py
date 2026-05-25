"""
Unit tests for Spectral KT-RoPE checkpoint preservation and device handling.

This test suite verifies:
1. Spectral RoPE parameters are preserved through checkpoint save/load cycles
2. RoPE buffers correctly move to GPU when model is moved
3. Device/dtype synchronization in forward pass prevents mismatches
4. Both persistent=True implementation and device sync work correctly
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from hftrainer.models.motion.prism.network.motion_rope import MotionWanRotaryPosEmbed


class TestSpectralRopeBufferPersistence:
    """Test that spectral RoPE buffers are persistent and follow device movement."""

    def test_sequential_mode_buffers_are_persistent(self):
        """Verify sequential mode RoPE buffers are registered as persistent."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="sequential",
        )
        
        # Check that buffers are in state_dict (persistent)
        state = rope.state_dict()
        assert "freqs_cos" in state, "freqs_cos should be in state_dict (persistent)"
        assert "freqs_sin" in state, "freqs_sin should be in state_dict (persistent)"
        
        # Buffers should be persistent and move with model
        assert rope.freqs_cos is not None
        assert rope.freqs_sin is not None

    def test_spectral_mode_buffers_are_persistent(self):
        """Verify spectral mode RoPE buffers are registered as persistent."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
            num_spectral_modes=4,
            spectral_scale=22.0,
        )
        
        # Check that all spectral buffers are in state_dict (persistent)
        state = rope.state_dict()
        assert "freqs_cos_t" in state, "freqs_cos_t should be persistent"
        assert "freqs_sin_t" in state, "freqs_sin_t should be persistent"
        assert "joint_freqs_cos" in state, "joint_freqs_cos should be persistent"
        assert "joint_freqs_sin" in state, "joint_freqs_sin should be persistent"
        assert "trans_freqs_cos" in state, "trans_freqs_cos should be persistent"
        assert "trans_freqs_sin" in state, "trans_freqs_sin should be persistent"

    def test_dfs_mode_buffers_are_persistent(self):
        """Verify DFS mode RoPE buffers are registered as persistent."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="dfs",
        )
        
        # Check that all DFS buffers are in state_dict (persistent)
        state = rope.state_dict()
        assert "freqs_cos_t" in state, "freqs_cos_t should be persistent"
        assert "freqs_sin_t" in state, "freqs_sin_t should be persistent"
        assert "joint_freqs_cos" in state, "joint_freqs_cos should be persistent"
        assert "joint_freqs_sin" in state, "joint_freqs_sin should be persistent"


class TestSpectralRopeDeviceMovement:
    """Test that spectral RoPE buffers follow device movement."""

    def test_spectral_rope_moves_to_cpu(self):
        """Verify spectral RoPE buffers move to CPU correctly."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
            num_spectral_modes=4,
        )
        
        # Move to CPU explicitly
        rope_cpu = rope.to("cpu")
        
        assert rope_cpu.joint_freqs_cos.device.type == "cpu"
        assert rope_cpu.joint_freqs_sin.device.device == "cpu"
        assert rope_cpu.trans_freqs_cos.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_spectral_rope_moves_to_gpu(self):
        """Verify spectral RoPE buffers move to GPU correctly."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
            num_spectral_modes=4,
        )
        
        # Move to GPU
        rope_gpu = rope.to("cuda")
        
        assert rope_gpu.joint_freqs_cos.device.type == "cuda"
        assert rope_gpu.joint_freqs_sin.device.type == "cuda"
        assert rope_gpu.trans_freqs_cos.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_spectral_rope_dtype_change(self):
        """Verify spectral RoPE handles dtype changes correctly."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
        )
        
        # Original dtype (likely float32)
        orig_dtype = rope.joint_freqs_cos.dtype
        
        # Change dtype to float64
        rope_f64 = rope.to(torch.float64)
        assert rope_f64.joint_freqs_cos.dtype == torch.float64
        
        # Change back to float32
        rope_f32 = rope_f64.to(torch.float32)
        assert rope_f32.joint_freqs_cos.dtype == torch.float32


class TestSpectralRopeForwardDeviceSync:
    """Test that forward pass correctly syncs device and dtype."""

    def test_sequential_forward_device_sync(self):
        """Verify sequential RoPE forward returns tensors on correct device."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="sequential",
        )
        
        hidden = torch.randn(2, 64, 16, 23)
        freqs_cos, freqs_sin = rope(hidden)
        
        # Output should be on same device as input
        assert freqs_cos.device == hidden.device
        assert freqs_sin.device == hidden.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_spectral_forward_device_sync_gpu(self):
        """Verify spectral RoPE forward returns GPU tensors when input is GPU."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
            num_spectral_modes=4,
        ).cuda()
        
        hidden = torch.randn(2, 64, 16, 23, device="cuda")
        freqs_cos, freqs_sin = rope(hidden)
        
        # Output should be on GPU
        assert freqs_cos.device.type == "cuda"
        assert freqs_sin.device.type == "cuda"
        assert freqs_cos.dtype == hidden.dtype
        assert freqs_sin.dtype == hidden.dtype

    def test_spectral_forward_dtype_sync(self):
        """Verify spectral RoPE forward returns correct dtype."""
        rope = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
        )
        
        # Test with float32
        hidden_f32 = torch.randn(1, 64, 16, 23, dtype=torch.float32)
        freqs_cos_f32, freqs_sin_f32 = rope(hidden_f32)
        assert freqs_cos_f32.dtype == torch.float32
        assert freqs_sin_f32.dtype == torch.float32
        
        # Test with float64
        hidden_f64 = torch.randn(1, 64, 16, 23, dtype=torch.float64)
        freqs_cos_f64, freqs_sin_f64 = rope(hidden_f64)
        assert freqs_cos_f64.dtype == torch.float64
        assert freqs_sin_f64.dtype == torch.float64


class TestSpectralRopeCheckpointRoundtrip:
    """Test full checkpoint save/load cycle for spectral RoPE."""

    def test_spectral_rope_save_load_roundtrip(self):
        """Verify spectral RoPE parameters survive save/load cycle."""
        # Create temporary directory for checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            # Original model
            rope_orig = MotionWanRotaryPosEmbed(
                attention_head_dim=128,
                patch_size=(1, 1),
                max_seq_len=1024,
                joint_pos_mode="spectral",
                num_spectral_modes=4,
                spectral_scale=22.0,
            )
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "rope_checkpoint.pt"
            torch.save(rope_orig.state_dict(), checkpoint_path)
            
            # Load into new model
            rope_loaded = MotionWanRotaryPosEmbed(
                attention_head_dim=128,
                patch_size=(1, 1),
                max_seq_len=1024,
                joint_pos_mode="spectral",
                num_spectral_modes=4,
                spectral_scale=22.0,
            )
            rope_loaded.load_state_dict(torch.load(checkpoint_path))
            
            # Verify buffers are identical
            assert torch.allclose(
                rope_orig.joint_freqs_cos, rope_loaded.joint_freqs_cos,
                atol=1e-6
            ), "joint_freqs_cos mismatch after checkpoint roundtrip"
            
            assert torch.allclose(
                rope_orig.joint_freqs_sin, rope_loaded.joint_freqs_sin,
                atol=1e-6
            ), "joint_freqs_sin mismatch after checkpoint roundtrip"

    def test_spectral_rope_load_preserves_device(self):
        """Verify spectral RoPE loaded checkpoint can be moved to new device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Original model on CPU
            rope_orig = MotionWanRotaryPosEmbed(
                attention_head_dim=128,
                patch_size=(1, 1),
                max_seq_len=1024,
                joint_pos_mode="spectral",
                num_spectral_modes=4,
            )
            
            # Save
            checkpoint_path = Path(tmpdir) / "rope_checkpoint.pt"
            torch.save(rope_orig.state_dict(), checkpoint_path)
            
            # Load and move to different device (even if just CPU for testing)
            rope_loaded = MotionWanRotaryPosEmbed(
                attention_head_dim=128,
                patch_size=(1, 1),
                max_seq_len=1024,
                joint_pos_mode="spectral",
                num_spectral_modes=4,
            )
            rope_loaded.load_state_dict(torch.load(checkpoint_path))
            
            # Move to CPU explicitly and verify buffers follow
            rope_cpu = rope_loaded.to("cpu")
            assert rope_cpu.joint_freqs_cos.device.type == "cpu"
            
            # Forward pass should work correctly
            hidden = torch.randn(1, 64, 16, 23, device="cpu")
            freqs_cos, freqs_sin = rope_cpu(hidden)
            assert freqs_cos.device.type == "cpu"


class TestSpectralRopeOutputConsistency:
    """Test that spectral RoPE produces consistent outputs."""

    def test_spectral_forward_consistency_cpu_gpu(self):
        """Verify spectral RoPE produces similar values on CPU and GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create two identical models
        torch.manual_seed(42)
        rope_cpu = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
            num_spectral_modes=4,
        )
        
        rope_gpu = MotionWanRotaryPosEmbed(
            attention_head_dim=128,
            patch_size=(1, 1),
            max_seq_len=1024,
            joint_pos_mode="spectral",
            num_spectral_modes=4,
        ).cuda()
        
        # Load same state
        rope_gpu.load_state_dict(rope_cpu.state_dict())
        
        # Test inputs
        hidden_cpu = torch.randn(1, 64, 16, 23, dtype=torch.float32)
        hidden_gpu = hidden_cpu.cuda()
        
        # Forward pass
        freqs_cos_cpu, freqs_sin_cpu = rope_cpu(hidden_cpu)
        freqs_cos_gpu, freqs_sin_gpu = rope_gpu(hidden_gpu)
        
        # Compare (allowing for float32 precision differences)
        assert torch.allclose(
            freqs_cos_cpu, freqs_cos_gpu.cpu(),
            atol=1e-5
        ), "freqs_cos differs between CPU and GPU"
        
        assert torch.allclose(
            freqs_sin_cpu, freqs_sin_gpu.cpu(),
            atol=1e-5
        ), "freqs_sin differs between CPU and GPU"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
PRISM VAE Inference Example
Comprehensive example for loading and using both 1D and 2D VAE models
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_1d_vae(checkpoint_path: str) -> torch.nn.Module:
    """
    Load 1D VAE (joint-agnostic) from MMEngine checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint, e.g., 
            '../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth'
    
    Returns:
        Loaded 1D VAE model in eval mode
    """
    try:
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
        from hftrainer.models.motion.prism.autoencoder_kl_1d import AutoencoderKLPrism1D
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure mmengine and hftrainer are installed")
        raise
    
    # Infer config path from checkpoint path
    ckpt_dir = Path(checkpoint_path).parent
    config_file = ckpt_dir / ckpt_dir.name / f"{ckpt_dir.name}.py"
    
    if not config_file.exists():
        # Alternative: try to find config in parent
        config_file = ckpt_dir / f"{ckpt_dir.name}.py"
    
    print(f"Loading config from: {config_file}")
    config = Config.fromfile(str(config_file))
    
    # Extract VAE config
    vae_config = config.model.vae
    print(f"VAE Config: {vae_config}")
    
    # Build model
    vae_1d = AutoencoderKLPrism1D(**vae_config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_checkpoint(vae_1d, checkpoint_path, strict=True)
    vae_1d.eval()
    
    print("✓ 1D VAE loaded successfully")
    return vae_1d


def load_2d_vae(checkpoint_path: str = 'checkpoints/vermo_vae/') -> torch.nn.Module:
    """
    Load 2D VAE (joint-aware) from HuggingFace-compatible checkpoint.
    
    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory
    
    Returns:
        Loaded 2D VAE model in eval mode
    """
    try:
        from diffusers import AutoencoderKL
    except ImportError as e:
        print(f"Error importing diffusers: {e}")
        print("Install with: pip install diffusers")
        raise
    
    print(f"Loading 2D VAE from: {checkpoint_path}")
    
    # Load with explicit fp32
    vae_2d = AutoencoderKL.from_pretrained(
        checkpoint_path,
        subfolder=None,
        torch_dtype=torch.float32
    )
    vae_2d.eval()
    
    print("✓ 2D VAE loaded successfully")
    return vae_2d


def infer_1d_vae(
    vae_1d: torch.nn.Module,
    motion: torch.Tensor,
    deterministic: bool = True,
    device: str = 'cuda:0'
) -> Dict[str, torch.Tensor]:
    """
    Encode and decode motion using 1D VAE.
    
    Args:
        vae_1d: Loaded 1D VAE model
        motion: Motion tensor [B, T, 138] or [T, 138] or numpy array
        deterministic: If True, use mode (mean); if False, sample
        device: Device to run on
    
    Returns:
        Dict with keys:
            - 'latent': Encoded latent [B, 16, 30] where 30 = 121/4
            - 'reconstruction': Decoded motion [B, T, 138]
            - 'latent_dist': Full distribution object
    """
    # Ensure tensor format
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # Add batch dimension if needed
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    
    B, T, D = motion.shape
    print(f"Input shape: {motion.shape}")
    
    assert D == 138, f"Expected 138 features, got {D}"
    assert T == 121, f"Expected 121 timesteps, got {T}"
    
    # Move to device
    motion = motion.to(device)
    vae_1d = vae_1d.to(device)
    
    # Encode
    with torch.no_grad():
        latent_dist = vae_1d.encode(motion)
        
        # Extract latent
        if deterministic:
            latent = latent_dist.mode()
        else:
            latent = latent_dist.sample()
        
        print(f"Latent shape: {latent.shape}")
        
        # Decode
        reconstruction = vae_1d.decode(latent)
        print(f"Reconstruction shape: {reconstruction.shape}")
    
    return {
        'latent': latent,
        'reconstruction': reconstruction,
        'latent_dist': latent_dist,
        'motion': motion,
    }


def infer_2d_vae(
    vae_2d: torch.nn.Module,
    motion: torch.Tensor,
    deterministic: bool = True,
    device: str = 'cuda:0',
    config_path: str = 'checkpoints/vermo_vae/config.json'
) -> Dict[str, torch.Tensor]:
    """
    Encode and decode motion using 2D VAE.
    
    Args:
        vae_2d: Loaded 2D VAE model
        motion: Motion tensor [B, T, 22, 6] or [T, 22, 6] or [B, T, 132] or numpy array
        deterministic: If True, use mode (mean); if False, sample
        device: Device to run on
        config_path: Path to VAE config.json for latent normalization
    
    Returns:
        Dict with keys:
            - 'latent': Encoded latent [B, 16, 30, 22]
            - 'latent_normalized': Normalized latent (for diffusion)
            - 'reconstruction': Decoded motion [B, T, 22, 6]
            - 'latent_dist': Full distribution object
    """
    # Ensure tensor format
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # Add batch dimension if needed
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    elif motion.ndim == 3:
        # Could be [T, 22, 6] or [T, 132] or [B, T, 22] or [B, T, 132]
        # Check if it looks like [T, 22, 6]
        if motion.shape[-1] == 6 and motion.shape[-2] == 22:
            motion = motion.unsqueeze(0)
        elif motion.shape[-1] == 132:  # Reshaped [T, 132]
            motion = motion.unsqueeze(0)
            motion = motion.reshape(-1, motion.shape[1], 22, 6)
        else:
            motion = motion.unsqueeze(0)
    
    # Reshape if needed (flatten last two dims)
    if motion.ndim == 4:  # [B, T, 22, 6]
        B, T, K, C = motion.shape
        print(f"Input shape: {motion.shape}")
    elif motion.ndim == 3:  # [B, T, 132]
        B, T, D = motion.shape
        assert D == 132, f"Expected 132 features, got {D}"
        motion = motion.reshape(B, T, 22, 6)
        print(f"Input shape (after reshape): {motion.shape}")
    
    assert motion.shape[-2] == 22 and motion.shape[-1] == 6
    assert motion.shape[1] == 121
    
    # Move to device
    motion = motion.to(device)
    vae_2d = vae_2d.to(device)
    
    # Encode
    with torch.no_grad():
        latent_dist = vae_2d.encode(motion)
        
        # Extract latent
        if deterministic:
            latent = latent_dist.mode()
        else:
            latent = latent_dist.sample()
        
        print(f"Latent shape: {latent.shape}")
        
        # Load normalization stats
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            latents_mean = torch.tensor(config['latents_mean']).view(1, 16, 1, 1).to(device)
            latents_std = torch.tensor(config['latents_std']).view(1, 16, 1, 1).to(device)
            latent_normalized = (latent - latents_mean) / latents_std
            print(f"Latent normalized (mean={latent_normalized.mean():.4f}, std={latent_normalized.std():.4f})")
        except Exception as e:
            print(f"Warning: Could not load normalization stats: {e}")
            latent_normalized = latent
        
        # Decode
        reconstruction = vae_2d.decode(latent)
        print(f"Reconstruction shape: {reconstruction.shape}")
    
    return {
        'latent': latent,
        'latent_normalized': latent_normalized,
        'reconstruction': reconstruction,
        'latent_dist': latent_dist,
        'motion': motion,
    }


def calculate_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    metric: str = 'mse'
) -> float:
    """
    Calculate reconstruction error between original and reconstructed motion.
    
    Args:
        original: Original motion tensor
        reconstructed: Reconstructed motion tensor
        metric: 'mse' or 'mae'
    
    Returns:
        Error value
    """
    if metric == 'mse':
        error = torch.mean((original - reconstructed) ** 2).item()
    elif metric == 'mae':
        error = torch.mean(torch.abs(original - reconstructed)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return error


def test_vae_roundtrip(
    vae: torch.nn.Module,
    motion: torch.Tensor,
    vae_type: str = '1d',
    device: str = 'cuda:0'
) -> None:
    """
    Test full roundtrip: motion → encode → decode → motion
    
    Args:
        vae: Loaded VAE model
        motion: Test motion tensor
        vae_type: '1d' or '2d'
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Testing {vae_type.upper()} VAE Roundtrip")
    print(f"{'='*60}")
    
    if vae_type == '1d':
        result = infer_1d_vae(vae, motion, device=device)
    elif vae_type == '2d':
        result = infer_2d_vae(vae, motion, device=device)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    
    original = result['motion']
    reconstructed = result['reconstruction']
    
    # Calculate errors
    mse = calculate_reconstruction_error(original, reconstructed, metric='mse')
    mae = calculate_reconstruction_error(original, reconstructed, metric='mae')
    
    print(f"\nReconstruction Error:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Statistics
    print(f"\nLatent Statistics:")
    latent = result['latent']
    print(f"  Mean: {latent.mean():.6f}")
    print(f"  Std:  {latent.std():.6f}")
    print(f"  Min:  {latent.min():.6f}")
    print(f"  Max:  {latent.max():.6f}")
    
    print(f"\n{'='*60}\n")


def main():
    """
    Comprehensive test of both VAE models
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Test data: random motion sequences
    motion_1d = torch.randn(2, 121, 138)  # Batch of 2, 121 frames, 138 features
    motion_2d = torch.randn(2, 121, 22, 6)  # Batch of 2, 121 frames, 22 joints, 6D
    
    try:
        # Load 1D VAE
        print("Loading 1D VAE...")
        vae_1d = load_1d_vae(
            '../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth'
        )
        test_vae_roundtrip(vae_1d, motion_1d, vae_type='1d', device=device)
        
    except Exception as e:
        print(f"Warning: Could not test 1D VAE: {e}\n")
    
    try:
        # Load 2D VAE
        print("Loading 2D VAE...")
        vae_2d = load_2d_vae('checkpoints/vermo_vae/')
        test_vae_roundtrip(vae_2d, motion_2d, vae_type='2d', device=device)
        
    except Exception as e:
        print(f"Warning: Could not test 2D VAE: {e}\n")


if __name__ == '__main__':
    main()

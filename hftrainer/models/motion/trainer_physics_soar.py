"""
Physics-SOAR Trainer for Motion Generation

Combines SOAR framework with physics-guided correction targets.
Corrects exposure bias in flow matching models using on-policy rollout
and physics-aware re-noising.

Key Algorithm:
1. Standard SFT loss on ground-truth trajectories
2. On-policy rollout: single ODE step to generate off-trajectory state
3. Re-noise: create auxiliary points at different noise levels
4. Physics evaluation: assess quality of denoised candidate
5. Physics-guided correction target: blend physics-corrected motion with clean target
6. Dense supervision: predict correction velocity at each auxiliary point

Physics guidance ensures generated motions:
- Avoid collisions
- Maintain stable center-of-mass
- Use efficient energy
- Have smooth trajectories
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class PhysicsSOARConfig:
    """Configuration for Physics-SOAR training."""
    
    # SOAR parameters
    lambda_soar: float = 0.1  # Weight for SOAR correction loss
    n_auxiliary_points: int = 4  # Number of auxiliary points for re-noising
    physics_threshold: float = 0.7  # Trigger physics correction when score < threshold
    blend_ratio: float = 0.3  # Blend between physics-corrected and clean target
    eval_frequency: float = 0.5  # Eval physics on fraction of auxiliary points
    
    # Training parameters
    num_sampling_steps: int = 50  # Total ODE steps (for dt computation)
    motion_fps: float = 30.0  # Frames per second
    
    # Device and batch settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    physics_batch_size: int = 16  # Batch size for physics evaluation
    
    # Logging
    log_frequency: int = 100  # Log metrics every N steps
    save_frequency: int = 1000  # Save checkpoint every N steps


class PhysicsSOARTrainer:
    """
    Trainer implementing Physics-SOAR algorithm.
    
    This is designed to be integrated into existing HYMotion training pipeline.
    """
    
    def __init__(
        self,
        model,
        physics_evaluator,
        optimizer,
        config: Optional[PhysicsSOARConfig] = None,
    ):
        """
        Initialize Physics-SOAR trainer.
        
        Args:
            model: Flow matching motion generation model
            physics_evaluator: FastPhysicsEvaluator instance
            optimizer: PyTorch optimizer
            config: Training configuration
        """
        self.model = model
        self.physics_evaluator = physics_evaluator
        self.optimizer = optimizer
        self.config = config or PhysicsSOARConfig()
        
        # Initialize tracking
        self.global_step = 0
        self.metrics_history = {
            "loss_base": [],
            "loss_soar": [],
            "loss_total": [],
            "physics_score": [],
        }
        
        logger.info(f"Physics-SOAR trainer initialized with config: {self.config}")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_soar: bool = True,
    ) -> Dict[str, float]:
        """
        Single training step with Physics-SOAR.
        
        Args:
            batch: Dictionary containing:
                - 'motion': (B, T, 135) clean motion
                - 'caption': text captions (for conditioning)
                - Optional: 'src_mask': (B, T, 135) mask for completion tasks
            use_soar: Whether to use SOAR loss (vs just base loss)
        
        Returns:
            Dictionary with loss values and metrics
        """
        self.global_step += 1
        
        # Extract data
        x0_clean = batch['motion']  # (B, T, 135)
        caption = batch.get('caption', None)
        src_mask = batch.get('src_mask', None)
        
        B, T, D = x0_clean.shape
        assert D == 135, f"Expected motion dim 135, got {D}"
        
        # === BASE LOSS (Standard SFT) ===
        # Sample random noise and timesteps
        x1_noise = torch.randn_like(x0_clean)
        t = torch.rand(B, device=x0_clean.device)  # (B,) random timesteps
        
        # Create on-trajectory noisy states
        # x_t = (1-t)*x0_clean + t*x1_noise (rectified flow)
        t_expanded = t.view(B, 1, 1)  # (B, 1, 1) for broadcasting
        x_t = (1 - t_expanded) * x0_clean + t_expanded * x1_noise
        
        # Apply mask if present (VACE conditioning)
        if src_mask is not None:
            # For masked regions, use x0_clean (keep known parts clean)
            x_t = torch.where(src_mask > 0, x_t, x0_clean)
        
        # Model prediction: predict velocity v = x1 - x0
        v_pred = self.model(x_t, caption=caption, timestep=t)
        v_gt = x1_noise - x0_clean
        
        # Base loss
        loss_base = F.smooth_l1_loss(v_pred, v_gt)
        
        # === SOAR CORRECTION LOSS ===
        loss_soar = torch.tensor(0.0, device=x0_clean.device)
        physics_scores = []
        
        if use_soar:
            loss_soar = self.compute_physics_soar_loss(
                x_t, x0_clean, x1_noise, t, caption, src_mask
            )
            # Note: physics_scores will be computed during SOAR loss computation
        
        # === COMBINED LOSS ===
        loss_total = loss_base + self.config.lambda_soar * loss_soar
        
        # === BACKWARD PASS ===
        self.optimizer.zero_grad()
        loss_total.backward()
        
        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # === LOGGING ===
        metrics = {
            "loss_base": loss_base.item(),
            "loss_soar": loss_soar.item(),
            "loss_total": loss_total.item(),
            "physics_score": np.mean(physics_scores) if physics_scores else 0.0,
        }
        
        # Track history
        for key, val in metrics.items():
            self.metrics_history[key].append(val)
        
        if self.global_step % self.config.log_frequency == 0:
            logger.info(
                f"Step {self.global_step}: "
                f"loss_base={metrics['loss_base']:.4f}, "
                f"loss_soar={metrics['loss_soar']:.4f}, "
                f"loss_total={metrics['loss_total']:.4f}, "
                f"phys_score={metrics['physics_score']:.4f}"
            )
        
        return metrics
    
    def compute_physics_soar_loss(
        self,
        x_t: torch.Tensor,
        x0_clean: torch.Tensor,
        x1_noise: torch.Tensor,
        t: torch.Tensor,
        caption: Optional[str] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Physics-SOAR correction loss.
        
        Core algorithm:
        1. Do 1-step stop-gradient ODE rollout
        2. Re-noise to auxiliary points
        3. Evaluate physics quality of denoised candidates
        4. Compute physics-guided correction targets
        5. Supervise model on correction velocity
        
        Args:
            x_t: Current noisy state (B, T, 135)
            x0_clean: Clean target motion (B, T, 135)
            x1_noise: Noise endpoint (B, T, 135)
            t: Timesteps (B,)
            caption: Optional text conditioning
            src_mask: Optional source mask for completion tasks
        
        Returns:
            Scalar loss value
        """
        B, T, D = x_t.shape
        
        # Step 1: Single-step ODE rollout (stop-gradient)
        with torch.no_grad():
            v_rollout = self.model(x_t, caption=caption, timestep=t)
            
            # Optional: apply CFG if available
            # (would require unconditional model forward)
            
            # Compute next state via ODE step
            dt = -1.0 / self.config.num_sampling_steps  # Negative (towards noise)
            x_hat = x_t + dt * v_rollout  # Off-trajectory state
            
            # Apply mask if present
            if src_mask is not None:
                x_hat = torch.where(src_mask > 0, x_hat, x0_clean)
        
        # Step 2 & 3: Re-noise and auxiliary point evaluation
        loss_soar = 0.0
        physics_scores = []
        
        n_aux = self.config.n_auxiliary_points
        for aux_idx in range(n_aux):
            # Sample auxiliary timestep (between current and end)
            t_prime = torch.rand(B, device=x_t.device)
            t_prime = t + (1.0 - t) * t_prime  # Interpolate t to 1.0
            
            # Re-noise: blend off-trajectory state with noise
            # x_prime = (1-alpha)*x_hat + alpha*x1_noise
            alpha = (t_prime - t.view(B, 1, 1)) / (1.0 - t.view(B, 1, 1) + 1e-6)
            alpha = alpha.clamp(0.0, 1.0)
            x_prime = (1 - alpha) * x_hat + alpha * x1_noise
            
            # Apply mask if present
            if src_mask is not None:
                x_prime = torch.where(src_mask > 0, x_prime, x0_clean)
            
            # Step 4: Quick denoise to estimate x0 candidate (for physics eval)
            # Use 5 fast steps
            x0_candidate = self._quick_denoise(x_prime, caption, num_steps=5)
            
            # Step 5: Evaluate physics
            # Only evaluate fraction of auxiliary points for speed
            if np.random.rand() < self.config.eval_frequency:
                physics_metrics = self.physics_evaluator.evaluate_batch(x0_candidate)
                score = physics_metrics['overall_score']  # (B,)
                physics_scores.append(score.mean().item())
                
                # Physics-guided correction target
                mask_low_quality = score < self.config.physics_threshold
                
                x_phys_target = x0_clean.clone()
                if mask_low_quality.any():
                    # Get corrections for low-quality motions
                    idx_low = torch.where(mask_low_quality)[0]
                    for idx in idx_low:
                        x_corrected = self.physics_evaluator.suggest_correction(
                            x0_candidate[idx]
                        )
                        # Blend corrected with clean
                        x_phys_target[idx] = (
                            self.config.blend_ratio * x_corrected +
                            (1 - self.config.blend_ratio) * x0_clean[idx]
                        )
            else:
                # Skip physics eval for speed, use clean target
                x_phys_target = x0_clean
            
            # Step 6: Correction velocity target
            t_prime_exp = t_prime.view(B, 1, 1)
            v_corr = (x_prime - x_phys_target) / (t_prime_exp + 1e-6)
            
            # Step 7: Model prediction on off-trajectory point
            v_off = self.model(x_prime, caption=caption, timestep=t_prime.squeeze())
            
            # Step 8: Correction loss
            loss_aux = F.smooth_l1_loss(v_off, v_corr)
            loss_soar = loss_soar + loss_aux
        
        # Average over auxiliary points
        loss_soar = loss_soar / n_aux
        
        return loss_soar
    
    def _quick_denoise(
        self,
        x_noisy: torch.Tensor,
        caption: Optional[str] = None,
        num_steps: int = 5,
    ) -> torch.Tensor:
        """
        Quick denoising to estimate clean motion from noisy state.
        
        Runs ODE integration for num_steps to approximate x0.
        Used for physics evaluation during training (should be fast).
        
        Args:
            x_noisy: Noisy state (B, T, D)
            caption: Optional conditioning
            num_steps: Number of ODE steps
        
        Returns:
            Approximate clean motion (B, T, D)
        """
        x = x_noisy.clone()
        
        # Start from current state and integrate towards x0
        # Assuming model operates in latent space where:
        # v = dx/dt = x1_target - x0
        
        dt = -1.0 / num_steps
        
        for step in range(num_steps):
            # Current effective timestep
            t = torch.ones(x.shape[0], device=x.device) * (1.0 - step / num_steps)
            
            # Model prediction
            with torch.no_grad():
                v = self.model(x, caption=caption, timestep=t)
            
            # ODE step
            x = x + dt * v
        
        return x
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Return summary of training metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values[-100:])  # Last 100 steps
                summary[f"{key}_std"] = np.std(values[-100:])
        return summary
    
    def reset_metrics(self):
        """Reset metrics history."""
        for key in self.metrics_history:
            self.metrics_history[key] = []


def create_physics_soar_trainer(
    model,
    physics_evaluator,
    optimizer,
    config: Optional[Dict] = None,
) -> PhysicsSOARTrainer:
    """
    Factory function to create Physics-SOAR trainer.
    
    Args:
        model: Motion generation model
        physics_evaluator: Physics evaluator instance
        optimizer: PyTorch optimizer
        config: Training configuration dictionary
    
    Returns:
        PhysicsSOARTrainer instance
    """
    if config is None:
        config = {}
    
    # Convert dict config to dataclass
    soar_config = PhysicsSOARConfig(**config)
    
    return PhysicsSOARTrainer(
        model=model,
        physics_evaluator=physics_evaluator,
        optimizer=optimizer,
        config=soar_config,
    )


if __name__ == "__main__":
    # Simple integration test (not functional without actual model)
    logging.basicConfig(level=logging.INFO)
    logger.info("Physics-SOAR trainer module loaded successfully")

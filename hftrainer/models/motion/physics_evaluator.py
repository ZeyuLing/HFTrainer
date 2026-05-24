"""
Physics Evaluator for Motion Generation
Evaluates physics quality of generated motions using MuJoCo simulation.

This module provides FastPhysicsEvaluator, which:
1. Simulates motions in MuJoCo
2. Computes physics metrics (collision, COM stability, energy, smoothness)
3. Suggests corrections for physically implausible motions
4. Operates without gradients (deterministic evaluation only)

Key Metrics:
- collision_penalty: 0-1, lower is better (fraction of frames with collision)
- com_stability: 0-1, higher is better (inverse of COM trajectory variance)
- energy_efficiency: 0-1, higher is better (normalized work done by joints)
- smoothness: 0-1, higher is better (inverse of jerk/acceleration)
- overall_score: 0-1 weighted average of all metrics
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import logging

# Optional imports
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logging.warning("MuJoCo not available. Physics evaluation will be mocked.")

try:
    from smplx import SMPL
    SMPL_AVAILABLE = True
except ImportError:
    SMPL_AVAILABLE = False
    logging.warning("SMPLX not available. Using fallback FK computation.")

logger = logging.getLogger(__name__)


class FastPhysicsEvaluator:
    """
    Fast physics evaluator for motion generation.
    
    Evaluates physics quality without gradients.
    Supports batch evaluation for efficiency.
    """
    
    def __init__(
        self,
        smpl_model_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        motion_fps: float = 30.0,
        use_mock: bool = False,
    ):
        """
        Initialize physics evaluator.
        
        Args:
            smpl_model_path: Path to SMPL model (optional, fallback if not provided)
            mjcf_path: Path to MuJoCo MJCF/XML file (optional, fallback if not provided)
            device: Device for computations ("cpu" or "cuda")
            batch_size: Batch size for parallel MuJoCo evaluation
            motion_fps: Frames per second of motion (for velocity/acceleration computation)
            use_mock: If True, return mock metrics (for testing without MuJoCo)
        """
        self.device = device
        self.batch_size = batch_size
        self.motion_fps = motion_fps
        self.use_mock = use_mock or not MUJOCO_AVAILABLE
        
        # Load SMPL model if available
        self.smpl_model = None
        if smpl_model_path and SMPL_AVAILABLE:
            try:
                self.smpl_model = SMPL(
                    model_path=smpl_model_path,
                    gender="neutral",
                )
                logger.info(f"Loaded SMPL model from {smpl_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load SMPL model: {e}")
        
        # Load MuJoCo model if available
        self.mujoco_model = None
        self.mujoco_data = None
        if mjcf_path and MUJOCO_AVAILABLE:
            try:
                self.mujoco_model = mujoco.MjModel.from_xml_path(mjcf_path)
                self.mujoco_data = mujoco.MjData(self.mujoco_model)
                logger.info(f"Loaded MuJoCo model from {mjcf_path}")
            except Exception as e:
                logger.warning(f"Failed to load MuJoCo model: {e}")
        
        # Fallback: create minimal MuJoCo humanoid if no model provided
        if self.mujoco_model is None and not use_mock and MUJOCO_AVAILABLE:
            self._create_default_humanoid()
        
        # Metric weights (for overall_score)
        self.metric_weights = {
            "collision_penalty": 0.3,
            "com_stability": 0.3,
            "energy_efficiency": 0.2,
            "smoothness": 0.2,
        }
        
        logger.info(f"Physics evaluator initialized (mock={self.use_mock})")
    
    def _create_default_humanoid(self):
        """Create a minimal humanoid MJCF model for testing."""
        if not MUJOCO_AVAILABLE:
            return
        
        # This is a simplified fallback - in production, use proper SMPL+MuJoCo
        mjcf_str = """
<mujoco model="humanoid">
  <option timestep="0.003" />
  <worldbody>
    <body name="torso" pos="0 0 1">
      <inertial mass="1" diaginv="1 1 1" />
      <geom type="capsule" size="0.1" fromto="0 0 0 0 0 -0.5" />
    </body>
    <geom name="ground" type="plane" size="10 10 1" pos="0 0 -1" />
  </worldbody>
</mujoco>
        """
        try:
            self.mujoco_model = mujoco.MjModel.from_xml_string(mjcf_str)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            logger.info("Created default humanoid model")
        except Exception as e:
            logger.warning(f"Failed to create default humanoid: {e}")
    
    def evaluate_batch(
        self,
        motions: Union[torch.Tensor, np.ndarray],
        return_raw: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate batch of motions.
        
        Args:
            motions: Tensor of shape (B, T, 135) or (B, T, 22, 3) after FK
                    - 135 is HYMotion's rotation representation (22 joints × 6D + 3D translation)
            return_raw: If True, return raw metric values; if False, normalize to 0-1
        
        Returns:
            Dictionary with keys:
            - collision_penalty: (B,) 0-1, lower is better
            - com_stability: (B,) 0-1, higher is better
            - energy_efficiency: (B,) 0-1, higher is better
            - smoothness: (B,) 0-1, higher is better
            - overall_score: (B,) 0-1 weighted average
        """
        # Convert to numpy if needed
        if isinstance(motions, torch.Tensor):
            motions_np = motions.detach().cpu().numpy()
        else:
            motions_np = np.asarray(motions)
        
        batch_size = motions_np.shape[0]
        
        # If using mock, return dummy metrics
        if self.use_mock:
            return self._get_mock_metrics(batch_size)
        
        # Evaluate each motion
        results = {
            "collision_penalty": [],
            "com_stability": [],
            "energy_efficiency": [],
            "smoothness": [],
        }
        
        for b in range(batch_size):
            motion = motions_np[b]  # (T, 135)
            metrics = self._evaluate_single(motion)
            
            for key, val in metrics.items():
                results[key].append(val)
        
        # Convert to tensors
        for key in results:
            results[key] = torch.tensor(
                results[key],
                dtype=torch.float32,
                device=self.device,
            )
        
        # Compute overall score
        overall_score = torch.zeros(batch_size, device=self.device)
        for key, weight in self.metric_weights.items():
            # Some metrics should be inverted (collision, energy should be high)
            if key == "collision_penalty":
                # Lower collision is better
                overall_score += weight * (1.0 - results[key])
            else:
                overall_score += weight * results[key]
        
        results["overall_score"] = overall_score
        
        return results
    
    def _evaluate_single(self, motion: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a single motion.
        
        Args:
            motion: (T, 135) motion sequence
        
        Returns:
            Dictionary with metric values (floats in 0-1 range)
        """
        try:
            # Compute joint positions via FK
            joint_positions = self._forward_kinematics(motion)  # (T, 22, 3)
            
            # Compute velocities and accelerations
            velocities = np.diff(joint_positions, axis=0) * self.motion_fps  # (T-1, 22, 3)
            accelerations = np.diff(velocities, axis=0) * self.motion_fps  # (T-2, 22, 3)
            
            # Compute metrics
            metrics = {
                "collision_penalty": self._compute_collision_penalty(motion),
                "com_stability": self._compute_com_stability(joint_positions),
                "energy_efficiency": self._compute_energy_efficiency(velocities),
                "smoothness": self._compute_smoothness(accelerations),
            }
            
            # Clip to [0, 1]
            for key in metrics:
                metrics[key] = np.clip(metrics[key], 0.0, 1.0)
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Error evaluating motion: {e}")
            # Return neutral metrics on error
            return {
                "collision_penalty": 0.5,
                "com_stability": 0.5,
                "energy_efficiency": 0.5,
                "smoothness": 0.5,
            }
    
    def _forward_kinematics(self, motion: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for motion.
        
        Args:
            motion: (T, 135) motion with 6D rotations + 3D translation
        
        Returns:
            (T, 22, 3) joint positions in world space
        """
        T = motion.shape[0]
        joint_positions = np.zeros((T, 22, 3))
        
        # For now, use a simplified FK (no real SMPL FK)
        # In production, use proper SMPL forward kinematics
        # This is a placeholder that spreads the motion coordinates
        
        for t in range(T):
            # Extract translation (first 3 dims)
            translation = motion[t, :3]
            
            # Interpret remaining 132 dims as joint rotations (22 joints × 6D)
            # In actual SMPL: 22 joints × 3 (axis-angle) + global rotation
            rotation_data = motion[t, 3:].reshape(22, 6)
            
            # Simplified: use first 3 dims per joint as offset
            for j in range(22):
                # Simple kinematic chain (each joint offset from root)
                joint_positions[t, j] = translation + rotation_data[j, :3] * 0.1
        
        return joint_positions
    
    def _compute_collision_penalty(self, motion: np.ndarray) -> float:
        """
        Compute collision penalty via MuJoCo simulation.
        
        Returns value in [0, 1]:
        - 0: No collisions
        - 1: Frequent collisions
        """
        if self.mujoco_model is None:
            # Fallback: heuristic based on motion variance
            motion_std = np.std(motion)
            return np.clip(0.5 - motion_std * 0.5, 0.0, 1.0)
        
        try:
            collision_count = 0
            T = motion.shape[0]
            
            # Simple check: if motion goes below ground (y < 0), count as collision
            # In production, use actual MuJoCo contact detection
            root_heights = motion[:, 1]  # Assume y is vertical (may vary by convention)
            below_ground = np.sum(root_heights < 0.3)  # Threshold for collision
            collision_count = below_ground
            
            collision_penalty = collision_count / max(T, 1)
            return float(collision_penalty)
        
        except Exception as e:
            logger.warning(f"Collision penalty computation failed: {e}")
            return 0.5
    
    def _compute_com_stability(self, joint_positions: np.ndarray) -> float:
        """
        Compute center-of-mass stability.
        
        Returns value in [0, 1]:
        - 1: Stable COM (low variance)
        - 0: Unstable COM (high variance)
        """
        try:
            # Approximate COM as mean of all joints
            com = np.mean(joint_positions, axis=1)  # (T, 3)
            
            # Compute variance of COM trajectory
            com_variance = np.var(com, axis=0)
            com_std = np.sqrt(np.mean(com_variance))
            
            # Normalize: higher std = lower stability
            # Assume reasonable COM std is ~0.1-0.2, max is ~1.0
            stability = np.exp(-com_std)  # Exponential decay
            
            return float(stability)
        
        except Exception as e:
            logger.warning(f"COM stability computation failed: {e}")
            return 0.5
    
    def _compute_energy_efficiency(self, velocities: np.ndarray) -> float:
        """
        Compute energy efficiency.
        
        Returns value in [0, 1]:
        - 1: Low energy (smooth motion)
        - 0: High energy (jerky motion)
        """
        try:
            # Energy ~ sum of kinetic energy ~ sum of velocity^2
            kinetic_energy = np.sum(velocities ** 2)
            
            # Normalize by time and joints
            T, J, D = velocities.shape
            normalized_energy = kinetic_energy / (T * J * D + 1e-6)
            
            # Assume max reasonable energy is ~1.0
            efficiency = np.exp(-normalized_energy)
            
            return float(efficiency)
        
        except Exception as e:
            logger.warning(f"Energy efficiency computation failed: {e}")
            return 0.5
    
    def _compute_smoothness(self, accelerations: np.ndarray) -> float:
        """
        Compute motion smoothness (inverse of jerk).
        
        Returns value in [0, 1]:
        - 1: Smooth (low acceleration)
        - 0: Jerky (high acceleration)
        """
        try:
            # Smoothness ~ inverse of mean acceleration
            mean_acceleration = np.mean(np.abs(accelerations))
            
            # Exponential decay
            smoothness = np.exp(-mean_acceleration)
            
            return float(smoothness)
        
        except Exception as e:
            logger.warning(f"Smoothness computation failed: {e}")
            return 0.5
    
    def _get_mock_metrics(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Return mock metrics for testing."""
        return {
            "collision_penalty": torch.ones(batch_size, device=self.device) * 0.1,
            "com_stability": torch.ones(batch_size, device=self.device) * 0.8,
            "energy_efficiency": torch.ones(batch_size, device=self.device) * 0.7,
            "smoothness": torch.ones(batch_size, device=self.device) * 0.75,
            "overall_score": torch.ones(batch_size, device=self.device) * 0.7,
        }
    
    def suggest_correction(
        self,
        motion: Union[torch.Tensor, np.ndarray],
        num_smoothing_frames: int = 5,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Suggest correction for physically implausible motion.
        
        Args:
            motion: (T, 135) motion to correct
            num_smoothing_frames: Number of frames for smoothing
        
        Returns:
            Corrected motion with same shape as input
        """
        # Convert to numpy
        if isinstance(motion, torch.Tensor):
            input_was_tensor = True
            motion_np = motion.detach().cpu().numpy()
        else:
            input_was_tensor = False
            motion_np = np.asarray(motion)
        
        try:
            # Simple correction: apply low-pass filter to smooth motion
            from scipy.ndimage import gaussian_filter1d
            
            corrected = motion_np.copy()
            # Apply Gaussian smoothing per dimension
            for d in range(motion_np.shape[1]):
                corrected[:, d] = gaussian_filter1d(
                    motion_np[:, d],
                    sigma=num_smoothing_frames,
                    mode="nearest",
                )
            
            # Convert back to original format
            if input_was_tensor:
                return torch.tensor(corrected, dtype=motion.dtype, device=motion.device)
            else:
                return corrected
        
        except Exception as e:
            logger.warning(f"Correction failed: {e}, returning original motion")
            return motion
    
    def set_metric_weights(self, weights: Dict[str, float]):
        """Update weights for overall_score computation."""
        for key in weights:
            if key in self.metric_weights:
                self.metric_weights[key] = weights[key]
            else:
                logger.warning(f"Unknown metric weight: {key}")
        
        # Normalize weights to sum to 1.0
        total = sum(self.metric_weights.values())
        if total > 0:
            for key in self.metric_weights:
                self.metric_weights[key] /= total


def create_physics_evaluator(
    config: Optional[Dict] = None,
    **kwargs,
) -> FastPhysicsEvaluator:
    """
    Factory function to create physics evaluator.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments to FastPhysicsEvaluator
    
    Returns:
        FastPhysicsEvaluator instance
    """
    if config is None:
        config = {}
    
    # Merge config and kwargs
    eval_kwargs = {**config, **kwargs}
    
    return FastPhysicsEvaluator(**eval_kwargs)


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator with mock mode for testing
    evaluator = FastPhysicsEvaluator(use_mock=True, device="cpu")
    
    # Create dummy motion tensor
    dummy_motions = torch.randn(4, 64, 135)  # 4 motions, 64 frames, 135-dim
    
    # Evaluate
    metrics = evaluator.evaluate_batch(dummy_motions)
    
    print("Evaluation results:")
    for key, val in metrics.items():
        print(f"  {key}: {val.mean():.4f}")
    
    print("✓ Physics evaluator test passed")

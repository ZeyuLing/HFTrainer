"""Isaac Gym / ASAP integration for G1 humanoid imitation learning.

This module provides wrappers to interface HyMotion-generated motions
(after retargeting to G1 joint space) with the ASAP / HumanoidVerse
reinforcement learning framework for motion imitation.

Pipeline overview:
  1. HyMotion T2M-Lite generates SMPL motion from text
  2. SMPLToG1Retargeter converts to G1 29-DOF joint angles
  3. This module creates Isaac Gym compatible configs and reference motions
  4. ASAP's PPO-based motion tracking trains a policy in simulation
  5. Policy can be deployed to sim2sim (MuJoCo) or sim2real

Dependencies (external, not bundled):
  - ASAP: https://github.com/LeCAR-Lab/ASAP
  - Isaac Gym Preview 4: https://developer.nvidia.com/isaac-gym-preview-4
  - MuJoCo (for sim2sim evaluation)

This module handles step 3: creating the bridge between our retargeted
motion and ASAP's training pipeline. It does NOT implement RL training
itself — that's handled by ASAP/HumanoidVerse.
"""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# Config Generator
# ============================================================================

class ASAPConfigGenerator:
    """Generate ASAP/HumanoidVerse configs for G1 motion tracking.

    Given a retargeted motion file (from SMPLToG1Retargeter.to_asap_pkl),
    this class generates the full set of configs and launch commands needed
    to train a motion tracking policy in Isaac Gym.
    """

    # Default ASAP repo path — user should override
    DEFAULT_ASAP_ROOT = os.environ.get(
        'ASAP_ROOT',
        os.path.expanduser('~/ASAP'),
    )

    # Default G1 robot config in ASAP
    DEFAULT_ROBOT_CFG = 'g1/g1_29dof_anneal_23dof'

    def __init__(
        self,
        asap_root: Optional[str] = None,
        robot_cfg: str = DEFAULT_ROBOT_CFG,
        num_envs: int = 4096,
        project_name: str = 'HyMotion_G1',
    ):
        self.asap_root = asap_root or self.DEFAULT_ASAP_ROOT
        self.robot_cfg = robot_cfg
        self.num_envs = num_envs
        self.project_name = project_name

    def generate_training_command(
        self,
        motion_file: str,
        experiment_name: str = 'hymotion_t2m_g1',
        headless: bool = True,
        reward_penalty_curriculum: bool = True,
        reward_penalty_degree: float = 0.00001,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the ASAP training command string.

        Args:
            motion_file: Path to the retargeted motion .pkl file.
            experiment_name: Name for W&B / logging.
            headless: Whether to run without GUI.
            reward_penalty_curriculum: Enable curriculum-based reward penalty.
            reward_penalty_degree: Penalty scaling factor.
            extra_args: Additional hydra overrides.

        Returns:
            Full shell command string.
        """
        cmd_parts = [
            f'cd {self.asap_root} &&',
            'python humanoidverse/train_agent.py',
            '+simulator=isaacgym',
            '+exp=motion_tracking',
            '+domain_rand=NO_domain_rand',
            '+rewards=motion_tracking/reward_motion_tracking_dm_2real',
            f'+robot={self.robot_cfg}',
            '+terrain=terrain_locomotion_plane',
            '+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history',
            f'num_envs={self.num_envs}',
            f'project_name={self.project_name}',
            f'experiment_name={experiment_name}',
            f'robot.motion.motion_file="{motion_file}"',
            f'headless={str(headless).lower()}',
        ]

        if reward_penalty_curriculum:
            cmd_parts.append('rewards.reward_penalty_curriculum=True')
            cmd_parts.append(f'rewards.reward_penalty_degree={reward_penalty_degree}')

        if extra_args:
            for k, v in extra_args.items():
                cmd_parts.append(f'{k}={v}')

        return ' \\\n  '.join(cmd_parts)

    def generate_eval_command(
        self,
        checkpoint_path: str,
        headless: bool = False,
    ) -> str:
        """Generate evaluation/visualization command."""
        return ' \\\n  '.join([
            f'cd {self.asap_root} &&',
            'python humanoidverse/eval_agent.py',
            f'+checkpoint={checkpoint_path}',
            f'headless={str(headless).lower()}',
        ])

    def generate_sim2sim_commands(
        self,
        policy_onnx_path: str,
        config_yaml: str = 'config/g1_29dof_hist.yaml',
    ) -> Dict[str, str]:
        """Generate sim2sim deployment commands.

        ASAP sim2sim uses two terminals:
          Terminal 1: MuJoCo simulator
          Terminal 2: RL policy inference

        Returns:
            Dict with 'simulator' and 'policy' command strings.
        """
        sim_dir = os.path.join(self.asap_root, 'sim2real')
        return {
            'simulator': (
                f'cd {sim_dir} && '
                f'python sim_env/base_sim.py --config={config_yaml}'
            ),
            'policy': (
                f'cd {sim_dir} && '
                f'python rl_policy/deepmimic_dec_loco_height.py '
                f'--config={config_yaml} '
                f'--mimic_model_paths={policy_onnx_path}'
            ),
        }

    def check_asap_installation(self) -> Dict[str, bool]:
        """Check if ASAP and dependencies are properly installed."""
        checks = {}

        # Check ASAP root exists
        checks['asap_root'] = os.path.isdir(self.asap_root)

        # Check Isaac Gym
        try:
            import isaacgym  # noqa: F401
            checks['isaac_gym'] = True
        except ImportError:
            checks['isaac_gym'] = False

        # Check MuJoCo
        try:
            import mujoco  # noqa: F401
            checks['mujoco'] = True
        except ImportError:
            checks['mujoco'] = False

        # Check hydra (used by ASAP)
        try:
            import hydra  # noqa: F401
            checks['hydra'] = True
        except ImportError:
            checks['hydra'] = False

        return checks


# ============================================================================
# Reference Motion Manager
# ============================================================================

class ReferenceMotionManager:
    """Manage reference motions for ASAP motion tracking training.

    Handles batching multiple retargeted motions into a single training
    dataset, with metadata for curriculum scheduling.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.motions: List[Dict[str, Any]] = []

    def add_motion(
        self,
        retarget_result: Dict[str, np.ndarray],
        name: str,
        text_prompt: str = '',
        difficulty: float = 0.5,
    ):
        """Add a retargeted motion to the collection.

        Args:
            retarget_result: Output from SMPLToG1Retargeter.retarget()
            name: Unique name for this motion clip
            text_prompt: Original text prompt used for generation
            difficulty: Estimated difficulty 0-1 for curriculum learning
        """
        from hftrainer.motion.retarget import SMPLToG1Retargeter

        retargeter = SMPLToG1Retargeter()
        pkl_path = os.path.join(self.output_dir, f'{name}.pkl')
        retargeter.to_asap_pkl(retarget_result, pkl_path)

        self.motions.append({
            'name': name,
            'path': pkl_path,
            'text_prompt': text_prompt,
            'difficulty': difficulty,
            'num_frames': retarget_result['joint_angles'].shape[0],
            'fps': retarget_result['fps'],
            'duration': retarget_result['joint_angles'].shape[0] / retarget_result['fps'],
        })

    def save_manifest(self) -> str:
        """Save manifest JSON listing all motions."""
        manifest_path = os.path.join(self.output_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump({
                'motions': self.motions,
                'total_motions': len(self.motions),
                'total_duration': sum(m['duration'] for m in self.motions),
            }, f, indent=2)
        return manifest_path

    def get_motion_file_for_asap(self) -> str:
        """Get the motion file path suitable for ASAP training.

        If single motion, returns the pkl directly.
        If multiple motions, creates a combined file.
        """
        if len(self.motions) == 1:
            return self.motions[0]['path']

        # For multiple motions, save as a list-format pkl
        import pickle
        combined = {
            'type': 'multi_motion',
            'motions': [],
        }
        for m in self.motions:
            with open(m['path'], 'rb') as f:
                data = pickle.load(f)
            combined['motions'].append({
                'name': m['name'],
                'data': data,
            })

        combined_path = os.path.join(self.output_dir, 'combined_motions.pkl')
        with open(combined_path, 'wb') as f:
            pickle.dump(combined, f)

        return combined_path

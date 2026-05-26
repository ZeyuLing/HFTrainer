"""PhysFlow Phase 1 Direction B Configuration — Gen→RL Training

This configuration runs Phase 1 Direction B: RL training on T2M-generated motions
to improve physics plausibility while maintaining generation quality.

Phase 1 Objective:
    Improve Physics Pass Rate (PPR) from baseline through RL training
    
Configuration ID: C1 (Direction B - Gen→RL pipeline)

Key Settings:
    - Model: HyMotionT2MBundle (from Phase 0 baseline)
    - Training: RL policy on T2M outputs (Direction B specific)
    - Evaluation: Compare against Phase 0 baseline metrics
    - Success Criteria: PPR gain ≥ 10% with FID < 0.7
"""

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

experiment_config = {
    'name': 'PhysFlow Phase 1 Direction B (Gen→RL)',
    'phase': 'Phase 1',
    'direction': 'B',  # Gen→RL
    'config_id': 'C1',
    'description': 'RL training on T2M generator outputs for physics improvement',
}

# ============================================================================
# PHASE 0 BASELINE REFERENCE
# ============================================================================

phase0_baseline = {
    'fid': 0.537,
    'ppr': 0.331,
    'r_precision@3': 0.395,
    'diversity': 0.716,
    'num_samples': 200,
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

data_config = {
    # Training data: HumanML3D
    'train_dataset': {
        'name': 'HumanML3D',
        'split': 'train',
        'num_samples': None,  # Use full training set
    },
    
    # Evaluation data: Same as Phase 0 for comparison
    'eval_dataset': {
        'name': 'HumanML3D',
        'split': 'test',
        'num_samples': 200,  # Match Phase 0
        'seed': 42,
    },
    
    # Motion representation
    'motion_representation': {
        'format': '135d',  # Absolute translation (3D) + local rotation (6D × 22 joints)
        'fps': 20,
        'duration': 'variable',  # 40-196 frames typically
    },
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

model_config = {
    # T2M Generator (from Phase 0, fine-tuned in Phase 1)
    't2m_model': {
        'name': 'HyMotionT2MBundle',
        'checkpoint': '/path/to/hymotion_t2m_1.8gb.pt',
        'trainable': True,  # Fine-tune T2M generator
        'freeze_backbone': False,  # Allow full fine-tuning
    },
    
    # RL Policy (NEW in Phase 1)
    'rl_policy': {
        'name': 'MotionRLPolicy',
        'action_space': 'joint_targets',  # 69-dim SMPL joint targets
        'state_space': 'extended_smpl_state',
        'hidden_dim': 256,
        'num_layers': 3,
        'activation': 'relu',
    },
    
    # Reward Function (Direction B specific)
    'reward_function': {
        'type': 'physics_guided',
        'components': {
            'physics_validity': {'weight': 0.5, 'target': 1.0},
            'tracking': {'weight': 0.3, 'target': 'minimize_error'},
            'smoothness': {'weight': 0.1, 'target': 'minimize_jerk'},
            'text_alignment': {'weight': 0.1, 'target': 'maintain_fid'},
        },
    },
    
    # Constraint System
    'constraints': {
        'use_mujoco': True,
        'contact_constraints': True,
        'joint_limits': True,
        'collision_avoidance': True,
    },
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

training_config = {
    'phase1_training': {
        'mode': 'direction_b_gen_rl',
        'duration': '200k_steps',
        'batch_size': 32,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'warmup_steps': 10000,
        'gradient_accumulation_steps': 2,
    },
    
    # T2M Generator fine-tuning
    't2m_training': {
        'enabled': True,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
    },
    
    # RL Training
    'rl_training': {
        'algorithm': 'PPO',
        'num_envs': 16,
        'horizon': 300,  # Max steps per episode
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
    },
    
    # Checkpointing
    'checkpointing': {
        'save_interval': 10000,
        'keep_best': True,
        'metric_to_track': 'ppr_improvement',
    },
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

inference_config = {
    'batch_size': 16,
    'temperature': 1.0,
    'guidance_scale': None,  # No classifier-free guidance in Phase 1
    'num_rollouts': 1,
    'use_rl_policy': True,  # Apply RL corrections to T2M outputs
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

evaluation_config = {
    # Metrics to compute
    'compute_fid': True,
    'compute_r_precision': True,
    'compute_diversity': True,
    'compute_ppr': True,  # Critical metric for Phase 1
    'compute_jerk': True,
    'compute_tracking_error': True,
    
    # Evaluation settings
    'num_eval_samples': 200,  # Match Phase 0 for direct comparison
    'eval_batch_size': 16,
    'save_per_sample_metrics': True,
    
    # Physics Evaluation (Direction B specific)
    'physics_evaluation': {
        'simulator': 'mujoco',
        'max_sim_steps': 300,
        'contact_threshold': 0.05,
        'joint_limit_margin': 0.05,
    },
}

# ============================================================================
# EXPERIMENT METADATA
# ============================================================================

experiment_metadata = {
    'phase': 'Phase 1',
    'direction': 'B',
    'config_id': 'C1',
    'name': 'Gen→RL Training Pipeline',
    
    # Expected performance improvements
    'expected_improvements': {
        'ppr_gain_min': 0.10,  # +10%
        'ppr_gain_target': 0.15,  # +15%
        'ppr_final_target': 0.43,  # From 0.331 + 0.10
        'fid_threshold': 0.70,  # Allow slight increase
        'diversity_maintained': True,
    },
    
    # Success criteria for Phase 1 gate
    'phase1_gate_criteria': {
        'criterion_1': 'PPR improvement ≥ 10%',
        'criterion_2': 'FID < 0.70',
        'criterion_3': 'Diversity > 0.70',
        'criterion_4': 'No training divergence (loss stable)',
    },
    
    # Comparison to baseline
    'baseline_reference': {
        'phase': 'Phase 0',
        'config_id': 'C0',
        'ppr_baseline': 0.331,
        'fid_baseline': 0.537,
        'r_precision_baseline': 0.395,
        'diversity_baseline': 0.716,
    },
    
    # Hyperparameter notes
    'hyperparameter_notes': {
        'learning_rate_t2m': '5e-5 for stable fine-tuning',
        'learning_rate_rl': '1e-4 for policy learning',
        'ppo_entropy': '0.01 to encourage exploration without instability',
        'physics_weight': '0.5 to balance plausibility vs generation quality',
    },
    
    # Expected timeline
    'expected_runtime': {
        'training': '2-4 hours',
        'evaluation': '0.5-1 hour',
        'total': '3-5 hours',
    },
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

output_config = {
    'output_base_dir': 'results/physflow_phase1/',
    'experiment_dir': 'results/physflow_phase1/c1_direction_b_gen_rl/',
    
    'save_format': {
        'metrics': 'json',
        'trajectories': 'npz',
        'models': 'pytorch',
    },
    
    'artifacts_to_save': [
        'final_metrics.json',
        'training_curves.json',
        'generated_motions.npz',
        'generated_videos.mp4',
        'model_checkpoint.pt',
        'config_snapshot.json',
    ],
}

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

logging_config = {
    'log_level': 'INFO',
    'tensorboard': True,
    'tensorboard_dir': 'runs/physflow_phase1_c1',
    'wandb': False,
    'log_interval': 100,
    'eval_interval': 5000,
}

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

reproducibility_config = {
    'seed': 42,
    'deterministic': True,
    'benchmark': False,  # Disable cuDNN benchmark for reproducibility
}

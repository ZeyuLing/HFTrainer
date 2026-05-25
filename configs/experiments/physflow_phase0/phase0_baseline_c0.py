# PhysFlow Phase 0 Baseline — Configuration C0
#
# **Experiment Objective**:
#   Establish baseline T2M generation performance using current HyMotion model
#   without any RL correction or bidirectional training.
#
# **Configuration Parameters**:
#   - C0: Pure T2M generation (no RL correction)
#   - No Direction A (RL→Gen) correction
#   - No Direction B (Gen→RL) training data generation
#   - Purpose: Establish upper bound on generation quality
#
# **Evaluation Sets**:
#   - GEN-STD: 4646 HumanML3D test prompts (standard evaluation)
#   - GEN-PHYS: 200 physics-sensitive prompts (physics metrics focus)
#   - TR-ID: 200 AMASS in-distribution test motions (for RL tracker if available)
#   - TR-OOD-H: 200 HumanML3D hard OOD motions (challenging generalization)
#
# **Expected Metrics**:
#   - FID (Fréchet Inception Distance): ~0.50-0.60 (baseline T2M quality)
#   - R-Prec (Recall@3): ~0.30-0.40 (text-motion relevance)
#   - PPR (Physics Pass Rate): ~30-50% (physics realism without RL correction)
#   - Diversity: ~0.70-0.80 (multimodality across generations)
#
# **Computation Requirements**:
#   - Single V100 32GB GPU
#   - Time: 2-4 hours for 200 generation samples
#   - Storage: ~1GB for generated motions + metrics
#

# ============================================================================
# 1. Data Configuration
# ============================================================================

data_test = dict(
    type='HumanML3DDataset',
    split='test',
    num_samples=4646,  # Full test split for GEN-STD evaluation
    motion_format='135d',  # HyMotion standard format (22 joints × 3 = 66 dims + scale/root)
    preprocess=dict(
        normalize_motion=True,
        flip_prob=0.0,  # No augmentation for evaluation
        mask_prob=0.0,
    ),
)

data_test_phys = dict(
    type='HumanML3DPhysicsDataset',  # Custom subset with physics-sensitive prompts
    split='test',
    num_samples=200,
    prompt_filter='physics_sensitive',  # Filter prompts requiring physical plausibility
    motion_format='135d',
    preprocess=dict(
        normalize_motion=True,
        flip_prob=0.0,
        mask_prob=0.0,
    ),
)

# ============================================================================
# 2. Model Configuration
# ============================================================================

_motion_dim = 135  # 22 joints × 3 dims + 69 root dims = 135

model = dict(
    type='HyMotionT2MBundle',
    motion_cond_mask_prob=0.0,  # No motion condition dropout during inference
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=False,  # Frozen for C0 baseline (no training)
        input_dim=_motion_dim,
        output_dim=_motion_dim,
        feat_dim=1024,
        num_layers=18,
        num_heads=16,
        mlp_ratio=4.0,
        mlp_act_type='gelu_tanh',
        norm_type='layer',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        use_fp32_upcast_attention=False,  # Not needed for inference
        init_cfg=None,
    ),
    text_encoder=dict(
        type='CLIPTextEncoder',
        model_name='openai/clip-vit-large-patch14',
        device='cuda',
        output_dim=768,
    ),
    caption_encoder=dict(
        type='T5TextEncoder',
        model_name='t5-large',
        max_seq_length=77,
        device='cuda',
    ),
)

# Load pretrained checkpoint
load_from = dict(
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)

# ============================================================================
# 3. Inference / Evaluation Configuration
# ============================================================================

inference = dict(
    batch_size=16,  # Process 16 samples at once
    num_samples_per_prompt=1,  # Single sample per text prompt for GEN-STD
    num_samples_per_prompt_diversity=10,  # 10 samples for diversity measurement
    temperature=1.0,  # Deterministic for reproducibility
    top_k=0,
    top_p=0.95,
    guidance_scale=1.0,  # No classifier-free guidance
    return_intermediates=False,
    device='cuda',
)

# ============================================================================
# 4. Evaluation / Metrics Configuration
# ============================================================================

evaluation = dict(
    # FID computation
    compute_fid=True,
    fid_reference_model='inception_v3',
    fid_batch_size=32,
    
    # Text-Motion alignment
    compute_r_precision=True,
    r_precision_top_k=[3, 6, 12],
    
    # Diversity metrics
    compute_diversity=True,
    diversity_method='std_dev',  # Standard deviation across generated motions
    
    # Physics metrics (if RL oracle available)
    compute_ppr=True,
    ppr_simulator='mujoco',  # Use MuJoCo for lightweight physics check
    ppr_timeout=5.0,  # Max 5 seconds per motion physics simulation
    ppr_num_trials=3,  # 3 forward steps to check stability
    
    # Kinematic metrics
    compute_mpjpe=False,  # Not applicable for generative evaluation
    compute_joint_limits=True,  # Check if joints exceed limits
)

# ============================================================================
# 5. Output Configuration
# ============================================================================

output_dir = 'results/physflow_phase0/c0_baseline_t2m'

save_config = dict(
    save_generated_motions=True,
    motion_format='pkl',  # Save as pickle for analysis
    save_metrics_csv=True,
    save_metrics_json=True,
    save_visualization=True,
    viz_format='mp4',  # Video format for comparison
)

# ============================================================================
# 6. Logging & Tracking
# ============================================================================

log_config = dict(
    interval=10,  # Log every 10 samples
    level='INFO',
)

# Optional: Weights & Biases tracking
wandb_config = dict(
    project='physflow',
    entity='motion-group',
    group='phase0_baseline',
    name='c0_t2m_baseline',
    tags=['phase0', 'baseline', 'c0', 't2m'],
    notes='Pure T2M generation without RL correction (baseline)',
)

# ============================================================================
# 7. Experimental Metadata
# ============================================================================

experiment_metadata = dict(
    phase='Phase 0',
    config_id='C0',
    experiment_name='Baseline T2M Generation',
    description='Establish baseline performance of HyMotion T2M without RL correction',
    objective='Measure current T2M generation quality (FID, R-Prec, PPR)',
    direction_a_enabled=False,
    direction_b_enabled=False,
    expected_ppr_range=[0.30, 0.50],
    expected_fid_range=[0.50, 0.60],
    expected_diversity=[0.70, 0.80],
)

# ============================================================================
# 8. Phase 0 Success Criteria
# ============================================================================

success_criteria = dict(
    # Must collect baseline metrics
    metrics_collected=['fid', 'r_precision', 'diversity', 'ppr'],
    
    # Baseline sanity checks
    fid_range=[0.40, 1.50],  # Reasonable bounds (may vary)
    diversity_range=[0.50, 1.00],  # Reasonable bounds
    ppr_range=[0.10, 0.70],  # PPR may be low without RL correction
    
    # Confirm forward pass works
    inference_speed_ok=True,  # Should complete in reasonable time
    no_nan_in_output=True,  # No NaN values in generations
    motion_shapes_valid=True,  # All motions valid shape (num_frames, 135)
    
    # Gate for Phase 1
    gate_to_phase1=dict(
        condition='ppr > 0.25 and fid < 1.0',
        description='Baseline must show some physical plausibility',
    ),
)

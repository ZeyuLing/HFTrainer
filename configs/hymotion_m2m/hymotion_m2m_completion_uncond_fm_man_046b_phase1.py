# HyMotion M2M 0.46B — Phase 1: Task Instruction Modulation + Mask-Aware Noise
#
# Phase 1 improvements:
#   1. Task Instruction Modulation: CLIP-encoded natural language task descriptions
#      injected into the timestep adapter signal (e.g. "complete motion from sparse 
#      random cells" for M1, "extend or bridge motion temporally" for M3, etc.)
#   2. Mask-Aware Noise (V4): known regions in x_t stay clean during training
#   3. 7 Mask Strategies (M1-M7): improved coverage of motion editing tasks
#
# Expected improvements (target):
#   - Boundary quality: +2-3% (task awareness helps known-region transition)
#   - Motion naturalness: +1-2% FID improvement (task instructions provide semantic guidance)
#   - Robustness across mask patterns: uniform improvement across all 7 strategies
#
# Architecture:
#   - Model: HyMotionMMDiT (0.46B) with task_emb injection into adapter signal
#   - Adapter: timestep_feat (1024) + vtxt_feat (1024) + task_emb (1024)
#   - Task encoding: CLIP-L (768-dim) → projection (768→1024) via HYTextModel's vtxt_encoder
#   - Strategy mapping: 7 strategies map to natural language via task_instruction.py
#
# Data flow:
#   1. Dataset → mask_strategy ∈ {m1_random_cell, ..., m7_scattered_joint}
#   2. Trainer → get_task_instruction(mask_strategy) → natural language text
#   3. Trainer → bundle.encode_task_instruction([instructions]) → {task_emb: (B, 1, 1024)}
#   4. MMDiT.forward(..., task_emb=task_emb) → adapter += task_emb
#   5. All ModulateDiT layers receive task-aware adapter signal
#
# Hyperparameter defaults (phase 1 baseline):
#   - Learning rate: 1e-4 (same as uncond_fm_man)
#   - Warmup steps: 5K
#   - EMA: enabled (decay=0.9999)
#   - Batch size: 8×32 = 256 per GPU (across 8 GPUs)
#   - Max epochs: 1000
#
# Known limitations:
#   - Task instructions are fixed natural language descriptions (not learned)
#   - Instruction→vector mapping (via CLIP encoder) is frozen from T2M pretraining
#   - No curriculum or weighting based on task difficulty yet (phase 2 improvement)
#
# Launch:
#   python tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b_phase1.py
#   bash tools/dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b_phase1.py 8
#   bash tools/taiji_dist_train.sh configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b_phase1.py

_base_ = './hymotion_m2m_completion_uncond_fm_man_046b.py'

work_dir = 'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b_phase1'

# Phase 1: Enable task instruction modulation in trainer
trainer = dict(
    mask_aware_noise=True,
    # NEW: Enable task instruction encoding for all training samples
    encode_task_instruction=True,
    # Task instruction settings
    task_instruction_cfg=dict(
        # Whether to encode task instructions (required for modulation)
        enabled=True,
        # Which strategies to encode (all 7 by default)
        strategies=['m1_random_cell', 'm2_random_block', 'm3_temporal_contiguous',
                    'm4_joint_contiguous', 'm5_full_mask', 'm6_keyframe_sparse', 
                    'm7_scattered_joint'],
        # Fallback instruction if strategy not found (shouldn't happen in normal operation)
        default_instruction='complete motion from sparse random cells',
    ),
)

# Optional: Monitor task embedding statistics during training
log_config = dict(
    interval=100,
    hooks=[
        dict(
            type='TextLoggerHook',
            by_epoch=False,
        ),
        dict(
            type='TensorboardLoggerHook',
            by_epoch=False,
        ),
    ],
)

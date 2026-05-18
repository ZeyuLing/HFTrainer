# HyMotion M2M v2 0.46B with T2M Pretrained Loading
#
# This config extends the base M2M v2 configuration with selective T2M pretrained
# checkpoint loading. It reuses text encoders and transformer blocks from HyMotion-T2M
# while reinitializing components that don't match the M2M v2 architecture.
#
# Key features:
#   - Loads T2M pretrained weights (text encoders, transformer blocks)
#   - Reinitializes input_encoder (135→594 VACE expansion) and final_layer (135→198)
#   - Supports freezing strategies: 'none', 'encoders', 'text_refiner', 'blocks', 'full'
#   - Default strategy 'encoders' freezes text understanding modules only
#   - Transformer blocks remain trainable for VACE/caption adaptation
#
# Usage:
#   python tools/train.py configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_pretrained.py
#
# Transfer learning rationale:
#   - T2M trained on 400h of text-to-motion data, has strong text understanding
#   - Text encoders (Qwen3, CLIP) stable across tasks, reuse saves compute
#   - Timestep/text embedding layers task-specific but architecturally identical
#   - Transformer blocks mostly task-agnostic (attention, MLP, layer norm patterns)
#   - Input/output layers highly task-specific (VACE vs standard, dim changes)

_base_ = '_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_046b_t2m_pretrained'

# ----- Model -----
# Extend base config with T2M pretrained checkpoint loading
model = dict(
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    # Freezing strategy for loaded T2M modules:
    # - 'none': all loaded modules remain trainable (default, no transfer learning benefit)
    # - 'encoders': freeze text encoders + timestep encoder (recommended)
    #   → Text understanding stable, transformer blocks adapt to VACE/caption
    # - 'text_refiner': also freeze text_refiner
    # - 'blocks': freeze transformer blocks only (train only embeddings/encoders)
    # - 'full': freeze all loaded modules except input/output layers
    t2m_freeze_strategy='encoders',
)

# ----- Logging -----
# Add logging for checkpoint loading statistics
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=100, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)

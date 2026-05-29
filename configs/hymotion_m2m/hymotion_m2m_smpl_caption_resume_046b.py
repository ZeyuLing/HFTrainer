# HyMotion M2M v2 — SMPL Root + Caption: Resume from E2 epoch_730
#
# **Resume training from E2 checkpoint** with proper two-stage loading:
#   1. T2M pretrained weights loaded first (clean text encoders) via t2m_pretrained_path
#   2. E2 checkpoint loaded second, but skip_frozen=True prevents overwriting
#      the frozen vtxt/ctxt/timestep encoders (which are collapsed in E2)
#
# This gives us:
#   - Transformer blocks, input_encoder, final_layer from E2 epoch_730 (trained weights)
#   - vtxt_encoder, ctxt_encoder, timestep_encoder from T2M pretrained (healthy encoders)
#   - caption_freeze_strategy='encoders' prevents encoder collapse going forward
#
# The key insight: E2's encoders have cos(text, null) > 0.98 (collapsed),
# so we must NOT load them from E2. skip_frozen ensures frozen params
# (set by caption_freeze_strategy) are not overwritten by the E2 checkpoint.
#
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_smpl_caption_r4 \
#     configs/hymotion_m2m/hymotion_m2m_smpl_caption_resume_046b.py --host_num 8

_base_ = './hymotion_m2m_smpl_caption_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_resume_E2'

# Two-stage loading:
# Stage 1 (in Bundle.__init__): T2M pretrained → clean encoders, then freeze
# Stage 2 (in _pre_prepare_load): E2 epoch_730 → everything else (skip frozen)
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_730',
    load_scope='model',
    skip_frozen=True,  # Don't overwrite frozen encoders with collapsed E2 values
)

model = dict(
    # T2M pretrained loaded in __init__ BEFORE caption_freeze_strategy takes effect
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    caption_freeze_strategy='encoders',  # Freeze vtxt/ctxt/timestep encoders
)

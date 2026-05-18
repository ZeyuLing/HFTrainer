# HyMotion M2M v2 — KIMODO Root + Caption + PerMo: Resume from E4 epoch_660
#
# **Resume training from E4 checkpoint** with proper two-stage loading:
#   1. T2M pretrained weights loaded first (clean text encoders) via t2m_pretrained_path
#   2. E4 checkpoint loaded second, but skip_frozen=True prevents overwriting
#      the frozen vtxt/ctxt/timestep encoders (which are collapsed in E4)
#
# This gives us:
#   - Transformer blocks, input_encoder, final_layer from E4 epoch_660 (trained weights)
#   - vtxt_encoder, ctxt_encoder, timestep_encoder from T2M pretrained (healthy encoders)
#   - caption_freeze_strategy='encoders' prevents encoder collapse going forward
#
# The key insight: E4's encoders have cos(text, null) > 0.98 (collapsed),
# so we must NOT load them from E4. skip_frozen ensures frozen params
# (set by caption_freeze_strategy) are not overwritten by the E4 checkpoint.
#
#
# Launch (Taiji, 64 GPUs):
#   python tools/taiji_submit.py m2m_v2_kimodo_caption_permo_r7 \
#     configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_resume_046b.py --host_num 8

_base_ = './hymotion_m2m_v2_kimodo_caption_permo_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4'

# Two-stage loading:
# Stage 1 (in Bundle.__init__): T2M pretrained → clean encoders, then freeze
# Stage 2 (in _pre_prepare_load): E4 epoch_660 → everything else (skip frozen)
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_660',
    load_scope='model',
    exclude_bundle_keys=['mean', 'std'],  # Prefer stats loaded from mean_std_dir in __init__
    skip_frozen=True,  # Don't overwrite frozen encoders with collapsed E4 values
)

model = dict(
    # T2M pretrained loaded in __init__ BEFORE caption_freeze_strategy takes effect
    t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    caption_freeze_strategy='encoders',  # Freeze vtxt/ctxt/timestep encoders
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',  # KIMODO stats
)

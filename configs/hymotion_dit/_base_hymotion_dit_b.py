# HyMotion DiT Base — Text-free DiT base config.
#
# Architecture: HunyuanMotionDiT (text-free, single-stream only)
# Size: feat_dim=1024, num_layers=18, num_heads=16 → ~288M params
# Motion: smpl_22 with rotation_6d, 135 dims (3 abs transl + 6*22 rot6d)
# VACE: input_dim = 135 + 3*135 = 540
#
# No pretrained weights — train from scratch.

_base_ = './_base_hymotion_dit_s.py'

work_dir = 'work_dirs/hymotion_dit_b'

model = dict(
    motion_transformer=dict(
        feat_dim=1024,
        num_layers=18,
        num_heads=16,
    ),
)

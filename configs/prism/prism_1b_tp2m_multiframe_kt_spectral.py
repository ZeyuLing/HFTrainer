# PRISM 1B text+pose-to-motion, multi-frame conditioning + KT-RoPE spectral
#
# KT-RoPE replaces flat sequential joint indices with Laplacian spectral
# coordinates from the SMPL-22 kinematic tree. This encodes kinematic distance
# as RoPE attention bias: parent-child joints (e.g., knee->ankle) receive
# high correlation, while unrelated joints (L_Foot<->R_Foot) receive proper
# separation. Correlation with tree distance: 0.849 vs 0.397 (sequential).
#
# Zero additional parameters -- spectral coordinates are precomputed constants.

_base_ = './prism_1b_tp2m_multiframe.py'

model = dict(
    transformer=dict(
        joint_pos_mode="spectral",  # KT-RoPE spectral mode
        num_spectral_modes=4,  # First 4 Laplacian eigenvectors
        spectral_scale=22.0,  # Scale spectral coords (default = num_joints)
        # module_dtype stays bf16 from base config; V100 emulates bf16 compute.
        # fp16 module_dtype causes NaN due to overflow from bf16-trained weights.
    ),
)

# Fix checkpoint saving: base config has interval=2000 but by_epoch auto-inherits
# True from train_cfg, making it "save every 2000 epochs" (= never).
# Override to save every 2000 iterations explicitly.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        max_keep_ckpts=5,
        save_last=True,
        by_epoch=False,  # save by iteration, not by epoch
    ),
)

# Keep bf16 compute (same as base config); fp16 mixed precision causes NaN
# with bf16-trained weights due to narrower dynamic range.
# Speed gains come from: persistent_workers, more num_workers, fixed batch_size.
accelerator = dict(
    mixed_precision='no',  # bf16 compute from base config; fp16 autocast causes NaN
    gradient_accumulation_steps=1,
    fsdp_plugin=dict(
        sharding_strategy='FULL_SHARD',
        backward_prefetch='BACKWARD_PRE',
        auto_wrap_policy='TRANSFORMER_BASED_WRAP',
        transformer_cls_names_to_wrap=['WanTransformerBlockWithMask'],
        state_dict_type='FULL_STATE_DICT',
        sync_module_states=True,
        use_orig_params=True,
        cpu_offload=False,
    ),
)

# Increase dataloader workers and enable persistent workers for faster data loading
train_dataloader = dict(
    batch_size=6,
    num_workers=12,
    persistent_workers=True,
    shuffle=True,
)

# Load weights from the sequential RoPE checkpoint (model weights only).
# RoPE buffers are non-persistent and will be recomputed with spectral coords.
load_from = dict(
    _delete_=True,
    path='work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000',
    load_scope='model',
)

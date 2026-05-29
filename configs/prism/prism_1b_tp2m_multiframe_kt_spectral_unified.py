# PRISM 1B text+pose-to-motion, multi-frame conditioning + KT-RoPE spectral_unified
#
# FIX for original "spectral" mode: uses the SAME frequency basis as sequential
# (full j_dim=64) with spectral-derived scalar positions. This is compatible
# with pretrained sequential weights — the attention distance metric uses
# identical frequency components, only the position values change.
#
# Original "spectral" mode splits j_dim into 4 independent 16-dim frequency
# spaces with different decay rates, making it incompatible with sequential
# pretrained weights and causing catastrophic translation errors.

_base_ = './prism_1b_tp2m_multiframe.py'

model = dict(
    transformer=dict(
        joint_pos_mode="spectral_unified",  # Fixed KT-RoPE spectral mode
        num_spectral_modes=4,  # First 4 Laplacian eigenvectors
        spectral_scale=22.0,  # Scale spectral coords (default = num_joints)
    ),
    # ``load_from`` overwrites these frozen modules with the sequential PRISM
    # checkpoint versions.  Keep them in model.pt so saved KT checkpoints replay
    # in the same latent/statistics space used during training.
    vae=dict(save_ckpt=True),
    smpl_pose_processor=dict(save_ckpt=True),
)

# Fix checkpoint saving
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        max_keep_ckpts=5,
        save_last=True,
        by_epoch=False,
    ),
)

# Keep bf16 compute
accelerator = dict(
    mixed_precision='no',
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

train_dataloader = dict(
    batch_size=6,
    num_workers=12,
    persistent_workers=True,
    shuffle=True,
)

# Load weights from the sequential RoPE checkpoint (model weights only).
# spectral_unified uses same buffer structure as spectral/dfs, RoPE buffers
# will be recomputed with the unified spectral positions.
load_from = dict(
    _delete_=True,
    path='work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000',
    load_scope='model',
)

work_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified'

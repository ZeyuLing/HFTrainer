# PRISM KT-RoPE spectral_unified overfit on 100 fixed samples (canonical).
#
# Goal: if the implementation is correct, text-only generation on the same
# 100 captions should reconstruct the paired GT motions.
#
# Verified 2026-05-29 (savefix / self-contained checkpoint era): epoch_260
# gives MPJRE~5deg and root-aligned MPJPE~36mm across all 100 samples (vs the
# old broken ~938mm). Inherits vae.save_ckpt / smpl_pose_processor.save_ckpt
# from the KT base so checkpoints replay in the same latent space.
# Inherits joint_pos_mode='spectral_unified' (modes=4, scale=22) from the base.

_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py'

trainer = dict(
    condition_num_frames=[1],
    frame_condition_rate=0.0,
    prompt_drop_rate=0.0,
    use_fp16_autocast=False,
    max_text_length=256,
    null_embedding_path='data/t5_feature/_null_embedding.pt',
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        anno_file='data/annotation/train_overfit_prism_100_valid.json',
        refetch=False,
        verbose=True,
        pipeline=[
            dict(
                type='LoadPreExtractedT5Feature',
                feature_dir='data/t5_feature',
                data_dir='data/motionhub',
                max_seq_length=256,
                allow_none=False,
                hidden_dim=4096,
                select_idx=0,
            ),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='abs_rel',
                smpl_type='smpl_22',
                rot6d_convention='column',
                transl_aug_prob=0.0,
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,
                pad_mode='replicate',
                allow_shorter=True,
                allow_longer=True,
            ),
            dict(
                type='PackInputs',
                keys=['motion', 'num_frames', 'caption', 't5_text_embeds', 't5_text_mask'],
                meta_keys=['motion_path', 'fps'],
                set_dummy_value=False,
            ),
        ],
    ),
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        interval=80,
        max_keep_ckpts=12,
        save_last=True,
        by_epoch=False,
    ),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=260,
    val_interval=1000000,
)

val_dataloader = None
val_evaluator = None
val_visualizer = None

auto_resume = False

work_dir = 'work_dirs/prism_overfit100_kt_spectral_unified'

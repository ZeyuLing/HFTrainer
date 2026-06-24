# Ablation A: isolate "MotionFix dirty data" vs "caption training dynamics".
# Cloned from the editfix snapshot config (work_dirs/.../20260528_174931/config.py).
# ONLY ONE variable changed vs editfix: training data source.
#   - anno_file -> clean train_hymotion_400h_hq_20260403.json (drop motionfix/permo_editing)
#   - load_from -> editfix epoch_960 (the checkpoint with the bad root jitter)
# Everything else (text branch, cond_mask_prob=0.1, motion_cond_mask_prob=0.3,
# losses, editing_prob=0.15, LoadEditingSourceMotion no-op on clean data) is identical.
# If root_hf_frac recovers after finetuning on clean data -> dirty data is the cause.
# If root_hf_frac stays high -> caption training dynamics (text branch / cond_mask) is the cause.
_motion_dim = 198
accelerator = dict(gradient_accumulation_steps=1, mixed_precision='no')
auto_resume = True
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=100, save_last=True,
        type='CheckpointHook'),
    ema=dict(decay=0.999, type='EMAHook', update_interval=1),
    logger=dict(interval=1, iter_interval=10, type='LoggerHook'))
load_from = dict(
    load_scope='model',
    path=
    'work_dirs/hymotion_m2m_v2_smpl_caption_editfix_from870_20260528/checkpoint-epoch_960/')
lr_scheduler = None
model = dict(
    body_model_path=None,
    caption_freeze_strategy='encoders',
    cond_mask_prob=0.1,
    ctxt_input_dim=4096,
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    losses_cfg=dict(
        aux_fk_consistency_warmup_steps=2000,
        aux_fk_consistency_weight=1500.0,
        aux_joint_pos_warmup_steps=2000,
        aux_joint_pos_weight=50.0,
        aux_joint_vel_warmup_steps=2000,
        aux_joint_vel_weight=500.0,
        aux_timestep_squared_weighting=True,
        fk_consistency_warmup_steps=2000,
        fk_consistency_weight=0.0,
        keypoints3d_weight=10.0,
        loss_type='smooth_l1',
        motion_smoothness_weight=0.5,
        trans_dim_weight=5.0,
        translation_weight=0.0,
        velocity_loss_reduction='component_mean',
        velocity_weight=1.0,
        x1_weight=0.0),
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    motion_cond_mask_prob=0.3,
    motion_transformer=dict(
        apply_rope_to_single_branch=False,
        ctxt_input_dim=4096,
        dropout=0.0,
        feat_dim=1024,
        final_layer_cfg=dict(act_type='silu'),
        input_dim=594,
        insert_start_token=False,
        mask_mode='narrowband',
        mlp_act_type='gelu_tanh',
        mlp_ratio=4.0,
        norm_type='layer',
        num_heads=16,
        num_layers=18,
        output_dim=198,
        qk_norm_type='rms',
        qkv_bias=True,
        text_refiner_cfg=dict(num_layers=2),
        time_factor=1000.0,
        trainable=True,
        type='HunyuanMotionMMDiT',
        vtxt_input_dim=768,
        with_long_skip_connection=False),
    motion_type='smpl_22',
    noise_scheduler_cfg=dict(method='euler'),
    pred_type='velocity',
    rotation_space='local',
    text_encoder=dict(),
    type='HyMotionM2MBundle',
    uncondition_mode=False,
    vace_condition_mode='no_inactive',
    vtxt_input_dim=768)
optimizer = dict(
    betas=[
        0.9,
        0.99,
    ], lr=0.0001, type='AdamW', weight_decay=0.0)
train_cfg = dict(
    by_epoch=True, max_epochs=10000, max_grad_norm=2.0, val_interval=10)
train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        data_dir='data/motionhub',
        motion_key='smplx',
        num_person=1,
        pipeline=[
            dict(allow_none=False, type='LoadCompatibleCaption'),
            dict(
                allow_none=True,
                key='caption',
                type='LoadPreExtractedTextEmbedding'),
            dict(
                key='motion',
                rot_type='rotation_6d',
                smpl_type='smpl_22',
                transl_type='abs',
                type='LoadSmplx55'),
            dict(key='motion', type='Compute198DimPosition'),
            dict(
                allow_shorter=True,
                clip_len=360,
                make_pad_mask=True,
                pad_mask_key='pad_mask',
                pad_mode='replicate',
                type='RandomCropPadding'),
            dict(
                corruptor_names=[
                    'jitter',
                    'joint_jump',
                    'sliding',
                    'limb_candy_wrapper',
                    'wrist_candy_wrapper',
                ],
                editing_prob=0.15,
                key='motion',
                max_corruptions=2,
                sampler_version='v3',
                type='PrepareM2Mv2Condition'),
            dict(type='LoadEditingSourceMotion'),
            dict(
                dummy_value=None,
                keys=[
                    'src_motion',
                    'tgt_motion',
                    'src_mask',
                    'tgt_length',
                    'src_length',
                    'edit_mode',
                    'text_vec_raw',
                    'text_ctxt_raw',
                    'text_ctxt_raw_length',
                ],
                meta_keys=[
                    'motion_path',
                    'fps',
                ],
                set_dummy_value=True,
                type='PackInputs'),
        ],
        refetch=True,
        task_mode='auto',
        type='MotionhubMultiTaskMultiAgentDataset',
        verbose=True),
    num_workers=8,
    persistent_workers=True,
    shuffle=True)
trainer = dict(
    mask_aware_noise=True, type='HyMotionM2MTrainer', val_num_steps=10)
val_dataloader = None
val_evaluator = None
val_visualizer = None
work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_cleandata_ablation'

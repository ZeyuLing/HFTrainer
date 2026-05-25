# PRISM smoke test configuration
# Tiny model for rapid smoke testing of PRISM trainer.
# Uses synthetic random data for text-to-motion generation.

_base_ = "../_base_/default_runtime.py"

model = dict(
    type="PrismBundle",
    transformer=dict(
        type="PrismTransformerMotionModel",
        trainable=True,
        gradient_checkpointing=False,
        module_dtype="bf16",
        patch_size=(1, 1),
        attention_head_dim=64,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=512,
        freq_dim=64,
        in_channels=16,
        num_attention_heads=4,
        num_layers=2,
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        joint_pos_mode="sequential",
        num_spectral_modes=4,
        spectral_scale=None,
        text_dim=4096,
    ),
    vae=dict(
        type="AutoencoderKLPrism2DTK",
        trainable=False,
        save_ckpt=False,
        module_dtype="fp32",
        from_pretrained=dict(pretrained_model_name_or_path="checkpoints/vermo_vae"),
    ),
    tokenizer=dict(
        type="T5Tokenizer",
        from_pretrained=dict(
            pretrained_model_name_or_path="checkpoints/Wan2.1-VACE-1.3B-diffusers",
            local_files_only=True,
            subfolder="tokenizer",
        ),
    ),
    text_encoder=dict(
        type="UMT5EncoderModel",
        trainable=False,
        save_ckpt=False,
        module_dtype="bf16",
        from_pretrained=dict(
            pretrained_model_name_or_path="checkpoints/Wan2.1-VACE-1.3B-diffusers",
            local_files_only=True,
            subfolder="text_encoder",
            low_cpu_mem_usage=False,
        ),
    ),
    scheduler=dict(
        type="FlowMatchEulerDiscreteScheduler",
        num_train_timesteps=1000,
        shift=5.0,
        use_dynamic_shifting=False,
        base_shift=0.5,
        max_shift=1.15,
    ),
    smpl_pose_processor=dict(
        type="SMPLPoseProcessor",
        trainable=False,
        save_ckpt=False,
        do_normalize=True,
        stats_file="data/statistic/smplx55_stats_hymotion_aug.json",
        rot_type="rotation_6d",
        transl_type="abs_rel",
        smpl_type="smpl_22",
        smpl_model=dict(
            type="SmplxLiteV437Coco17",
            model_path="checkpoints/smpl_models/smplx",
            smplx2smpl_path="checkpoints/smpl_models/smplx2smpl_sparse.pt",
            coco17_regressor_path="checkpoints/smpl_models/smpl_coco17_J_regressor.pt",
            smplx_verts437_path="checkpoints/smpl_models/smplx_verts437.pt",
            gender="neutral",
            num_betas=10,
        ),
    ),
)

trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=128,
    val_prompts=["a person stands"],
    num_val_inference_steps=2,
    guidance_scale=5.0,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type="MotionHubSingleAgentTextDataset",
        motion_key="smplx",
        data_dir="data/motionhub",
        anno_file="data/annotation/train_hq_motionhub_hymotion.json",
        pipeline=[
            dict(type="LoadCompatibleCaption", allow_none=False),
            dict(
                type="LoadSmplx55",
                key="motion",
                rot_type="rotation_6d",
                transl_type="abs_rel",
                smpl_type="smpl_22",
                rot6d_convention="column",
                transl_aug_prob=0.0,
            ),
            dict(
                type="RandomCropPadding",
                clip_len=32,
                pad_mode="replicate",
                allow_shorter=True,
                allow_longer=False,
            ),
            dict(
                type="PackInputs",
                keys=["motion", "num_frames", "caption"],
                meta_keys=["motion_path", "fps"],
                set_dummy_value=False,
            ),
        ],
        verbose=False,
        refetch=False,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

lr_scheduler = None

accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    by_epoch=False,
    max_iters=5,
    val_interval=100,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1),
    checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=1, save_last=True),
)

val_dataloader = None
val_evaluator = None
val_visualizer = None

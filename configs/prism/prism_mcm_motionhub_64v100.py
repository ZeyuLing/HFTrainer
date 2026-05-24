# PRISM MCM audio-conditioned training — 64×V100 (8 nodes)
#
# Key changes from 16v100 config:
#   1. 8 nodes × 8 GPUs = 64 GPUs (effective batch = 256)
#   2. persistent_workers=True for faster data loading
#   3. lr scaled 1e-4 → 2e-4 (moderate linear scaling)
#   4. Checkpoint saved every epoch
#   5. soundfile-accelerated audio loading (via updated LoadAudio)
#
# Submit:
#   python3 tools/taiji_submit.py prism_mcm_64v100 configs/prism/prism_mcm_motionhub_64v100.py --host_num 8

_base_ = "../_base_/default_runtime.py"

work_dir = "work_dirs/prism_mcm_motionhub_64v100"

# Load pretrained PRISM multiframe transformer weights.
# The MCM bundle will automatically re-init control branch from loaded main branch.
load_from = dict(
    _delete_=True,
    path="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000",
    load_scope="model",
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=1),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,          # save every epoch
        max_keep_ckpts=10,
        save_last=True,
    ),
)

model = dict(
    type="PrismMCMBundle",
    init_control_from_main=True,
    transformer=dict(
        type="PrismTransformerMotionModel",
        trainable=False,
        save_ckpt=False,
        gradient_checkpointing=True,
        module_dtype="bf16",
        patch_size=(1, 1),
        attention_head_dim=128,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=16,
        num_attention_heads=12,
        num_layers=30,
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        text_dim=4096,
    ),
    control_transformer=dict(
        type="PrismVACEControlTransformer",
        trainable=True,
        gradient_checkpointing=True,
        # Trainable module stays fp32 for stable training.
        # No module_dtype — defaults to fp32.
        patch_size=(1, 1),
        attention_head_dim=128,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=16,
        num_attention_heads=12,
        num_layers=30,
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        text_dim=4096,
        audio_feature_dim=768,
        # WanVACE-style: 8 sparse blocks at evenly-spaced layers
        vace_layers=[0, 4, 8, 12, 16, 20, 24, 28],
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
    audio_encoder=dict(
        type="AudioEncoderWrapper",
        trainable=False,
        save_ckpt=False,
        encoder_type="beats",
        pretrained="checkpoints/BEATs_iter3_plus_AS2M.pt",
        feature_dim=768,
        target_sr=16000,
    ),
)

trainer = dict(
    type="PrismMCMTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.5,
    prompt_drop_rate=0.1,
    audio_drop_rate=0.1,
    max_text_length=256,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type="MotionhubMultiTaskMultiAgentDataset",
        motion_key="smplx",
        data_dir="data/motionhub",
        anno_file="data/annotation/train_audio_motionhub_hymotion.json",
        task_mode="preset",
        preset_tasks=["m2d", "s2g", "t2md", "g2md", "n2md", "t2sg", "n2sg"],
        num_person=1,
        pipeline=[
            dict(type="LoadHierarchicalCaption", allow_none=True),
            dict(
                type="LoadSmplx55",
                key="motion",
                rot_type="rotation_6d",
                transl_type="abs_rel",
                smpl_type="smpl_22",
                rot6d_convention="column",
            ),
            # Load full audio/music waveforms first (soundfile-accelerated)
            dict(type="LoadAudio", key="audio", target_sr=16000, allow_none=True),
            dict(type="LoadAudio", key="music", target_sr=16000, allow_none=True),
            # Random crop motion to 512 frames; pad if shorter
            dict(
                type="RandomCropPadding",
                clip_len=512,
                pad_mode="replicate",
                allow_shorter=True,
            ),
            # Sync audio/music to the same time window as the cropped motion.
            # Uses start_frame and num_frames set by RandomCropPadding.
            dict(
                type="CropAudioToMotion",
                audio_keys=["audio", "music"],
                target_sr=16000,
            ),
            dict(
                type="PackInputs",
                keys=["motion", "num_frames", "caption", "audio", "music"],
                meta_keys=["motion_path", "fps"],
                set_dummy_value=True,
            ),
        ],
        verbose=True,
        refetch=True,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

lr_scheduler = None

accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
    fsdp_plugin=dict(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        transformer_cls_names_to_wrap=["PrismVACEControlBlock"],
        state_dict_type="FULL_STATE_DICT",
        sync_module_states=True,
        use_orig_params=True,
        cpu_offload=False,
    ),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=50,
    max_grad_norm=1.0,
)

val_dataloader = None
val_evaluator = None
val_visualizer = None

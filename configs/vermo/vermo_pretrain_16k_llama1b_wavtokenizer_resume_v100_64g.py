accelerator = dict(gradient_accumulation_steps=1, mixed_precision='fp16')
auto_resume = True
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=5,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=1, iter_interval=1, type='LoggerHook'))
load_from = None
lr_scheduler = None
model = dict(
    lm=dict(
        from_pretrained=dict(
            attn_implementation='eager',
            pretrained_model_name_or_path='checkpoints/Llama-3.2-1B-Instruct'),
        gradient_checkpointing=True,
        module_dtype='fp32',
        trainable=True,
        type='VermoLlamaForCausalLM'),
    mean_init_embeddings=False,
    processor=dict(
        audio_codebook_size=4096,
        audio_tokenizer=dict(
            pretrained='checkpoints/WavTokenizer-large-unify-40token',
            type='WavTokenizer'),
        instruction_stage=True,
        max_seq_len=2048,
        module_dtype='fp32',
        motion_tokenizer=dict(
            from_pretrained=dict(
                pretrained_model_name_or_path=
                'checkpoints/vermo_vqvae2d_16k_rescale_iter47k'),
            type='VQVAEWanMotion2DTK'),
        optional_input_modal_mode='random',
        pretrained_text_tokenizer=dict(
            from_pretrained=dict(pretrained_model_name_or_path=
                                 'checkpoints/Llama-3.2-1B-Instruct'),
            type='AutoTokenizer'),
        save_ckpt=False,
        smpl_pose_processor=dict(
            do_normalize=True,
            rot_type='rotation_6d',
            smpl_model=dict(
                coco17_regressor_path=
                'checkpoints/smpl_models/smpl_coco17_J_regressor.pt',
                gender='neutral',
                model_path='checkpoints/smpl_models/smplx',
                num_betas=10,
                smplx2smpl_path='checkpoints/smpl_models/smplx2smpl_sparse.pt',
                smplx_verts437_path='checkpoints/smpl_models/smplx_verts437.pt',
                type='SmplxLiteV437Coco17'),
            smpl_type='smpl_22',
            stats_file='data/statistic/smplx55_stats_hymotion_aug.json',
            transl_type='abs_rel',
            type='SMPLPoseProcessor'),
        trainable=False,
        type='VermoProcessor'),
    type='VermoBundle')
optimizer = dict(
    betas=[
        0.9,
        0.99,
    ], lr=3e-05, type='AdamW', weight_decay=0.0)
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        anno_file=
        'data/annotation/train_hq_motionhub_hymotion_motionclip2p_20260604.json',
        data_dir='data/motionhub',
        log_task_iter=1000,
        motion_key='smplx',
        pipeline=[
            dict(allow_none=True, type='LoadCompatibleCaption'),
            dict(allow_none=True, key='speech_script', type='LoadTxt'),
            dict(
                key='motion',
                rot6d_convention='column',
                rot_type='rotation_6d',
                smpl_type='smpl_22',
                transl_aug_offset_std=(
                    1.0,
                    0.0,
                    1.0,
                ),
                transl_aug_prob=0.75,
                transl_aug_yaw_deg=180.0,
                transl_type='abs_rel',
                type='LoadSmplx55'),
            dict(
                allow_none=True,
                key='audio',
                target_sr=24000,
                type='LoadAudio'),
            dict(
                allow_none=True,
                key='music',
                target_sr=24000,
                type='LoadAudio'),
            dict(
                collision_check=True,
                compose_prob=0.2,
                max_persons=3,
                min_persons=2,
                placement_radius_range=(
                    1.0,
                    3.0,
                ),
                skip_with_audio=True,
                type='ComposeMultiPerson',
                yaw_range=180.0),
            dict(
                audio_key='audio',
                max_duration=12.0,
                motion_key='motion',
                pair_only=True,
                type='MotionAudioMaxDurationFilter'),
            dict(
                audio_key='music',
                max_duration=12.0,
                motion_key='motion',
                pair_only=True,
                type='MotionAudioMaxDurationFilter'),
            dict(
                audio_key=None,
                max_duration=12.0,
                motion_key='motion',
                pair_only=False,
                type='MotionAudioMaxDurationFilter'),
            dict(
                key='motion',
                min_future_frames=17,
                past_ratio=0.4,
                random_ratio=False,
                single_frame_prob=0.25,
                type='SplitPrediction'),
            dict(
                future_ratio=0.2,
                keys='motion',
                min_edge_frames=4,
                min_middle_frames=4,
                past_ratio=0.2,
                random_ratio=False,
                single_frame_pair_prob=0.25,
                type='SplitInbetween'),
            dict(key='motion', single_frame_prob=1.0, type='SplitMotionForAR'),
            dict(key='music', type='SplitMusicForAR'),
            dict(
                dummy_value=None,
                keys=[
                    'task',
                    'motion',
                    'past_motion',
                    'future_motion',
                    'middle_motion',
                    'num_frames',
                    'duration',
                    'audio',
                    'music',
                    'past_music',
                    'future_music',
                    'caption',
                    'person_captions',
                    'speech_script',
                    'num_person',
                    'genre',
                    'per_person_num_frames',
                    'past_per_person_num_frames',
                    'future_per_person_num_frames',
                    'middle_per_person_num_frames',
                ],
                meta_keys=[
                    'motion_path',
                    'fps',
                    'person_captions',
                ],
                set_dummy_value=True,
                type='PackInputs'),
        ],
        refetch=True,
        task_mode='auto',
        type='MotionhubMultiTaskMultiAgentDataset',
        verbose=False),
    num_workers=8,
    persistent_workers=True,
    shuffle=True)
trainer = dict(type='VermoTrainer')
val_dataloader = None
val_evaluator = None
val_visualizer = None
work_dir = 'work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_v100_64g_ampfp16_bsz1_seq2048_eager'

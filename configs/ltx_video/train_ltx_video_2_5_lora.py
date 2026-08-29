"""Managed LTX-2.5 text-to-audio-video LoRA training recipe."""

root = __import__('os').environ.get('LTX25_CHECKPOINT_ROOT', 'checkpoints/LTX-2.5')
data_root = __import__('os').environ.get(
    'LTX25_PREPROCESSED_DATA',
    'data/ltx_video_2_5/.precomputed',
)


custom_imports = dict(
    imports=['hftrainer.trainers.ltx_video'],
    allow_failed_imports=False,
)

work_dir = __import__('os').environ.get(
    'HFTRAINER_WORK_DIR',
    'outputs/training/ltx_video_2_5_lora',
)

trainer = dict(
    type='LTXVideoTrainer',
    require_linux=True,
    require_cuda=True,
    require_files=True,
    strict_checkpoint_roles=True,
    native_config=dict(
        model=dict(
            model_path=(
                f'{root}/diffusion_models/'
                'ltx-2.5-22b-dev-transformer-bf16.safetensors'
            ),
            text_encoder_path=(
                f'{root}/text_encoders/'
                'gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
            ),
            video_vae_path=f'{root}/vae/ltx-2.5-video-vae-bf16.safetensors',
            audio_vae_path=f'{root}/vae/ltx-2.5-audio-vae-bf16.safetensors',
            training_mode='lora',
            load_checkpoint=None,
        ),
        lora=dict(
            rank=32,
            alpha=32,
            dropout=0.0,
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        ),
        training_strategy=dict(
            name='flexible',
            video=dict(is_generated=True, latents_dir='latents'),
            audio=dict(is_generated=True, latents_dir='audio_latents'),
        ),
        optimization=dict(
            learning_rate=1e-4,
            steps=2000,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            optimizer_type='adamw',
            scheduler_type='linear',
            scheduler_params={},
            enable_gradient_checkpointing=True,
        ),
        acceleration=dict(
            mixed_precision_mode='bf16',
            quantization=None,
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        ),
        data=dict(
            preprocessed_data_root=data_root,
            num_dataloader_workers=2,
        ),
        validation=dict(
            samples=[],
            interval=None,
            video_dims=(960, 544, 89),
            frame_rate=24.0,
            generate_video=True,
            generate_audio=True,
        ),
        checkpoints=dict(
            interval=250,
            keep_last_n=3,
            precision='bfloat16',
            no_resume=False,
            save_training_state='minimal',
        ),
        flow_matching=dict(
            timestep_sampling_mode='shifted_logit_normal',
            timestep_sampling_params={},
        ),
        hub=dict(push_to_hub=False, hub_model_id=None),
        wandb=dict(
            enabled=False,
            project='hftrainer-ltx-2.5',
            entity=None,
            tags=['ltx-2.5', 'lora', 't2av'],
            log_validation_videos=True,
        ),
        seed=42,
        output_dir=work_dir,
    ),
)

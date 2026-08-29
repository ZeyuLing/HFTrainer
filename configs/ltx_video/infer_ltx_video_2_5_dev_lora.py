"""Guided LTX-2.5 dev inference, including a user-trained LoRA."""

root = __import__('os').environ.get('LTX25_CHECKPOINT_ROOT', 'checkpoints/LTX-2.5')
user_lora = __import__('os').environ.get(
    'LTX25_USER_LORA',
    (
        'outputs/training/ltx_video_2_5_lora/checkpoints/'
        'lora_weights_step_02000.safetensors'
    ),
)


custom_imports = dict(
    imports=[
        'hftrainer.models.ltx_video',
        'hftrainer.pipelines.ltx_video',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='LTXVideoBundle',
    mode='dev_two_stage',
    transformer_path=(
        f'{root}/diffusion_models/'
        'ltx-2.5-22b-dev-transformer-bf16.safetensors'
    ),
    text_encoder_path=(
        f'{root}/text_encoders/'
        'gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
    ),
    video_vae_path=f'{root}/vae/ltx-2.5-video-vae-bf16.safetensors',
    audio_vae_path=f'{root}/vae/ltx-2.5-audio-vae-bf16.safetensors',
    duration_head_path=(
        f'{root}/model_patches/ltx-2.5-duration-head-bf16.safetensors'
    ),
    spatial_upsampler_path=(
        f'{root}/latent_upscale_models/'
        'ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'
    ),
    distilled_lora_path=(
        f'{root}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors'
    ),
    loras=[dict(path=user_lora, strength=1.0)],
    device='cuda',
    offload_mode='none',
)

pipeline = dict(
    type='LTXVideoPipeline',
    height=512,
    width=768,
    num_frames=121,
    frame_rate=24.0,
    seed=42,
    num_inference_steps=30,
    negative_prompt='blurry, distorted, low quality, artifacts',
)

inference = dict(task='text_to_video')

"""LTX-2.5 fixed-eight-step native inference."""

# Avoid a top-level ``import os``: MMEngine treats any imported name as a lazy
# config object, which cannot be called while the config is being evaluated.
root = __import__('os').environ.get('LTX25_CHECKPOINT_ROOT', 'checkpoints/LTX-2.5')


custom_imports = dict(
    imports=[
        'hftrainer.models.ltx_video',
        'hftrainer.pipelines.ltx_video',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='LTXVideoBundle',
    mode='distilled',
    transformer_path=(
        f'{root}/diffusion_models/'
        'ltx-2.5-22b-distilled-transformer-bf16.safetensors'
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
    device='cuda',
    offload_mode='none',
    diffvae_optimization='chunked_eager',
)

pipeline = dict(
    type='LTXVideoPipeline',
    height=512,
    width=768,
    num_frames=121,
    frame_rate=24.0,
    seed=42,
)

inference = dict(task='text_to_video')

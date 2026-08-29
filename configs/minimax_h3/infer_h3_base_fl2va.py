"""MiniMax-H3 Base text/first-last-frame synchronized A/V inference."""

# ruff: noqa: C408 - MMEngine config files conventionally use dict(...).

root = __import__("os").environ.get("MINIMAX_H3_ROOT", "checkpoints/MiniMax-H3")
load_device = __import__("os").environ.get("MINIMAX_H3_LOAD_DEVICE", "cpu")
transformer_device = __import__("os").environ.get(
    "MINIMAX_H3_TRANSFORMER_DEVICE", load_device
)
conditioner_device = __import__("os").environ.get(
    "MINIMAX_H3_CONDITIONER_DEVICE", load_device
)
codec_device = __import__("os").environ.get("MINIMAX_H3_CODEC_DEVICE", load_device)

custom_imports = dict(
    imports=[
        "hftrainer.models.minimax_h3",
        "hftrainer.pipelines.minimax_h3",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="MiniMaxH3Bundle",
    variant="fl2va",
    conditioning_layer=50,
    tokenizer_path=f"{root}/tokenizer",
    processor_path=f"{root}/processor",
    transformer=dict(
        type="MiniMaxH3Transformer3DModel",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="transformer",
            torch_dtype="bf16",
            low_cpu_mem_usage=True,
            device=transformer_device,
            strict=True,
        ),
        trainable=False,
        save_ckpt=False,
    ),
    text_encoder=dict(
        type="MiniMaxH3Qwen3VLEncoder",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="text_encoder",
            torch_dtype="bf16",
            low_cpu_mem_usage=True,
            device=conditioner_device,
            strict=True,
        ),
        trainable=False,
        save_ckpt=False,
    ),
    vae=dict(
        type="AutoencoderKLMiniMaxH3",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="vae",
            torch_dtype="fp32",
            low_cpu_mem_usage=True,
            device=codec_device,
            strict=True,
        ),
        trainable=False,
        save_ckpt=False,
    ),
    audio_vae=dict(
        type="AutoencoderKLMiniMaxH3Audio",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="audio_vae",
            torch_dtype="fp32",
            low_cpu_mem_usage=True,
            device=codec_device,
            strict=True,
        ),
        trainable=False,
        save_ckpt=False,
    ),
    scheduler=dict(
        type="MiniMaxH3Scheduler",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="scheduler",
        ),
        trainable=False,
        save_ckpt=False,
    ),
    audio_scheduler=dict(
        type="MiniMaxH3Scheduler",
        from_pretrained=dict(
            pretrained_model_name_or_path=root,
            subfolder="audio_scheduler",
        ),
        trainable=False,
        save_ckpt=False,
    ),
)

pipeline = dict(
    type="MiniMaxH3Pipeline",
    num_inference_steps=50,
    canvas_short_edge=768,
    canvas_max_pixels=768 * 1344,
    reference_image_short_edge=2048,
    min_duration=5.0,
    max_duration=15.0,
)

inference = dict(task="multimodal_to_audio_video")

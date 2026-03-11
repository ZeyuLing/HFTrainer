_base_ = ['../_base_/default_runtime.py']

TEACHER_PATH = 'checkpoints/stable-diffusion-v1-5'

model = dict(
    type='DMDBundle',
    text_encoder=dict(
        type='CLIPTextModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=TEACHER_PATH,
            subfolder='text_encoder',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=dict(
            pretrained_model_name_or_path=TEACHER_PATH,
            subfolder='vae',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    real_score_unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=TEACHER_PATH,
            subfolder='unet',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    fake_score_unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=TEACHER_PATH,
            subfolder='unet',
        ),
        trainable=True,
        save_ckpt=True,
    ),
    generator_unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=dict(
            pretrained_model_name_or_path=TEACHER_PATH,
            subfolder='unet',
        ),
        trainable=True,
        save_ckpt=True,
    ),
    scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=dict(
            pretrained_model_name_or_path=TEACHER_PATH,
            subfolder='scheduler',
        ),
        trainable=False,
        save_ckpt=False,
    ),
    tokenizer_path=TEACHER_PATH,
    image_size=512,
    conditioning_timestep=999,
    dm_min_timestep_percent=0.02,
    dm_max_timestep_percent=0.98,
    generator_guidance_scale=1.0,
    real_score_guidance_scale=7.5,
    fake_score_guidance_scale=1.0,
    regression_guidance_scale=7.5,
)

trainer = dict(
    type='DMDTrainer',
    dm_weight=1.0,
    regression_weight=1.0,
    fake_score_weight=1.0,
    # Official DMD uses precomputed teacher pairs for regression; this
    # reference config can also generate the targets online for convenience.
    online_regression_num_inference_steps=20,
    score_start_step=0,
    score_warmup_steps=0,
    score_update_interval=1,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='DMDImagePairDataset',
        data_root='data/text2image/demo',
        image_size=512,
    ),
)

optimizer = dict(
    generator=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=1e-2,
        params=['generator_unet'],
    ),
    fake_score=dict(
        type='AdamW',
        lr=4e-5,
        weight_decay=1e-2,
        params=['fake_score_unet'],
    ),
)

lr_scheduler = dict(
    generator=dict(type='cosine_with_warmup', num_warmup_steps=500),
    fake_score=dict(type='constant'),
)

train_cfg = dict(by_epoch=False, max_iters=10000)
work_dir = 'work_dirs/dmd_demo'

_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='StyleGAN2Bundle',
    generator=dict(
        type='StyleGAN2Generator',
        z_dim=512,
        w_dim=512,
        img_resolution=64,
        img_channels=3,
        channel_base=16384,
        channel_max=512,
        mapping_layers=8,
        style_mixing_prob=0.9,
        trainable=True,
        save_ckpt=True,
    ),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        img_resolution=64,
        img_channels=3,
        channel_base=16384,
        channel_max=512,
        mbstd_group_size=4,
        trainable=True,
        save_ckpt=True,
    ),
)

trainer = dict(
    type='GANTrainer',
    d_steps_per_g_step=1,
    gan_loss_type='stylegan2',
    r1_gamma=10.0,
    d_reg_interval=16,
    pl_weight=2.0,
    g_reg_interval=4,
    disc_start_step=0,
    disc_warmup_steps=0,
    disc_update_interval=1,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type='ImageFolderGANDataset',
        data_root='data/classification/demo/images',
        image_size=64,
        random_horizontal_flip=True,
    ),
)

optimizer = dict(
    generator=dict(type='Adam', lr=2e-3, betas=(0.0, 0.99)),
    discriminator=dict(type='Adam', lr=2e-3, betas=(0.0, 0.99)),
)

lr_scheduler = dict(
    generator=dict(type='constant'),
    discriminator=dict(type='constant'),
)

train_cfg = dict(by_epoch=False, max_iters=10000)
work_dir = 'work_dirs/stylegan2_demo'

"""Two-step full-weight LTX validation on the local real-video subset."""
_base_ = ['train_ltx_video_2_5_lora.py']
trainer = dict(native_config=dict(
    lora=dict(rank=2, alpha=2),
    optimization=dict(steps=2),
    data=dict(num_dataloader_workers=0),
    validation=dict(
        samples=[dict(prompt='A dancer performs a dance routine with music.', seed=42)],
        interval=1,
        video_dims=(256, 256, 49),
    ),
    checkpoints=dict(interval=1, keep_last_n=3, save_training_state='full'),
))

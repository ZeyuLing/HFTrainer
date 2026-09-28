"""Public CREMA-D audio/video LoRA training, 20 steps."""
_base_ = ['../pretrained/train_ltx_video.py']
work_dir = 'work_dirs/public_data/ltx_video'
trainer = dict(native_config=dict(
    data=dict(preprocessed_data_root='data/hftrainer_crema_d_mini/.cache/ltx/train'),
    optimization=dict(steps=20),
    validation=dict(samples=[], interval=None),
    checkpoints=dict(interval=20, keep_last_n=1, save_training_state='full'),
    output_dir=work_dir,
))

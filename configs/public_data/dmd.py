"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_dmd.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/dmd')

work_dir = 'work_dirs/public_data/dmd'
auto_resume = False
randomness = dict(seed=42)
train_cfg = dict(max_iters=20, val_interval=None)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
train_dataloader = dict(num_workers=0, dataset=dict(type='DMDImagePairDataset', data_root='data/hftrainer_smithsonian_butterflies/train', image_size=512))
trainer = dict(online_regression_num_inference_steps=20)
lr_scheduler = dict(generator=dict(type='constant', _delete_=True), fake_score=dict(type='constant'))
accelerator = dict(mixed_precision='bf16', gradient_accumulation_steps=1)

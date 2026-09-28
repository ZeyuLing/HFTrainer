"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_stylegan2.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/stylegan2')

work_dir = 'work_dirs/public_data/stylegan2'
auto_resume = False
randomness = dict(seed=42)
train_cfg = dict(max_iters=20, val_interval=None)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
train_dataloader = dict(num_workers=0, dataset=dict(data_root='data/hftrainer_smithsonian_butterflies/train/images', image_size=1024))
optimizer = dict(generator=dict(lr=1e-4), discriminator=dict(lr=1e-4))

"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_sd15.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/sd15')

work_dir = 'work_dirs/public_data/sd15'
auto_resume = False
randomness = dict(seed=42)
train_cfg = dict(max_iters=20, val_interval=None)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
train_dataloader = dict(data_root='data/hftrainer_smithsonian_butterflies/train', image_size=512, max_samples=None)
trainer = dict(val_prompts=['A photograph of a butterfly, Paper Kite.'])

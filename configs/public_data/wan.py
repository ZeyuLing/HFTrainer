"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_wan.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/wan')

work_dir = 'work_dirs/public_data/wan'
auto_resume = False
randomness = dict(seed=42)
train_cfg = dict(max_iters=20, val_interval=None)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
train_dataloader = dict(data_root='data/hftrainer_crema_d_mini/train', synthetic=False, max_samples=None, num_frames=17, height=128, width=128)
trainer = dict(val_num_frames=17, val_height=128, val_width=128, val_prompts=['A person speaks with a happy expression.'])

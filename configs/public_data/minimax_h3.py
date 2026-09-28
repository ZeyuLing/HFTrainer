"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_minimax_h3.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/minimax_h3')

work_dir = 'work_dirs/public_data/minimax_h3'
auto_resume = False
randomness = dict(seed=42)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
train_dataloader = dict(dataset=dict(manifest='data/hftrainer_crema_d_mini/.cache/h3/train.jsonl'))
val_dataloader = dict(dataset=dict(manifest='data/hftrainer_crema_d_mini/.cache/h3/validation.jsonl'))
train_cfg = dict(max_iters=20, val_interval=20)

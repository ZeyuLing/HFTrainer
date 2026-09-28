"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_llama.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/llama')

work_dir = 'work_dirs/public_data/llama'
auto_resume = False
randomness = dict(seed=42)
train_cfg = dict(max_iters=20, val_interval=None)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
train_dataloader = dict(_delete_=True, type='PlainTextDataset', data_path='data/hftrainer_wikitext2/train.jsonl', tokenizer_name_or_path=root, max_length=256, max_samples=None, batch_size=2, num_workers=0, shuffle=True)

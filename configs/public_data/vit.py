"""Public-data 20-step pipeline validation; not a convergence recipe."""
_base_ = ['../pretrained/train_vit.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/vit')

work_dir = 'work_dirs/public_data/vit'
auto_resume = False
randomness = dict(seed=42)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=False), logger=dict(type='LoggerHook', interval=1))
model = dict(from_pretrained=dict(num_labels=3))
train_dataloader = dict(data_root='data/hftrainer_beans/train', max_samples=None, batch_size=8)
val_dataloader = dict(data_root='data/hftrainer_beans/validation', max_samples=None, batch_size=8)
train_cfg = dict(max_iters=20, val_interval=20)

# Match the normalization shipped with the pretrained ViT processor.
input_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeImage', size=(224, 224)),
    dict(type='HFTrainerImageToTensor', output_key='pixel_values'),
    dict(type='NormalizeTensor', key='pixel_values', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    dict(type='RenameKeys', mapping={'label': 'labels'}),
    dict(type='PackMetaKeys', meta_keys=('img_path', 'class_name', 'sample_idx')),
]
train_dataloader['pipeline'] = input_pipeline
val_dataloader['pipeline'] = input_pipeline
val_visualizer = None

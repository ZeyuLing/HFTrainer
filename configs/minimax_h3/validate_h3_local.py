"""Real-weight, real-video single-H200 smoke run; not a convergence recipe."""
_base_ = ['train_h3_base_lora.py']
custom_imports = dict(
    imports=['hftrainer.models.minimax_h3', 'hftrainer.trainers.minimax_h3',
             'hftrainer.datasets.synchronized_audio_video', 'tools.validation.h3_metrics'],
    allow_failed_imports=False,
)
model = dict(transformer=dict(
    from_pretrained=dict(device='cuda'),
    lora_cfg=dict(rank=2, alpha=2),
))
train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(
    dataset=dict(type='MiniMaxH3CachedFeatureDataset',
                 manifest='data/video_validation_20260928/h3/validation.jsonl'),
    batch_size=1, num_workers=0, shuffle=False,
)
val_evaluator = dict(type='H3SmokeLossEvaluator')
train_cfg = dict(max_iters=2, val_interval=1)
lr_scheduler = dict(_delete_=True, type='constant')
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_last=True),
    logger=dict(interval=1),
)
auto_resume = False

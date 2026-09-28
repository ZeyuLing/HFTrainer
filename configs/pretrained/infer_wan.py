"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/wan')
custom_imports = dict(imports=['hftrainer.models.wan', 'hftrainer.pipelines.wan'], allow_failed_imports=False)
model = dict(type='WanBundle', from_pretrained=root)
pipeline = dict(type='WanPipeline')
inference = dict(task='text_to_video')

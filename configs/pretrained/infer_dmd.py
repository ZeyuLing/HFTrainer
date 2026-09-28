"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/dmd')
custom_imports = dict(imports=['hftrainer.models.dmd', 'hftrainer.pipelines.dmd'], allow_failed_imports=False)
model = dict(type='DMDBundle', from_pretrained=root)
pipeline = dict(type='DMDPipeline')
inference = dict(task='text_to_image')

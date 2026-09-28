"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/sd15')
custom_imports = dict(imports=['hftrainer.models.sd15', 'hftrainer.pipelines.sd15'], allow_failed_imports=False)
model = dict(type='SD15Bundle', from_pretrained=root)
pipeline = dict(type='SD15Pipeline')
inference = dict(task='text_to_image')

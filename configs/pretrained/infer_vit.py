"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/vit')
custom_imports = dict(imports=['hftrainer.models.vit', 'hftrainer.tasks.image_classification'], allow_failed_imports=False)
model = dict(type='ViTBundle', from_pretrained=root)
pipeline = dict(type='ClassificationPipeline')
inference = dict(task='image_classification')

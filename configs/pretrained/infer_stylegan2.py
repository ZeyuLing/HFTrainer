"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/stylegan2')
custom_imports = dict(imports=['hftrainer.models.stylegan2', 'hftrainer.pipelines.stylegan2'], allow_failed_imports=False)
model = dict(type='StyleGAN2Bundle', from_pretrained=root)
pipeline = dict(type='StyleGAN2Pipeline')
inference = dict(task='unconditional_image_generation')

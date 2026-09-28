"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/minimax_h3')
custom_imports = dict(imports=['hftrainer.models.minimax_h3', 'hftrainer.pipelines.minimax_h3'], allow_failed_imports=False)
model = dict(type='MiniMaxH3Bundle', from_pretrained=root)
pipeline = dict(type='MiniMaxH3Pipeline')
inference = dict(task='multimodal_to_audio_video')

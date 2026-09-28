"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/ltx_video')
custom_imports = dict(imports=['hftrainer.models.ltx_video', 'hftrainer.pipelines.ltx_video'], allow_failed_imports=False)
model = dict(type='LTXVideoBundle', from_pretrained=root)
pipeline = dict(type='LTXVideoPipeline', height=512, width=768, num_frames=121, num_inference_steps=30, negative_prompt='blurry, distorted, low quality, artifacts')
inference = dict(task='text_to_video')

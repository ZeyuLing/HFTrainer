"""Inference from a complete downloaded HFTrainer artifact."""
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/llama')
custom_imports = dict(imports=['hftrainer.models.llama', 'hftrainer.tasks.causal_language_modeling'], allow_failed_imports=False)
model = dict(type='LlamaBundle', from_pretrained=root)
pipeline = dict(type='CausalLMPipeline')
inference = dict(task='text_generation')

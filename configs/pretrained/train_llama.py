"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../llama/llama_lora_demo.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/llama')
model = dict(_delete_=True, type='LlamaBundle', from_pretrained=dict(pretrained_model_name_or_path=root, model_overrides=dict(trainable='lora', checkpoint_format='lora', lora_cfg=dict(rank=16, alpha=32, target_modules='all-linear'))))
train_dataloader = dict(tokenizer_name_or_path=root)

"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../vit/vit_base_demo.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/vit')
model = dict(_delete_=True, type='ViTBundle', from_pretrained=dict(pretrained_model_name_or_path=root, num_labels=4))

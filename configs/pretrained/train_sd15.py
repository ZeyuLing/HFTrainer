"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../sd15/sd15_demo.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/sd15')
model = dict(_delete_=True, type='SD15Bundle', from_pretrained=root)

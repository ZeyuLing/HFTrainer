"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../dmd/dmd_demo.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/dmd')
model = dict(_delete_=True, type='DMDBundle', from_pretrained=root)

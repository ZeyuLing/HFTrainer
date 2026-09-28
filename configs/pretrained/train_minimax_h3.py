"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../minimax_h3/demo_loss.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/minimax_h3')
model = dict(_delete_=True, type='MiniMaxH3Bundle', from_pretrained=dict(pretrained_model_name_or_path=root, trainable='lora', load_conditioner=False, load_vaes=False, gradient_checkpointing=True, lora_cfg=dict(rank=2, alpha=2, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'])))

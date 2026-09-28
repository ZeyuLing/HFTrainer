"""Full artifact export for the rank-2 demo adapter, including frozen components."""
_base_ = ['infer_minimax_h3.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/minimax_h3')
model = dict(_delete_=True, type='MiniMaxH3Bundle', from_pretrained=dict(
    pretrained_model_name_or_path=root, trainable='lora',
    lora_cfg=dict(rank=2, alpha=2, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'])))

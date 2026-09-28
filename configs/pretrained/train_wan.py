"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../wan/wan_demo.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/wan')
model = dict(_delete_=True, type='WanBundle', from_pretrained=root,
             shared_pretrained_kwargs=dict(torch_dtype='bf16'),
             config_overrides=dict(
                 text_encoder=dict(trainable=False, save_ckpt=False),
                 vae=dict(trainable=False, save_ckpt=False),
                 transformer=dict(trainable=True, save_ckpt=True, gradient_checkpointing=True),
                 scheduler=dict(trainable=False, save_ckpt=False)))
train_dataloader = dict(num_frames=5)
trainer = dict(val_num_frames=5)

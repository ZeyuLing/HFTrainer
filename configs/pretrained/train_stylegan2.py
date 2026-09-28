"""Fine-tuning from a complete downloaded HFTrainer artifact; prepare data first."""
_base_ = ['../stylegan2/stylegan2_demo.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/stylegan2')
model = dict(_delete_=True, type='StyleGAN2Bundle', from_pretrained=root)
train_dataloader = dict(batch_size=1, dataset=dict(image_size=1024))

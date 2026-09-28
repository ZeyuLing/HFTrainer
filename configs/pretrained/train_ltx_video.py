"""Train LTX from the same complete artifact used for inference."""
_base_ = ['../ltx_video/demo_loss.py']
root = __import__('os').environ.get('HFTRAINER_CHECKPOINT', 'checkpoints/hftrainer/ltx_video')
custom_imports = dict(imports=['hftrainer.models.ltx_video', 'hftrainer.trainers.ltx_video'], allow_failed_imports=False)
model = dict(type='LTXVideoBundle', from_pretrained=root)

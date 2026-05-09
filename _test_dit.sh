#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH

python3 -c "
import torch
from hftrainer.models.motion.hymotion_m2m.network.hymotion_dit import HunyuanMotionDiT

model = HunyuanMotionDiT(
    input_dim=540, feat_dim=512, output_dim=135,
    num_layers=12, num_heads=8, mlp_ratio=4.0,
    mlp_act_type='gelu_tanh', qk_norm_type='rms',
    qkv_bias=True, dropout=0.0,
    final_layer_cfg=dict(act_type='silu'),
    mask_mode='narrowband', time_factor=1000.0,
)
model = model.cuda()

B, L, D = 4, 360, 540
x = torch.randn(B, L, D, device='cuda')
t = torch.rand(B, device='cuda')
mask = torch.ones(B, L, dtype=torch.bool, device='cuda')

out = model(x, t, mask)
print('Input:', x.shape, 'Output:', out.shape)
total = sum(p.numel() for p in model.parameters())
print('Total params: %.1fM' % (total/1e6))
print('Forward pass OK!')
"

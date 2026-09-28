import json, torch
from pathlib import Path
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
x=torch.randn((9,56,128),generator=torch.Generator().manual_seed(42)).to('cuda',torch.bfloat16)
w=torch.randn((128,),generator=torch.Generator().manual_seed(7)).to('cuda',torch.bfloat16)
y=x.clone();z=x.clone();fused_inplace_qknorm(y,z,w,w,eps=1e-5,head_dim=128)
ref=torch.nn.functional.rms_norm(x,(128,),w,1e-5)
u=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)
variants={'torch_rms_norm':ref,'fp32_normalize_round_then_weight':u.bfloat16()*w,'fp32_normalize_and_weight':(u*w.float()).bfloat16()}
r={k:{'max':(v.float()-y.float()).abs().max().item(),'rmse':(v.float()-y.float()).square().mean().sqrt().item(),'equal':torch.equal(v,y)} for k,v in variants.items()}
Path('outputs/validation_alignment_20260928/h3_qknorm.json').write_text(json.dumps(r,indent=2));print(r)

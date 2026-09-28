"""Isolate first H3 refiner divergence using independent raw-weight loading."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from sglang.multimodal_gen.runtime.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3TokenRefiner as Official
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import FlashAttentionBackend
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from hftrainer.models.minimax_h3.network.transformer import MiniMaxH3TokenRefiner as Local, MiniMaxH3Transformer3DModel

out=Path('outputs/validation_alignment_20260928')
root=Path('checkpoints/MiniMax-H3/transformer')
config=json.loads((root/'config.json').read_text());index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='tcp://127.0.0.1:29687',backend='nccl')
initialize_model_parallel(sequence_parallel_degree=1,tensor_parallel_degree=1)
with torch.device('cuda'):
    reference=Official(MiniMaxH3DiTArchConfig(),None,prefix='token_refiner').eval()
with torch.device('meta'):
    local=Local(hidden_size=5376,num_attention_heads=56,attention_head_dim=128,ffn_dim=14336,num_layers=2,norm_eps=1e-5,qk_norm_eps=1e-5,final_norm_eps=1e-5).eval()
params=dict(reference.named_parameters());state={}
for shard in sorted({v for k,v in index.items() if k.startswith('token_refiner.')}):
    with safe_open(str(root/shard),framework='pt',device='cpu') as f:
        for key in f.keys():
            if not key.startswith('token_refiner.'):continue
            weight=f.get_tensor(key);name=key.removeprefix('token_refiner.');param=params[name]
            with torch.no_grad():
                if hasattr(param,'weight_loader'):param.weight_loader(param,weight)
                else:param.copy_(weight.to(param.dtype))
            for k,v in MiniMaxH3Transformer3DModel._convert_checkpoint_tensor(key,weight,config).items():
                state[k.removeprefix('token_refiner.')]=v.to('cuda',torch.bfloat16)
local.load_state_dict(state,strict=True,assign=True);del state
for block in reference.blocks:block.attn._set_attention_backend(FlashAttentionBackend)

def fa3(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,**kw):
    n=q.shape[2];cu=torch.tensor([0,n],dtype=torch.int32,device=q.device)
    return flash_attn_varlen_func(q[0].transpose(0,1).contiguous(),k[0].transpose(0,1).contiguous(),v[0].transpose(0,1).contiguous(),cu,cu,n,n,causal=is_causal).transpose(0,1)[None]
torch.nn.functional.scaled_dot_product_attention=fa3
trace=torch.load(out/'h3_layer_trace.pt',weights_only=True,map_location='cpu')
x=trace['token_refiner.input'].cuda();expected=trace['token_refiner.output']
values={}
def hook(name):
    def capture(m,a,value):
        if isinstance(value,tuple):value=value[0]
        if isinstance(value,torch.Tensor):values[name]=value.detach().cpu().clone().squeeze(0)
    return capture
for prefix,model in [('official',reference),('local',local)]:
    for name,m in model.named_modules():m.register_forward_hook(hook(prefix+'.'+name))
with torch.no_grad():
    cu=torch.tensor([0,len(x),len(x)],dtype=torch.int32,device='cuda')
    result=reference(x,cu_seqlens=cu,cu_seqlens_host=(0,len(x),len(x)),max_seqlen=len(x))
    ours=local(x[None])

def metric(a,b):
    a,b=a.float().cpu(),b.float().cpu();d=a-b
    return dict(max_abs=d.abs().max().item(),relative_rmse=(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-10)).item())
report={'official_reproduces_saved':metric(result,expected),'local':metric(ours[0],expected),'modules':{}}
for name,v in values.items():
    if not name.startswith('local.'):continue
    mapped=name.replace('local.','official.',1).replace('refiner_blocks.','blocks.').replace('.ff.net.2','.mlp.fc2').replace('.ff','.mlp').replace('.attn.to_out.0','.attn.out_proj').replace('.attn.norm_q','.attn.q_norm').replace('.attn.norm_k','.attn.k_norm')
    if mapped in values and values[mapped].shape==v.shape:report['modules'][name]=metric(v,values[mapped])
# Fused QKV versus three projections can choose different GEMM accumulation paths.
for i in range(2):
    qkv=values[f'official.blocks.{i}.attn.qkv_proj']
    for j,k in enumerate(('q','k','v')):report['modules'][f'qkv_{i}_{k}']=metric(values[f'local.refiner_blocks.{i}.attn.to_{k}'],qkv.chunk(3,dim=-1)[j])
# One-variable ablations against the same original refiner output.
from hftrainer.models.minimax_h3.network.transformer import MiniMaxH3AttnProcessor
base_processor=MiniMaxH3AttnProcessor()
def fused_qkv(attn,hidden_states,rotary_emb=None,attention_mask=None):
    projections=(attn.to_q,attn.to_k,attn.to_v)
    projected=torch.nn.functional.linear(hidden_states,torch.cat([m.weight for m in projections])).chunk(3,dim=-1)
    forwards=[m.forward for m in projections]
    try:
        for module,value in zip(projections,projected):module.forward=lambda *a,value=value:value
        return base_processor(attn,hidden_states,rotary_emb,attention_mask)
    finally:
        for module,forward in zip(projections,forwards):module.forward=forward
for block in local.refiner_blocks:block.attn.set_processor(fused_qkv)
with torch.no_grad():report['fused_qkv']=metric(local(x[None])[0],expected)
# The original MLP checkpoint and GEMM use gate-first rows.
import types
def gate_first(self,hidden):
    value_weight,gate_weight=self.proj.weight.chunk(2,dim=0)
    gate,value=torch.nn.functional.linear(hidden,torch.cat((gate_weight,value_weight))).chunk(2,dim=-1)
    return torch.nn.functional.silu(gate)*value
for block in local.refiner_blocks:block.ff.net[0].forward=types.MethodType(gate_first,block.ff.net[0])
with torch.no_grad():report['fused_qkv_gate_first']=metric(local(x[None])[0],expected)
torch.save(values,out/'h3_refiner_trace.pt');(out/'h3_refiner_ablations.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
torch.distributed.destroy_process_group()

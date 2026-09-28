"""Compare identical SGLang inputs with the local full pretrained denoiser."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel
from hftrainer.models.minimax_h3.network.layout import build_fl2va_layout, build_row_timesteps
from hftrainer.models.minimax_h3.network.scheduler import MiniMaxH3Scheduler


def metric(a, b):
    a,b=a.float().cpu(),b.float().cpu()
    d=a-b
    return dict(max_abs=d.abs().max().item(),rmse=d.square().mean().sqrt().item(),relative_rmse=(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-10)).item())


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--sglang-attention',action='store_true');args=parser.parse_args()
    if args.sglang_attention:
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
        def fa3(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,**kwargs):
            assert q.shape[0]==1 and attn_mask is None and dropout_p==0. and not kwargs
            n=q.shape[2];cu=torch.tensor([0,n],dtype=torch.int32,device=q.device)
            out=flash_attn_varlen_func(q[0].transpose(0,1).contiguous(),k[0].transpose(0,1).contiguous(),v[0].transpose(0,1).contiguous(),cu,cu,n,n,causal=is_causal)
            return out.transpose(0,1)[None]
        torch.nn.functional.scaled_dot_product_attention=fa3
    root=Path('outputs/validation_h3_diagnosis')
    source=torch.load(root/'sglang_inputs.pt',map_location='cpu',weights_only=True)
    expected=torch.load(root/'sglang_step0.pt',map_location='cpu',weights_only=True)
    cache=load_file('data/video_validation_20260928/h3/02.safetensors')
    report={'torch':str(torch.__version__),'attention':'sglang_fa3' if args.sglang_attention else 'torch_sdpa','text':metric(cache['prompt_embeds'],source['text']),'schedules':{},'comparisons':{}}
    for modality,shift in [('video',12),('audio',3)]:
        scheduler=MiniMaxH3Scheduler(shift=shift);scheduler.set_timesteps(50)
        report['schedules'][modality]=metric(scheduler.sigmas,torch.tensor(source['sigmas'][modality]))
    model=MiniMaxH3Transformer3DModel.from_pretrained('checkpoints/MiniMax-H3',subfolder='transformer',device='cuda',torch_dtype=torch.bfloat16).eval()
    official=torch.load('outputs/validation_alignment_20260928/h3_layer_trace.pt',map_location='cpu',weights_only=True)
    trace={}
    rows=torch.tensor([0,8,9,100,422,423,500,10000,21734])
    def capture(name,value):
        if name in trace:return
        if isinstance(value,tuple):value=value[0]
        if not isinstance(value,torch.Tensor):return
        value=value.detach()
        if value.ndim==3 and value.shape[0]==1:value=value[0]
        if value.ndim>=2 and value.shape[0]>=21735:value=value.index_select(0,rows.to(value.device))
        trace[name]=value.cpu().clone()
    def input_hook(name):
        return lambda module,values:capture(name+'.input',values[0]) if values else None
    def output_hook(name):
        return lambda module,values,output:capture(name+'.output',output)
    maps={'proj_in':'video_patch_proj','audio_proj_in':'audio_patch_proj','context_embedder':'condition_proj','time_embedder':'time_embedder','token_refiner':'token_refiner','proj_out':'final_layer'}
    for name,module in model.named_modules():
        mapped=maps.get(name,name.replace('transformer_blocks.','blocks.').replace('.ff.','.mlp.').replace('.ff','.mlp').replace('.attn.to_out.0','.attn.out_proj'))
        if mapped+'.input' in official or mapped+'.output' in official:
            module.register_forward_pre_hook(input_hook(mapped))
            module.register_forward_hook(output_hook(mapped))
    layout=build_fl2va_layout(source['tags'],num_latent_frames=37,latent_height=48,latent_width=48,num_audio_latents=207).to('cuda')
    times,indices=build_row_timesteps(layout,video_timestep=0.,audio_timestep=0.)
    for label,text in [('sglang_text',source['text'])]:
        velocities=model(hidden_states=source['video'][None].cuda(),audio_hidden_states=source['audio'][None].cuda(),
            encoder_hidden_states=text[None].cuda().bfloat16(),timestep=times.cuda(),timestep_indices=indices.cuda(),
            token_tags=layout.token_tags,position_ids=layout.position_ids,video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,text_indices=layout.text_indices,return_dict=False)
        result={}
        for modality,velocity in zip(('video','audio'),velocities):
            sigma=torch.tensor(source['sigmas'][modality],dtype=torch.float32,device='cuda')
            state=source[modality].cuda(); ratio=sigma[1]/sigma[0]
            denoised=state+sigma[0]*velocity[0].float()
            updated=ratio*state+(1.-ratio)*denoised
            result[modality]=metric(updated,expected[modality])
            # Divide out the small first-step scale to measure the model delta.
            result[modality]['velocity_rmse_from_update']=result[modality]['rmse']/abs((sigma[0]-sigma[1]).item())
        report['comparisons'][label]=result
    report['layers']={name:metric(value,official[name]) for name,value in trace.items() if name in official and value.shape==official[name].shape}
    torch.save(trace,Path(args.output).with_suffix('.pt'))
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

"""Full pretrained denoiser parity against pinned Diffusers, evaluation only.

The optional reference package is never imported by HFTrainer runtime code.
"""
import json
import time
from pathlib import Path
import torch
from safetensors.torch import load_file
from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel as Local
from hftrainer.models.minimax_h3.network.layout import build_fl2va_layout, build_row_timesteps, patchify_video_latents
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel as Reference


@torch.no_grad()
def main():
    torch.manual_seed(42)
    start=time.monotonic()
    local=Local.from_pretrained('checkpoints/MiniMax-H3',subfolder='transformer',device='cuda',torch_dtype=torch.bfloat16).eval()
    with torch.device('meta'):
        reference=Reference(**dict(local.config)).eval()
    reference.load_state_dict(local.state_dict(),strict=True,assign=True)
    reference.rope.inv_freq=local.rope.inv_freq.cuda().clone()
    cache=load_file('data/video_validation_20260928/h3/02.safetensors',device='cuda')
    clean_v=patchify_video_latents(cache['video_latents'][None],(1,2,2)).float()
    clean_a=cache['audio_latents'].permute(0,2,1).reshape(1,-1,32).float()
    noise_v,noise_a=torch.randn_like(clean_v),torch.randn_like(clean_a)
    layout=build_fl2va_layout(cache['text_token_tags'],num_latent_frames=37,latent_height=8,latent_width=8,num_audio_latents=207).to('cuda')
    records=[]
    for t in [0.0,0.5,0.9]:
        times,indices=build_row_timesteps(layout,video_timestep=t,audio_timestep=t)
        args=dict(hidden_states=t*clean_v+(1-t)*noise_v,audio_hidden_states=t*clean_a+(1-t)*noise_a,
                  encoder_hidden_states=cache['prompt_embeds'][None].bfloat16(),timestep=times.cuda(),timestep_indices=indices.cuda(),
                  token_tags=layout.token_tags,position_ids=layout.position_ids,video_indices=layout.video_indices,
                  audio_indices=layout.audio_indices,text_indices=layout.text_indices,return_dict=False)
        ours=local(**args)
        theirs=reference(**args)
        for name,a,b in zip(['video','audio'],ours,theirs):
            delta=(a.float()-b.float())
            rel=delta.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8)
            row=dict(timestep=t,modality=name,max_abs=delta.abs().max().item(),relative_rmse=rel.item(),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
            records.append(row);print(row,flush=True)
    result=dict(reference_commit='c1bf18c92c6285334adcaac7e75ef8946a227f49',shared_pretrained_parameters=True,
                elapsed_seconds=time.monotonic()-start,comparisons=records,
                passed=all(x['finite'] and x['relative_rmse']<0.01 for x in records),relative_rmse_tolerance=0.01)
    Path('outputs/validation_followup/h3_reference_parity.json').write_text(json.dumps(result,indent=2))
    if not result['passed']: raise RuntimeError('H3 reference numerical parity failed')


if __name__ == '__main__': main()

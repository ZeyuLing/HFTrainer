"""Compare pretrained local codec decoders with the pinned reference classes."""
import gc
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKLMiniMaxH3 as RefVideo, AutoencoderKLMiniMaxH3Audio as RefAudio
from hftrainer.models.minimax_h3.network import AutoencoderKLMiniMaxH3 as LocalVideo, AutoencoderKLMiniMaxH3Audio as LocalAudio
from hftrainer.models.minimax_h3.bundle import MiniMaxH3Bundle

@torch.no_grad()
def main():
    cache=load_file('data/video_validation_20260928/h3/02.safetensors',device='cuda')
    results=[]
    for name,local_cls,ref_cls,key in [('vae',LocalVideo,RefVideo,'video_latents'),('audio_vae',LocalAudio,RefAudio,'audio_latents')]:
        local=local_cls.from_pretrained('checkpoints/MiniMax-H3',subfolder=name,device='cuda',torch_dtype=torch.float32).eval()
        ref=ref_cls.from_config(dict(local.config)).eval().cuda()
        ref.load_state_dict(local.state_dict(),strict=True)
        z=cache[key]
        if name=='vae': z=z[None]
        mean,std=MiniMaxH3Bundle._latent_stats(local.config,z.shape[1],device=z.device,dtype=z.dtype,ndim=z.ndim)
        z=z*std+mean
        with torch.autocast('cuda',dtype=torch.float16,enabled=name=='vae'):
            a=local.decode(z).sample.float()
            b=ref.decode(z).sample.float()
        delta=a-b
        row=dict(component=name,shape=list(a.shape),max_abs=delta.abs().max().item(),relative_rmse=(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)).item(),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        print(row,flush=True);results.append(row)
        del local,ref,a,b,z
        gc.collect();torch.cuda.empty_cache()
    Path('outputs/validation_followup/h3_codec_reference.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

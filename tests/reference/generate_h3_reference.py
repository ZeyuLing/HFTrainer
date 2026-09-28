"""Reference denoising/layout/schedulers/codecs with shared converted H3 weights.

Text features are the same cached input as local inference, deliberately isolating
pipeline parity from conditioner differences. This is NOT independent validation
of the original-to-Diffusers weight conversion.
"""
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel as Reference
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
from hftrainer.models.minimax_h3.network import MiniMaxH3Transformer3DModel as Local
from tools.infer import _save_audio_video

@torch.no_grad()
def main():
    start=time.monotonic()
    root='checkpoints/MiniMax-H3'
    local=Local.from_pretrained(root,subfolder='transformer',device='cuda',torch_dtype=torch.bfloat16).eval()
    with torch.device('meta'):
        ref=Reference(**dict(local.config)).eval()
    ref.load_state_dict(local.state_dict(),strict=True,assign=True)
    ref.rope.inv_freq=local.rope.inv_freq.cuda().clone()
    del local
    blocks=MiniMaxH3Blocks().get_workflow('t2va')
    blocks.sub_blocks.pop('text_encoder')
    pipe=MiniMaxH3ModularPipeline(blocks=blocks)
    pipe.update_components(transformer=ref,
        vae=AutoencoderKLMiniMaxH3.from_pretrained(root,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).cuda(),
        audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(root,subfolder='audio_vae',torch_dtype=torch.float32,local_files_only=True).cuda(),
        scheduler=MiniMaxH3Scheduler.from_pretrained(root,subfolder='scheduler',local_files_only=True),
        audio_scheduler=MiniMaxH3Scheduler.from_pretrained(root,subfolder='audio_scheduler',local_files_only=True))
    cached=load_file('data/video_validation_20260928/h3/02.safetensors',device='cuda')
    print('Reference pipeline sampling',flush=True)
    result=pipe(height=768,width=768,num_frames=124,num_inference_steps=50,
        generator=torch.Generator('cpu').manual_seed(42),prompt_embeds=cached['prompt_embeds'][None],
        text_token_tags=cached['text_token_tags'].cpu(),output_type='pt',output=['videos','audio'])
    # Both Diffusers and the repository writer use B,T,C,H,W.
    video=result['videos']
    output='outputs/validation_followup/h3_reference_seed42.mp4'
    _save_audio_video(video,result['audio'],output,24,32000)
    Path(output).with_suffix('.json').write_text(json.dumps(dict(reference_commit='c1bf18c92c6285334adcaac7e75ef8946a227f49',elapsed_seconds=time.monotonic()-start,video_shape=list(video.shape),audio_shape=list(result['audio'].shape)),indent=2))
    print(output,flush=True)

if __name__=='__main__': main()

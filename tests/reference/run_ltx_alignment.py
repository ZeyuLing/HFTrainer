"""Run independent upstream or public HFTrainer inference with observation hooks."""
import argparse
import importlib
import json
import runpy
import sys
import time
from pathlib import Path
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--backend',choices=['official','local'],required=True)
parser.add_argument('--output-dir',required=True)
parser.add_argument('--prompt',default='A dancer performs a dance routine with music.')
parser.add_argument('--frames',type=int,default=121)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--text-only',action='store_true')
args=parser.parse_args()
out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
prefix='ltx_pipelines' if args.backend=='official' else 'hftrainer.pipelines.ltx_video.backend'
core='ltx_core' if args.backend=='official' else 'hftrainer.models.ltx_video.network'
blocks=importlib.import_module(prefix+'.utils.blocks')
original_audio=blocks.AudioDecoder.__call__
def audio_decode(self,*a,**kw):
    # Control backend nondeterminism equally in both implementations.
    with torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        result=original_audio(self,*a,**kw)
    torch.save(result.waveform.cpu(),out/'audio_waveform.pt')
    return result
blocks.AudioDecoder.__call__=audio_decode
encoder=importlib.import_module(core+'.text_encoders.gemma.encoders.base_encoder')
original_encode=encoder.LTXGemmaTextEncoder.encode

def encode(self,prompts):
    result=original_encode(self,prompts)
    # Observe actual unpadded tokens at selected layers, without changing returned data.
    torch.save([{'mask':mask.cpu(),'layers':{i:hs[i][:,mask[0].bool()].cpu() for i in (0,1,len(hs)//2,len(hs)-1)},'tokens':self.tokenizer.tokenize_with_weights(prompt)['gemma']} for prompt,(hs,mask) in zip(prompts,result,strict=True)],out/'text_layers.pt')
    return result
encoder.LTXGemmaTextEncoder.encode=encode
original_prompt=blocks.PromptEncoder.__call__
def prompt(self,*a,**kw):
    result=original_prompt(self,*a,**kw)
    torch.save([{'video':x.video_encoding.cpu(),'audio':x.audio_encoding.cpu()} for x in result],out/'conditioning.pt')
    if args.text_only:
        (out/'text_only.json').write_text(json.dumps({'backend':args.backend,'prompt':args.prompt,'torch':str(torch.__version__)}))
        raise SystemExit(0)
    return result
blocks.PromptEncoder.__call__=prompt
original_stage=blocks.DiffusionStage.__call__
stage_count=0
def stage(self,*a,**kw):
    global stage_count
    result=original_stage(self,*a,**kw);stage_count+=1
    torch.save({'video':result[0].latent.cpu(),'audio':result[1].latent.cpu()},out/f'stage{stage_count}.pt')
    return result
blocks.DiffusionStage.__call__=stage
root=Path('checkpoints/LTX-2.5').resolve()
common=['--prompt',args.prompt,'--seed',str(args.seed),'--height','512','--width','768','--num-frames',str(args.frames),'--num-inference-steps','30','--negative-prompt','blurry, distorted, low quality, artifacts']
start=time.monotonic()
if args.backend=='official':
    components={'transformer-path':'diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors','text-encoder-path':'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors','video-vae-path':'vae/ltx-2.5-video-vae-bf16.safetensors','audio-vae-path':'vae/ltx-2.5-audio-vae-bf16.safetensors','spatial-upsampler-path':'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors','distilled-lora':'loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors'}
    sys.argv=['ltx_pipelines.ti2vid_two_stages',*[v for k,path in components.items() for v in ('--'+k,str(root/path))],*common,'--output-path',str(out/'generated.mp4'),'--diffvae-optimization','chunked_eager']
    runpy.run_module('ltx_pipelines.ti2vid_two_stages',run_name='__main__')
else:
    sys.argv=['tools/infer_ltx_video.py','configs/ltx_video/infer_ltx_video_2_5_dev_lora.py',*common,'--output',str(out/'generated.mp4'),'--cfg-options','model.loras=[]','model.duration_head_path=None']
    runpy.run_path('tools/infer_ltx_video.py',run_name='__main__')
(out/'run.json').write_text(json.dumps({'backend':args.backend,'prompt':args.prompt,'frames':args.frames,'seed':args.seed,'elapsed_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'torch':str(torch.__version__)},indent=2))

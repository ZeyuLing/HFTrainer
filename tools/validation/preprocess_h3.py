"""Cache real local A/V clips with the repository's pretrained H3 components.

Run from the repository root with python -m tools.validation.preprocess_h3.
Components are loaded sequentially so a single H200 can prepare the cache.
"""
import argparse
import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from hftrainer.models.minimax_h3.bundle import MiniMaxH3Bundle
from hftrainer.models.minimax_h3.network import (
    AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Processor, MiniMaxH3Qwen3VLEncoder, MiniMaxH3Tokenizer,
)


def normalize(z, config):
    mean, std = MiniMaxH3Bundle._latent_stats(
        config, z.shape[1], device=z.device, dtype=z.dtype, ndim=z.ndim,
    )
    return (z - mean) / std


def release(model):
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--frames', type=int, default=124)
    args = parser.parse_args()
    if args.size % 32 or (args.frames - 5) % 17:
        raise ValueError('H3 requires spatial multiples of 32 and 17*n+5 frames')
    root, out = Path(args.model_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(s) for s in Path(args.manifest).read_text().splitlines() if s.strip()]
    records = [{} for _ in items]
    torch.manual_seed(42)
    print('Loading pretrained video VAE', flush=True)
    model = AutoencoderKLMiniMaxH3.from_pretrained(root, subfolder='vae', device='cuda', torch_dtype=torch.float32).eval()
    for i, item in enumerate(items):
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', item['video'], '-vf', f'fps=24,scale={args.size}:{args.size}', '-frames:v', str(args.frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
        arr = np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1,args.size,args.size,3)
        if len(arr) != args.frames:
            raise ValueError(f"{item['video']}: insufficient frames {len(arr)}")
        pixels = torch.from_numpy(arr).permute(3,0,1,2)[None].cuda().float()/255
        mean = pixels.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
        std = pixels.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
        z = normalize(model.encode((pixels-mean)/std).latent_dist.mode(), model.config)
        records[i]['video_latents'] = z[0].cpu().contiguous()
        print('video',i,tuple(z.shape), bool(torch.isfinite(z).all()),flush=True)
    release(model); del model
    print('Loading pretrained audio VAE', flush=True)
    model = AutoencoderKLMiniMaxH3Audio.from_pretrained(root, subfolder='audio_vae', device='cuda', torch_dtype=torch.float32).eval()
    # Match the complete audio hop duration to H3's generated A/V contract.
    samples = ((args.frames*32000//24+799)//800)*800
    for i,item in enumerate(items):
        raw = subprocess.check_output(['ffmpeg','-v','error','-i',item['video'],'-t',str(samples/32000),'-f','f32le','-ac','2','-ar','32000','-'])
        wave = np.frombuffer(raw,dtype=np.float32).copy().reshape(-1,2).T
        if wave.shape[-1] < samples:
            raise ValueError(f"{item['video']}: insufficient audio")
        wave = torch.from_numpy(wave[:,:samples].copy()).cuda()[:,None]
        z = normalize(model.encode(wave).latent_dist.mode(),model.config)
        records[i]['audio_latents'] = z[:, :, :].cpu().contiguous()
        print('audio',i,tuple(z.shape), bool(torch.isfinite(z).all()),flush=True)
    release(model); del model
    # Preserve codec work even when the conditioner fails to load.
    for i, record in enumerate(records): save_file(record,str(out/f'{i:02d}.codecs.safetensors'))
    print('Loading pretrained Qwen3-VL-32B', flush=True)
    tokenizer = MiniMaxH3Tokenizer.from_pretrained(root/'tokenizer')
    processor = MiniMaxH3Processor.from_pretrained(root/'processor',tokenizer=tokenizer)
    model = MiniMaxH3Qwen3VLEncoder.from_pretrained(root,subfolder='text_encoder',device='cuda',torch_dtype=torch.bfloat16).eval()
    for i,item in enumerate(items):
        p = processor.encode_presentation(item['caption'],mode='t2va')
        z = model.encode(p.token_ids,processor=processor,conditioning_layer=50)
        records[i].update(prompt_embeds=z[0].cpu().contiguous(),text_token_tags=torch.tensor(p.token_tags,dtype=torch.long))
        if not all(torch.isfinite(t).all() for t in records[i].values()): raise ValueError('Nonfinite cached feature')
        save_file(records[i],str(out/f'{i:02d}.safetensors'))
        print('prompt',i,tuple(z.shape),flush=True)
    for split in ['train','validation']:
        rows=[{'feature_file':f'{i:02d}.safetensors','source_video':item['video'],'caption':item['caption']} for i,item in enumerate(items) if item['split']==split]
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('CACHE COMPLETE',flush=True)


if __name__ == '__main__': main()

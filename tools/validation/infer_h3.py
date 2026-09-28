"""Complete local H3 A/V inference, loading the conditioner before the denoiser."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from mmengine.config import Config
from safetensors.torch import load_file, save_file

import hftrainer.models.minimax_h3
import hftrainer.pipelines.minimax_h3
from hftrainer.pipelines.builder import build_pipeline_from_cfg
from tools.infer import _save_audio_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Omit to evaluate the untouched pretrained base')
    parser.add_argument('--feature', help='Optional existing text cache; omit to encode the prompt locally')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--audio-seed', type=int, help='Controlled independent audio-noise ablation')
    parser.add_argument('--attention-backend', choices=['sdpa', 'flash_attention_3'], default='sdpa')
    parser.add_argument('--prompt', default='A dancer performs a dance routine with music.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cfg = Config.fromfile('configs/minimax_h3/validate_h3_local.py')
    cfg.model.transformer.gradient_checkpointing = False
    if args.checkpoint is None:
        cfg.model.transformer.trainable = False
        cfg.model.transformer.pop('lora_cfg', None)
        cfg.model.transformer.pop('checkpoint_format', None)
        cfg.model.transformer.save_ckpt = False
    for component, cls in [('vae', 'AutoencoderKLMiniMaxH3'), ('audio_vae', 'AutoencoderKLMiniMaxH3Audio')]:
        cfg.model[component] = dict(type=cls, from_pretrained=dict(
            pretrained_model_name_or_path=cfg.root, subfolder=component,
            device='cuda', torch_dtype='fp32', strict=True,
        ), trainable=False, save_ckpt=False)
    cfg.pipeline = dict(type='MiniMaxH3Pipeline',num_inference_steps=args.steps,attention_backend=args.attention_backend)
    start = time.monotonic()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.feature:
        cached = load_file(args.feature)
    else:
        from hftrainer.models.minimax_h3.network import MiniMaxH3Tokenizer, MiniMaxH3Processor, MiniMaxH3Qwen3VLEncoder
        from hftrainer.models.minimax_h3.network.attention import set_attention_backend

        root = Path(cfg.root)
        tokenizer = MiniMaxH3Tokenizer.from_pretrained(root / 'tokenizer')
        processor = MiniMaxH3Processor.from_pretrained(root / 'processor', tokenizer=tokenizer)
        encoder = MiniMaxH3Qwen3VLEncoder.from_pretrained(
            root, subfolder='text_encoder', device='cuda', torch_dtype=torch.bfloat16,
        ).eval()
        set_attention_backend(encoder, args.attention_backend)
        presentation = processor.encode_presentation(args.prompt, mode='t2va')
        with torch.no_grad():
            embeds = encoder.encode(presentation.token_ids, processor=processor, conditioning_layer=50)
        cached = dict(prompt_embeds=embeds[0].cpu().contiguous(), text_token_tags=torch.tensor(presentation.token_tags, dtype=torch.long))
        save_file(cached, str(Path(args.output).with_suffix('.conditioning.safetensors')))
        del encoder, embeds, processor, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print('Encoded prompt locally; released conditioner GPU memory', flush=True)
    pipeline = build_pipeline_from_cfg(cfg, checkpoint_path=args.checkpoint, merge_lora=True, strict_checkpoint=True)
    print('Loaded ' + ('trained adapter' if args.checkpoint else 'untouched base') + '; sampling', flush=True)
    predict = pipeline.bundle.predict_velocity
    counter = [0]
    def logged_predict(*values, **kwargs):
        result = predict(*values, **kwargs)
        counter[0] += 1
        if counter[0] % 5 == 0:
            print(f'Denoising step {counter[0]} elapsed {time.monotonic()-start:.1f}s', flush=True)
        return result
    pipeline.bundle.predict_velocity = logged_predict
    noise_kwargs = {}
    if args.audio_seed is not None:
        from hftrainer.models.minimax_h3.network.layout import audio_latent_num_frames
        audio_t = audio_latent_num_frames(124)
        noise_kwargs['audio_latents'] = torch.randn(
            (2 * audio_t, 32), generator=torch.Generator('cpu').manual_seed(args.audio_seed),
        ).reshape(2, audio_t, 32).permute(0, 2, 1)
    with torch.no_grad():
        result = pipeline(prompt=args.prompt,
                          mode='t2va',duration=5,height=args.size,width=args.size,
                          seed=args.seed, output_type='pt',
                          prompt_embeds=cached['prompt_embeds'][None],
                          text_token_tags=cached['text_token_tags'], **noise_kwargs)
    if not torch.isfinite(result.videos).all() or not torch.isfinite(result.audio).all():
        raise ValueError('Nonfinite generated media')
    torch.save({'video':result.video_latents,'audio':result.audio_latents},Path(args.output).with_suffix('.latents.pt'))
    output = _save_audio_video(result.videos,result.audio,args.output,24,32000)
    report = dict(output=output,video_shape=list(result.videos.shape),audio_shape=list(result.audio.shape),
                  elapsed_seconds=time.monotonic()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,
                  steps=args.steps,checkpoint=args.checkpoint,audio_seed=args.audio_seed,
                  attention_backend=args.attention_backend,prompt=args.prompt,seed=args.seed,
                  text_source=args.feature or 'local_conditioner')
    Path(args.output).with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()

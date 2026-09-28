"""Compare raw codec output using official final latents and independent weights."""
import gc,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.distributed.parallel_state import init_distributed_environment,initialize_model_parallel
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE,MiniMaxH3AudioVAE
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig,MiniMaxH3VideoVAEArchConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig,MiniMaxH3AudioVAEArchConfig
from hftrainer.models.minimax_h3.network import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
from hftrainer.models.minimax_h3.bundle import MiniMaxH3Bundle
from hftrainer.models.minimax_h3.network.layout import unpatchify_video_latents
init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='tcp://127.0.0.1:29689',backend='nccl');initialize_model_parallel(sequence_parallel_degree=1,tensor_parallel_degree=1)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
out=Path('outputs/validation_alignment_20260928');root=Path('/efs/nicorhli/project/video/MiniMax-H3/models/MiniMax-H3/FL2VA')
s=torch.load('outputs/validation_h3_diagnosis/sglang_step48.pt',weights_only=True,map_location='cuda')
inputs={'video_vae':unpatchify_video_latents(s['video'],channels=24,num_frames=37,height=48,width=48,patch_size=(1,2,2)), 'audio_vae':s['audio'].reshape(2,207,32).permute(0,2,1)}
report={}
with torch.no_grad():
 for name,ref_cls,cfg_cls,arch_cls,local_cls,local_folder in [('audio_vae',MiniMaxH3AudioVAE,MiniMaxH3AudioVAEConfig,MiniMaxH3AudioVAEArchConfig,AutoencoderKLMiniMaxH3Audio,'audio_vae'),('video_vae',MiniMaxH3VideoVAE,MiniMaxH3VideoVAEConfig,MiniMaxH3VideoVAEArchConfig,AutoencoderKLMiniMaxH3,'vae')]:
  stats=json.loads((root/name/'config.json').read_text());cfg=cfg_cls(arch_config=arch_cls(latents_mean=stats['latents_mean'],latents_std=stats['latents_std']))
  model=ref_cls(cfg).cuda().eval();state=load_file(str(root/name/('source/model.safetensors' if name=='video_vae' else 'model.safetensors')));model.load_state_dict(state,strict=True);del state
  z=inputs[name];mean,std=MiniMaxH3Bundle._latent_stats(type('Stats',(),stats),z.shape[1],device=z.device,dtype=z.dtype,ndim=z.ndim);z=z*std+mean
  if name=='video_vae':model.prepare_decoder_autocast_weights(torch.float16)
  with torch.autocast('cuda',dtype=torch.float16,enabled=name=='video_vae'):
   a=model.decode_base(z) if name=='video_vae' else model.decode(z)
  a=a.float().cpu();del model;gc.collect();torch.cuda.empty_cache()
  model=local_cls.from_pretrained('checkpoints/MiniMax-H3',subfolder=local_folder,device='cuda',torch_dtype=torch.float32).eval()
  with torch.autocast('cuda',dtype=torch.float16,enabled=name=='video_vae'):b=model.decode(z).sample.float().cpu()
  print(name,a.shape,b.shape,flush=True)
  d=a-b;report[name]={'exact':torch.equal(a,b),'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item()}
  print(report[name],flush=True);del model,a,b;gc.collect();torch.cuda.empty_cache()
(out/'h3_sglang_codecs.json').write_text(json.dumps(report,indent=2));torch.distributed.destroy_process_group()

"""Decode saved official final latents through HFTrainer's exact public codecs."""
from pathlib import Path
import torch
from hftrainer.models.minimax_h3.bundle import MiniMaxH3Bundle
from hftrainer.models.minimax_h3.network import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
from hftrainer.models.minimax_h3.network.layout import unpatchify_video_latents
from tools.infer import _save_audio_video
class Codecs:
 _latent_stats=staticmethod(MiniMaxH3Bundle._latent_stats)
 decode_video=MiniMaxH3Bundle.decode_video
 decode_audio=MiniMaxH3Bundle.decode_audio
 def require_components(self,*args): pass
c=Codecs();root='checkpoints/MiniMax-H3'
c.vae=AutoencoderKLMiniMaxH3.from_pretrained(root,subfolder='vae',device='cuda',torch_dtype=torch.float32).eval()
c.audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(root,subfolder='audio_vae',device='cuda',torch_dtype=torch.float32).eval()
s=torch.load('outputs/validation_h3_diagnosis/sglang_step48.pt',weights_only=True,map_location='cuda')
z=unpatchify_video_latents(s['video'],channels=24,num_frames=37,height=48,width=48,patch_size=(1,2,2))
a=s['audio'].reshape(2,207,32).permute(0,2,1)[None]
with torch.no_grad():v=c.decode_video(z).permute(0,2,1,3,4).cpu();a=c.decode_audio(a).cpu()
_save_audio_video(v,a,'outputs/validation_alignment_20260928/h3_official_latents_local_codec.mp4',24,32000)
print('done',flush=True)

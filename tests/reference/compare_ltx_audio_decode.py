"""Independent audio decoding of identical official first-stage latents."""
import json
from pathlib import Path
import torch
from ltx_pipelines.utils.blocks import AudioDecoder as Official
from hftrainer.pipelines.ltx_video.backend.utils.blocks import AudioDecoder as Local
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
root=Path('outputs/validation_alignment_20260928')
z=torch.load(root/'ltx_official_dance/stage1.pt',weights_only=True)['audio'].cuda()
path=str(Path('checkpoints/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors').resolve())
outputs={}
for name,cls in [('official',Official),('official_repeat',Official),('local',Local)]:
 with torch.inference_mode():
  audio=cls(path,torch.bfloat16,torch.device('cuda'))(z.clone())
 outputs[name]=audio.waveform.cpu()
 torch.save(outputs[name],root/f'ltx_audio_{name}.pt')
 print(name,tuple(audio.waveform.shape),audio.sampling_rate,flush=True)
print('official_repeat_max', (outputs['official_repeat'].float()-outputs['official'].float()).abs().max().item(),flush=True)
d=outputs['local'].float()-outputs['official'].float()
r={'exact':torch.equal(outputs['local'],outputs['official']),'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item()}
(root/'ltx_audio_decode.json').write_text(json.dumps(r,indent=2));print(r)

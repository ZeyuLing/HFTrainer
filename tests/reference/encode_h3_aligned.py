"""Encode through the local public conditioner, with official kernel settings."""
import argparse,json,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from hftrainer.models.minimax_h3.network import MiniMaxH3Tokenizer,MiniMaxH3Processor,MiniMaxH3Qwen3VLEncoder
from hftrainer.models.minimax_h3.network.attention import set_attention_backend
p=argparse.ArgumentParser();p.add_argument('--prompt',default='A dancer performs a dance routine with music.');p.add_argument('--output',required=True);args=p.parse_args()
root=Path('checkpoints/MiniMax-H3');start=time.monotonic()
tokenizer=MiniMaxH3Tokenizer.from_pretrained(root/'tokenizer');processor=MiniMaxH3Processor.from_pretrained(root/'processor',tokenizer=tokenizer)
model=MiniMaxH3Qwen3VLEncoder.from_pretrained(root,subfolder='text_encoder',device='cuda',torch_dtype=torch.bfloat16).eval()
set_attention_backend(model,'flash_attention_3')
presentation=processor.encode_presentation(args.prompt,mode='t2va')
with torch.no_grad():z=model.encode(presentation.token_ids,processor=processor,conditioning_layer=50)
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
save_file({'prompt_embeds':z[0].cpu().contiguous(),'text_token_tags':torch.tensor(presentation.token_tags,dtype=torch.long)},str(output))
report={'prompt':args.prompt,'torch':str(torch.__version__),'seconds':time.monotonic()-start}
if args.prompt=='A dancer performs a dance routine with music.':
 ref=torch.load('outputs/validation_h3_diagnosis/sglang_inputs.pt',weights_only=True,map_location='cpu')['text'];d=z[0].cpu().float()-ref.float()
 report.update(exact=torch.equal(z[0].cpu(),ref),max_abs=d.abs().max().item(),rmse=d.square().mean().sqrt().item())
output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

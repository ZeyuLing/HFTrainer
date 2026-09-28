"""Independent upstream Qwen text conditioning on CPU while GPU checks run."""
import json
import time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration, Qwen2TokenizerFast

start = time.monotonic()
root = Path('checkpoints/MiniMax-H3')
tokenizer = Qwen2TokenizerFast.from_pretrained(root / 'tokenizer', local_files_only=True)
ids = tokenizer('A dancer performs a dance routine with music.', add_special_tokens=False, return_tensors='pt')['input_ids']
print('Loading reference Qwen on CPU', flush=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(root / 'text_encoder', dtype=torch.bfloat16, device_map='cpu', local_files_only=True, attn_implementation='sdpa').eval()
print('Reference text forward', flush=True)
with torch.no_grad():
    output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.zeros_like(ids), use_cache=False, output_hidden_states=True).hidden_states[50].float()
cached = load_file('data/video_validation_20260928/h3/02.safetensors')['prompt_embeds'][None].float()
delta = output-cached
report = dict(reference_commit='a08ace4bbd97e721c98751deec37d87b026acadc', reference_device='cpu', local_device='cuda', token_ids=ids.tolist(), max_abs=delta.abs().max().item(), relative_rmse=(delta.square().mean().sqrt()/output.square().mean().sqrt()).item(), elapsed_seconds=time.monotonic()-start)
Path('outputs/validation_followup/h3_conditioner_reference.json').write_text(json.dumps(report, indent=2))
print(report, flush=True)

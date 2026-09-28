"""Differential tokenization against the actual official Gemma assets."""
import json
from pathlib import Path
from ltx_core.text_encoders.gemma.gemma_assets import GemmaAssets as OfficialAssets,build_gemma_hf_tokenizer
from ltx_core.text_encoders.gemma.tokenizer import LTXGemmaTokenizer as OfficialTokenizer
from hftrainer.models.ltx_video.network.text_encoders.gemma.gemma_assets import GemmaAssets,build_local_gemma_tokenizer as build_local
from hftrainer.models.ltx_video.network.text_encoders.gemma.tokenizer import LTXGemmaTokenizer
path='checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
o=OfficialTokenizer(build_gemma_hf_tokenizer(OfficialAssets.load(path)),max_length=1024)
l=LTXGemmaTokenizer(build_local(GemmaAssets.load(path)),max_length=1024)
texts=['A dancer performs a dance routine with music.','blurry, distorted, low quality, artifacts','A woman smiles at the camera and says, "Hello!"','一位舞者在舞台上跳舞，面部清晰。','café cafe\u0301 naïve résumé','  leading  spaces\tand\nnewlines  ','123 123.45 2026-09-28','🙂🫠🧑🏽‍🚀\U00031350','hello...world! <bos> <eos>','\n\n','  ','one '+('two '*1100)]
records=[]
for text in texts:
 a=o.tokenize_with_weights(text)['gemma'];b=l.tokenize_with_weights(text)['gemma']
 records.append({'text':text[:100],'equal':a==b,'reference_active':[int(t) for t,m in a if m],'local_active':[int(t) for t,m in b if m]})
Path('outputs/validation_alignment_20260928/ltx_tokenizer.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print([(r['text'][:45],r['equal']) for r in records]);assert all(r['equal'] for r in records)

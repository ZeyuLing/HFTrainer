import ast, torch, json
from pathlib import Path
from hftrainer.models.minimax_h3.network.layout import build_fl2va_layout
p=Path('/tmp/h3-official-python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py')
tree=ast.parse(p.read_text()); tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('sglang.'))]
ns={'MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT':64,'MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES':((0,),(-1,),(0,-1))};exec(compile(tree,str(p),'exec'),ns)
s=ns['minimax_h3_packed_sequence'](text_len=9,latent_t=37,latent_h=48,latent_w=48,audio_t=207,include_keyframe_cond=False)
l=build_fl2va_layout(torch.ones(9,dtype=torch.long),num_latent_frames=37,latent_height=48,latent_width=48,num_audio_latents=207)
r={}
for k,a,b in [('position_ids',l.position_ids,s['img_position_ids'][:l.position_ids.shape[0]]),('video_indices',l.video_indices,s['img_pos']),('audio_indices',l.audio_indices,s['audio_pos']),('text_indices',l.text_indices,s['text_pos']),('token_tags',l.token_tags,s['token_tags'][:l.position_ids.shape[0]])]:
 r[k]={'equal':torch.equal(a,b),'max_abs':(a-b).abs().max().item()}
r['sglang_attention_segments']=s['cu_seqlens'].tolist();r['local_seq_len']=l.position_ids.shape[0]
print(json.dumps(r,indent=2));Path('outputs/validation_h3_diagnosis/layout_comparison.json').write_text(json.dumps(r,indent=2))

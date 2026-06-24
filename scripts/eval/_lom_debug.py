import os, sys
from pathlib import Path
import torch
REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
LOM_ROOT = REPO / "ref_repo" / "language_of_motion"
os.chdir(LOM_ROOT); sys.path.insert(0, str(LOM_ROOT))
import smplx
_oc = smplx.create
class _Stub(torch.nn.Module):
    def __init__(self,*a,**k): super().__init__()
smplx.create = lambda p,**k: (_oc(p,**k) if os.path.exists(p) else _Stub())
smplx.FLAME = lambda *a,**k: _Stub()
import logging; logging.basicConfig(level=logging.ERROR)
import pytorch_lightning as pl
from lom.config import parse_args
from lom.models.build_model import build_model
from lom.utils.load_checkpoint import load_pretrained_vae, load_pretrained_lm
sys.argv = ["demo.py","--cfg","configs/demo_text2motion.yaml","--task","text2motion","--text","configs/demo_text2motion.yaml"]
cfg = parse_args(phase="demo")
cfg.model.params.lm.params.model_path = "google/flan-t5-base"
cfg.model.params.lm.params.flash_attention = False
pl.seed_everything(42)
device = torch.device("cuda")
model = build_model(cfg)
load_pretrained_vae(cfg, model, logging.getLogger(), phase="demo")
load_pretrained_lm(cfg, model, logging.getLogger(), phase="demo")
model.to(device).eval()
model.lm.device = device
lm = model.lm
cap = "a man is crouching then raises his left hand to his head the lowers it back down."
print("max_length", lm.max_length)

def run(prompt, do_sample):
    enc = lm.tokenizer(prompt, padding='max_length', max_length=lm.max_length, truncation=True,
                       return_attention_mask=True, add_special_tokens=True, return_tensors="pt")
    out = lm.language_model.generate(enc.input_ids.to(device), max_length=512, num_beams=1, do_sample=do_sample)
    s = lm.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return s

for ds in (False, True):
    s = run([cap], ds)
    print(f"\n=== do_sample={ds} raw_out(len={len(s)}): {s[:300]}")

# try instruction prefixes
for pref in ["Generate motion: ", "Generate a motion sequence from the following text: ",
             "Generate upper and lower body motion from given caption: ",
             "<Caption_Placeholder>"]:
    p = (pref + cap) if pref != "<Caption_Placeholder>" else cap
    s = run([p], False)
    print(f"\n=== prefix='{pref[:40]}' greedy out(len={len(s)}): {s[:200]}")

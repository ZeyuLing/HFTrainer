#!/usr/bin/env python3
"""Per-stage timing probe for the T2M HumanML3D-263 generation pipeline.

Splits wall time into: model build, ckpt load, text-encode, diffusion sample,
and motion198_to_humanml263 (FK + resample + process_file) so we can see which
stage dominates the ~tens-of-seconds-per-sample throughput.
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PROBE_GPU", "0")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
from hftrainer.datasets.motion.representation import (
    motion198_to_humanml263, setup_process_globals,
)

CFG = "configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py"
CKPT = ("work_dirs/hymotion_m2m_v2_smpl_caption_editfix_from870_20260528/"
        "checkpoint-epoch_230")
device = "cuda:0"


def t():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


t0 = t()
cfg = Config.fromfile(CFG)
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
if bundle._text_encoder_cfg is None:
    bundle._text_encoder_cfg = {
        "llm_type": "qwen3_embedding", "max_length_llm": 512,
        "sentence_emb_type": "clipl", "max_length_sentence_emb": 77,
        "enable_llm_padding": False,  # pad-to-actual-len, not 512 -> ~17x faster
    }
print(f"[build] {t()-t0:.1f}s", flush=True)

t0 = t()
sd = load_checkpoint(CKPT, map_location="cpu")
bundle.load_state_dict_selective(sd); del sd
bundle.eval().to(device)
print(f"[ckpt+to_gpu] {t()-t0:.1f}s", flush=True)

setup_process_globals()
pipe = HyMotionM2MPipeline(bundle=bundle, num_steps=50,
                           text_guidance_scale=2.0, replacement_guidance="none")

texts = ["A person is walking forward in a straight line.",
         "A person jumps and then sits down.",
         "A man waves his right hand."]
gt_lens = [120, 150, 100]
D = 198
for i, (text, gt_len) in enumerate(zip(texts, gt_lens)):
    T30 = min(360, int(round((gt_len + 2) * 30 / 20)) + 2)
    L = min(360, ((T30 + 3) // 4) * 4)
    src_mask = torch.zeros(1, L, D, device=device); src_mask[:, :T30, :] = 1.0
    src_motion = torch.zeros(1, L, D, device=device)
    batch = {"src_motion": src_motion, "src_mask": src_mask,
             "src_length": [T30], "tgt_length": [T30]}

    t0 = t()
    to = bundle.encode_text([text])
    batch["text_vec_raw"] = to["text_vec_raw"].to(device)
    batch["text_ctxt_raw"] = to["text_ctxt_raw"].to(device)
    batch["text_ctxt_raw_length"] = to["text_ctxt_raw_length"].to(device)
    t_enc = t() - t0

    t0 = t()
    with torch.no_grad():
        out = pipe(batch)
    denorm = bundle.denormalize_motion(out["latent"])[0].cpu()[:T30]
    t_gen = t() - t0

    t0 = t()
    m263, _ = motion198_to_humanml263(denorm.numpy(), rotation_space="local",
                                      src_fps=30.0, dst_fps=20.0, ensure_globals=False)
    t_cvt = t() - t0

    print(f"[sample {i}] L={L} T30={T30} | encode={t_enc:.2f}s "
          f"gen={t_gen:.2f}s convert={t_cvt:.2f}s | m263={m263.shape}", flush=True)

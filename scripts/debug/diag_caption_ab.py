#!/usr/bin/env python3
"""Decisive A/B test: does the caption actually drive generation?

Loads one M2M model, generates the SAME (mask-everything) clip under several
very different captions + an unconditional baseline, and reports how much the
generated motion differs between captions. If the per-caption outputs are
near-identical, text conditioning is broken in the generation path (not the
model). If they differ substantially, the model simply follows text weakly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_resume_046b.py")
    p.add_argument("--work-dir", default="work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4")
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint, find_latest_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    device = "cuda:0"
    cfg = Config.fromfile(args.config)
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    ckpt = find_latest_checkpoint(args.work_dir)
    print(f"[+] loading {ckpt}")
    sd = load_checkpoint(ckpt, map_location="cpu")
    bundle.load_state_dict_selective(sd)
    del sd
    if getattr(bundle, "_text_encoder_cfg", None) is None:
        bundle._text_encoder_cfg = {
            "llm_type": "qwen3_embedding", "max_length_llm": 512,
            "sentence_emb_type": "clipl", "max_length_sentence_emb": 77,
        }
    bundle.eval().to(device)

    pipeline = HyMotionM2MPipeline(
        bundle=bundle, num_steps=50,
        text_guidance_scale=args.cfg_scale, replacement_guidance="none",
    )

    D = 198
    L = args.frames
    captions = [
        "a person walks forward in a straight line",
        "a person is sitting down on a chair",
        "a person jumps high into the air with both arms raised",
        "a person spins around in a circle while waving",
    ]

    def gen(text):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        src_mask = torch.zeros(1, L, D, device=device)
        src_mask[:, :L, :] = 1.0
        src_motion = torch.zeros(1, L, D, device=device)
        batch = {"src_motion": src_motion, "src_mask": src_mask,
                 "src_length": [L], "tgt_length": [L]}
        if text is not None:
            t = bundle.encode_text([text])
            batch["text_vec_raw"] = t["text_vec_raw"].to(device)
            batch["text_ctxt_raw"] = t["text_ctxt_raw"].to(device)
            batch["text_ctxt_raw_length"] = t["text_ctxt_raw_length"].to(device)
        with torch.no_grad():
            out = pipeline(batch)
        return bundle.denormalize_motion(out["latent"])[0].cpu().float().numpy()[:L]

    outs = {c: gen(c) for c in captions}
    uncond = gen(None)

    ref = outs[captions[0]]
    print("\n=== per-caption difference vs caption[0] (mean abs over all 198 dims) ===")
    print(f"  uncond           : {np.mean(np.abs(uncond - ref)):.4f}")
    for c in captions:
        print(f"  {c[:34]:34s}: {np.mean(np.abs(outs[c] - ref)):.4f}")

    # focus on pose channels [3:135] (rotations) and trans [0:3]
    print("\n=== translation[0:3] mean abs diff vs caption[0] ===")
    for c in captions[1:]:
        print(f"  {c[:34]:34s}: {np.mean(np.abs(outs[c][:, :3] - ref[:, :3])):.4f}")
    print("=== rotation[3:135] mean abs diff vs caption[0] ===")
    for c in captions[1:]:
        print(f"  {c[:34]:34s}: {np.mean(np.abs(outs[c][:, 3:135] - ref[:, 3:135])):.4f}")
    print("\nINTERPRET: diffs ~0 (<1e-3) => text NOT conditioning (pipeline bug).")
    print("           diffs clearly >0 => text conditions; model just follows weakly.")


if __name__ == "__main__":
    main()

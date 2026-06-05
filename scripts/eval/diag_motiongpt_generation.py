#!/usr/bin/env python3
"""Print raw MotionGPT text generations for a few prompt paths."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from motiongpt_infer_hml3d263 import (  # noqa: E402
    MOTIONGPT_ROOT,
    DummyHumanML3DDataModule,
    force_untied_t5_lm_head,
    load_cfg,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5-path", default="google/flan-t5-base")
    ap.add_argument("--checkpoint", default=str(MOTIONGPT_ROOT / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"))
    ap.add_argument("--cfg", default=str(MOTIONGPT_ROOT / "configs" / "config_h3d_stage3.yaml"))
    ap.add_argument("--out-dir", default="/tmp/motiongpt_diag")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    os.chdir(MOTIONGPT_ROOT)
    force_untied_t5_lm_head()
    from mGPT.models.base import BaseModel  # noqa: WPS433
    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    from mGPT.models.build_model import build_model  # noqa: WPS433

    cfg = load_cfg(args)
    dm = DummyHumanML3DDataModule(
        MOTIONGPT_ROOT / "assets" / "meta" / "mean.npy",
        MOTIONGPT_ROOT / "assets" / "meta" / "std.npy",
    )
    model = build_model(cfg, dm).eval()
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    texts = [
        "a person walks forward",
        "a person jumps and turns around",
        "a person waves both arms",
    ]
    lengths = [80, 80, 80]
    modes = ["official_nolen", "official_len", "instruction", "direct"]
    for mode in modes:
        print(f"\n== {mode}", flush=True)
        if mode == "official_nolen":
            tasks = [{"input": ["Generate motion: <Caption_Placeholder>"], "output": [""]}] * len(texts)
            prompts, _ = model.lm.template_fulfill(tasks, [0] * len(texts), [""] * len(texts), texts, "test")
        elif mode == "official_len":
            tasks = [{"input": ["Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>"], "output": [""]}] * len(texts)
            prompts, _ = model.lm.template_fulfill(tasks, lengths, [""] * len(texts), texts, "test")
        elif mode == "instruction":
            instr = json.load(open(MOTIONGPT_ROOT / "prepare" / "instructions" / "template_instructions.json"))["Text-to-Motion"]["caption_framelen"]
            prompts, _ = model.lm.template_fulfill([instr] * len(texts), lengths, [""] * len(texts), texts, "test")
        else:
            prompts = texts
        with torch.no_grad():
            tokens, cleaned = model.lm.generate_direct(
                prompts,
                max_length=128,
                num_beams=1,
                do_sample=not args.greedy,
            )
        for prompt, output, token_ids in zip(prompts, cleaned, tokens):
            print("PROMPT:", prompt[:220].replace("\n", " "), flush=True)
            print("TOK_LEN:", len(token_ids), "TOK_HEAD:", token_ids[:12].detach().cpu().tolist(), flush=True)
            print("OUT:", output[:600].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()

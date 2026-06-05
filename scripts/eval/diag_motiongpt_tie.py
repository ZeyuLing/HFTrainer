#!/usr/bin/env python3
"""Diagnose MotionGPT T5 input/output embedding tying."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, T5ForConditionalGeneration

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from motiongpt_infer_hml3d263 import (  # noqa: E402
    MOTIONGPT_ROOT,
    DummyHumanML3DDataModule,
    load_cfg,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5-path", default="google/flan-t5-base")
    ap.add_argument("--checkpoint", default=str(MOTIONGPT_ROOT / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"))
    ap.add_argument("--cfg", default=str(MOTIONGPT_ROOT / "configs" / "config_h3d_stage3.yaml"))
    ap.add_argument("--out-dir", default="/tmp/motiongpt_tie_diag")
    ap.add_argument("--force-untie", action="store_true")
    args = ap.parse_args()

    if args.force_untie:
        original = T5ForConditionalGeneration.from_pretrained

        def from_pretrained_untied(model_path, *pargs, **kwargs):
            cfg = kwargs.pop("config", None)
            if cfg is None:
                cfg = AutoConfig.from_pretrained(model_path)
            cfg.tie_word_embeddings = False
            return original(model_path, *pargs, config=cfg, **kwargs)

        T5ForConditionalGeneration.from_pretrained = from_pretrained_untied

    os.chdir(MOTIONGPT_ROOT)
    from mGPT.models.base import BaseModel  # noqa: WPS433
    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    from mGPT.models.build_model import build_model  # noqa: WPS433

    cfg = load_cfg(args)
    dm = DummyHumanML3DDataModule(
        MOTIONGPT_ROOT / "assets" / "meta" / "mean.npy",
        MOTIONGPT_ROOT / "assets" / "meta" / "std.npy",
    )
    model = build_model(cfg, dm).eval()
    lm = model.lm.language_model
    print("config.tie_word_embeddings before", lm.config.tie_word_embeddings, flush=True)
    print("ptr_same before", lm.shared.weight.data_ptr() == lm.lm_head.weight.data_ptr(), flush=True)

    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    print("ptr_same after", lm.shared.weight.data_ptr() == lm.lm_head.weight.data_ptr(), flush=True)
    print(
        "means after",
        "shared_last", float(lm.shared.weight[-1].abs().mean()),
        "head_last", float(lm.lm_head.weight[-1].abs().mean()),
        "diff_mean", float((lm.shared.weight - lm.lm_head.weight).abs().mean()),
        flush=True,
    )

    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    prompts = [
        "I need a motion that represents a person walks forward. Can you generate it for me?",
        "Give me a gesture that corresponds to a person jumps and turns around",
    ]
    outputs, cleaned = model.lm.generate_direct(prompts, max_length=128, num_beams=1, do_sample=True)
    for prompt, out, toks in zip(prompts, cleaned, outputs):
        print("PROMPT", prompt, flush=True)
        print("TOK_LEN", len(toks), "TOK_HEAD", toks[:12].detach().cpu().tolist(), flush=True)
        print("OUT", out[:500].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect whether MotionGPT language-model weights load exactly."""
from __future__ import annotations

import argparse
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
    ap.add_argument("--mean", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "mean.npy"))
    ap.add_argument("--std", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "std.npy"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "evaluation" / "humanml3d" / "_inspect_motiongpt"))
    ap.add_argument("--tie-word-embeddings", action="store_true")
    args = ap.parse_args()

    if not args.tie_word_embeddings:
        force_untied_t5_lm_head()

    os.chdir(MOTIONGPT_ROOT)
    from mGPT.models.base import BaseModel  # noqa: WPS433
    from mGPT.models.build_model import build_model  # noqa: WPS433

    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    cfg = load_cfg(args)
    dm = DummyHumanML3DDataModule(Path(args.mean), Path(args.std))
    model = build_model(cfg, dm).eval()
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    result = model.load_state_dict(state, strict=False)
    print(f"load_result={result}")

    names = [
        "lm.language_model.shared.weight",
        "lm.language_model.encoder.embed_tokens.weight",
        "lm.language_model.decoder.embed_tokens.weight",
        "lm.language_model.lm_head.weight",
    ]
    params = dict(model.named_parameters())
    for name in names:
        param = params.get(name)
        ckpt = state.get(name)
        if param is None or ckpt is None:
            print(f"{name}: param={param is not None} ckpt={ckpt is not None}")
            continue
        p = param.detach().cpu()
        diff = (p - ckpt).abs().max().item()
        print(
            f"{name}: shape={tuple(p.shape)} maxdiff={diff:.6g} "
            f"mean={p.float().mean().item():.6g} std={p.float().std().item():.6g}"
        )

    lm = model.lm.language_model
    print(f"ptr shared/enc={lm.shared.weight.data_ptr() == lm.encoder.embed_tokens.weight.data_ptr()}")
    print(f"ptr shared/dec={lm.shared.weight.data_ptr() == lm.decoder.embed_tokens.weight.data_ptr()}")
    print(f"ptr shared/head={lm.shared.weight.data_ptr() == lm.lm_head.weight.data_ptr()}")
    print(f"config.tie_word_embeddings={lm.config.tie_word_embeddings}")
    print(f"tokenizer_len={len(model.lm.tokenizer)} vocab_size={lm.config.vocab_size}")


if __name__ == "__main__":
    main()

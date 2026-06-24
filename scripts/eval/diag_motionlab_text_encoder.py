#!/usr/bin/env python3
"""Inspect MotionLab text encoder outputs for wrapper debugging."""
from __future__ import annotations

import torch

from motionlab_infer_hml3d263 import _load_cfg, _load_modules


class Args:
    cfg = "configs/config_rfmotion_text.yaml"
    cfg_assets = "configs/assets.yaml"
    checkpoint = "checkpoints/motionflow/motionflow/motionflow.ckpt"
    cfg_from_checkpoint = True
    clip_path = "openai/clip-vit-large-patch14"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(Args())
    print("target", cfg.model.text_encoder)
    text_encoder, _denoiser, _scheduler = _load_modules(cfg, device)
    print("class", text_encoder.__class__)
    print("attrs", getattr(text_encoder, "name", None), getattr(text_encoder, "max_length", None))
    out = text_encoder(["", "a person walks forward"])
    print("out_type", type(out))
    if isinstance(out, (tuple, list)):
        print("out_len", len(out))
        for idx, item in enumerate(out):
            print("item", idx, type(item), getattr(item, "shape", None))
    else:
        keys = list(out.keys()) if hasattr(out, "keys") else []
        print("out_keys", keys)
        print(
            "last",
            getattr(getattr(out, "last_hidden_state", None), "shape", None),
            "pool",
            getattr(getattr(out, "pooler_output", None), "shape", None),
        )


if __name__ == "__main__":
    main()

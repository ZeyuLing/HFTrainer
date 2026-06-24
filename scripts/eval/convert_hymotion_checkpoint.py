#!/usr/bin/env python3
"""Convert upstream HY-Motion T2M checkpoints into hftrainer artifacts.

The released HY-Motion checkpoints are ``latest.ckpt`` files with a
``model_state_dict`` that contains the MMDiT transformer, classifier-free null
embeddings, and 201-dim Mean/Std stats. This CLI re-exports them as a compact
hftrainer artifact readable by ``HyMotionT2MBundle.from_pretrained``:

    <out>/hymotion_t2m_config.json
    <out>/motion_transformer.safetensors
    <out>/Mean.npy
    <out>/Std.npy
    <out>/text_encoder/llm/
    <out>/text_encoder/sentence/

By default the artifact is self-contained and includes the frozen Qwen3-8B and
CLIP-L text encoders. Use ``--no_text_encoder`` only for legacy lightweight
exports.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = "configs/hymotion_t2m/hymotion_t2m_201dim_full.py"
DEFAULT_CKPT = "checkpoints/HY-Motion-1.0/HY-Motion-1.0/latest.ckpt"
DEFAULT_TEXT = "a person walks forward in a straight line"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _state_for_artifact(bundle) -> dict[str, torch.Tensor]:
    state = {
        f"motion_transformer.{k}": v.detach().cpu()
        for k, v in bundle.motion_transformer.state_dict().items()
    }
    state["null_vtxt_feat"] = bundle.null_vtxt_feat.detach().cpu()
    state["null_ctxt_input"] = bundle.null_ctxt_input.detach().cpu()
    return state


def _build_raw_bundle(config_path: str, ckpt_path: str, text_dtype: str):
    import hftrainer  # noqa: F401  (registry side effects)
    import hftrainer.models.motion.components.hunyuan_motion  # noqa: F401
    import hftrainer.models.motion.hymotion_t2m.bundle  # noqa: F401
    from hftrainer.models.motion.hymotion_t2m.bundle import _DEFAULT_TEXT_ENCODER_CFG
    from hftrainer.registry import MODEL_BUNDLES
    from mmengine.config import Config

    cfg = Config.fromfile(str(REPO / config_path))
    model_cfg = dict(cfg.model)
    if not model_cfg.get("text_encoder"):
        model_cfg["text_encoder"] = dict(_DEFAULT_TEXT_ENCODER_CFG)
    model_cfg["text_dtype"] = text_dtype
    bundle = MODEL_BUNDLES.build(model_cfg).eval()

    ckpt_file = REPO / ckpt_path
    print(f"[convert] loading raw checkpoint: {ckpt_file}", flush=True)
    ck = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    missing, unexpected = bundle.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("special_game")]
    print(
        f"[convert] raw load: missing={len(missing)} "
        f"unexpected(non-special_game)={len(unexpected)}",
        flush=True,
    )
    if missing:
        raise RuntimeError(f"raw checkpoint missing keys: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"raw checkpoint unexpected keys: {unexpected[:10]}")
    return bundle


def _verify_state(raw_bundle, reloaded_bundle) -> float:
    raw_state = _state_for_artifact(raw_bundle)
    rel_state = _state_for_artifact(reloaded_bundle)
    if raw_state.keys() != rel_state.keys():
        missing = sorted(raw_state.keys() - rel_state.keys())
        extra = sorted(rel_state.keys() - raw_state.keys())
        raise AssertionError(f"artifact key mismatch: missing={missing[:5]} extra={extra[:5]}")

    max_diff = 0.0
    worst = None
    for key in raw_state:
        a = raw_state[key]
        b = rel_state[key]
        if a.shape != b.shape:
            raise AssertionError(f"{key} shape mismatch: {a.shape} vs {b.shape}")
        diff = float((a.float() - b.float()).abs().max())
        if diff > max_diff:
            max_diff = diff
            worst = key
    print(f"[verify] state max-abs-diff = {max_diff} ({worst})", flush=True)
    if max_diff != 0.0:
        raise AssertionError("artifact state diverged from raw checkpoint")
    return max_diff


def _clear_bundle_runtime(bundle) -> None:
    if hasattr(bundle, "_text_encoder"):
        bundle._text_encoder = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _generate_one(bundle, device: str, seed: int, text: str, length: int) -> np.ndarray:
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

    bundle.to(torch.device(device))
    bundle.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    pipe = HyMotionT2MPipeline(
        bundle,
        num_steps=int(bundle.validation_steps),
        text_guidance_scale=5.0,
        should_apply_smoothing=True,
    )
    with torch.no_grad():
        out = pipe({"caption": [text], "tgt_length": [length]})
    latent = out["latent"].detach().cpu().float().numpy()
    _clear_bundle_runtime(bundle)
    bundle.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return latent


def _verify_generation(raw_bundle, reloaded_bundle, args) -> float:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    print(
        f"[verify] generation text={args.verify_text!r} length={args.verify_length} "
        f"device={device}",
        flush=True,
    )
    a = _generate_one(
        raw_bundle,
        device=device,
        seed=args.seed,
        text=args.verify_text,
        length=args.verify_length,
    )
    b = _generate_one(
        reloaded_bundle,
        device=device,
        seed=args.seed,
        text=args.verify_text,
        length=args.verify_length,
    )
    if a.shape != b.shape:
        raise AssertionError(f"generation shape mismatch: {a.shape} vs {b.shape}")
    diff = float(np.abs(a - b).max())
    print(f"[verify] generation max-abs-diff = {diff}", flush=True)
    if diff > 1e-5:
        raise AssertionError("artifact generation diverged from raw checkpoint")
    return diff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--variant", default="1.0b")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--device", default=_default_device())
    p.add_argument("--text_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--no_text_encoder",
        action="store_true",
        help="legacy export: do not copy Qwen3-8B / CLIP-L into the artifact",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify_text", default=DEFAULT_TEXT)
    p.add_argument("--verify_length", type=int, default=80)
    args = p.parse_args()

    from hftrainer.models.motion.hymotion_t2m import HyMotionT2MBundle

    print(
        f"[convert] config={args.config} ckpt={args.ckpt} variant={args.variant}",
        flush=True,
    )
    raw_bundle = _build_raw_bundle(args.config, args.ckpt, args.text_dtype)
    raw_bundle.save_pretrained(
        args.out_dir,
        variant=args.variant,
        include_text_encoder=not args.no_text_encoder,
    )
    out_dir = Path(args.out_dir)
    print(f"[convert] wrote artifact -> {out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in out_dir.iterdir())}", flush=True)

    if args.verify:
        reloaded = HyMotionT2MBundle.from_pretrained(
            args.out_dir,
            device=None,
            text_dtype=args.text_dtype,
        )
        _verify_state(raw_bundle, reloaded)
        _verify_generation(raw_bundle, reloaded, args)
        print("[verify] OK: artifact round-trip matches the raw checkpoint.", flush=True)


if __name__ == "__main__":
    main()

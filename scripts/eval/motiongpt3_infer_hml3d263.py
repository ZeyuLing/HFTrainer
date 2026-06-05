#!/usr/bin/env python3
"""Run MotionGPT3 text-to-motion inference and save HumanML3D-263 features."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
MGPT3_ROOT = REPO / "ref_repo" / "MotionGPT3"
sys.path.insert(0, str(MGPT3_ROOT))


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_entries(raw):
    if isinstance(raw, dict) and "data_list" in raw:
        data = raw["data_list"]
        if isinstance(data, dict):
            for name, entry in data.items():
                yield str(name), entry
        else:
            for i, entry in enumerate(data):
                yield str(entry.get("motion_id") or entry.get("id") or i), entry
    elif isinstance(raw, list):
        for i, entry in enumerate(raw):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry
    else:
        raise ValueError("Unrecognized annotation format")


def _load_rewritten(path: Optional[Path]):
    if path is None:
        return {}
    raw = _load_json(path)
    if isinstance(raw, dict) and "data_list" in raw:
        raw = raw["data_list"]
    if not isinstance(raw, dict):
        raise ValueError(f"rewritten caption file must be a dict: {path}")
    out = {}
    for key, value in raw.items():
        if isinstance(value, str):
            cap = value
        elif isinstance(value, dict):
            cap = value.get("caption") or value.get("text") or value.get("short_caption")
        else:
            cap = None
        if isinstance(cap, str) and cap.strip():
            out[str(key)] = cap.strip()
    return out


def _load_caption_from_json(path: Path):
    try:
        data = _load_json(path)
    except Exception:
        return None
    pool = []
    if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
        for group in ("macro", "meso", "micro"):
            pool.extend(v.strip() for v in data[group] if isinstance(v, str) and v.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                vals = item.get(key)
                if isinstance(vals, list):
                    pool.extend(v.strip() for v in vals if isinstance(v, str) and v.strip())
                    break
            else:
                for key in ("short_caption", "short caption"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        pool.append(val.strip())
                        break
    return pool[0] if pool else None


def _load_annotation_pairs(anno_file: Path, data_dir: Path, rewritten_file: Optional[Path],
                           caption_protocol: str, default_fps: float):
    rewritten = _load_rewritten(rewritten_file)
    pairs = []
    for name, entry in _iter_entries(_load_json(anno_file)):
        caption = None
        if caption_protocol == "rewritten":
            caption = rewritten.get(name)
        if not caption and caption_protocol in {"original", "fallback"} and entry.get("hierarchical_caption_path"):
            caption = _load_caption_from_json(data_dir / entry["hierarchical_caption_path"])
        if not caption and caption_protocol in {"rewritten", "fallback"}:
            caption = rewritten.get(name)
        if not isinstance(caption, str) or not caption.strip():
            continue
        src_fps = float(entry.get("fps") or default_fps)
        length_src = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * src_fps))
        if length_src <= 0:
            continue
        pairs.append((name, caption.strip(), length_src, src_fps))
    return pairs


def _select_shard(items, num_shards: int, shard_index: int):
    if num_shards <= 1:
        return items
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


def _safe_name(name: str) -> str:
    return str(name).replace("/", "__")


class _DummyDataModule:
    name = "humanml3d"
    njoints = 22
    fps = 20
    is_mm = False

    def __init__(self):
        mean_path = MGPT3_ROOT / "datasets" / "humanml3d" / "Mean.npy"
        std_path = MGPT3_ROOT / "datasets" / "humanml3d" / "Std.npy"
        self.mean = torch.from_numpy(np.load(mean_path)).float()
        self.std = torch.from_numpy(np.load(std_path)).float()

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(features.device, features.dtype)
        std = self.std.to(features.device, features.dtype)
        return features * std + mean

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*features.shape[:2], self.njoints, 3), device=features.device, dtype=features.dtype)


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    for param in module.parameters(recurse=True):
        return param.dtype
    return torch.float32


def _convert_head_mask_to_5d(module: torch.nn.Module, head_mask: torch.Tensor,
                             num_hidden_layers: int) -> torch.Tensor:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if head_mask.dim() != 5:
        raise ValueError(f"head_mask.dim != 5, instead {head_mask.dim()}")
    return head_mask.to(dtype=_module_dtype(module))


def _get_head_mask(module: torch.nn.Module, head_mask: Optional[torch.Tensor],
                   num_hidden_layers: int, is_attention_chunked: bool = False):
    if head_mask is None:
        return [None] * num_hidden_layers
    head_mask = _convert_head_mask_to_5d(module, head_mask, num_hidden_layers)
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask


def _patch_motiongpt3_transformers_compat(model: torch.nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "MoTGPT2Model":
            continue
        if not hasattr(module, "model_parallel"):
            module.model_parallel = False
        if not hasattr(module, "device_map"):
            module.device_map = None
        if not hasattr(module, "get_head_mask"):
            module.get_head_mask = types.MethodType(_get_head_mask, module)
            patched += 1
    return patched


def _load_cfg(args):
    from motGPT.config import get_module_config

    cwd = os.getcwd()
    os.chdir(MGPT3_ROOT)
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        cfg_assets = OmegaConf.load("./configs/assets.yaml")
        cfg_base = OmegaConf.load("./configs/default.yaml")
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
        if not cfg_exp.FULL_CONFIG:
            cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
        cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    finally:
        os.chdir(cwd)

    cfg.DEBUG = False
    cfg.DEVICE = [0]
    cfg.TEST.CHECKPOINTS = str(Path(args.checkpoint).resolve())
    cfg.METRIC.TYPE = []
    cfg.model.params.metrics_dict = []
    cfg.model.params.guidance_scale = args.guidance_scale
    cfg.lm_ablation.model_guidance_scale = args.guidance_scale
    cfg.FOLDER = str(Path(args.out_dir).resolve())
    cfg.TIME = _dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg.FOLDER_EXP = str((Path(args.out_dir).resolve() / "_motiongpt3_runtime"))
    return cfg


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(MGPT3_ROOT / "configs" / "test.yaml"))
    parser.add_argument("--checkpoint", default=str(MGPT3_ROOT / "checkpoints" / "motiongpt3.ckpt"))
    parser.add_argument("--anno_file", required=True)
    parser.add_argument("--anno_data_dir", default=str(REPO / "data" / "motionhub"))
    parser.add_argument("--rewritten_file", default=None)
    parser.add_argument("--caption_protocol", choices=["rewritten", "original", "fallback"], default="rewritten")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--gt_fps", type=float, default=30.0)
    parser.add_argument("--model_fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    return parser.parse_args()


def main():
    args = build_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_annotation_pairs(
        Path(args.anno_file),
        Path(args.anno_data_dir),
        Path(args.rewritten_file) if args.rewritten_file else None,
        args.caption_protocol,
        args.gt_fps,
    )
    pairs = _select_shard(pairs, args.num_shards, args.shard_index)
    if args.max_samples:
        pairs = pairs[:args.max_samples]

    indexed = []
    src_fps_values = []
    for sid, caption, length_src, src_fps in pairs:
        ml = int(round(length_src * args.model_fps / src_fps))
        ml = (ml // 4) * 4
        ml = max(40, min(196, ml))
        indexed.append((sid, caption, ml))
        src_fps_values.append(src_fps)
    print(f"[setup] shard={args.shard_index}/{args.num_shards} pairs={len(indexed)} out={out_dir}", flush=True)
    if src_fps_values:
        print(f"[setup] source_fps min/median/max="
              f"{min(src_fps_values):.3g}/{float(np.median(src_fps_values)):.3g}/{max(src_fps_values):.3g}; "
              f"model_fps={args.model_fps:.3g}", flush=True)

    os.chdir(MGPT3_ROOT)
    from motGPT.models.base import BaseModel
    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    from motGPT.models.build_model import build_model

    cfg = _load_cfg(args)
    datamodule = _DummyDataModule()
    model = build_model(cfg, datamodule).eval()
    patched = _patch_motiongpt3_transformers_compat(model)
    if patched:
        print(f"[compat] patched get_head_mask on {patched} MoTGPT2Model module(s)", flush=True)
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    load_result = model.load_state_dict(state, strict=False)
    if load_result is None:
        print("[load] checkpoint loaded", flush=True)
    else:
        missing, unexpected = load_result
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    written = skipped = failed = 0
    bs = max(1, args.batch_size)
    for start in range(0, len(indexed), bs):
        chunk = indexed[start:start + bs]
        todo = []
        for sid, caption, ml in chunk:
            if args.skip_existing and (out_dir / f"{_safe_name(sid)}.npy").exists():
                skipped += 1
            else:
                todo.append((sid, caption, ml))
        if not todo:
            continue

        texts = [x[1] for x in todo]
        lengths = [x[2] for x in todo]
        try:
            with torch.no_grad():
                outputs = model.lm.generate_conditional(
                    texts,
                    lengths=lengths,
                    stage="test",
                    tasks=None,
                )
                sampled_token_latents, motion_mask = model.lm.sample_tokens(
                    outputs,
                    model.lm.device,
                    temperature=1.0,
                    cfg=model.guidance_scale,
                    vae_mean_std_inv=model.vae.mean_std_inv,
                )
                z = sampled_token_latents.reshape(len(lengths), model.vae.latent_size, -1).permute(1, 0, 2)
                feats = model.vae.decode(z, lengths=lengths)
                if motion_mask is not None:
                    feats = feats.clone()
                    mask = motion_mask.to(device=feats.device, dtype=torch.bool)
                    while mask.ndim < feats.ndim:
                        mask = mask.unsqueeze(-1)
                    feats = torch.where(mask, torch.zeros_like(feats), feats)
                feats = datamodule.denormalize(feats).detach().cpu().numpy().astype(np.float32)
            if feats.ndim != 3 or feats.shape[-1] != 263:
                raise RuntimeError(f"unexpected feature shape: {feats.shape}")
            for k, (sid, _, ml) in enumerate(todo):
                np.save(out_dir / f"{_safe_name(sid)}.npy", feats[k, :ml].astype(np.float32))
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch {start}-{start + len(todo)}: {type(exc).__name__}: {exc}", flush=True)

        if (start // bs + 1) % 5 == 0:
            print(f"[progress] seen={min(start + bs, len(indexed))}/{len(indexed)} "
                  f"written={written} skipped={skipped} failed={failed}", flush=True)

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run MotionLCM's MLD checkpoint and save unstandardized HumanML3D-263 features."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
MLD_ROOT = REPO / "ref_repo" / "MotionLCM"
sys.path.insert(0, str(MLD_ROOT))

from mld.config import get_module_config  # noqa: E402
from mld.models.modeltype.mld import MLD  # noqa: E402
from mld.utils.utils import set_seed  # noqa: E402


class _DummyDataModule:
    is_mm = False

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        return features


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


def _load_cfg(args):
    cfg_path = Path(args.cfg).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    cwd = os.getcwd()
    os.chdir(MLD_ROOT)
    try:
        cfg = OmegaConf.load(cfg_path)
        cfg_model = get_module_config(cfg.model, cfg.model.target)
        cfg = OmegaConf.merge(cfg, cfg_model)
    finally:
        os.chdir(cwd)

    cfg.TEST.CHECKPOINTS = str(checkpoint_path)
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22
    cfg.METRIC.TYPE = []
    cfg.model.t5_path = args.text_encoder
    cfg.model.t2m_path = str((REPO / "ref_repo" / "MDM" / "t2m").resolve())
    cfg.model.guidance_scale = args.guidance_scale
    if "guidance_uncondp" not in cfg.model:
        cfg.model.guidance_uncondp = 0.0
    cfg.model.is_controlnet = False
    cfg.model.scheduler.num_inference_timesteps = args.num_inference_timesteps
    return cfg


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(MLD_ROOT / "configs" / "mld_t2m_infer.yaml"))
    parser.add_argument("--checkpoint", default=str(MLD_ROOT / "experiments_t2m" / "mld_humanml" / "mld_humanml_v1.ckpt"))
    parser.add_argument("--text_encoder", default="sentence-transformers/sentence-t5-large")
    parser.add_argument("--anno_file", required=True)
    parser.add_argument("--anno_data_dir", default=str(REPO / "data" / "motionhub"))
    parser.add_argument("--rewritten_file", default=None)
    parser.add_argument("--caption_protocol", choices=["rewritten", "original", "fallback"], default="rewritten")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--mean_path",
        default=str(REPO / "ref_repo" / "Momask" / "weights" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "meta" / "mean.npy"),
    )
    parser.add_argument(
        "--std_path",
        default=str(REPO / "ref_repo" / "Momask" / "weights" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2" / "meta" / "std.npy"),
    )
    parser.add_argument("--gt_fps", type=float, default=30.0)
    parser.add_argument("--mld_fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_timesteps", type=int, default=50)
    return parser.parse_args()


def main():
    args = build_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = np.load(args.mean_path).astype(np.float32)
    std = np.load(args.std_path).astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")

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
        ml = int(round(length_src * args.mld_fps / src_fps))
        ml = (ml // 4) * 4
        ml = max(40, min(196, ml))
        indexed.append((sid, caption, ml))
        src_fps_values.append(src_fps)
    print(f"[setup] shard={args.shard_index}/{args.num_shards} pairs={len(indexed)} out={out_dir}", flush=True)
    if src_fps_values:
        print(f"[setup] source_fps min/median/max="
              f"{min(src_fps_values):.3g}/{float(np.median(src_fps_values)):.3g}/{max(src_fps_values):.3g}; "
              f"model_fps={args.mld_fps:.3g}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(args)

    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    if "vae.global_motion_token" in state:
        token_count = int(state["vae.global_motion_token"].shape[0])
        hidden_dim = int(state["vae.global_motion_token"].shape[1])
        if "vae.latent_pre.weight" in state:
            cfg.model.latent_dim = [
                token_count // 2,
                int(state["vae.latent_pre.weight"].shape[0]),
                hidden_dim,
            ]
        else:
            cfg.model.latent_dim = [token_count // 2, hidden_dim]
    lcm_key = "denoiser.time_embedding.cond_proj.weight"
    if lcm_key in state:
        cfg.model.denoiser.params.time_cond_proj_dim = state[lcm_key].shape[1]
    state = {k: v for k, v in state.items() if not k.startswith("t2m_")}

    MLD._get_t2m_evaluator = lambda self, cfg: None
    model = MLD(cfg, _DummyDataModule())
    model.to(device)
    model.float()
    model.eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    diffusion_dtype = next(model.denoiser.parameters()).dtype

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

        batch = {"text": [x[1] for x in todo], "length": [x[2] for x in todo]}
        try:
            with torch.no_grad():
                texts = batch["text"]
                if model.do_classifier_free_guidance:
                    texts = [""] * len(texts) + texts
                text_emb = model.text_encoder(texts).to(device=device, dtype=diffusion_dtype)
                z = model._diffusion_reverse(text_emb)
                feats = model.vae.decode(z, batch["length"]).detach().cpu().numpy().astype(np.float32)
                feats = feats * std[None, None, :] + mean[None, None, :]
            if feats.ndim != 3 or feats.shape[-1] != 263:
                raise RuntimeError(f"unexpected feature shape: {feats.shape}")
            for k, (sid, _, ml) in enumerate(todo):
                np.save(out_dir / f"{_safe_name(sid)}.npy", feats[k, :ml].astype(np.float32))
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch {start}-{start + len(todo)}: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)

        if (start // bs + 1) % 5 == 0:
            print(f"[progress] seen={min(start + bs, len(indexed))}/{len(indexed)} "
                  f"written={written} skipped={skipped} failed={failed}", flush=True)

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

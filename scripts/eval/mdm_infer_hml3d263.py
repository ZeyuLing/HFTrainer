#!/usr/bin/env python3
"""Run MDM T2M inference and save unstandardized HumanML3D-263 features."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
MDM_ROOT = REPO / "ref_repo" / "MDM"
sys.path.insert(0, str(MDM_ROOT))

from data_loaders.tensors import collate  # noqa: E402
from utils import dist_util  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from utils.model_util import create_model_and_diffusion, load_saved_model  # noqa: E402
from utils.parser_util import (  # noqa: E402
    add_base_options,
    add_generate_options,
    add_sampling_options,
    get_cond_mode,
    parse_and_load_from_model,
)
from utils.sampler_util import ClassifierFreeSampleModel  # noqa: E402


class _DummyDataset:
    num_actions = 1


class _DummyData:
    dataset = _DummyDataset()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_motionhub_entries(raw):
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
    for name, entry in _iter_motionhub_entries(_load_json(anno_file)):
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


def build_args():
    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    parser.add_argument("--anno_file", required=True)
    parser.add_argument("--anno_data_dir", default="data/motionhub")
    parser.add_argument("--rewritten_file", default=None)
    parser.add_argument("--caption_protocol", choices=["rewritten", "original", "fallback"], default="rewritten")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--mean_path",
        default="ref_repo/Momask/weights/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/mean.npy",
    )
    parser.add_argument(
        "--std_path",
        default="ref_repo/Momask/weights/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/std.npy",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--gt_fps", type=float, default=30.0)
    parser.add_argument("--mdm_fps", type=float, default=20.0)
    args = parse_and_load_from_model(parser)
    if get_cond_mode(args) != "text":
        raise RuntimeError("MDM checkpoint is not text-conditional")
    return args


def main():
    args = build_args()
    fixseed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
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
        ml = int(round(length_src * args.mdm_fps / src_fps))
        ml = (ml // 4) * 4
        ml = max(40, min(196, ml))
        indexed.append((sid, caption, ml))
        src_fps_values.append(src_fps)
    print(f"[setup] shard={args.shard_index}/{args.num_shards} pairs={len(indexed)} out={out_dir}", flush=True)
    if src_fps_values:
        print(f"[setup] source_fps min/median/max="
              f"{min(src_fps_values):.3g}/{float(np.median(src_fps_values)):.3g}/{max(src_fps_values):.3g}; "
              f"model_fps={args.mdm_fps:.3g}", flush=True)

    dist_util.setup_dist(args.device)
    args.batch_size = min(args.batch_size, max(1, len(indexed)))
    data = _DummyData()
    mean = np.load(args.mean_path).astype(np.float32)
    std = np.load(args.std_path).astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")

    print("Creating model and diffusion...", flush=True)
    model, diffusion = create_model_and_diffusion(args, data)
    print(f"Loading checkpoints from [{args.model_path}]...", flush=True)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    written = skipped = failed = 0
    bs = args.batch_size
    for start in range(0, len(indexed), bs):
        chunk = indexed[start:start + bs]
        todo = []
        for sid, caption, ml in chunk:
            if args.skip_existing and (out_dir / f"{sid}.npy").exists():
                skipped += 1
            else:
                todo.append((sid, caption, ml))
        if not todo:
            continue

        n_frames = max(x[2] for x in todo)
        collate_args = [
            {
                "inp": torch.zeros(263, 1, n_frames),
                "tokens": None,
                "lengths": ml,
                "text": caption,
            }
            for _, caption, ml in todo
        ]
        motion, model_kwargs = collate(collate_args)
        model_kwargs["y"] = {
            key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = torch.ones(len(todo), device=dist_util.dev()) * args.guidance_param
        if "text" in model_kwargs["y"]:
            model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

        motion_shape = tuple(motion.shape)
        try:
            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model,
                    motion_shape,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
            if model.data_rep != "hml_vec":
                raise RuntimeError(f"expected hml_vec data_rep, got {model.data_rep}")
            sample_np = sample.detach().cpu().numpy().astype(np.float32)
            if sample_np.ndim != 4 or sample_np.shape[1] != 263 or sample_np.shape[2] != 1:
                raise RuntimeError(f"unexpected MDM sample shape: {sample_np.shape}")
            m263 = sample_np[:, :, 0, :].transpose(0, 2, 1)
            m263 = (m263 * std[None, None, :]) + mean[None, None, :]
            for k, (sid, _, ml) in enumerate(todo):
                np.save(out_dir / f"{sid}.npy", m263[k, :ml].astype(np.float32))
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

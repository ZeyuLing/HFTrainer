#!/usr/bin/env python3
"""Generate MLD HumanML3D-263 T2M outputs under the official HumanML3D protocol.

This is the hftrainer-native path: standard HumanML3D test split, one generated
motion per id, first caption, and unstandardized 263-dim output files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.mld import MLDBundle  # noqa: E402
from hftrainer.pipelines.mld import MLDPipeline  # noqa: E402

DEFAULT_DATA_ROOT = REPO / "ref_repo/CondMDI/dataset/HumanML3D"
DEFAULT_MODEL = REPO / "checkpoints/mld/humanml3d"
DEFAULT_BASE = REPO / "outputs/evaluation/t2m/humanml3d_official_test"
DEFAULT_SELECTED_ANNO = (
    DEFAULT_BASE
    / "captions/humanml3d_official_corrected"
    / "test_hml3d_official272_gtlen_official_caption.json"
)
DEFAULT_OFFICIAL_ANNO = REPO / "data/annotation/test_hml3d_official272_gtlen.json"
DEFAULT_ANNO = DEFAULT_SELECTED_ANNO if DEFAULT_SELECTED_ANNO.exists() else DEFAULT_OFFICIAL_ANNO


def first_caption(text_file: Path):
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if parts and parts[0].strip():
            return parts[0].strip()
    return None


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_entries(raw):
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        for name, entry in data.items():
            yield str(name), entry
        return
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            yield str(entry.get("motion_id") or entry.get("id") or idx), entry
        return
    raise ValueError("Unrecognized annotation format")


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
                for key in ("short_caption", "short caption", "caption", "text"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        pool.append(val.strip())
                        break
    elif isinstance(data, dict):
        for key in ("caption", "text", "short_caption", "short caption"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                pool.append(val.strip())
                break
    return pool[0] if pool else None


def resolve_annotation_items(
    anno_file: Path,
    data_dir: Path,
    *,
    source_fps_default: float = 30.0,
    model_fps: float = 20.0,
):
    items = []
    for name, entry in _iter_entries(_load_json(anno_file)):
        cap_path = entry.get("hierarchical_caption_path")
        if not cap_path:
            continue
        cap = _load_caption_from_json(data_dir / cap_path)
        if not cap:
            continue
        src_fps = float(entry.get("fps") or source_fps_default)
        src_len = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * src_fps))
        if src_len <= 0:
            continue
        gt_len = int(round(src_len * model_fps / src_fps))
        items.append((name, cap, max(1, gt_len)))
    return items


def _resolve_one(arg):
    name, motion_dir, text_dir = arg
    try:
        gt_len = int(np.load(motion_dir / f"{name}.npy", mmap_mode="r").shape[0])
    except (FileNotFoundError, OSError, ValueError):
        return None
    try:
        cap = first_caption(text_dir / f"{name}.txt")
    except (FileNotFoundError, OSError):
        return None
    if not cap:
        return None
    return (name, cap, gt_len)


def resolve_items(data_root: Path, names, workers: int = 32):
    from concurrent.futures import ThreadPoolExecutor

    motion_dir = data_root / "new_joint_vecs"
    text_dir = data_root / "texts"
    tasks = [(n, motion_dir, text_dir) for n in names]
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_resolve_one, tasks):
            if res is not None:
                out[res[0]] = res
    return [out[n] for n in names if n in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--anno_file", default=str(DEFAULT_ANNO),
                   help="Annotation JSON used for ids, corrected captions, and target lengths.")
    p.add_argument("--anno_data_dir", default=str(REPO))
    p.add_argument("--gt_fps", type=float, default=30.0)
    p.add_argument("--model_fps", type=float, default=20.0)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL),
                   help="hftrainer MLD artifact dir or HuggingFace repo id")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.anno_file:
        items = resolve_annotation_items(
            Path(args.anno_file),
            Path(args.anno_data_dir),
            source_fps_default=args.gt_fps,
            model_fps=args.model_fps,
        )
        if args.num_shards > 1:
            items = [item for i, item in enumerate(items) if i % args.num_shards == args.shard_index]
    else:
        data_root = Path(args.data_root)
        names = [n.strip() for n in (data_root / "test.txt").read_text().splitlines() if n.strip()]
        if args.num_shards > 1:
            names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard_index]
        cand = names if not args.max_samples else names[: args.max_samples * 3]
        items = resolve_items(data_root, cand)
    if args.max_samples:
        items = items[: args.max_samples]
    print(f"[setup] shard={args.shard_index}/{args.num_shards} items={len(items)} "
          f"steps={args.num_inference_steps} out={out_dir} (resolve {time.time() - t0:.1f}s)", flush=True)

    print(f"[setup] building MLD bundle/pipeline from {args.model_path} ...", flush=True)
    bundle = MLDBundle.from_pretrained(
        args.model_path,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )
    pipe = MLDPipeline(bundle, device=args.device)

    written = skipped = failed = 0
    bs = args.batch_size
    for start in range(0, len(items), bs):
        chunk = items[start : start + bs]
        todo = []
        for name, caption, gt_len in chunk:
            if args.skip_existing and (out_dir / f"{name}.npy").exists():
                skipped += 1
                continue
            todo.append((name, caption, pipe.clamp_length(gt_len)))
        if not todo:
            continue
        names_b = [t[0] for t in todo]
        captions = [t[1] for t in todo]
        lengths = [t[2] for t in todo]
        try:
            motions = pipe.infer_t2m(
                captions,
                lengths,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )
            for name, m in zip(names_b, motions):
                np.save(out_dir / f"{name}.npy", m.astype(np.float32))
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch {start}: {type(exc).__name__}: {exc}", flush=True)
        if (start // bs + 1) % 5 == 0:
            print(f"[progress] seen={min(start + bs, len(items))}/{len(items)} "
                  f"written={written} skipped={skipped} failed={failed}", flush=True)

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

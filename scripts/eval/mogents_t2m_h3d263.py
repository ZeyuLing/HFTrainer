#!/usr/bin/env python3
"""Generate MoGenTS HumanML3D-263 T2M outputs under the official protocol."""
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

from hftrainer.models.motion.mogents import MoGenTSBundle
from hftrainer.pipelines.mogents import MoGenTSPipeline

DEFAULT_DATA_ROOT = REPO / "ref_repo/CondMDI/dataset/HumanML3D"
DEFAULT_MODEL = REPO / "checkpoints/mogents/humanml3d"


def first_caption(text_file: Path):
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if parts and parts[0].strip():
            return parts[0].strip()
    return None


def first_hier_caption(caption_file: Path):
    data = json.loads(caption_file.read_text())
    for key in ("macro", "meso", "micro"):
        vals = data.get(key) or []
        for val in vals:
            cap = str(val).strip()
            if cap:
                return cap
    return None


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


def resolve_anno_items(anno_file: Path, names, output_fps: float = 20.0):
    payload = json.loads(anno_file.read_text())
    anno = payload.get("data_list", payload)
    out = {}
    for name in names:
        entry = anno.get(name)
        if not entry:
            continue
        cap_path = Path(entry.get("hierarchical_caption_path", ""))
        if not cap_path.is_absolute():
            cap_path = REPO / cap_path
        if not cap_path.exists():
            continue
        cap = first_hier_caption(cap_path)
        if not cap:
            continue
        fps = float(entry.get("fps") or 30.0)
        num_frames = int(entry.get("num_frames") or 0)
        if fps <= 0 or num_frames <= 0:
            continue
        hml_len = max(1, int(round(num_frames * output_fps / fps)))
        out[name] = (name, cap, hml_len)
    return out


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
    p.add_argument("--model_path", default=str(DEFAULT_MODEL),
                   help="hftrainer MoGenTS artifact dir OR raw upstream weights root")
    p.add_argument("--length_root", default="checkpoints",
                   help="raw-checkpoint mode only: root containing length_estimator")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--id_file", default=None,
                   help="optional one-id-per-line subset; defaults to data_root/test.txt")
    p.add_argument("--anno_file", default=None,
                   help="optional official 272 annotation fallback for ids missing "
                        "from CondMDI HumanML3D files")
    p.add_argument("--anno_output_fps", type=float, default=20.0,
                   help="output fps used when converting official annotation length "
                        "to MoGenTS/HML263 length")
    p.add_argument("--cond_scale", type=float, default=4.0)
    p.add_argument("--time_steps", type=int, default=10)
    p.add_argument("--topkr", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--res_cond_scale", type=float, default=5.0)
    p.add_argument("--gumbel_sample", action="store_true")
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

    data_root = Path(args.data_root)
    t0 = time.time()
    list_file = Path(args.id_file) if args.id_file else (data_root / "test.txt")
    names = [n.strip() for n in list_file.read_text().splitlines() if n.strip()]
    if args.num_shards > 1:
        names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard_index]
    cand = names if not args.max_samples else names[: args.max_samples * 3]
    items_by_name = {}
    for item in resolve_items(data_root, cand):
        items_by_name[item[0]] = item
    if args.anno_file:
        anno_items = resolve_anno_items(
            Path(args.anno_file), cand, output_fps=args.anno_output_fps)
        for name, item in anno_items.items():
            items_by_name.setdefault(name, item)
    items = [items_by_name[n] for n in cand if n in items_by_name]
    if args.max_samples:
        items = items[: args.max_samples]
    print(f"[setup] shard={args.shard_index}/{args.num_shards} items={len(items)} "
          f"out={out_dir} (resolve {time.time() - t0:.1f}s)", flush=True)

    mp = Path(args.model_path)
    print(f"[setup] building MoGenTS bundle/pipeline from {mp} ...", flush=True)
    if (mp / "mogents_config.json").exists():
        bundle = MoGenTSBundle.from_pretrained(str(mp), load_length_estimator=False)
    else:
        bundle = MoGenTSBundle(
            weights_root=str(mp),
            length_root=args.length_root,
            load_length_estimator=False,
        )
    pipe = MoGenTSPipeline(bundle, device=args.device)

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
                cond_scale=args.cond_scale,
                time_steps=args.time_steps,
                topkr=args.topkr,
                temperature=args.temperature,
                res_cond_scale=args.res_cond_scale,
                gumbel_sample=args.gumbel_sample,
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

#!/usr/bin/env python3
"""Generate MDM HumanML3D-263 T2M outputs under the *official* HumanML3D protocol.

Unlike ``mdm_t2m_humanml272.py`` (which drives generation from the 30 fps
MotionStreamer-272 test subset), this script drives from the **standard
HumanML3D test split** (263-dim @ 20 fps native, the distribution MDM's paper
metrics are computed on). For each test id we read the GT length (20 fps, no
resampling) and the primary caption, generate one motion with the
hftrainer-native vendored ``MDMPipeline``, and save the un-standardized
HumanML3D-263 features keyed by id.

The saved 263 files are scored with ``HumanML263Evaluator`` (caption='first',
i.e. gen-caption == retrieval-caption) against the same split, reproducing the
MDM paper HumanML3D row (FID / R-Precision / MM-Dist / Diversity).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.mdm import MDMBundle
from hftrainer.pipelines.mdm import MDMPipeline

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "ref_repo/CondMDI/dataset/HumanML3D"
DEFAULT_MODEL = REPO / "checkpoints/mdm/humanml_trans_enc_512"


def first_caption(text_file: Path):
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if parts and parts[0].strip():
            return parts[0].strip()
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
                   help="hftrainer MDM artifact dir OR raw .pt")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--guidance_param", type=float, default=2.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    t0 = time.time()
    names = [n.strip() for n in (data_root / "test.txt").read_text().splitlines() if n.strip()]
    if args.num_shards > 1:
        names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard_index]
    cand = names if not args.max_samples else names[: args.max_samples * 3]
    items = resolve_items(data_root, cand)
    if args.max_samples:
        items = items[: args.max_samples]
    print(f"[setup] shard={args.shard_index}/{args.num_shards} items={len(items)} "
          f"out={out_dir} (resolve {time.time() - t0:.1f}s)", flush=True)

    # Load the self-contained hftrainer artifact (or fall back to a raw .pt).
    mp = Path(args.model_path)
    print(f"[setup] building MDM bundle/pipeline from {mp} ...", flush=True)
    if (mp / "mdm_config.json").exists():
        bundle = MDMBundle.from_pretrained(str(mp), guidance_param=args.guidance_param)
    else:
        bundle = MDMBundle(model_path=str(mp), guidance_param=args.guidance_param)
    pipe = MDMPipeline(bundle, device=args.device)

    written = skipped = failed = 0
    bs = args.batch_size
    for start in range(0, len(items), bs):
        chunk = items[start : start + bs]
        todo = []
        for name, caption, gt_len in chunk:
            if args.skip_existing and (out_dir / f"{name}.npy").exists():
                skipped += 1
                continue
            todo.append((name, caption, pipe.clamp_length(gt_len)))  # 20 fps native
        if not todo:
            continue
        names_b = [t[0] for t in todo]
        captions = [t[1] for t in todo]
        lengths = [t[2] for t in todo]
        try:
            motions = pipe.infer_t2m(captions, lengths, progress=args.progress)
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

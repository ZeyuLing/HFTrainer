#!/usr/bin/env python3
"""Generate T2M-GPT HumanML3D-263 T2M outputs under the *official* protocol.

Mirrors ``scripts/eval/mdm_t2m_h3d263.py`` but for the vendored T2M-GPT
reproduction. Drives generation from the **standard HumanML3D test split**
(263-dim @ 20 fps native): for each test id we read the primary caption,
generate one motion with the hftrainer-native ``T2MGPTPipeline`` and save the
un-standardized HumanML3D-263 features keyed by id.

Unlike a diffusion model, T2M-GPT determines the sequence length itself via the
GPT EOS token, so we do **not** feed a target length by default (matching the
gold-standard ``t2mgpt_infer_hml3d263.py``). Pass ``--truncate_to_gt`` to clip
each prediction to the GT length instead.

The saved 263 files are scored with ``HumanML263Evaluator`` (caption='first')
against the same split.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.t2mgpt import T2MGPTBundle
from hftrainer.pipelines.t2mgpt import T2MGPTPipeline

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "ref_repo/CondMDI/dataset/HumanML3D"
DEFAULT_MODEL = REPO / "checkpoints/t2mgpt/humanml3d"


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
                   help="hftrainer T2M-GPT artifact dir OR raw checkpoint dir")
    p.add_argument("--vq_path", default=None, help="raw VQ-VAE .pth (overrides model_path)")
    p.add_argument("--gpt_path", default=None, help="raw GPT .pth (overrides model_path)")
    p.add_argument("--mean_path", default=None)
    p.add_argument("--std_path", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--truncate_to_gt", action="store_true",
                   help="clip each prediction to the GT length (default: model length)")
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

    # Load the self-contained hftrainer artifact (or fall back to raw .pth).
    mp = Path(args.model_path)
    print(f"[setup] building T2M-GPT bundle/pipeline ...", flush=True)
    if args.vq_path or args.gpt_path:
        bundle = T2MGPTBundle(
            vq_path=args.vq_path, gpt_path=args.gpt_path,
            mean_path=args.mean_path, std_path=args.std_path,
        )
    elif (mp / "t2mgpt_config.json").exists():
        bundle = T2MGPTBundle.from_pretrained(str(mp))
    else:
        bundle = T2MGPTBundle(mean_path=args.mean_path, std_path=args.std_path)
    pipe = T2MGPTPipeline(bundle, device=args.device)

    written = skipped = failed = 0
    bs = args.batch_size
    for start in range(0, len(items), bs):
        chunk = items[start: start + bs]
        todo = []
        for name, caption, gt_len in chunk:
            if args.skip_existing and (out_dir / f"{name}.npy").exists():
                skipped += 1
                continue
            todo.append((name, caption, gt_len))
        if not todo:
            continue
        names_b = [t[0] for t in todo]
        captions = [t[1] for t in todo]
        lengths = [t[2] for t in todo] if args.truncate_to_gt else None
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

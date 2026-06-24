#!/usr/bin/env python3
"""Generate MDM HumanML3D-263 T2M outputs under the MotionStreamer-272 protocol.

For each test id in the humanml3d_272 split we read the primary caption and the
GT motion length, generate one motion with the hftrainer-native MDMPipeline, and
save the un-standardized HumanML3D-263 features (MDM native 20 fps) keyed by id.

The saved 263 files feed the shared retarget/encode pipeline:
  Stage A  scripts/eval/hml263_to_smpl_ik.py        (263 -> SMPL motion_135, 20->30 fps)
  Stage B  scripts/data/convert_motion135_to_h3d272.py (135 -> 272)
  Stage C  ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from hftrainer.models.motion.mdm import MDMBundle
from hftrainer.pipelines.mdm import MDMPipeline

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"
DEFAULT_MODEL = REPO / "ref_repo/MDM/save/humanml_enc_512_50steps/model000750000.pt"

GT_FPS = 30.0
MDM_FPS = 20.0
MIN_MOTION_30 = 60
MAX_MOTION_30 = 300


def first_caption(text_file: Path):
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if not parts or not parts[0].strip():
            continue
        return parts[0].strip()
    return None


def split_names(data_root: Path):
    split = (data_root / "split" / "test.txt").read_text().splitlines()
    return [n.strip() for n in split if n.strip()]


def _resolve_one(args):
    name, motion_dir, text_dir = args
    m_file = motion_dir / f"{name}.npy"
    t_file = text_dir / f"{name}.txt"
    try:
        gt_len = int(np.load(m_file, mmap_mode="r").shape[0])
    except (FileNotFoundError, OSError, ValueError):
        return None
    if gt_len < MIN_MOTION_30 or gt_len >= MAX_MOTION_30:
        return None
    try:
        caption = first_caption(t_file)
    except (FileNotFoundError, OSError):
        return None
    if not caption:
        return None
    return (name, caption, gt_len)


def resolve_items(data_root: Path, names, workers: int = 32):
    """Read GT length + primary caption for candidate names (parallel I/O).

    The 272 test set lives on shared storage with high per-file latency, so we
    fan out the metadata reads across a thread pool (file I/O releases the GIL).
    """
    from concurrent.futures import ThreadPoolExecutor

    motion_dir = data_root / "motion_data"
    text_dir = data_root / "texts"
    tasks = [(n, motion_dir, text_dir) for n in names]
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_resolve_one, tasks):
            if res is not None:
                out[res[0]] = res
    # Preserve split order for deterministic sharding/batching.
    return [out[n] for n in names if n in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--guidance_param", type=float, default=2.5)
    p.add_argument("--batch_size", type=int, default=32)
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

    import time

    t0 = time.time()
    names = split_names(Path(args.data_root))
    if args.num_shards > 1:
        names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard_index]
    # When limiting samples, only resolve a small candidate window to avoid
    # statting all test files on slow shared storage.
    cand = names if not args.max_samples else names[: args.max_samples * 3]
    items = resolve_items(Path(args.data_root), cand)
    if args.max_samples:
        items = items[: args.max_samples]
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} items={len(items)} "
        f"out={out_dir} (resolve {time.time() - t0:.1f}s)",
        flush=True,
    )

    print("[setup] building MDM bundle/pipeline ...", flush=True)
    bundle = MDMBundle(model_path=args.model_path, guidance_param=args.guidance_param)
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
            mdm_len = pipe.clamp_length(round(gt_len * MDM_FPS / GT_FPS))
            todo.append((name, caption, mdm_len))
        if not todo:
            continue

        names = [t[0] for t in todo]
        captions = [t[1] for t in todo]
        lengths = [t[2] for t in todo]
        try:
            motions = pipe.infer_t2m(captions, lengths, progress=args.progress)
            for name, m in zip(names, motions):
                np.save(out_dir / f"{name}.npy", m.astype(np.float32))
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch {start}: {type(exc).__name__}: {exc}", flush=True)

        if (start // bs + 1) % 5 == 0:
            print(
                f"[progress] seen={min(start + bs, len(items))}/{len(items)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

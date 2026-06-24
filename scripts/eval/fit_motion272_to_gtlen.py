#!/usr/bin/env python3
"""Trim or last-frame-pad raw MotionStreamer-272 clips to official GT lengths."""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np


def fit_motion_length(motion: np.ndarray, target_len: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    target_len = int(target_len)
    if motion.shape[0] == target_len:
        return motion
    if motion.shape[0] > target_len:
        return motion[:target_len]
    if motion.shape[0] <= 0:
        return motion
    pad = np.repeat(motion[-1:], target_len - motion.shape[0], axis=0)
    return np.concatenate([motion, pad], axis=0).astype(np.float32)


def load_motion272(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            return np.asarray(z["motion_272"], dtype=np.float32)
    return np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)


def worker(task):
    src, dst, gt_path, output_format = task
    try:
        if dst.exists():
            return "skip"
        motion = load_motion272(src)
        target_len = int(np.load(gt_path, mmap_mode="r").shape[0])
        motion = fit_motion_length(motion, target_len)
        if output_format == "npz":
            np.savez_compressed(dst, motion_272=motion)
        else:
            np.save(dst, motion)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{src}:{type(exc).__name__}:{exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output-format", choices=["npy", "npz"], default="npy")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    gt_dir = Path(args.gt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src in sorted(list(src_dir.glob("*.npy")) + list(src_dir.glob("*.npz"))):
        gt_path = gt_dir / f"{src.stem}.npy"
        if not gt_path.exists():
            continue
        suffix = ".npz" if args.output_format == "npz" else ".npy"
        tasks.append((src, out_dir / f"{src.stem}{suffix}", gt_path, args.output_format))
    print(f"[fit272] tasks={len(tasks)} src={src_dir} out={out_dir}", flush=True)

    ok = skip = fail = 0
    with mp.Pool(max(1, args.workers)) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, tasks, chunksize=16), 1):
            if res == "ok":
                ok += 1
            elif res == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 10:
                    print(res, flush=True)
            if i % 1000 == 0:
                print(f"[fit272] {i}/{len(tasks)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"[fit272] DONE ok={ok} skip={skip} fail={fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trim or last-frame-pad evaluator-ready npz clips to official GT lengths."""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np


PREFERRED_KEYS = ("motion_135", "motion_272", "transl", "global_orient", "body_pose")


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


def worker(task):
    src, dst, gt_path, overwrite = task
    try:
        if dst.exists() and not overwrite:
            return "skip"
        with np.load(src, allow_pickle=True) as z:
            keys = list(z.files)
            main_key = next((key for key in PREFERRED_KEYS if key in z.files), keys[0] if keys else None)
            if main_key is None:
                return f"fail:{src}:empty npz"
            target_len = int(np.load(gt_path, mmap_mode="r").shape[0])
            payload = {}
            for key in keys:
                arr = np.asarray(z[key])
                if arr.ndim >= 1 and arr.shape[0] == np.asarray(z[main_key]).shape[0]:
                    payload[key] = fit_motion_length(arr, target_len)
                else:
                    payload[key] = arr
        np.savez_compressed(dst, **payload)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{src}:{type(exc).__name__}:{exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    gt_dir = Path(args.gt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src in sorted(src_dir.glob("*.npz")):
        gt_path = gt_dir / f"{src.stem}.npy"
        if gt_path.exists():
            tasks.append((src, out_dir / src.name, gt_path, args.overwrite))
    print(f"[fit-eval-npz] tasks={len(tasks)} src={src_dir} out={out_dir}", flush=True)

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
                print(f"[fit-eval-npz] {i}/{len(tasks)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"[fit-eval-npz] DONE ok={ok} skip={skip} fail={fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

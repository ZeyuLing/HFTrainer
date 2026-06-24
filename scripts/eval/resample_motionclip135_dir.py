#!/usr/bin/env python3
"""Resample a directory of MotionCLIP 135D motion arrays between frame rates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def resample_motion(motion: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"fps must be positive, got src={src_fps}, dst={dst_fps}")
    motion = np.asarray(motion, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-6 or len(motion) < 2:
        return motion
    out_len = max(1, int(round(len(motion) * dst_fps / src_fps)))
    src_t = np.linspace(0.0, 1.0, len(motion), dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    out = np.empty((out_len, motion.shape[1]), dtype=np.float32)
    for dim in range(motion.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, motion[:, dim])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--src-fps", type=float, default=20.0)
    parser.add_argument("--dst-fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy files under {src_dir}")

    ok = failed = skipped = 0
    for idx, path in enumerate(files, 1):
        out_path = out_dir / path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            motion = np.load(str(path)).astype(np.float32)
            if motion.ndim != 2 or motion.shape[-1] != 135:
                raise ValueError(f"expected (T,135), got {motion.shape}")
            np.save(str(out_path), resample_motion(motion, args.src_fps, args.dst_fps))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 10:
                print(f"[fail] {path}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 500 == 0 or idx == len(files):
            print(f"[progress] {idx}/{len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)
    print(f"[done] src={src_dir} out={out_dir} files={len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

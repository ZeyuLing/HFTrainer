#!/usr/bin/env python3
"""Repack ``motion_135`` rotation-6D convention between row and column layouts."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.motion.representation.rotation import repack_6d  # noqa: E402


def repack_one(src: Path, out_dir: Path, src_conv: str, dst_conv: str, skip_existing: bool) -> str:
    dst = out_dir / src.name
    if skip_existing and dst.exists():
        return "skip"
    d = np.load(src, allow_pickle=True)
    if "motion_135" not in d.files:
        return "no_motion_135"
    payload = {k: d[k] for k in d.files}
    m = np.asarray(d["motion_135"], dtype=np.float32)
    if m.ndim != 2 or m.shape[-1] != 135:
        return "bad_shape"
    rot6d = m[:, 3:].reshape(m.shape[0], 22, 6)
    repacked = repack_6d(rot6d, src=src_conv, dst=dst_conv).reshape(m.shape[0], 132)
    payload["motion_135"] = np.concatenate([m[:, :3], repacked], axis=1).astype(np.float32)
    payload["rot6d_convention"] = np.array(dst_conv)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **payload)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--src", choices=["row", "column"], required=True)
    ap.add_argument("--dst", choices=["row", "column"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    files = sorted(in_dir.glob("*.npz"))
    print(f"[start] repack {len(files)} files {args.src}->{args.dst} -> {out_dir}", flush=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        iterator = ex.map(
            lambda p: repack_one(p, out_dir, args.src, args.dst, args.skip_existing),
            files,
        )
        for i, status in enumerate(iterator, 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0 or i == len(files):
                print(f"[progress] {i}/{len(files)} {counts}", flush=True)
    print(f"[done] {counts}", flush=True)


if __name__ == "__main__":
    main()

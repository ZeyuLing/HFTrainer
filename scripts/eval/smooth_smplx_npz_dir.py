#!/usr/bin/env python3
"""Apply HY-Motion-style temporal smoothing to a directory of SMPLX npz files."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.motion.processing.temporal_smoothing import smooth_smplx_dict_hymotion


def _smooth_one(args: Tuple[str, str, bool]) -> Tuple[str, str]:
    src_path, dst_path, skip_existing = args
    if skip_existing and os.path.isfile(dst_path):
        return src_path, "skip"

    with np.load(src_path, allow_pickle=True) as data:
        smplx_dict = {k: data[k] for k in data.files}

    smoothed = smooth_smplx_dict_hymotion(smplx_dict)
    tmp_path = dst_path + ".tmp"
    np.savez_compressed(tmp_path, **smoothed)
    if not tmp_path.endswith(".npz"):
        tmp_path += ".npz"
    os.replace(tmp_path, dst_path)
    return src_path, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smooth generated SMPLX npz files without rerunning generation."
    )
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.glob("*.npz") if p.is_file())
    tasks = [
        (str(p), str(out_dir / p.name), bool(args.skip_existing))
        for p in files
    ]

    ok = skip = fail = 0
    if args.workers <= 1:
        iterator = map(_smooth_one, tasks)
        for src, status in iterator:
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[fail] {src}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_smooth_one, t) for t in tasks]
            for fut in as_completed(futs):
                try:
                    src, status = fut.result()
                except Exception as exc:
                    fail += 1
                    print(f"[fail] {exc}")
                    continue
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    print(f"[fail] {src}: {status}")

    print(f"DONE files={len(files)} ok={ok} skip={skip} fail={fail} out={out_dir}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

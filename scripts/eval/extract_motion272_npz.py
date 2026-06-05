#!/usr/bin/env python3
"""Extract ``motion_272`` arrays from npz files to evaluator-ready npy files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    src = Path(args.in_dir)
    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(src.glob("*.npz")):
        data = np.load(str(path))
        np.save(str(dst / f"{path.stem}.npy"), data["motion_272"].astype(np.float32))
        count += 1
    print(f"[extract done] pred272={count}", flush=True)


if __name__ == "__main__":
    main()

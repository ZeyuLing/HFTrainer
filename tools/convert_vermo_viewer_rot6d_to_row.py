#!/usr/bin/env python3
"""Convert existing VerMo viewer motion_135 NPZ artifacts to row-major rot6d."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np


def read_scalar(value, default: str = "") -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        return str(arr.tolist())
    except Exception:
        return default


def motion135_column_to_row(motion135: np.ndarray) -> np.ndarray:
    motion135 = np.asarray(motion135, dtype=np.float32).copy()
    if motion135.shape[-1] < 135:
        raise ValueError(f"Expected motion_135 last dim >=135, got {motion135.shape}")
    rot = motion135[..., 3:135].reshape(*motion135.shape[:-1], 22, 6)
    motion135[..., 3:135] = rot[..., [0, 3, 1, 4, 2, 5]].reshape(*motion135.shape[:-1], 132)
    return motion135


def convert_file(path: str) -> bool:
    loaded = np.load(path, allow_pickle=True)
    if "motion_135" not in loaded.files:
        return False
    data: Dict[str, np.ndarray] = {key: loaded[key] for key in loaded.files}
    convention = read_scalar(data.get("rot6d_convention"), default="")
    if convention == "row":
        return False
    data["motion_135"] = motion135_column_to_row(data["motion_135"])
    data["rot6d_convention"] = np.asarray("row")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    converted = 0
    seen = 0
    for root in args.roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".npz"):
                    continue
                path = os.path.join(dirpath, filename)
                seen += 1
                if convert_file(path):
                    converted += 1
                    if converted % 100 == 0:
                        print(f"[rot6d] converted {converted}/{seen}", flush=True)
    print(f"[rot6d] converted={converted} scanned_npz={seen}", flush=True)


if __name__ == "__main__":
    main()

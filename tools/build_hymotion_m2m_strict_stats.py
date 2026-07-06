#!/usr/bin/env python3
"""Build HYMotion M2M strict-198 stats from official HYMotion 201 stats."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-stats-dir",
        default="checkpoints/HY-Motion-1.0/stats",
        help="Directory containing official 201-dim Mean.npy and Std.npy.",
    )
    parser.add_argument(
        "--out-201-dir",
        default="data/hymotion_m2m_data/_stats_201dim",
        help="Output directory for strict 201-dim stats.",
    )
    parser.add_argument(
        "--out-198-dir",
        default="data/hymotion_m2m_data/_stats_198dim",
        help="Output directory for strict 198-dim stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.official_stats_dir)
    out201 = Path(args.out_201_dir)
    out198 = Path(args.out_198_dir)

    mean201 = np.load(src / "Mean.npy").astype(np.float32)
    std201 = np.load(src / "Std.npy").astype(np.float32)
    if mean201.shape != (201,) or std201.shape != (201,):
        raise ValueError(f"Expected official stats shape (201,), got {mean201.shape}, {std201.shape}")
    if np.max(np.abs(mean201[135:138])) > 1e-8 or np.max(np.abs(std201[135:138])) > 1e-8:
        raise ValueError("Official pelvis RIC stats [135:138] must be exactly zero.")

    idx = np.r_[0:135, 138:201]
    out201.mkdir(parents=True, exist_ok=True)
    out198.mkdir(parents=True, exist_ok=True)
    np.save(out201 / "Mean.npy", mean201)
    np.save(out201 / "Std.npy", std201)
    np.save(out198 / "Mean.npy", mean201[idx].astype(np.float32))
    np.save(out198 / "Std.npy", std201[idx].astype(np.float32))

    print(f"wrote {out201} and {out198}")


if __name__ == "__main__":
    main()

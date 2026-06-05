#!/usr/bin/env python3
"""Visualize HML263 decoded joints against SMPL IK retargeted joints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.visualize_t2m_baseline_outputs import (  # noqa: E402
    BONE_PAIRS,
    _draw_skeleton,
    _set_axes,
    read_first_caption,
)


def render_strip(target: np.ndarray, fitted: np.ndarray, mpjpe: np.ndarray,
                 out_path: Path, caption: str, sid: str):
    rows = [
        ("HML263 decoded target", target, "#2ca25f"),
        ("SMPL IK fitted", fitted, "#756bb1"),
    ]
    n_cols = 6
    fig = plt.figure(figsize=(2.65 * n_cols, 2.2 * len(rows)))
    for r, (label, pts, color) in enumerate(rows):
        for c, frac in enumerate(np.linspace(0.0, 1.0, n_cols)):
            t = int(round(frac * (len(pts) - 1)))
            ax = fig.add_subplot(len(rows), n_cols, r * n_cols + c + 1, projection="3d")
            _draw_skeleton(ax, pts[t], color)
            suffix = f"  {mpjpe[t]:.0f}mm" if r == 1 and len(mpjpe) == len(pts) else ""
            _set_axes(ax, pts, f"{label}  f{t}/{len(pts) - 1}{suffix}")
    fig.suptitle(f"{sid}  {caption[:130]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_trajectory(target: np.ndarray, fitted: np.ndarray, out_path: Path, sid: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for label, pts, color in [
        ("HML263 target root", target[:, 0, [0, 2]], "#2ca25f"),
        ("SMPL fitted root", fitted[:, 0, [0, 2]], "#756bb1"),
    ]:
        ax.plot(pts[:, 0], pts[:, 1], label=label, color=color, lw=2.2)
        ax.scatter([pts[0, 0]], [pts[0, 1]], color=color, s=20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{sid} HML263 -> SMPL root XZ")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["motionlab", "flowmdm"], required=True)
    parser.add_argument("--ids", default="000021,000612,001003")
    parser.add_argument("--smpl-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/evaluation/visual_diagnostics/t2m_baselines")
    args = parser.parse_args()

    smpl_dir = PROJECT_ROOT / args.smpl_dir
    text_dir = (
        PROJECT_ROOT / "ref_repo" / "MotionStreamer" / "MotionStreamer" /
        "humanml3d_272" / "texts"
    )
    out_dir = PROJECT_ROOT / args.out_dir / f"{args.method}_smpl_retarget"

    for sid in [x.strip() for x in args.ids.split(",") if x.strip()]:
        npz_path = smpl_dir / f"{sid}.npz"
        if not npz_path.exists():
            print(f"[skip] {sid}: missing {npz_path}")
            continue
        data = np.load(npz_path)
        target = np.asarray(data["target_joints"], dtype=np.float32)
        fitted = np.asarray(data["fitted_joints"], dtype=np.float32)
        mpjpe = np.asarray(data["fit_mpjpe_mm"], dtype=np.float32)
        caption = read_first_caption(text_dir / f"{sid}.txt")

        strip = out_dir / f"{args.method}_{sid}_hml263_vs_smpl_strip.png"
        traj = out_dir / f"{args.method}_{sid}_hml263_vs_smpl_trajectory.png"
        render_strip(target, fitted, mpjpe, strip, caption, sid)
        render_trajectory(target, fitted, traj, sid)
        print(f"[strip] {strip}")
        print(f"[traj]  {traj}")


if __name__ == "__main__":
    main()

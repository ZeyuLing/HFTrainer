#!/usr/bin/env python3
"""Visualize Minimal In-Betweening (E2 both_1f) eval results.

For each eval NPZ (saved by ``eval_m2m_v2_all_tasks.py --save-npz``), renders a
side-by-side comparison of GT vs predicted 22-joint SMPL skeleton. The two
preserved/condition frames (first + last, derived from ``src_mask``) are drawn
in green; the generated in-between frames in blue. GT is always gray.

Outputs (per sample):
  - ``<id>_compare.gif``  : animated GT | Pred
  - ``<id>_strip.png``    : static keyframe strip (GT row / Pred row)

Usage:
  python scripts/analysis/visualize_mib_results.py \
      --npz-dir output/evaluation/mib_h3d_full/kimodo_caption_editfix_ep240/cfg20/rep0/kimodo_caption_editfix_ep240/E2_both_1f/npz \
      --out-dir output/viz/mib_kimodo_cfg20 --samples 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# SMPL 22-joint bone connectivity (matches phys_metrics BODY_BONE_PAIRS).
BONE_PAIRS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
    (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]
BONE_OFFSETS_PATH = "data/hymotion_m2m_data/bone_offsets_22.pt"


def fk_positions(motion_135: np.ndarray, bone_offsets: np.ndarray) -> np.ndarray:
    from hftrainer.evaluation.motion.m2m_eval_metrics import (
        motion135_to_positions_np,
    )
    return motion135_to_positions_np(motion_135, bone_offsets)


def _draw_skeleton(ax, joints, color, lw=2.0, alpha=1.0, marker_size=8):
    # plot mapping: horizontal = X (0), depth = Z (2), vertical = Y (1, up)
    x, z, y = joints[:, 0], joints[:, 2], joints[:, 1]
    for a, b in BONE_PAIRS:
        ax.plot([x[a], x[b]], [z[a], z[b]], [y[a], y[b]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(x, z, y, color=color, s=marker_size, alpha=alpha)


def _set_axes(ax, all_pts, title):
    cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
    cz = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
    rng = max(all_pts[:, 0].ptp(), all_pts[:, 2].ptp(),
              all_pts[:, 1].ptp(), 1.0) * 0.55
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cz - rng, cz + rng)
    ax.set_zlim(all_pts[:, 1].min() - 0.05, all_pts[:, 1].min() + 2 * rng)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=12, azim=-70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _preserve_frames(src_mask: np.ndarray, T: int) -> np.ndarray:
    fm = src_mask.max(axis=-1) > 0.5   # True=generated
    if fm.shape[0] != T:
        fm = fm[:T] if fm.shape[0] > T else np.pad(fm, (0, T - fm.shape[0]))
    return ~fm                          # True=preserved/condition


def render_gif(pred, gt, preserved, out_path, caption="", fps=20):
    T = pred.shape[0]
    allp = np.concatenate([pred.reshape(-1, 3), gt.reshape(-1, 3)], 0)
    frames = []
    step = max(1, T // 120)             # cap ~120 rendered frames
    for t in range(0, T, step):
        fig = plt.figure(figsize=(8, 4.2))
        is_pres = bool(preserved[t])
        for i, (pos, name, base) in enumerate(
                [(gt, "GT", "0.45"), (pred, "Pred", None)]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            if base is not None:
                _draw_skeleton(ax, pos[t], base)
            else:
                col = "#2ca02c" if is_pres else "#1f77b4"
                _draw_skeleton(ax, pos[t], col, lw=2.5)
            tag = "  [PRESERVED]" if (is_pres and i == 1) else ""
            _set_axes(ax, allp, f"{name}{tag}")
        fig.suptitle(f"frame {t}/{T-1}   {caption[:70]}", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(buf.copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=min(fps, 20), loop=0)


def render_strip(pred, gt, preserved, out_path, caption="", n=6):
    T = pred.shape[0]
    idx = np.linspace(0, T - 1, n).round().astype(int)
    allp = np.concatenate([pred.reshape(-1, 3), gt.reshape(-1, 3)], 0)
    fig = plt.figure(figsize=(3.0 * n, 6.2))
    for c, t in enumerate(idx):
        is_pres = bool(preserved[t])
        ax = fig.add_subplot(2, n, c + 1, projection="3d")
        _draw_skeleton(ax, gt[t], "0.45")
        _set_axes(ax, allp, f"GT  f{t}")
        ax2 = fig.add_subplot(2, n, n + c + 1, projection="3d")
        col = "#2ca02c" if is_pres else "#1f77b4"
        _draw_skeleton(ax2, pred[t], col, lw=2.5)
        _set_axes(ax2, allp, f"Pred f{t}" + ("  [PRES]" if is_pres else ""))
    fig.suptitle(f"MIB GT(top) vs Pred(bottom)   {caption[:90]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None, help="single NPZ file")
    ap.add_argument("--npz-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--bone-offsets", type=str, default=BONE_OFFSETS_PATH)
    ap.add_argument("--mode", choices=["gif", "strip", "both"], default="both")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import torch
    bo = torch.load(args.bone_offsets, map_location="cpu").float().numpy()

    if args.npz:
        files = [Path(args.npz)]
    else:
        files = sorted(Path(args.npz_dir).glob("*.npz"))[: args.samples]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        d = np.load(f, allow_pickle=True)
        pred = d["positions"] if "positions" in d.files else \
            fk_positions(d["motion_135"], bo)
        gt = fk_positions(d["gt_motion_135"], bo)
        T = min(pred.shape[0], gt.shape[0])
        pred, gt = pred[:T], gt[:T]
        preserved = _preserve_frames(d["src_mask"], T)
        cap = str(d["caption"]) if "caption" in d.files else ""
        stem = f.stem
        if args.mode in ("strip", "both"):
            p = out_dir / f"{stem}_strip.png"
            render_strip(pred, gt, preserved, p, cap)
            print(f"[strip] {p}")
        if args.mode in ("gif", "both"):
            p = out_dir / f"{stem}_compare.gif"
            render_gif(pred, gt, preserved, p, cap, fps=args.fps)
            print(f"[gif]   {p}")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()

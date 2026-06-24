#!/usr/bin/env python3
"""Render MotionFix edit triplets as skeleton montages: Source / Edited(Ours) / Target(GT).

Each \ours eval NPZ embeds source_motion_135, motion_135 (edited), gt_motion_135 and
the edit instruction (caption). We FK all three (SMPL-22) to joint positions and draw
N evenly-sampled keyframes per row, with the instruction as the title.

Usage:
    .venv_t2m_a100/bin/python scripts/eval/_viz_motionfix_edit.py \
        --npz-dir <.../E16_style_edit/npz> --indices 00012,00003 \
        --out-dir outputs/visualization/motionfix_edit --n-frames 6
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not _REPO.exists():
    _REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
ROWS = [("source_motion_135", "Source (before edit)", "#1f77b4"),
        ("motion_135", "Edited — Ours", "#d62728"),
        ("gt_motion_135", "Target (GT)", "#2ca02c")]


def fk(motion_135, bone_offsets):
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
    return motion135_to_positions_np(motion_135.astype(np.float32), bone_offsets)  # (T,22,3)


def sample_idxs(T, n):
    return np.linspace(0, T - 1, n).round().astype(int)


def draw_row(ax, pos, n_frames, base_color, label, h_ax, floor_y, ceil_y, stride):
    """2D skeleton montage on one axes, using a SHARED scale across rows."""
    idxs = sample_idxs(pos.shape[0], n_frames)
    cmap = plt.cm.viridis
    for col, fi in enumerate(idxs):
        xoff = col * stride
        j = pos[fi]                      # (22,3)
        ph = j[:, h_ax] - j[0, h_ax]     # center pelvis horizontally at column
        xs = ph + xoff
        ys = j[:, 1] - floor_y           # height above shared floor
        for c in range(1, 22):
            par = SMPL22_PARENTS[c]
            ax.plot([xs[c], xs[par]], [ys[c], ys[par]],
                    color=base_color, lw=2.2, alpha=0.92, solid_capstyle="round")
        ax.scatter(xs, ys, c=[cmap(col / max(1, n_frames - 1))], s=14,
                   zorder=3, edgecolors="none")
        ax.text(xoff, -0.14 * ceil_y, f"t={fi}", fontsize=8, ha="center", color="#555")
    ax.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-stride * 0.7, stride * (n_frames - 0.3))
    ax.set_ylim(-0.22 * ceil_y, ceil_y * 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(label, fontsize=11, fontweight="bold", color=base_color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--indices", required=True, help="comma list of npz stems, e.g. 00012,00003")
    ap.add_argument("--out-dir", default="outputs/visualization/motionfix_edit")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--view", default="front", choices=["front", "side"])
    ap.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    args = ap.parse_args()

    import torch
    bo = torch.load(str(_REPO / args.bone_offsets), map_location="cpu").numpy()
    npz_dir = Path(args.npz_dir)
    out_dir = _REPO / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in args.indices.split(","):
        stem = stem.strip()
        f = npz_dir / f"{stem}.npz"
        if not f.exists():
            print(f"[skip] {f} missing"); continue
        d = np.load(f, allow_pickle=True)
        caption = str(d["caption"])
        h_ax = 0 if args.view == "front" else 2
        # FK all three rows, then derive a SHARED scale so the same body is the
        # same size in every row (robust to acrobatic/outlier frames via the
        # sampled keyframes only).
        row_pos = [fk(d[key], bo) for key, _, _ in ROWS]
        samp = np.concatenate([p[sample_idxs(p.shape[0], args.n_frames)]
                               for p in row_pos], axis=0)  # (3n,22,3)
        floor_y = float(samp[:, :, 1].min())
        ceil_y = float(samp[:, :, 1].max()) - floor_y
        body_w = float(np.percentile(
            samp[:, :, h_ax].max(1) - samp[:, :, h_ax].min(1), 90)) + 0.05
        stride = max(0.9, 1.4 * body_w)
        fig, axes = plt.subplots(3, 1, figsize=(1.7 * args.n_frames + 1.0, 6.6))
        fig.suptitle(f"MotionFix {stem}   |   instruction: \u201c{caption}\u201d",
                     fontsize=13, fontweight="bold", y=0.985)
        for ax, pos, (key, label, color) in zip(axes, row_pos, ROWS):
            draw_row(ax, pos, args.n_frames, color, label, h_ax, floor_y, ceil_y, stride)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.02, hspace=0.12)
        out = out_dir / f"motionfix_{stem}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[ok] {out}")


if __name__ == "__main__":
    main()

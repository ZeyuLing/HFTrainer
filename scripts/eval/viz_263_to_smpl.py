#!/usr/bin/env python3
"""Visualize the 263 -> SMPL IK conversion quality.

For each clip the IK npz stores:
  target_joints  : (T,22,3) the HumanML3D-263-decoded GT joints (the INPUT)
  fitted_joints  : (T,22,3) the SMPL-FK joints after IK (the OUTPUT used for 272)
We overlay them (gray = 263 input, colored = SMPL output) so the IK fidelity is
directly visible, and also render the SMPL output alone. A frame strip (PNG) and
an animation (GIF) are produced per clip.
"""
import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# HumanML3D / SMPL-22 kinematic tree
CHAINS = [[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
CCOL = ["#d62728","#1f77b4","#2ca02c","#ff7f0e","#9467bd"]


def _prep(j):
    """center on root xz, put feet on floor; return (T,22,3) in plot coords (x,z,y up)."""
    j = np.asarray(j, dtype=np.float32).copy()
    j[..., [1, 2]] = j[..., [2, 1]]  # swap so axis2 is up (HumanML3D y-up -> mpl z-up)
    j[:, :, 0] -= j[0, 0, 0]
    j[:, :, 1] -= j[0, 0, 1]
    j[:, :, 2] -= j[:, :, 2].min()
    return j


def _draw(ax, j_t, color_chain=True, alpha=1.0, lw=2.0, base="#888888"):
    for ci, chain in enumerate(CHAINS):
        c = CCOL[ci] if color_chain else base
        ax.plot(j_t[chain, 0], j_t[chain, 1], j_t[chain, 2],
                "-o", color=c, lw=lw, ms=2.0, alpha=alpha)


def _setlim(ax, j):
    r = 1.0
    cx, cy = 0.0, 0.0
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_zlim(0, 1.8)
    ax.set_box_aspect((1, 1, 0.9)); ax.view_init(elev=12, azim=-70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def strip(tgt, fit, mpjpe, sid, out_png, nframes=6):
    T = tgt.shape[0]
    idx = np.linspace(0, T - 1, nframes).astype(int)
    fig = plt.figure(figsize=(3.0 * nframes, 6.2))
    for k, fr in enumerate(idx):
        # top: overlay
        ax = fig.add_subplot(2, nframes, k + 1, projection="3d")
        _draw(ax, tgt[fr], color_chain=False, alpha=0.55, lw=3.2, base="#bbbbbb")
        _draw(ax, fit[fr], color_chain=True, alpha=1.0, lw=1.6)
        _setlim(ax, fit); ax.set_title(f"f{fr}", fontsize=8)
        if k == 0: ax.text2D(0.0, 1.08, "overlay: gray=263 input, color=SMPL out",
                             transform=ax.transAxes, fontsize=9)
        # bottom: SMPL output only
        ax2 = fig.add_subplot(2, nframes, nframes + k + 1, projection="3d")
        _draw(ax2, fit[fr], color_chain=True, alpha=1.0, lw=2.0)
        _setlim(ax2, fit)
        if k == 0: ax2.text2D(0.0, 1.08, "SMPL output (-> 272)",
                              transform=ax2.transAxes, fontsize=9)
    fig.suptitle(f"263->SMPL IK  id={sid}  fit MPJPE={mpjpe:.1f} mm  (T={T})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=90); plt.close(fig)


def gif(tgt, fit, mpjpe, sid, out_gif, fps=20, stride=2):
    frames = list(range(0, tgt.shape[0], stride))
    fig = plt.figure(figsize=(9, 4.6))
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axB = fig.add_subplot(1, 2, 2, projection="3d")

    def upd(fr):
        axA.cla(); axB.cla()
        _draw(axA, tgt[fr], color_chain=False, alpha=0.55, lw=3.2, base="#bbbbbb")
        _draw(axA, fit[fr], color_chain=True, alpha=1.0, lw=1.6)
        _setlim(axA, fit); axA.set_title("gray=263 input | color=SMPL out", fontsize=9)
        _draw(axB, fit[fr], color_chain=True, alpha=1.0, lw=2.0)
        _setlim(axB, fit); axB.set_title("SMPL output (-> 272)", fontsize=9)
        fig.suptitle(f"id={sid}  fit MPJPE={mpjpe:.1f} mm  frame {fr}", fontsize=11)
    ani = FuncAnimation(fig, upd, frames=frames, interval=1000 / fps)
    ani.save(out_gif, writer=PillowWriter(fps=fps)); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx-dir", default="output/evaluation/mib_ms272_ikfix/gtctrl_full/smplx")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out-dir", default="output/evaluation/mib_ms272_ikfix/_viz")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for sid in args.ids:
        p = os.path.join(args.smplx_dir, sid + ".npz")
        if not os.path.exists(p):
            print("MISS", p); continue
        z = np.load(p)
        tgt = _prep(z["target_joints"]); fit = _prep(z["fitted_joints"])
        m = float(np.asarray(z["fit_mpjpe_mm"]).mean())
        png = os.path.join(args.out_dir, f"{sid}_strip.png")
        strip(tgt, fit, m, sid, png); print("wrote", png)
        if args.gif:
            g = os.path.join(args.out_dir, f"{sid}.gif")
            gif(tgt, fit, m, sid, g); print("wrote", g)


if __name__ == "__main__":
    main()

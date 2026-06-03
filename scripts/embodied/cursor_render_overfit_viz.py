#!/usr/bin/env python3
"""Render reference-vs-tracked visualizations for the overfit run.

Loads a ProtoMotions ``predicted_motion_lib_epoch_*.pt`` (tracked rollout, field
``gts`` = global rigid-body positions [total_frames, 33, 3]) plus the reference
KIMODO ``.motion`` files it points to (``rigid_body_pos``), and produces, for a
handful of motions:
  1) a skeleton montage (reference gray + tracked color, overlaid, vs time),
  2) per-motion overlay GIFs,
  3) a root-xy trajectory plot (reference vs early-epoch vs late-epoch tracked)
     — the direct visual of global-translation reconstruction.

Pure torch.load + matplotlib (no IsaacGym needed); run with py3.10 (`python3`).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v2 as imageio

# g1 33-body order (from resolved_configs.yaml body_names)
BODY_NAMES = [
    "pelvis", "head",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "left_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_rubber_hand",
]
CHAINS = [
    ["pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
     "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link"],
    ["pelvis", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
     "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link"],
    ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link", "head"],
    ["torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
     "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
     "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand"],
    ["torso_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
     "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
     "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand"],
]
_IDX = {n: i for i, n in enumerate(BODY_NAMES)}
EDGES = [(_IDX[a], _IDX[b]) for ch in CHAINS for a, b in zip(ch[:-1], ch[1:])]
REF_COLOR = "#9aa0a6"   # gray = reference (KIMODO target)
TRK_COLOR = "#2ca02c"   # green = tracked (reconstruction)


def per_motion_frames(d):
    total = d["gts"].shape[0]
    ls = d["length_starts"].long().tolist()
    nm = len(ls)
    nf = [(ls[m + 1] - ls[m]) if m + 1 < nm else (total - ls[m]) for m in range(nm)]
    return ls, nf


def tracked_motion(d, m, ref_root_xy0=None):
    """Return tracked global body positions [T,33,3].

    IsaacGym lays parallel envs on an XY grid, so recorded ``gts`` carry a
    constant per-env XY origin offset. We remove it by anchoring the tracked
    pelvis at t=0 to the reference pelvis at t=0 (z is shared, untouched). The
    residual path is then real tracking drift (consistent with eval/gt_error)."""
    ls, nf = per_motion_frames(d)
    s, n = ls[m], nf[m]
    g = d["gts"][s:s + n].numpy().copy()  # [T,33,3]
    if ref_root_xy0 is not None and len(g):
        env_off_xy = g[0, 0, :2] - ref_root_xy0  # constant env-grid origin
        g[..., :2] -= env_off_xy
    return g


def ref_motion(path):
    r = torch.load(path, map_location="cpu", weights_only=False)
    return r["rigid_body_pos"].numpy(), int(r.get("fps", 30))


def draw_skel(ax, P, color, alpha=1.0, lw=2.0, s=6):
    for (i, j) in EDGES:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=color, s=s, alpha=alpha)


def set_axes(ax, center, span=0.9):
    cx, cy = center
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(0, 1.8)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def montage(ref, trk, title, out, n=6):
    T = min(len(ref), len(trk))
    ticks = np.linspace(0, T - 1, n).astype(int)
    fig = plt.figure(figsize=(2.6 * n, 3.0))
    for col, ti in enumerate(ticks):
        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        center = (ref[ti, 0, 0], ref[ti, 0, 1])
        draw_skel(ax, ref[ti], REF_COLOR, alpha=0.55, lw=1.5)
        draw_skel(ax, trk[ti], TRK_COLOR, alpha=1.0, lw=2.0)
        set_axes(ax, center)
        ax.set_title(f"t={ti}", fontsize=9)
    fig.suptitle(f"{title}   (gray=reference / green=tracked)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def gif(ref, trk, title, out, stride=2, fps_out=20):
    T = min(len(ref), len(trk))
    frames = []
    for ti in range(0, T, stride):
        fig = plt.figure(figsize=(4.6, 4.4))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        center = (ref[ti, 0, 0], ref[ti, 0, 1])
        draw_skel(ax, ref[ti], REF_COLOR, alpha=0.5, lw=1.5)
        draw_skel(ax, trk[ti], TRK_COLOR, alpha=1.0, lw=2.2)
        set_axes(ax, center)
        ax.set_title(f"{title}\nt={ti}  gray=ref / green=tracked", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out, frames, fps=fps_out, loop=0)
    print("saved", out, len(frames), "frames")


def root_xy_plot(items, out, n_cols=4):
    # items: list of (name, ref_xy, {label: trk_xy})
    n = len(items)
    rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(rows, n_cols, figsize=(3.4 * n_cols, 3.2 * rows),
                             squeeze=False)
    for k, (name, refxy, trks) in enumerate(items):
        ax = axes[k // n_cols][k % n_cols]
        ax.plot(refxy[:, 0], refxy[:, 1], color="k", lw=2.4, label="reference", zorder=3)
        ax.scatter([refxy[0, 0]], [refxy[0, 1]], c="k", s=30, zorder=4)
        for lab, xy, col in trks:
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.8, alpha=0.9, label=lab)
        ax.set_title(name[:30], fontsize=8)
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for k in range(n, rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.suptitle("Root (pelvis) XY path: reference vs tracked — global translation reconstruction",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="run results/ dir containing predicted_motion_lib_epoch_*.pt")
    ap.add_argument("--late-epoch", type=int, default=740)
    ap.add_argument("--early-epoch", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=6, help="num motions to render")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    late = torch.load(rd / f"predicted_motion_lib_epoch_{args.late_epoch}.pt",
                      map_location="cpu", weights_only=False)
    early_path = rd / f"predicted_motion_lib_epoch_{args.early_epoch}.pt"
    early = torch.load(early_path, map_location="cpu", weights_only=False) if early_path.exists() else None

    files = list(late["motion_files"])
    nm = len(files)

    # rank motions by reference root-xy path length (most translation first)
    ls, nf = per_motion_frames(late)
    disp = []
    refs = {}
    for m in range(nm):
        try:
            rp, fps = ref_motion(files[m])
        except Exception as e:
            print("skip ref", m, e); disp.append((m, -1.0)); continue
        refs[m] = rp
        xy = rp[:, 0, :2]
        plen = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0
        disp.append((m, plen))
    ranked = [m for m, _ in sorted(disp, key=lambda x: -x[1])]
    # pick: top-(num-2) translation + 2 low-translation (stationary) motions
    k = max(1, args.num - 2)
    pick = ranked[:k] + ranked[-2:]
    pick = list(dict.fromkeys(pick))[:args.num]
    print("rendering motions:", [(m, round(dict(disp)[m], 2), Path(files[m]).stem) for m in pick])

    xy_items = []
    for m in pick:
        if m not in refs:
            continue
        rp = refs[m]
        ref_xy0 = rp[0, 0, :2].copy()
        trk = tracked_motion(late, m, ref_root_xy0=ref_xy0)
        stem = Path(files[m]).stem
        T = min(len(rp), len(trk))
        montage(rp[:T], trk[:T], stem, out / f"montage_{stem}.png")
        if m == pick[0] or m == pick[1]:
            gif(rp[:T], trk[:T], stem, out / f"gif_{stem}.gif")
        trks = [(f"tracked e{args.late_epoch}", trk[:T, 0, :2], "#2ca02c")]
        if early is not None:
            te = tracked_motion(early, m, ref_root_xy0=ref_xy0)
            Te = min(len(rp), len(te))
            trks.append((f"tracked e{args.early_epoch}", te[:Te, 0, :2], "#d62728"))
        xy_items.append((stem, rp[:, 0, :2], trks))
    root_xy_plot(xy_items, out / "root_xy_paths.png", n_cols=min(3, len(xy_items)))
    print("DONE_VIZ ->", out)


if __name__ == "__main__":
    main()

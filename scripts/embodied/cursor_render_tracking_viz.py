#!/usr/bin/env python3
"""Render G1 tracking visualizations from g1_tracker_json/*.json.

Compares baseline tracker vs fine-tuned A_e609 tracker on the held-out eval
motions: (1) pelvis-height-over-time (falls), (2) skeleton montage + GIF.
"""
import json
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v2 as imageio

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
BASE_DIR = ROOT / "output/physflow_kimodo_g1/cursor_iter1_eval_baseline/g1_tracker_json"
A_DIR = ROOT / "output/physflow_kimodo_g1/eval_after_A_e609/g1_tracker_json"
OUT = ROOT / "output/physflow_kimodo_g1/viz"
OUT.mkdir(parents=True, exist_ok=True)

# G1 kinematic chains (by body name) -> edges
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


def load(json_path):
    d = json.load(open(json_path))
    names = [b["name"] for b in d["bodies"]]
    pos = np.array([f["body_pos"] for f in d["frames"]], dtype=np.float32)  # [T,B,3]
    return pos, names, float(d["fps"])


def edges_from_names(names):
    idx = {n: i for i, n in enumerate(names)}
    e = []
    for chain in CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            if a in idx and b in idx:
                e.append((idx[a], idx[b]))
    return e


def draw_skel(ax, P, edges, color, alpha=1.0, lw=2.0):
    # P: [B,3]  mujoco z-up
    for (i, j) in edges:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=color, s=6, alpha=alpha)


def set_axes(ax, center, span=0.9):
    cx, cy = center
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(0, 1.8)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def height_plot(stems):
    fig, axes = plt.subplots(1, len(stems), figsize=(3.2 * len(stems), 3.0), squeeze=False)
    for k, stem in enumerate(stems):
        ax = axes[0][k]
        for d, color, lab in [(BASE_DIR, "#d62728", "baseline"), (A_DIR, "#2ca02c", "A_e609")]:
            fp = sorted(glob.glob(str(d / f"{stem}*.json")))
            if not fp:
                continue
            P, names, fps = load(fp[0])
            z = P[:, 0, 2]  # pelvis z
            t = np.arange(len(z)) / fps
            ax.plot(t, z, color=color, lw=2, label=lab)
        ax.axhline(0.4, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(stem.replace("eval_", "").replace("_", " ")[:18], fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        if k == 0:
            ax.set_ylabel("pelvis height (m)", fontsize=8)
            ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=7)
    fig.suptitle("G1 pelvis height over tracking rollout (dashed = fall threshold 0.4m)", fontsize=10)
    fig.tight_layout()
    out = OUT / "pelvis_height_baseline_vs_A_e609.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def montage(stem, n=6):
    fb = sorted(glob.glob(str(BASE_DIR / f"{stem}*.json")))
    fa = sorted(glob.glob(str(A_DIR / f"{stem}*.json")))
    if not fb or not fa:
        print("missing", stem); return
    Pb, names, fps = load(fb[0])
    Pa, _, _ = load(fa[0])
    edges = edges_from_names(names)
    T = max(len(Pb), len(Pa))
    ticks = np.linspace(0, T - 1, n).astype(int)
    fig = plt.figure(figsize=(2.5 * n, 5.2))
    for col, ti in enumerate(ticks):
        for row, (P, color, lab) in enumerate([(Pb, "#d62728", "baseline"), (Pa, "#2ca02c", "A_e609")]):
            ax = fig.add_subplot(2, n, row * n + col + 1, projection="3d")
            fi = min(ti, len(P) - 1)
            center = (P[fi, 0, 0], P[fi, 0, 1])
            draw_skel(ax, P[fi], edges, color)
            set_axes(ax, center)
            if row == 0:
                ax.set_title(f"t={ti/fps:.1f}s", fontsize=9)
            if col == 0:
                ax.text2D(-0.1, 0.5, lab, transform=ax.transAxes, fontsize=11,
                          rotation=90, va="center", color=color, fontweight="bold")
    fig.suptitle(f"Tracking: {stem.replace('eval_','').replace('_',' ')}  (top baseline / bottom A_e609)", fontsize=12)
    fig.tight_layout()
    out = OUT / f"montage_{stem}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def gif(stem, fps_out=20):
    fb = sorted(glob.glob(str(BASE_DIR / f"{stem}*.json")))
    fa = sorted(glob.glob(str(A_DIR / f"{stem}*.json")))
    if not fb or not fa:
        return
    Pb, names, fps = load(fb[0])
    Pa, _, _ = load(fa[0])
    edges = edges_from_names(names)
    T = max(len(Pb), len(Pa))
    frames = []
    for ti in range(0, T, 2):
        fig = plt.figure(figsize=(8, 4))
        for col, (P, color, lab) in enumerate([(Pb, "#d62728", "baseline"), (Pa, "#2ca02c", "A_e609")]):
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            fi = min(ti, len(P) - 1)
            center = (P[fi, 0, 0], P[fi, 0, 1])
            draw_skel(ax, P[fi], edges, color)
            set_axes(ax, center)
            ax.set_title(lab, color=color, fontsize=12, fontweight="bold")
        fig.suptitle(f"{stem.replace('eval_','').replace('_',' ')}  t={ti/fps:.2f}s", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)
    out = OUT / f"gif_{stem}.gif"
    imageio.mimsave(out, frames, fps=fps_out, loop=0)
    print("saved", out, len(frames), "frames")


if __name__ == "__main__":
    stems = ["eval_left_leg_balance", "eval_boxing", "eval_circle_walk",
             "eval_backward_walk", "eval_robot_dance"]
    height_plot(stems)
    for s in ["eval_left_leg_balance", "eval_circle_walk", "eval_backward_walk"]:
        montage(s)
    gif("eval_left_leg_balance")
    gif("eval_circle_walk")

#!/usr/bin/env python3
"""Visual diagnostics for T2M baseline HML263 and converted H3D272 outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOMASK_ROOT = PROJECT_ROOT / "ref_repo" / "Momask" / "momask-codes"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (PROJECT_ROOT, MOMASK_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from convert_momask263_to_h3d272 import (  # noqa: E402
    decode_263_to_pose,
    encode_h3d272,
    linear_resample_positions,
    slerp_rotations,
)
from utils.motion_process import recover_from_ric  # noqa: E402
from utils.paramUtil import t2m_kinematic_chain  # noqa: E402


BONE_PAIRS = []
for chain in t2m_kinematic_chain:
    BONE_PAIRS.extend(zip(chain[:-1], chain[1:]))


def read_first_caption(text_file: Path) -> str:
    if not text_file.exists():
        return ""
    for line in text_file.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) >= 4 and parts[0].strip():
            try:
                f_tag = 0.0 if parts[2] == "nan" else float(parts[2])
                t_tag = 0.0 if parts[3] == "nan" else float(parts[3])
            except ValueError:
                continue
            if f_tag == 0.0 and t_tag == 0.0:
                return parts[0].strip()
    return ""


def hml263_recover_positions(m263: np.ndarray) -> np.ndarray:
    data = torch.from_numpy(m263.astype(np.float32)).unsqueeze(0)
    return recover_from_ric(data, 22).squeeze(0).numpy()


def hml263_local_positions(m263: np.ndarray) -> np.ndarray:
    root = np.zeros((len(m263), 1, 3), dtype=np.float32)
    root[:, 0, 1] = m263[:, 3]
    nonroot = m263[:, 4:67].reshape(len(m263), 21, 3)
    return np.concatenate([root, nonroot], axis=1)


def m272_local_positions(m272: np.ndarray) -> np.ndarray:
    return m272[:, 8:74].reshape(len(m272), 22, 3)


def proper_m272_from_hml263(m263: np.ndarray) -> np.ndarray:
    pos20, rot20, _feet = decode_263_to_pose(m263)
    pos30 = linear_resample_positions(pos20, 20.0, 30.0)
    rot30 = slerp_rotations(rot20, 20.0, 30.0)
    return encode_h3d272(pos30, rot30).astype(np.float32)


def integrate_root_xz(m272: np.ndarray) -> np.ndarray:
    xz = np.zeros((len(m272), 2), dtype=np.float32)
    if len(m272) > 1:
        xz[1:] = np.cumsum(m272[1:, :2], axis=0)
    return xz


def _draw_skeleton(ax, joints: np.ndarray, color: str):
    x, z, y = joints[:, 0], joints[:, 2], joints[:, 1]
    for a, b in BONE_PAIRS:
        ax.plot([x[a], x[b]], [z[a], z[b]], [y[a], y[b]], color=color, lw=2.1)
    ax.scatter(x, z, y, color=color, s=8)


def _set_axes(ax, pts: np.ndarray, title: str):
    flat = pts.reshape(-1, 3)
    center = (flat.min(axis=0) + flat.max(axis=0)) / 2.0
    span = np.ptp(flat, axis=0)
    radius = max(float(span.max()) * 0.58, 0.8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(max(0.0, flat[:, 1].min() - 0.05), flat[:, 1].min() + 2.0 * radius)
    ax.view_init(elev=12, azim=-70)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def render_strip(rows: list[tuple[str, np.ndarray, str]], out_path: Path, caption: str, sid: str):
    n_cols = 6
    fig = plt.figure(figsize=(2.65 * n_cols, 2.15 * len(rows)))
    for r, (label, pts, color) in enumerate(rows):
        for c, frac in enumerate(np.linspace(0.0, 1.0, n_cols)):
            t = int(round(frac * (len(pts) - 1)))
            ax = fig.add_subplot(len(rows), n_cols, r * n_cols + c + 1, projection="3d")
            _draw_skeleton(ax, pts[t], color)
            _set_axes(ax, pts, f"{label}  f{t}/{len(pts) - 1}")
    fig.suptitle(f"{sid}  {caption[:130]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_trajectory(gt272: np.ndarray, old272: np.ndarray, proper272: np.ndarray,
                      hml_pos: np.ndarray, out_path: Path, sid: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    gt = integrate_root_xz(gt272)
    old = integrate_root_xz(old272)
    proper = integrate_root_xz(proper272)
    hml_root = hml_pos[:, 0, [0, 2]]
    for label, xy, color, lw in [
        ("GT 272 root", gt, "#555555", 2.3),
        ("HML263 recovered root", hml_root, "#2ca25f", 2.0),
        ("current evaluated 272", old, "#de2d26", 2.0),
        ("proper pose-encoded 272", proper, "#3182bd", 2.0),
    ]:
        ax.plot(xy[:, 0], xy[:, 1], label=label, color=color, lw=lw)
        ax.scatter([xy[0, 0]], [xy[0, 1]], color=color, s=20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{sid} root XZ trajectory")
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
    parser.add_argument("--out-dir", default="outputs/evaluation/visual_diagnostics/t2m_baselines")
    args = parser.parse_args()

    pred263_dir = PROJECT_ROOT / "outputs" / "evaluation" / "humanml3d_hml3d263" / args.method
    pred272_dir = PROJECT_ROOT / "outputs" / "evaluation" / "humanml3d" / args.method
    gt272_dir = PROJECT_ROOT / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272" / "motion_data"
    text_dir = PROJECT_ROOT / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272" / "texts"
    out_dir = PROJECT_ROOT / args.out_dir / args.method

    for sid in [x.strip() for x in args.ids.split(",") if x.strip()]:
        m263_file = pred263_dir / f"{sid}.npy"
        old272_file = pred272_dir / f"{sid}.npy"
        gt272_file = gt272_dir / f"{sid}.npy"
        if not (m263_file.exists() and old272_file.exists() and gt272_file.exists()):
            print(f"[skip] {sid}: missing one of pred263/current272/gt272")
            continue
        m263 = np.load(m263_file)
        old272 = np.load(old272_file)
        gt272 = np.load(gt272_file)
        proper272 = proper_m272_from_hml263(m263)

        hml_global = hml263_recover_positions(m263)
        rows = [
            ("GT 272 local", m272_local_positions(gt272), "#555555"),
            ("HML263 recover", hml_global, "#2ca25f"),
            ("current 272", m272_local_positions(old272), "#de2d26"),
            ("proper 272", m272_local_positions(proper272), "#3182bd"),
        ]
        caption = read_first_caption(text_dir / f"{sid}.txt")
        strip_path = out_dir / f"{args.method}_{sid}_strip.png"
        traj_path = out_dir / f"{args.method}_{sid}_trajectory.png"
        render_strip(rows, strip_path, caption, sid)
        render_trajectory(gt272, old272, proper272, hml_global, traj_path, sid)
        print(f"[strip] {strip_path}")
        print(f"[traj]  {traj_path}")


if __name__ == "__main__":
    main()

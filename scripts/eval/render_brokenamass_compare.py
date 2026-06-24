"""Render a 5-panel skeleton comparison (GT | corrupted | StableMotion |
ours-strict | ours-combo) for BrokenAMASS* repair cases.

Produces, per selected case:
  - an animated GIF (all panels share one world frame so global drift / foot
    sliding is visually comparable), and
  - a key-frame montage PNG (rows = methods, cols = sampled frames).

Cases are ranked by a foot-sliding score on the corrupted input so the most
visibly-corrupted clips are shown first (the corruption the user can see by
eye but that barely moves MPJPE).

Usage:
    python3 scripts/eval/render_brokenamass_compare.py \
        --sm   ref_repo/StableMotion/output/brokenamass_star_sm_enhanced/results.npy \
        --gt   ref_repo/StableMotion/output/brokenamass_star_clean_v2/results_collected.npy \
        --ours-strict ref_repo/StableMotion/output/brokenamass_star_ours_strict_sd/results.npy \
        --ours-combo  ref_repo/StableMotion/output/brokenamass_star_ours_combo_all_self_t0_frame/results.npy \
        --out-dir output/eval/brokenamass_compare_viz --n-cases 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stablemotion_e9 import smpldata_to_m2m135  # noqa: E402
from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    motion135_to_positions_np,
)

# SMPL-22 kinematic parents for bone drawing.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
ANKLES = [7, 8, 10, 11]  # ankles + toes, for the foot-sliding score

PANELS = [
    ("GT (clean)", "gt", "#2ca02c"),
    ("corrupted (input)", "corrupted", "#7f7f7f"),
    ("StableMotion", "sm", "#d62728"),
    ("ours strict\n(lock/joint/τ0.5)", "ours_strict", "#1f77b4"),
    ("ours combo\n(all/self/τ0/frame)", "ours_combo", "#9467bd"),
]


def _ten(x):
    return x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x)).float()


def smpldata_to_positions(sd, bone_offsets, L):
    """smpldata (y-up; SM results are already y-up) -> (L,22,3) world joints."""
    sd_y = {k: _ten(sd[k])[:L] for k in ("poses", "trans", "joints")}
    m135 = smpldata_to_m2m135(sd_y, bone_offsets)
    return motion135_to_positions_np(np.asarray(m135, np.float32), bone_offsets.numpy())


def foot_slide_score(pos):
    """Mean horizontal ankle speed while the ankle is low (foot planted)."""
    h = pos[:, ANKLES, 1]                       # (T,4) height (y-up)
    xz = pos[:, ANKLES][:, :, [0, 2]]           # (T,4,2)
    spd = np.linalg.norm(np.diff(xz, axis=0), axis=-1)  # (T-1,4)
    low = (h[:-1] < (h.min() + 0.08))           # planted frames
    return float((spd * low).sum() / (low.sum() + 1e-6))


def draw_skel(ax, pos_t, color, lims):
    ax.clear()
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        ax.plot([pos_t[j, 0], pos_t[p, 0]], [pos_t[j, 2], pos_t[p, 2]],
                [pos_t[j, 1], pos_t[p, 1]], c=color, lw=2)
    ax.scatter(pos_t[:, 0], pos_t[:, 2], pos_t[:, 1], c=color, s=8)
    (xmn, xmx), (zmn, zmx), (ymn, ymx) = lims
    ax.set_xlim(xmn, xmx); ax.set_ylim(zmn, zmx); ax.set_zlim(ymn, ymx)
    ax.set_box_aspect((xmx - xmn, zmx - zmn, ymx - ymn))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=12, azim=-70)


def shared_lims(pos_list):
    allp = np.concatenate(pos_list, axis=0)     # (sumT,22,3)
    x, y, z = allp[..., 0], allp[..., 1], allp[..., 2]
    pad = 0.2
    return ((x.min() - pad, x.max() + pad),
            (z.min() - pad, z.max() + pad),
            (y.min() - pad, y.max() + pad))


def render_case(idx, posd, out_dir):
    keys = [k for _, k, _ in PANELS]
    pos_list = [posd[k] for k in keys]
    T = min(p.shape[0] for p in pos_list)
    pos_list = [p[:T] for p in pos_list]
    lims = shared_lims(pos_list)
    n = len(PANELS)

    # --- animated GIF ---
    fig = plt.figure(figsize=(3.0 * n, 3.4))
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]

    def update(f):
        for ax, (label, k, color) in zip(axes, PANELS):
            draw_skel(ax, posd[k][f], color, lims)
            ax.set_title(label, fontsize=9)
        fig.suptitle(f"BrokenAMASS* case {idx:05d}   frame {f}/{T-1}", fontsize=10)
        return axes

    step = max(1, T // 100)
    frames = list(range(0, T, step))
    anim = FuncAnimation(fig, update, frames=frames, interval=50)
    gif_path = out_dir / f"case_{idx:05d}.gif"
    anim.save(str(gif_path), writer=PillowWriter(fps=20))
    plt.close(fig)

    # --- key-frame montage PNG (rows=methods, cols=sampled frames) ---
    cols = 6
    fcols = np.linspace(0, T - 1, cols).round().astype(int)
    figm = plt.figure(figsize=(2.4 * cols, 2.6 * n))
    for r, (label, k, color) in enumerate(PANELS):
        for c, f in enumerate(fcols):
            ax = figm.add_subplot(n, cols, r * cols + c + 1, projection="3d")
            draw_skel(ax, posd[k][f], color, lims)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:
                ax.set_title(f"frame {f}", fontsize=8)
    figm.suptitle(f"BrokenAMASS* case {idx:05d}  (rows: GT / corrupted / "
                  f"StableMotion / ours-strict / ours-combo)", fontsize=11)
    figm.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = out_dir / f"case_{idx:05d}_montage.png"
    figm.savefig(str(png_path), dpi=90)
    plt.close(figm)
    return gif_path, png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ours-strict", required=True)
    ap.add_argument("--ours-combo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-cases", type=int, default=4)
    ap.add_argument("--case-ids", type=str, default="",
                    help="comma-separated explicit case ids (overrides ranking)")
    args = ap.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / "data/hymotion_m2m_data/bone_offsets_22.pt"),
        map_location="cpu", weights_only=False,
    ).float()

    sm = np.load(args.sm, allow_pickle=True).item()
    gt = np.load(args.gt, allow_pickle=True).item()
    os_ = np.load(args.ours_strict, allow_pickle=True).item()
    oc = np.load(args.ours_combo, allow_pickle=True).item()
    lengths = np.asarray(sm["lengths"]).reshape(-1)
    N = min(len(sm["motion"]), len(gt["motion"]),
            len(os_["motion_fix"]), len(oc["motion_fix"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select cases.
    if args.case_ids:
        sel = [int(x) for x in args.case_ids.split(",")]
    else:
        scores = []
        for i in range(N):
            L = int(min(lengths[i], _ten(sm["motion"][i]["poses"]).shape[0]))
            try:
                cp = smpldata_to_positions(sm["motion"][i], bone_offsets, L)
                scores.append((foot_slide_score(cp), i))
            except Exception:
                continue
        scores.sort(reverse=True)
        sel = [i for _, i in scores[:args.n_cases]]
    print(f"[render] selected cases: {sel}")

    for idx in sel:
        L = int(min(lengths[idx], _ten(sm["motion"][idx]["poses"]).shape[0]))
        posd = {
            "gt": smpldata_to_positions(gt["motion"][idx], bone_offsets, L),
            "corrupted": smpldata_to_positions(sm["motion"][idx], bone_offsets, L),
            "sm": smpldata_to_positions(sm["motion_fix"][idx], bone_offsets, L),
            "ours_strict": smpldata_to_positions(os_["motion_fix"][idx], bone_offsets, L),
            "ours_combo": smpldata_to_positions(oc["motion_fix"][idx], bone_offsets, L),
        }
        gp, pp = render_case(idx, posd, out_dir)
        print(f"[render] case {idx}: {gp.name}, {pp.name}")
    print("[done]")


if __name__ == "__main__":
    sys.exit(main())

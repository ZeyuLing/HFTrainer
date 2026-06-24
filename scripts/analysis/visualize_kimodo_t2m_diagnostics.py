#!/usr/bin/env python3
"""Render KIMODO T2M diagnostic visuals across the full eval bridge.

Rows per sample:
  GT-272 local joints -> raw KIMODO 22-joint output -> SMPL-IK fitted joints
  -> final MotionStreamer-272 local joints.

The page is meant for debugging whether a bad metric comes from KIMODO
generation, SMPL retargeting, or motion135->MS272 conversion.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
MOMASK_CODES = REPO / "ref_repo" / "Momask" / "momask-codes"
for _path in (REPO, MOMASK_CODES):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from utils.paramUtil import t2m_kinematic_chain  # noqa: E402

BONE_PAIRS: list[tuple[int, int]] = []
for chain in t2m_kinematic_chain:
    BONE_PAIRS.extend(zip(chain[:-1], chain[1:]))

COLORS = {
    "GT-272": "#d9d9d9",
    "KIMODO raw22": "#2b8cbe",
    "SMPL-IK fitted": "#f03b20",
    "Final pred272": "#31a354",
}


def localize(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32).copy()
    if len(pos) == 0:
        return pos
    root = pos[:, 0:1, [0, 2]]
    pos[:, :, 0] -= root[:, :, 0]
    pos[:, :, 2] -= root[:, :, 1]
    pos[..., 1] -= float(np.nanmin(pos[..., 1]))
    return pos


def m272_local_positions(m272: np.ndarray) -> np.ndarray:
    return localize(np.asarray(m272[:, 8:74], dtype=np.float32).reshape(len(m272), 22, 3))


def resample_positions(pos: np.ndarray, target_len: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if len(pos) == target_len or len(pos) < 2:
        return pos[:target_len]
    src = np.linspace(0.0, 1.0, len(pos), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    flat = pos.reshape(len(pos), -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape(target_len, *pos.shape[1:])


def mean_joint_accel(pos: np.ndarray) -> float:
    if len(pos) < 3:
        return 0.0
    acc = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]
    return float(np.linalg.norm(acc, axis=-1).mean())


def path_index(root: Path, suffix: str) -> dict[str, Path]:
    return {p.stem: p for p in root.rglob(f"*{suffix}") if p.is_file()}


def load_caption(data_root: Path, sid: str, debug_npz: Path | None) -> str:
    if debug_npz and debug_npz.exists():
        try:
            with np.load(debug_npz, allow_pickle=True) as data:
                if "caption" in data.files:
                    return str(np.asarray(data["caption"]).item())
        except Exception:
            pass
    text_path = data_root / "texts" / f"{sid}.txt"
    if text_path.exists():
        for raw in text_path.read_text(encoding="utf-8").splitlines():
            cap = raw.split("#", 1)[0].strip()
            if cap:
                return cap
    return ""


def sample_ids(args, pred_index: dict[str, Path], smpl_index: dict[str, Path]) -> list[str]:
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        return [sid for sid in ids if sid in pred_index and sid in smpl_index]
    candidates = sorted(set(pred_index) & set(smpl_index))
    if args.selection == "first":
        return candidates[: args.num]
    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        return candidates[: args.num]

    scored: list[tuple[float, str]] = []
    for sid in candidates:
        try:
            with np.load(smpl_index[sid], allow_pickle=True) as data:
                mpjpe = np.asarray(data["fit_mpjpe_mm"], dtype=np.float32)
            scored.append((float(np.nanmean(mpjpe)), sid))
        except Exception:
            continue
    scored.sort(reverse=args.selection == "worst")
    return [sid for _score, sid in scored[: args.num]]


def draw_pose(ax, joints: np.ndarray, color: str) -> None:
    x, z, y = joints[:, 0], joints[:, 2], joints[:, 1]
    for a, b in BONE_PAIRS:
        ax.plot([x[a], x[b]], [z[a], z[b]], [y[a], y[b]], color=color, lw=2.0)
    ax.scatter(x, z, y, color=color, s=8)


def setup_axes(ax, pts: np.ndarray, title: str) -> None:
    flat = pts.reshape(-1, 3)
    lo = np.nanmin(flat, axis=0)
    hi = np.nanmax(flat, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.nanmax(hi - lo)) * 0.58, 0.8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(max(0.0, lo[1] - 0.05), max(1.0, lo[1] + 2.0 * radius))
    ax.view_init(elev=12, azim=-70)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def render_strip(rows, out_path: Path, sid: str, caption: str, cols: int = 8) -> None:
    fig = plt.figure(figsize=(2.35 * cols, 2.05 * len(rows)))
    for r, (label, pts, color) in enumerate(rows):
        for c, frac in enumerate(np.linspace(0.0, 1.0, cols)):
            ax = fig.add_subplot(len(rows), cols, r * cols + c + 1, projection="3d")
            if pts is None or len(pts) == 0:
                ax.text2D(0.5, 0.5, "N/A", ha="center", va="center", color="red")
                continue
            t = int(round(frac * (len(pts) - 1)))
            draw_pose(ax, pts[t], color)
            setup_axes(ax, pts, f"{label} f{t}")
    fig.suptitle(f"{sid}  {caption[:120]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def render_gif(rows, out_path: Path, sid: str, caption: str, fps: int = 15) -> None:
    avail = [(lab, p, col) for lab, p, col in rows if p is not None and len(p) > 0]
    n = min(len(p) for _lab, p, _col in avail)
    step = max(1, int(round(30 / fps)))
    frames = list(range(0, n, step))
    fig = plt.figure(figsize=(4.0 * len(avail), 4.3))
    axes = [fig.add_subplot(1, len(avail), i + 1, projection="3d") for i in range(len(avail))]

    def update(t: int) -> None:
        for ax, (label, pts, color) in zip(axes, avail):
            ax.cla()
            draw_pose(ax, pts[min(t, len(pts) - 1)], color)
            setup_axes(ax, pts, f"{label} f{t}")
        fig.suptitle(f"{sid}  {caption[:90]}", fontsize=10)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="outputs/evaluation/kimodo_hml3d_smpl_ms272_20260615_1gpu_v7")
    parser.add_argument("--data-root", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--out-dir", default="outputs/evaluation/visual_diagnostics/kimodo_v7_bridge")
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--ids", default="")
    parser.add_argument("--selection", choices=["worst", "best", "random", "first"], default="worst")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    eval_root = REPO / args.eval_root
    data_root = REPO / args.data_root
    out_dir = REPO / args.out_dir
    pos_dir = eval_root / "positions22"
    debug_dir = eval_root / "debug_npz"
    pred_dir = eval_root / "pred272_all"
    smpl_dir = eval_root / "smpl135_parts"

    pred_index = path_index(pred_dir, ".npy")
    smpl_index = path_index(smpl_dir, ".npz")
    raw_index = path_index(pos_dir, ".npy")
    debug_index = path_index(debug_dir, ".npz")
    ids = sample_ids(args, pred_index, smpl_index)
    if not ids:
        raise SystemExit("no sample ids found")

    print(f"[diagnostic] selected {len(ids)} ids: {ids}", flush=True)
    cards = []
    for sid in ids:
        gt_path = data_root / "motion_data" / f"{sid}.npy"
        if not gt_path.exists():
            print(f"[skip] missing GT {sid}", flush=True)
            continue

        gt = m272_local_positions(np.load(gt_path))
        raw = localize(np.load(raw_index[sid]))
        pred = m272_local_positions(np.load(pred_index[sid]))
        with np.load(smpl_index[sid], allow_pickle=True) as data:
            fitted = localize(np.asarray(data["fitted_joints"], dtype=np.float32))
            target = localize(np.asarray(data["target_joints"], dtype=np.float32))
            mpjpe = np.asarray(data["fit_mpjpe_mm"], dtype=np.float32)

        n = min(len(gt), len(raw), len(fitted), len(pred))
        gt, raw, fitted, pred, target = [x[:n] for x in (gt, raw, fitted, pred, target)]
        raw_to_gt = float(np.linalg.norm(raw - gt, axis=-1).mean() * 1000.0)
        fitted_to_raw = float(np.linalg.norm(fitted - target, axis=-1).mean() * 1000.0)
        pred_to_fitted = float(np.linalg.norm(pred - fitted, axis=-1).mean() * 1000.0)
        accel_gt = mean_joint_accel(gt)
        accel = {
            "raw": mean_joint_accel(raw) / max(accel_gt, 1e-8),
            "fitted": mean_joint_accel(fitted) / max(accel_gt, 1e-8),
            "pred": mean_joint_accel(pred) / max(accel_gt, 1e-8),
        }

        caption = load_caption(data_root, sid, debug_index.get(sid))
        rows = [
            ("GT-272", gt, COLORS["GT-272"]),
            ("KIMODO raw22", raw, COLORS["KIMODO raw22"]),
            ("SMPL-IK fitted", fitted, COLORS["SMPL-IK fitted"]),
            ("Final pred272", pred, COLORS["Final pred272"]),
        ]
        strip = out_dir / f"{sid}_strip.png"
        render_strip(rows, strip, sid, caption)
        gif_name = None
        if not args.no_gif:
            gif = out_dir / f"{sid}_anim.gif"
            render_gif(rows, gif, sid, caption)
            gif_name = gif.name
        cards.append({
            "sid": sid,
            "caption": caption,
            "frames": n,
            "strip": strip.name,
            "gif": gif_name,
            "fit_mpjpe_mean": float(np.nanmean(mpjpe)),
            "fit_mpjpe_p95": float(np.nanpercentile(mpjpe, 95)),
            "raw_to_gt_mm": raw_to_gt,
            "fitted_to_raw_mm": fitted_to_raw,
            "pred_to_fitted_mm": pred_to_fitted,
            "accel": accel,
        })
        print(
            f"[ok] {sid} raw_gt={raw_to_gt:.1f} fit={fitted_to_raw:.1f} "
            f"pred_fit={pred_to_fitted:.1f} accel_pred={accel['pred']:.2f}x",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(cards, indent=2))

    html_parts = [
        "<html><head><meta charset='utf-8'><title>KIMODO T2M diagnostics</title>",
        "<style>body{font-family:sans-serif;background:#101014;color:#eee;margin:24px}"
        ".card{background:#1d1d22;border:1px solid #333;border-radius:8px;margin:24px 0;padding:18px}"
        "img{display:block;max-width:100%;background:#000;margin:12px 0;border-radius:6px}"
        "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #444;padding:4px 9px}"
        ".meta{color:#aaa}.cap{color:#ddd;margin:8px 0 12px}</style></head><body>",
        "<h1>KIMODO T2M bridge diagnostics</h1>",
        "<p class='meta'>Rows: GT-272, KIMODO raw22, SMPL-IK fitted, final pred272. "
        "All views are root-centered for posture comparison.</p>",
    ]
    for c in cards:
        html_parts.append(f"<div class='card'><h2>{html.escape(c['sid'])} "
                          f"<span class='meta'>frames={c['frames']}</span></h2>")
        html_parts.append(f"<div class='cap'>{html.escape(c['caption'])}</div>")
        html_parts.append(
            "<table><tr><th>raw vs GT mm</th><th>IK fit mm</th>"
            "<th>pred272 vs IK mm</th><th>pred accel/GT</th></tr>"
            f"<tr><td>{c['raw_to_gt_mm']:.1f}</td><td>{c['fit_mpjpe_mean']:.1f}"
            f" / p95 {c['fit_mpjpe_p95']:.1f}</td><td>{c['pred_to_fitted_mm']:.1f}</td>"
            f"<td>{c['accel']['pred']:.2f}x</td></tr></table>"
        )
        if c["gif"]:
            html_parts.append(f"<img src='{c['gif']}' alt='animation'>")
        html_parts.append(f"<img src='{c['strip']}' alt='strip'>")
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html_parts))
    print(f"[done] {out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

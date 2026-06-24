#!/usr/bin/env python3
"""Visual confirmation of HY-Motion-1.0 T2M reproduction (MS-272 outputs).

For a handful of deterministic test pairs it renders, side by side, the **GT**
272-dim motion and the **HY-Motion prediction** saved by
``scripts/eval/hymotion_t2m_h3d272.py`` (keyed by pair index ``<idx:06d>.npy``):

* an animated skeleton GIF (GT | HY), local/root-centered pose -> shows jitter;
* a static 8-frame skeleton strip PNG (embeddable);
* a per-frame mean joint-acceleration curve (GT vs HY) -> quantifies jitter;
* a root-XZ trajectory plot (GT vs HY).

It then writes an ``index.html`` linking everything for side-by-side review.

The pair enumeration mirrors ``MotionStreamer272Evaluator.load_test_pairs``
exactly (same split order + length/tag filtering) so ``<idx>.npy`` lines up with
the prediction files, WITHOUT loading the heavy TMR evaluator network.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]

# Filter constants — must match hftrainer/evaluation/evaluators/motionstreamer_272.py
MIN_MOTION_LENGTH = 60
MAX_MOTION_LENGTH = 300
UNIT_LENGTH = 4
FPS = 30

# Standard HumanML3D 22-joint kinematic chain (no ref_repo dependency).
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
BONE_PAIRS = []
for _chain in T2M_KINEMATIC_CHAIN:
    BONE_PAIRS.extend(zip(_chain[:-1], _chain[1:]))


def load_test_pairs(data_root: Path):
    """Deterministic (name, caption, gt272, ml) pairs; mirrors the evaluator."""
    motion_dir = data_root / "motion_data"
    text_dir = data_root / "texts"
    split = (data_root / "split" / "test.txt").read_text().splitlines()
    pairs = []
    for name in split:
        name = name.strip()
        if not name:
            continue
        m_file = motion_dir / f"{name}.npy"
        t_file = text_dir / f"{name}.txt"
        if not (m_file.exists() and t_file.exists()):
            continue
        motion = np.load(m_file)
        if len(motion) < MIN_MOTION_LENGTH or len(motion) >= MAX_MOTION_LENGTH:
            continue
        for line in t_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 4:
                continue
            caption = parts[0]
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
            if f_tag == 0.0 and t_tag == 0.0:
                m = motion
            else:
                m = motion[int(f_tag * FPS): int(t_tag * FPS)]
                if len(m) < MIN_MOTION_LENGTH or len(m) >= MAX_MOTION_LENGTH:
                    continue
            ml = (len(m) // UNIT_LENGTH) * UNIT_LENGTH
            if ml < MIN_MOTION_LENGTH:
                continue
            pairs.append((name, caption, m[:ml], ml))
    return pairs


def m272_local_positions(m272: np.ndarray) -> np.ndarray:
    return m272[:, 8:74].reshape(len(m272), 22, 3)


def integrate_root_xz(m272: np.ndarray) -> np.ndarray:
    xz = np.zeros((len(m272), 2), dtype=np.float32)
    if len(m272) > 1:
        xz[1:] = np.cumsum(m272[1:, :2], axis=0)
    return xz


def mean_joint_accel(pos: np.ndarray) -> np.ndarray:
    """Per-frame mean joint acceleration magnitude (jitter proxy)."""
    if len(pos) < 3:
        return np.zeros(len(pos), dtype=np.float32)
    acc = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]      # (T-2, 22, 3)
    mag = np.linalg.norm(acc, axis=-1).mean(axis=-1)  # (T-2,)
    return np.concatenate([[0.0], mag, [0.0]]).astype(np.float32)


def _draw(ax, joints, color):
    x, z, y = joints[:, 0], joints[:, 2], joints[:, 1]
    for a, b in BONE_PAIRS:
        ax.plot([x[a], x[b]], [z[a], z[b]], [y[a], y[b]], color=color, lw=2.0)
    ax.scatter(x, z, y, color=color, s=7)


def _axes(ax, pts, title):
    flat = pts.reshape(-1, 3)
    center = (flat.min(0) + flat.max(0)) / 2.0
    radius = max(float(np.ptp(flat, 0).max()) * 0.58, 0.8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(max(0.0, flat[:, 1].min() - 0.05), flat[:, 1].min() + 2.0 * radius)
    ax.view_init(elev=12, azim=-70)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def render_strip(gt_pos, hy_pos, out_path, caption, sid):
    n_cols = 8
    rows = [("GT", gt_pos, "#444444"), ("HY-Motion", hy_pos, "#de2d26")]
    fig = plt.figure(figsize=(2.4 * n_cols, 2.2 * len(rows)))
    for r, (label, pts, color) in enumerate(rows):
        for c, frac in enumerate(np.linspace(0.0, 1.0, n_cols)):
            t = int(round(frac * (len(pts) - 1)))
            ax = fig.add_subplot(len(rows), n_cols, r * n_cols + c + 1, projection="3d")
            _draw(ax, pts[t], color)
            _axes(ax, pts, f"{label} f{t}")
    fig.suptitle(f"{sid}  {caption[:120]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_gif(gt_pos, hy_pos, out_path, caption, sid, fps=20):
    fig = plt.figure(figsize=(8.4, 4.4))
    axg = fig.add_subplot(1, 2, 1, projection="3d")
    axh = fig.add_subplot(1, 2, 2, projection="3d")
    n = min(len(gt_pos), len(hy_pos))

    def update(t):
        axg.cla(); axh.cla()
        _draw(axg, gt_pos[t], "#444444"); _axes(axg, gt_pos, f"GT  f{t}/{n - 1}")
        _draw(axh, hy_pos[t], "#de2d26"); _axes(axh, hy_pos, f"HY-Motion  f{t}/{n - 1}")
        fig.suptitle(f"{sid}  {caption[:90]}", fontsize=10)

    anim = FuncAnimation(fig, update, frames=range(0, n), interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_diag(gt272, hy272, gt_pos, hy_pos, out_path, sid):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # acceleration curve
    a1.plot(mean_joint_accel(gt_pos), color="#444444", lw=1.6, label="GT")
    a1.plot(mean_joint_accel(hy_pos), color="#de2d26", lw=1.4, label="HY-Motion")
    a1.set_title(f"{sid}  mean joint accel /frame (jitter proxy)")
    a1.set_xlabel("frame"); a1.set_ylabel("|accel|"); a1.legend(fontsize=8)
    a1.grid(True, alpha=0.25)
    # trajectory
    g = integrate_root_xz(gt272); h = integrate_root_xz(hy272)
    a2.plot(g[:, 0], g[:, 1], color="#444444", lw=2.0, label="GT root")
    a2.plot(h[:, 0], h[:, 1], color="#de2d26", lw=2.0, label="HY root")
    a2.scatter([g[0, 0]], [g[0, 1]], color="#444444", s=20)
    a2.scatter([h[0, 0]], [h[0, 1]], color="#de2d26", s=20)
    a2.set_aspect("equal", adjustable="box")
    a2.set_title("root XZ trajectory"); a2.legend(fontsize=8); a2.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir",
                    default="outputs/evaluation/hymotion_h3d272/hy_272_fp32cfg")
    ap.add_argument("--data-root", default="data/evaluators/humanml3d_272")
    ap.add_argument("--out-dir",
                    default="outputs/evaluation/visual_diagnostics/hymotion_repro")
    ap.add_argument("--ids", default="", help="comma pair indices; empty=evenly spaced")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    pred_dir = REPO / args.pred_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[viz] loading deterministic test pairs...", flush=True)
    pairs = load_test_pairs(REPO / args.data_root)
    print(f"[viz] {len(pairs)} pairs total", flush=True)

    if args.ids.strip():
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = [int(round(i)) for i in np.linspace(0, len(pairs) - 1, args.num)]

    cards = []
    for idx in ids:
        pred_file = pred_dir / f"{idx:06d}.npy"
        if idx >= len(pairs) or not pred_file.exists():
            print(f"[skip] idx={idx} (missing pred or out of range)", flush=True)
            continue
        name, caption, gt272, ml = pairs[idx]
        hy272 = np.load(pred_file)
        n = min(len(gt272), len(hy272))
        gt272, hy272 = gt272[:n], hy272[:n]
        gt_pos = m272_local_positions(gt272)
        hy_pos = m272_local_positions(hy272)
        sid = f"{idx:06d}({name})"

        strip = out_dir / f"{idx:06d}_strip.png"
        diag = out_dir / f"{idx:06d}_diag.png"
        render_strip(gt_pos, hy_pos, strip, caption, sid)
        render_diag(gt272, hy272, gt_pos, hy_pos, diag, sid)
        gif_rel = None
        if not args.no_gif:
            gif = out_dir / f"{idx:06d}_anim.gif"
            try:
                render_gif(gt_pos, hy_pos, gif, caption, sid)
                gif_rel = gif.name
            except Exception as e:  # noqa: BLE001
                print(f"[gif-fail] {idx}: {e}", flush=True)

        # jitter ratio HY/GT
        ja_gt = float(mean_joint_accel(gt_pos).mean())
        ja_hy = float(mean_joint_accel(hy_pos).mean())
        ratio = ja_hy / max(ja_gt, 1e-8)
        cards.append((idx, name, caption, strip.name, diag.name, gif_rel, ratio))
        print(f"[ok] idx={idx} {name} jitter HY/GT={ratio:.2f}x  '{caption[:60]}'",
              flush=True)

    # index.html
    html = ["<html><head><meta charset='utf-8'><title>HY-Motion repro</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}"
            "h2{border-bottom:1px solid #444;padding-bottom:4px}"
            ".card{margin:28px 0;padding:16px;background:#1c1c1c;border-radius:8px}"
            "img{max-width:100%;display:block;margin:8px 0;background:#000}"
            ".r{color:#ff8080;font-weight:bold}</style></head><body>",
            "<h1>HY-Motion-1.0 T2M reproduction check (GT vs HY, MS-272)</h1>",
            "<p>jitter = mean joint accel ratio HY/GT (1.0 = as smooth as GT).</p>"]
    for idx, name, cap, strip, diag, gif, ratio in cards:
        html.append(f"<div class='card'><h2>#{idx} &nbsp; {name} &nbsp; "
                    f"<span class='r'>jitter {ratio:.2f}x</span></h2>")
        html.append(f"<p>{cap}</p>")
        if gif:
            html.append(f"<img src='{gif}' alt='anim'>")
        html.append(f"<img src='{strip}' alt='strip'>")
        html.append(f"<img src='{diag}' alt='diag'>")
        html.append("</div>")
    html.append("</body></html>")
    idx_html = out_dir / "index.html"
    idx_html.write_text("\n".join(html))
    print(f"\n[done] {len(cards)} samples -> {idx_html}", flush=True)
    if cards:
        avg = float(np.mean([c[6] for c in cards]))
        print(f"[summary] avg jitter HY/GT = {avg:.2f}x (1.0=GT-smooth)", flush=True)


if __name__ == "__main__":
    main()

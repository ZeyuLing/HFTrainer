#!/usr/bin/env python3
"""Unified T2M comparison web page: GT vs HY-Motion-1.0 vs T2M-GPT vs MoMask.

For a set of HumanML3D test ids (aligned across all models) it renders, side by
side, the **GT** motion and the three model predictions as root-centered
(local) skeleton animations + static strips, then writes a dark-themed
``index.html`` for review.

Data sources
------------
* GT / T2M-GPT / MoMask: HumanML3D-263 @ 20fps -> ``recover_from_ric`` -> (T,22,3),
  then root-XZ-centered per frame (Y kept).
* HY-Motion: pre-generated MS-272 @ 30fps under
  ``outputs/evaluation/hymotion_h3d272/hy_272_fp32cfg/<pair_idx:06d>.npy``;
  ``m272_local_positions`` ([:, 8:74]) is already local.

The HY pair index is obtained by enumerating the deterministic 272 test pairs
(``load_test_pairs``), identically to ``visualize_hymotion_repro.py``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
MOMASK_CODES = REPO / "ref_repo" / "Momask" / "momask-codes"
for _p in (REPO, MOMASK_CODES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.motion_process import recover_from_ric  # noqa: E402
from utils.paramUtil import t2m_kinematic_chain  # noqa: E402

# ---------------------------------------------------------------------------
# constants / kinematic chain
# ---------------------------------------------------------------------------
BONE_PAIRS = []
for _chain in t2m_kinematic_chain:
    BONE_PAIRS.extend(zip(_chain[:-1], _chain[1:]))

# 272 test-pair enumeration filters (mirror motionstreamer_272 evaluator).
MIN_MOTION_LENGTH = 60
MAX_MOTION_LENGTH = 300
UNIT_LENGTH = 4
FPS_272 = 30

MODEL_COLORS = {
    "GT": "#bdbdbd",
    "HY-Motion": "#de2d26",
    "T2M-GPT": "#3182bd",
    "MoMask": "#31a354",
}


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
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
                m = motion[int(f_tag * FPS_272): int(t_tag * FPS_272)]
                if len(m) < MIN_MOTION_LENGTH or len(m) >= MAX_MOTION_LENGTH:
                    continue
            ml = (len(m) // UNIT_LENGTH) * UNIT_LENGTH
            if ml < MIN_MOTION_LENGTH:
                continue
            pairs.append((name, caption, m[:ml], ml))
    return pairs


def first_caption(text_file: Path):
    if not text_file.exists():
        return None
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if parts and parts[0].strip():
            return parts[0].strip()
    return None


def m263_local_positions(m263: np.ndarray) -> np.ndarray:
    """263 -> (T,22,3) global via recover_from_ric, then root-XZ centered."""
    data = torch.from_numpy(np.asarray(m263, dtype=np.float32)).unsqueeze(0)
    pos = recover_from_ric(data, 22).squeeze(0).numpy()  # (T,22,3) global
    pos = pos.copy()
    root_xz = pos[:, 0:1, [0, 2]]  # (T,1,2)
    pos[:, :, 0] -= root_xz[:, :, 0]
    pos[:, :, 2] -= root_xz[:, :, 1]
    return pos


def m272_local_positions(m272: np.ndarray) -> np.ndarray:
    return m272[:, 8:74].reshape(len(m272), 22, 3)


def mean_joint_accel(pos: np.ndarray) -> np.ndarray:
    if len(pos) < 3:
        return np.zeros(len(pos), dtype=np.float32)
    acc = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]
    mag = np.linalg.norm(acc, axis=-1).mean(axis=-1)
    return np.concatenate([[0.0], mag, [0.0]]).astype(np.float32)


# ---------------------------------------------------------------------------
# rendering (style follows visualize_hymotion_repro.py)
# ---------------------------------------------------------------------------
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


def render_strip(rows, out_path: Path, caption: str, sid: str, n_cols: int = 8):
    """rows: list of (label, pos|None, color)."""
    fig = plt.figure(figsize=(2.3 * n_cols, 2.1 * len(rows)))
    for r, (label, pts, color) in enumerate(rows):
        for c, frac in enumerate(np.linspace(0.0, 1.0, n_cols)):
            ax = fig.add_subplot(len(rows), n_cols, r * n_cols + c + 1, projection="3d")
            if pts is None or len(pts) == 0:
                ax.text2D(0.5, 0.5, "N/A", ha="center", va="center", color="red")
                ax.set_title(f"{label}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                continue
            t = int(round(frac * (len(pts) - 1)))
            _draw(ax, pts[t], color)
            _axes(ax, pts, f"{label} f{t}")
    fig.suptitle(f"{sid}  {caption[:120]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def render_gif(rows, out_path: Path, caption: str, sid: str, fps: int = 20):
    """rows: list of (label, pos|None, color). One panel per available row."""
    avail = [(lab, p, col) for (lab, p, col) in rows if p is not None and len(p) > 0]
    if not avail:
        return False
    ncol = len(avail)
    fig = plt.figure(figsize=(4.0 * ncol, 4.4))
    axes = [fig.add_subplot(1, ncol, i + 1, projection="3d") for i in range(ncol)]
    n = min(len(p) for _, p, _ in avail)

    def update(t):
        for ax, (lab, p, col) in zip(axes, avail):
            ax.cla()
            _draw(ax, p[t], col)
            _axes(ax, p, f"{lab}  f{t}/{n - 1}")
        fig.suptitle(f"{sid}  {caption[:90]}", fontsize=10)

    anim = FuncAnimation(fig, update, frames=range(0, n), interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def generate_t2mgpt(captions, lengths, out_dir: Path, ids, device: str):
    from hftrainer.models.motion.t2mgpt import T2MGPTBundle
    from hftrainer.pipelines.t2mgpt import T2MGPTPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = T2MGPTBundle(
        vq_path=str(REPO / "ref_repo/T2M-GPT/pretrained/VQVAE/net_last.pth"),
        gpt_path=str(REPO / "ref_repo/T2M-GPT/pretrained/VQTransformer_corruption05/net_best_fid.pth"),
        mean_path=str(REPO / "work_dirs/h3d263_eval/h3d263_test_recon_fk/Mean.npy"),
        std_path=str(REPO / "work_dirs/h3d263_eval/h3d263_test_recon_fk/Std.npy"),
    )
    pipe = T2MGPTPipeline(bundle, device=device)
    motions = pipe.infer_t2m(list(captions), list(lengths))
    for sid, m in zip(ids, motions):
        np.save(out_dir / f"{sid}.npy", np.asarray(m, dtype=np.float32))
    return {sid: out_dir / f"{sid}.npy" for sid in ids}


def generate_momask(captions, lengths, out_dir: Path, ids, device: str):
    from hftrainer.models.motion.momask import MoMaskBundle
    from hftrainer.pipelines.momask import MoMaskPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = REPO / "checkpoints/momask/humanml3d"
    if (artifact / "momask_config.json").exists():
        bundle = MoMaskBundle.from_pretrained(str(artifact), load_length_estimator=False)
    else:
        bundle = MoMaskBundle(
            weights_root=str(REPO / "ref_repo/Momask/weights"),
            load_length_estimator=False,
        )
    pipe = MoMaskPipeline(bundle, device=device)
    clamped = [pipe.clamp_length(int(l)) for l in lengths]
    motions = pipe.infer_t2m(list(captions), clamped)
    for sid, m in zip(ids, motions):
        np.save(out_dir / f"{sid}.npy", np.asarray(m, dtype=np.float32))
    return {sid: out_dir / f"{sid}.npy" for sid in ids}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data263", default="ref_repo/CondMDI/dataset/HumanML3D")
    ap.add_argument("--data272", default="data/evaluators/humanml3d_272")
    ap.add_argument("--hy-dir", default="outputs/evaluation/hymotion_h3d272/hy_272_fp32cfg")
    ap.add_argument("--out-dir", default="outputs/evaluation/visual_diagnostics/web_t2m_compare")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--skip-gen", action="store_true", help="reuse existing pred npys")
    args = ap.parse_args()

    data263 = REPO / args.data263
    data272 = REPO / args.data272
    hy_dir = REPO / args.hy_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- step 1: select aligned ids ----------------------------------------
    print("[step1] enumerating 272 test pairs ...", flush=True)
    pairs = load_test_pairs(data272)
    name_to_idx = {}
    for idx, (name, _cap, _gt, _ml) in enumerate(pairs):
        name_to_idx.setdefault(name, idx)  # first pair idx per name
    print(f"[step1] {len(pairs)} pairs, {len(name_to_idx)} unique names", flush=True)

    test_ids = [n.strip() for n in (data263 / "test.txt").read_text().splitlines() if n.strip()]
    njv = data263 / "new_joint_vecs"
    txt = data263 / "texts"

    selected = []  # (sid, caption, gt_len, hy_idx)
    for sid in test_ids:
        if sid not in name_to_idx:
            continue
        if not (njv / f"{sid}.npy").exists():
            continue
        hy_idx = name_to_idx[sid]
        if not (hy_dir / f"{hy_idx:06d}.npy").exists():
            continue
        cap = first_caption(txt / f"{sid}.txt")
        if not cap:
            continue
        gt_len = int(np.load(njv / f"{sid}.npy", mmap_mode="r").shape[0])
        selected.append((sid, cap, gt_len, hy_idx))
        if len(selected) >= args.num:
            break

    if not selected:
        raise SystemExit("[fatal] no aligned ids found")
    print(f"[step1] selected {len(selected)} ids:", flush=True)
    for sid, cap, gt_len, hy_idx in selected:
        print(f"   {sid}  len={gt_len}  hy_idx={hy_idx:06d}  '{cap[:60]}'", flush=True)

    ids = [s[0] for s in selected]
    caps = [s[1] for s in selected]
    lens = [s[2] for s in selected]
    hy_idxs = {s[0]: s[3] for s in selected}

    # ---- step 2: generate T2M-GPT and MoMask -------------------------------
    status = {"T2M-GPT": "ok", "MoMask": "ok"}
    t2mgpt_dir = out_dir / "t2mgpt"
    momask_dir = out_dir / "momask"

    if args.skip_gen:
        print("[step2] --skip-gen: reusing existing preds", flush=True)
    else:
        t0 = time.time()
        print("[step2] generating T2M-GPT ...", flush=True)
        try:
            generate_t2mgpt(caps, lens, t2mgpt_dir, ids, args.device)
            print(f"[step2] T2M-GPT done ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            status["T2M-GPT"] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[step2][fail] T2M-GPT: {exc}", flush=True)

        t0 = time.time()
        print("[step2] generating MoMask ...", flush=True)
        try:
            generate_momask(caps, lens, momask_dir, ids, args.device)
            print(f"[step2] MoMask done ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            status["MoMask"] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[step2][fail] MoMask: {exc}", flush=True)

    # ---- step 3 + 4: render + html -----------------------------------------
    print("[step3] rendering ...", flush=True)
    cards = []
    for sid, cap, gt_len, hy_idx in selected:
        gt263 = np.load(njv / f"{sid}.npy")
        gt_pos = m263_local_positions(gt263)

        hy272 = np.load(hy_dir / f"{hy_idx:06d}.npy")
        hy_pos = m272_local_positions(hy272)

        def _load_local(d: Path):
            f = d / f"{sid}.npy"
            if not f.exists():
                return None
            try:
                return m263_local_positions(np.load(f))
            except Exception as e:  # noqa: BLE001
                print(f"[render][warn] {sid} {d.name}: {e}", flush=True)
                return None

        t2m_pos = _load_local(t2mgpt_dir)
        mom_pos = _load_local(momask_dir)

        rows = [
            ("GT", gt_pos, MODEL_COLORS["GT"]),
            ("HY-Motion", hy_pos, MODEL_COLORS["HY-Motion"]),
            ("T2M-GPT", t2m_pos, MODEL_COLORS["T2M-GPT"]),
            ("MoMask", mom_pos, MODEL_COLORS["MoMask"]),
        ]

        strip = out_dir / f"{sid}_strip.png"
        render_strip(rows, strip, cap, sid)

        gif_rel = None
        if not args.no_gif:
            gif = out_dir / f"{sid}_anim.gif"
            try:
                if render_gif(rows, gif, cap, sid):
                    gif_rel = gif.name
            except Exception as e:  # noqa: BLE001
                print(f"[gif-fail] {sid}: {e}", flush=True)

        ja_gt = float(mean_joint_accel(gt_pos).mean())
        jitter = {}
        for lab, pos in (("HY-Motion", hy_pos), ("T2M-GPT", t2m_pos), ("MoMask", mom_pos)):
            if pos is None:
                jitter[lab] = None
            else:
                jitter[lab] = float(mean_joint_accel(pos).mean()) / max(ja_gt, 1e-8)

        cards.append({
            "sid": sid, "cap": cap, "hy_idx": hy_idx, "gt_len": gt_len,
            "strip": strip.name, "gif": gif_rel,
            "t2m_ok": t2m_pos is not None, "mom_ok": mom_pos is not None,
            "jitter": jitter,
        })
        print(f"[ok] {sid} jitter HY/GT={jitter['HY-Motion']:.2f} "
              f"T2MGPT/GT={jitter['T2M-GPT'] if jitter['T2M-GPT'] is None else round(jitter['T2M-GPT'],2)} "
              f"MoMask/GT={jitter['MoMask'] if jitter['MoMask'] is None else round(jitter['MoMask'],2)}",
              flush=True)

    # ---- html --------------------------------------------------------------
    def _jit(v):
        return "N/A" if v is None else f"{v:.2f}x"

    html = [
        "<html><head><meta charset='utf-8'><title>T2M comparison: GT vs HY vs T2M-GPT vs MoMask</title>",
        "<style>body{font-family:sans-serif;background:#0e0e10;color:#eaeaea;margin:24px}"
        "h1{font-weight:600}h2{border-bottom:1px solid #333;padding-bottom:6px;margin-top:8px}"
        ".card{margin:28px 0;padding:18px;background:#1a1a1d;border-radius:10px}"
        ".cap{color:#cfcfcf;font-size:15px;margin:6px 0 10px}"
        "img{max-width:100%;display:block;margin:10px 0;background:#000;border-radius:6px}"
        ".meta{color:#9aa;font-size:13px}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:6px;margin-right:8px;font-size:12px}"
        ".ok{background:#1d3b24;color:#7fdf9a}.bad{background:#3b1d1d;color:#ff8080}"
        "table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #333;padding:3px 10px;font-size:13px}"
        "</style></head><body>",
        "<h1>HumanML3D Text-to-Motion comparison</h1>",
        "<p class='meta'>Columns: <b>GT | HY-Motion-1.0 | T2M-GPT | MoMask</b>. "
        "Poses are root-centered (local) for fair posture comparison. "
        "jitter = mean joint-accel ratio model/GT (1.0 = as smooth as GT).</p>",
    ]
    for st_name, st in status.items():
        cls = "ok" if st == "ok" else "bad"
        html.append(f"<span class='badge {cls}'>{st_name}: {st}</span>")

    for c in cards:
        html.append(f"<div class='card'><h2>{c['sid']} "
                    f"<span class='meta'>(hy_idx {c['hy_idx']:06d}, GT len {c['gt_len']})</span></h2>")
        html.append(f"<div class='cap'>{c['cap']}</div>")
        t2m_badge = "ok" if c["t2m_ok"] else "bad"
        mom_badge = "ok" if c["mom_ok"] else "bad"
        html.append(
            "<span class='badge ok'>GT</span>"
            "<span class='badge ok'>HY-Motion</span>"
            f"<span class='badge {t2m_badge}'>T2M-GPT {'' if c['t2m_ok'] else 'MISSING'}</span>"
            f"<span class='badge {mom_badge}'>MoMask {'' if c['mom_ok'] else 'MISSING'}</span>"
        )
        j = c["jitter"]
        html.append(
            "<table><tr><th>model</th><th>HY-Motion</th><th>T2M-GPT</th><th>MoMask</th></tr>"
            f"<tr><td>jitter / GT</td><td>{_jit(j['HY-Motion'])}</td>"
            f"<td>{_jit(j['T2M-GPT'])}</td><td>{_jit(j['MoMask'])}</td></tr></table>"
        )
        if c["gif"]:
            html.append(f"<img src='{c['gif']}' alt='anim'>")
        html.append(f"<img src='{c['strip']}' alt='strip'>")
        html.append("</div>")
    html.append("</body></html>")

    idx_html = out_dir / "index.html"
    idx_html.write_text("\n".join(html))
    print(f"\n[done] {len(cards)} cards -> {idx_html}", flush=True)
    print(f"[status] {status}", flush=True)


if __name__ == "__main__":
    main()

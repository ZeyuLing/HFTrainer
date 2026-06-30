#!/usr/bin/env python3
"""Render MotionHub/HYMotion translation-alignment cases with matplotlib.

This is an independent checker for the Three.js viewer.  It reads the same
comparison manifest, runs the repository SMPL-H/SMPL-X LBS code, and draws
mesh vertices in the stored world coordinate system:

    vertices_world = smpl_vertices_without_translation + stored_transl

No floor alignment, canonicalization, or camera-side y-shifting is applied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSATILE_ROOT = PROJECT_ROOT.parent / "versatilemotion"
if str(VERSATILE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSATILE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COLORS = {
    "hymotion": "#4c78a8",
    "aist": "#59a14f",
    "finedance": "#e15759",
    "combatmotion": "#f28e2b",
    "fit3d": "#76b7b2",
    "humansc3d": "#b07aa1",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_cases(path: Path) -> List[Dict[str, Any]]:
    raw = read_json(path)
    cases = raw.get("cases", raw) if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        raise TypeError(f"expected list cases in {path}, got {type(cases).__name__}")
    return [case for case in cases if isinstance(case, dict) and "key" in case]


def as_repeated(arr: Any, frames: int, width: int) -> np.ndarray:
    if arr is None:
        return np.zeros((frames, width), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    if out.ndim == 1:
        out = out[None, :]
    if out.shape[0] == 1 and frames > 1:
        out = np.repeat(out, frames, axis=0)
    if out.shape[0] != frames:
        out = out[:frames]
        if 0 < out.shape[0] < frames:
            out = np.concatenate(
                [out, np.repeat(out[-1:], frames - out.shape[0], axis=0)],
                axis=0,
            )
    if out.shape[1] < width:
        out = np.concatenate(
            [out, np.zeros((out.shape[0], width - out.shape[1]), dtype=np.float32)],
            axis=1,
        )
    return out[:, :width].astype(np.float32, copy=False)


def frame_indices(frames: int, fracs: Iterable[float]) -> List[int]:
    if frames <= 0:
        raise ValueError("empty motion")
    idxs = []
    for frac in fracs:
        clamped = min(1.0, max(0.0, float(frac)))
        idxs.append(int(round(clamped * (frames - 1))))
    return idxs


def evenly_spaced_frame_indices(frames: int, count: int) -> List[int]:
    if frames <= 0:
        raise ValueError("empty motion")
    if count <= 1:
        return [0]
    return [int(round(x)) for x in np.linspace(0, frames - 1, count)]


def infer_smpl_type(poses: np.ndarray, motion: Dict[str, Any]) -> str:
    if motion.get("smpl_type"):
        return str(motion["smpl_type"]).lower()
    if poses.shape[1] == 156:
        return "smplh"
    if poses.shape[1] == 165:
        return "smplx"
    raise ValueError(f"unsupported pose shape {poses.shape}")


def load_motion_components(motion: Dict[str, Any], idxs: List[int]) -> Dict[str, Any]:
    path = Path(str(motion["smpl_path"]))
    data = dict(np.load(path, allow_pickle=True))
    poses = np.asarray(data["poses"], dtype=np.float32)
    transl = np.asarray(data.get("transl", data.get("trans")), dtype=np.float32)
    if transl.ndim != 2 or transl.shape[1] < 3:
        raise ValueError(f"bad transl shape {transl.shape} in {path}")
    frames = min(poses.shape[0], transl.shape[0])
    poses = poses[:frames]
    transl = transl[:frames, :3]
    safe_idxs = [min(frames - 1, max(0, int(i))) for i in idxs]
    smpl_type = infer_smpl_type(poses, motion)
    beta_width = 16 if smpl_type == "smplh" else 10
    betas = as_repeated(data.get("betas"), frames, beta_width)
    return {
        "path": path,
        "label": str(motion.get("label", motion.get("id", path.stem))),
        "id": str(motion.get("id", path.stem)),
        "smpl_type": smpl_type,
        "frames": frames,
        "idxs": safe_idxs,
        "poses": poses[safe_idxs],
        "transl": transl[safe_idxs],
        "betas": betas[safe_idxs],
    }


def build_models(device: torch.device) -> Dict[str, torch.nn.Module]:
    from mmotion.models.body_models.smplx_lite import SmplLite, SmplxLite

    models = {
        "smplx": SmplxLite(
            model_path=str(VERSATILE_ROOT / "checkpoints/smpl_models/smplx"),
            gender="neutral",
            num_betas=10,
        ).to(device=device, dtype=torch.float32).eval(),
        "smplh": SmplLite(
            model_path=str(VERSATILE_ROOT / "checkpoints/smpl_models/smplh"),
            gender="neutral",
            num_betas=16,
        ).to(device=device, dtype=torch.float32).eval(),
    }
    return models


def lbs_vertices(
    comp: Dict[str, Any],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    poses = comp["poses"]
    transl = torch.from_numpy(comp["transl"]).to(device=device, dtype=torch.float32)
    betas = torch.from_numpy(comp["betas"]).to(device=device, dtype=torch.float32)
    go = torch.from_numpy(poses[:, :3]).to(device=device, dtype=torch.float32)
    model = models[comp["smpl_type"]]

    with torch.no_grad():
        if comp["smpl_type"] == "smplx":
            bp = torch.from_numpy(poses[:, 3:66]).to(device=device, dtype=torch.float32)
            verts = model(
                body_pose=bp,
                betas=betas,
                global_orient=go,
                transl=transl,
            )
        else:
            bp = torch.from_numpy(poses[:, 3:156]).to(device=device, dtype=torch.float32)
            verts = model(
                body_pose=bp,
                betas=betas,
                global_orient=go,
                transl=transl,
                rotation_mode="aa",
            )
    return verts.detach().cpu().numpy().astype(np.float32), np.asarray(model.faces, dtype=np.int64)


def plot_coords(verts: np.ndarray) -> np.ndarray:
    # Matplotlib's z-axis is vertical, so map original (x, y, z) to (x, z, y).
    return np.stack([verts[..., 0], verts[..., 2], verts[..., 1]], axis=-1)


def decimate_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    idx = np.linspace(0, faces.shape[0] - 1, max_faces, dtype=np.int64)
    return faces[idx]


def draw_floor(ax: Any, bounds: Tuple[float, float, float, float, float, float]) -> None:
    xmin, xmax, zmin, zmax, _, _ = bounds
    xgrid = np.linspace(xmin, xmax, 7)
    ygrid = np.linspace(zmin, zmax, 7)
    for x in xgrid:
        ax.plot([x, x], [zmin, zmax], [0.0, 0.0], color="#c8c8c8", linewidth=0.45, alpha=0.45)
    for y in ygrid:
        ax.plot([xmin, xmax], [y, y], [0.0, 0.0], color="#c8c8c8", linewidth=0.45, alpha=0.45)
    ax.plot([xmin, xmax], [zmin, zmin], [0.0, 0.0], color="#7a7a7a", linewidth=0.8, alpha=0.65)


def padded_bounds(all_plot_verts: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    flat = all_plot_verts.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    mins[2] = min(mins[2], 0.0)
    maxs[2] = max(maxs[2], 0.0)
    spans = np.maximum(maxs - mins, 0.1)
    pad = np.maximum(spans * 0.12, np.array([0.15, 0.15, 0.08], dtype=np.float32))
    mins -= pad
    maxs += pad
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


def set_common_axes(ax: Any, bounds: Tuple[float, float, float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_xlabel("x", labelpad=-7, fontsize=7)
    ax.set_ylabel("z", labelpad=-7, fontsize=7)
    ax.set_zlabel("y", labelpad=-6, fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=6, pad=-2)
    ax.view_init(elev=18, azim=-68)
    ax.grid(False)


def add_mesh_axis(
    fig: Any,
    position: int,
    n_rows: int,
    n_cols: int,
    verts_plot: np.ndarray,
    faces: np.ndarray,
    color: str,
    bounds: Tuple[float, float, float, float, float, float],
    title: str,
    title_size: int,
) -> Any:
    ax = fig.add_subplot(n_rows, n_cols, position, projection="3d")
    mesh = Poly3DCollection(
        verts_plot[faces],
        facecolor=color,
        edgecolor="none",
        linewidths=0.0,
        alpha=0.86,
    )
    ax.add_collection3d(mesh)
    draw_floor(ax, bounds)
    set_common_axes(ax, bounds)
    ax.set_title(title, fontsize=title_size, pad=2)
    return ax


def render_case(
    case: Dict[str, Any],
    out_dir: Path,
    fracs: List[float],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    max_faces: int,
    dpi: int,
) -> Dict[str, Any]:
    motions = [m for m in case.get("motions", []) if isinstance(m, dict) and m.get("smpl_path")]
    if not motions:
        raise ValueError(f"case {case.get('key')} has no motions")

    comps: List[Dict[str, Any]] = []
    for motion in motions:
        # Use the same normalized time fractions for each method, even when
        # source clip lengths differ.
        frames = int(np.load(Path(str(motion["smpl_path"])), allow_pickle=True)["poses"].shape[0])
        comp = load_motion_components(motion, frame_indices(frames, fracs))
        verts, faces = lbs_vertices(comp, models, device)
        comp["verts"] = verts
        comp["faces"] = decimate_faces(faces, max_faces)
        comp["plot_verts"] = plot_coords(verts)
        comps.append(comp)

    all_plot_verts = np.concatenate([comp["plot_verts"].reshape(-1, 3) for comp in comps], axis=0)
    bounds = padded_bounds(all_plot_verts)
    n_rows = len(fracs)
    n_cols = len(comps)
    fig = plt.figure(figsize=(4.1 * n_cols, 3.25 * n_rows), dpi=dpi)
    fig.suptitle(
        f"{case.get('key')} · matplotlib LBS mesh · no frontend canonicalization",
        fontsize=12,
        y=0.99,
    )

    rows: List[Dict[str, Any]] = []
    for row_idx, frac in enumerate(fracs):
        for col_idx, comp in enumerate(comps):
            frame_in_comp = row_idx
            verts_plot = comp["plot_verts"][frame_in_comp]
            faces = comp["faces"]
            color = COLORS.get(comp["id"], "#777777")
            min_y = float(comp["verts"][frame_in_comp, :, 1].min())
            add_mesh_axis(
                fig,
                row_idx * n_cols + col_idx + 1,
                n_rows,
                n_cols,
                verts_plot,
                faces,
                color,
                bounds,
                f"{comp['label']}\nfrac={frac:.2f} frame={comp['idxs'][frame_in_comp]} min_y={min_y:+.4f}",
                8,
            )
            rows.append({
                "motion_id": comp["id"],
                "label": comp["label"],
                "smpl_type": comp["smpl_type"],
                "path": str(comp["path"]),
                "frac": float(frac),
                "frame": int(comp["idxs"][frame_in_comp]),
                "mesh_min_y": min_y,
                "stored_transl_y": float(comp["transl"][frame_in_comp, 1]),
            })

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case['key']}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "key": str(case["key"]),
        "output": str(out_path),
        "bounds": bounds,
        "frames": rows,
    }


def canvas_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def render_video_case(
    case: Dict[str, Any],
    out_dir: Path,
    video_frames: int,
    fps: int,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    max_faces: int,
    dpi: int,
) -> Dict[str, Any]:
    import imageio.v2 as imageio

    try:
        import imageio_ffmpeg

        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    motions = [m for m in case.get("motions", []) if isinstance(m, dict) and m.get("smpl_path")]
    if not motions:
        raise ValueError(f"case {case.get('key')} has no motions")

    comps: List[Dict[str, Any]] = []
    for motion in motions:
        frames = int(np.load(Path(str(motion["smpl_path"])), allow_pickle=True)["poses"].shape[0])
        comp = load_motion_components(
            motion,
            evenly_spaced_frame_indices(frames, max(1, video_frames)),
        )
        verts, faces = lbs_vertices(comp, models, device)
        comp["verts"] = verts
        comp["faces"] = decimate_faces(faces, max_faces)
        comp["plot_verts"] = plot_coords(verts)
        comps.append(comp)

    all_plot_verts = np.concatenate([comp["plot_verts"].reshape(-1, 3) for comp in comps], axis=0)
    bounds = padded_bounds(all_plot_verts)
    n_cols = len(comps)
    fig = plt.figure(figsize=(12.8, 4.8), dpi=dpi)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{case['key']}.mp4"
    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )

    frame_rows: List[Dict[str, Any]] = []
    try:
        for t in range(max(1, video_frames)):
            fig.clf()
            frac = 0.0 if video_frames <= 1 else t / float(video_frames - 1)
            fig.suptitle(
                f"{case.get('key')} · matplotlib LBS mesh video · progress={frac:.2f}",
                fontsize=12,
                y=0.98,
            )
            for col_idx, comp in enumerate(comps):
                color = COLORS.get(comp["id"], "#777777")
                min_y = float(comp["verts"][t, :, 1].min())
                if t in {0, video_frames // 2, video_frames - 1}:
                    frame_rows.append({
                        "motion_id": comp["id"],
                        "label": comp["label"],
                        "smpl_type": comp["smpl_type"],
                        "path": str(comp["path"]),
                        "progress": float(frac),
                        "frame": int(comp["idxs"][t]),
                        "mesh_min_y": min_y,
                        "stored_transl_y": float(comp["transl"][t, 1]),
                    })
                add_mesh_axis(
                    fig,
                    col_idx + 1,
                    1,
                    n_cols,
                    comp["plot_verts"][t],
                    comp["faces"],
                    color,
                    bounds,
                    f"{comp['label']}\nframe={comp['idxs'][t]} min_y={min_y:+.4f}",
                    8,
                )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            writer.append_data(canvas_rgb(fig))
    finally:
        writer.close()
        plt.close(fig)

    return {
        "key": str(case["key"]),
        "output": str(video_path),
        "bounds": bounds,
        "fps": int(fps),
        "num_video_frames": int(video_frames),
        "duration_sec": float(video_frames) / float(fps),
        "sampled_frames": frame_rows,
    }


def write_video_index(summary: Dict[str, Any], out_dir: Path) -> Path:
    index_path = out_dir / "index.html"
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotionHub Matplotlib Video Audit</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d1115;
      --panel: #151b22;
      --line: #2a333d;
      --text: #ecf1f5;
      --muted: #9ca9b6;
      --accent: #d8b45f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .shell {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #10161c;
      padding: 18px 14px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    h1 {
      margin: 0 0 14px;
      font-size: 18px;
      font-weight: 650;
    }
    .meta {
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .case-btn {
      width: 100%;
      border: 1px solid var(--line);
      background: transparent;
      color: var(--text);
      text-align: left;
      padding: 9px 10px;
      margin: 0 0 8px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
    }
    .case-btn.active {
      border-color: var(--accent);
      background: rgba(216, 180, 95, 0.12);
    }
    main {
      padding: 20px;
      min-width: 0;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 12px;
    }
    h2 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
    }
    .path {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    video {
      width: 100%;
      max-height: calc(100vh - 150px);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: block;
    }
    .facts {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .fact {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 10px;
      min-height: 58px;
    }
    .label {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
    }
    .value {
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Matplotlib Video Audit</h1>
      <div class="meta" id="meta"></div>
      <div id="caseList"></div>
    </aside>
    <main>
      <div class="topbar">
        <h2 id="title"></h2>
        <div class="path" id="videoPath"></div>
      </div>
      <video id="video" controls loop playsinline></video>
      <div class="facts">
        <div class="fact"><div class="label">fps</div><div class="value" id="fps"></div></div>
        <div class="fact"><div class="label">duration</div><div class="value" id="duration"></div></div>
        <div class="fact"><div class="label">min-y range</div><div class="value" id="miny"></div></div>
        <div class="fact"><div class="label">motions</div><div class="value">HYMotion · AIST++ · FineDance</div></div>
      </div>
    </main>
  </div>
  <script>
    let summary = null;
    let current = 0;
    const els = {
      meta: document.getElementById('meta'),
      caseList: document.getElementById('caseList'),
      title: document.getElementById('title'),
      videoPath: document.getElementById('videoPath'),
      video: document.getElementById('video'),
      fps: document.getElementById('fps'),
      duration: document.getElementById('duration'),
      miny: document.getElementById('miny')
    };
    function fmt(x) {
      return Number.isFinite(Number(x)) ? Number(x).toFixed(4) : 'n/a';
    }
    function choose(i) {
      current = i;
      const c = summary.videos[i];
      document.querySelectorAll('.case-btn').forEach((btn, idx) => {
        btn.classList.toggle('active', idx === i);
      });
      els.title.textContent = c.key;
      els.videoPath.textContent = c.output;
      els.video.src = c.output.split('/').pop();
      els.fps.textContent = c.fps;
      els.duration.textContent = `${Number(c.duration_sec).toFixed(2)} s`;
      const vals = [];
      for (const row of c.sampled_frames || []) vals.push(row.mesh_min_y);
      els.miny.textContent = vals.length ? `${fmt(Math.min(...vals))} to ${fmt(Math.max(...vals))}` : 'n/a';
      els.video.load();
    }
    fetch('summary.json').then(r => r.json()).then(data => {
      summary = data;
      els.meta.textContent = `${data.videos.length} cases · ${data.video_frames} frames/video · no canonicalization`;
      els.caseList.innerHTML = '';
      data.videos.forEach((c, i) => {
        const btn = document.createElement('button');
        btn.className = 'case-btn';
        btn.textContent = c.key;
        btn.addEventListener('click', () => choose(i));
        els.caseList.appendChild(btn);
      });
      choose(0);
    });
  </script>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")
    return index_path


def parse_fracs(raw: str) -> List[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--frame-fracs cannot be empty")
    return values


def pick_cases(all_cases: List[Dict[str, Any]], wanted: List[str], max_cases: int) -> List[Dict[str, Any]]:
    if wanted:
        by_key = {str(case["key"]): case for case in all_cases}
        missing = [key for key in wanted if key not in by_key]
        if missing:
            raise KeyError(f"case(s) not found in manifest: {missing}")
        return [by_key[key] for key in wanted]
    return all_cases[:max_cases]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Comparison manifest JSON")
    parser.add_argument("--output-dir", default="outputs/temp/motionhub_matplotlib_compare")
    parser.add_argument("--case", action="append", default=[], help="Case key; may repeat")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--frame-fracs", default="0,0.5,0.95")
    parser.add_argument("--max-faces", type=int, default=5000, help="Subsample mesh faces per body")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--render-images", action="store_true", help="Render static PNG panels")
    parser.add_argument("--render-videos", action="store_true", help="Render synchronized MP4 panels")
    parser.add_argument("--video-frames", type=int, default=72)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-max-faces", type=int, default=1800)
    parser.add_argument("--video-dpi", type=int, default=100)
    parser.add_argument("--write-html", action="store_true", help="Write index.html for video playback")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cases = pick_cases(manifest_cases(Path(args.manifest)), args.case, max(1, args.max_cases))
    fracs = parse_fracs(args.frame_fracs)
    models = build_models(device)
    out_dir = Path(args.output_dir)

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(out_dir.resolve()),
        "device": str(device),
        "frame_fracs": fracs,
        "max_faces": args.max_faces,
        "video_frames": args.video_frames,
        "video_fps": args.video_fps,
        "cases": [],
        "videos": [],
    }
    render_images = args.render_images or not args.render_videos
    for idx, case in enumerate(cases, start=1):
        print(f"[render] {idx}/{len(cases)} {case['key']}", flush=True)
        if render_images:
            summary["cases"].append(
                render_case(case, out_dir, fracs, models, device, args.max_faces, args.dpi)
            )
        if args.render_videos:
            summary["videos"].append(
                render_video_case(
                    case,
                    out_dir,
                    max(1, args.video_frames),
                    max(1, args.video_fps),
                    models,
                    device,
                    args.video_max_faces,
                    args.video_dpi,
                )
            )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    index_path = None
    if args.write_html:
        index_path = write_video_index(summary, out_dir)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "index": None if index_path is None else str(index_path),
                "images": [c["output"] for c in summary["cases"]],
                "videos": [c["output"] for c in summary["videos"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

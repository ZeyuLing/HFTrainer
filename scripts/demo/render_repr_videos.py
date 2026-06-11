#!/usr/bin/env python3
"""Render side-by-side demo videos for the multi-representation conversion demo.

Reuses the exact data the web viewer consumes
(``motion_annot_web/repr_convert_demo/data``): HML263 skeleton joints, SMPL/SOMA
mesh vertex binaries + faces, and Unitree-G1 per-link STL meshes + world
transforms. Each rep is rendered in its own panel (own camera fit), all panels
are stitched into one synchronized 30 fps MP4 per case (the "before vs after
retarget" strip used in docs/motion).

Offscreen rendering via pyrender + EGL. Output: docs/motion/assets/repr_demo/.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "motion_annot_web" / "repr_convert_demo" / "data"
VERTS_DIR = DATA_DIR / "verts"
G1_STL_DIR = REPO_ROOT / "ref_repo" / "ProtoMotions" / "protomotions" / "data" / "assets" / "mesh" / "G1"
OUT_DIR = REPO_ROOT / "docs" / "motion" / "assets" / "repr_demo"

PANEL_W, PANEL_H = 600, 680
LABEL_H = 46
BG = np.array([245, 246, 248], np.uint8)


# ----------------------------------------------------------------------------- math
def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def resample_idx(t: int, n_out: int, n_src: int) -> float:
    if n_out <= 1:
        return 0.0
    return t * (n_src - 1) / (n_out - 1)


# ----------------------------------------------------------------------------- panels
class Panel:
    """Base panel: holds a fixed camera fit + renders frame t to an RGB array."""

    def __init__(self, label: str, color):
        self.label = label
        self.color = np.array(color, float)
        self.center = np.zeros(3)
        self.radius = 1.0

    def fit_camera(self, seq: np.ndarray):
        """Fit zoom to the *per-frame* subject extent and store per-frame centers
        so the camera can follow a moving subject (jog/circle) instead of zooming
        out to the whole trajectory. ``seq`` is ``(T, N, 3)``."""
        seq = np.asarray(seq, float)
        if seq.ndim == 2:  # (N,3) single frame
            seq = seq[None]
        lo = seq.min(1)
        hi = seq.max(1)  # (T,3)
        self.centers = (lo + hi) / 2.0  # (T,3)
        ext = hi - lo  # (T,3)
        self.half_h = float(ext[:, 1].max()) / 2.0 + 1e-6
        self.half_w = float(np.maximum(ext[:, 0], ext[:, 2]).max()) / 2.0 + 1e-6
        self.radius = float(np.hypot(self.half_h, self.half_w)) + 1e-6
        self._y_center = float(np.median(self.centers[:, 1]))  # fixed vertical (no bob)
        self.center = self.centers.mean(0)
        self._cur_center = self.center.copy()

    def set_frame_center(self, fidx: float):
        n = len(self.centers)
        a = max(0.0, min(float(fidx), n - 1))
        i0 = int(np.floor(a))
        i1 = min(i0 + 1, n - 1)
        w = a - i0
        c = self.centers[i0] * (1 - w) + self.centers[i1] * w
        c = c.copy()
        c[1] = self._y_center
        self._cur_center = c

    def _cam_pose(self):
        yfov = np.deg2rad(45)
        aspect = PANEL_W / PANEL_H
        # required vertical half-extent to fit both height and (width / aspect)
        half = max(self.half_h, self.half_w / aspect) * 1.12  # tight margin
        c = getattr(self, "_cur_center", self.center)
        dist = half / np.tan(yfov / 2) + self.half_w  # +depth so subject isn't clipped
        eye = c + np.array([0.0, 0.0, dist])
        return look_at(eye, c), yfov

    def _render_scene(self, renderer, geoms):
        # lower ambient so directional shading reveals the 3D surface relief
        scene = pyrender.Scene(bg_color=[*(BG / 255.0), 1.0], ambient_light=[0.28, 0.28, 0.30])
        for g in geoms:
            scene.add(g)
        cam_pose, yfov = self._cam_pose()
        scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=PANEL_W / PANEL_H), pose=cam_pose)
        c = getattr(self, "_cur_center", self.center)
        R = max(self.radius, 1e-3)
        # 3-point rig: lights sit on the +Z (camera/front) side so the subject's
        # front is well lit, but off-axis so shading shows the form (not a flat silhouette).
        key = look_at(c + np.array([-1.0, 1.3, 1.7]) * R, c)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.2), pose=key)
        fill = look_at(c + np.array([1.5, 0.3, 1.1]) * R, c)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.8), pose=fill)
        rim = look_at(c + np.array([0.2, 1.1, -1.6]) * R, c)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.6), pose=rim)
        color, _ = renderer.render(scene)
        return color


class SkeletonPanel(Panel):
    def __init__(self, label, color, positions, edges):
        super().__init__(label, color)
        self.pos = positions  # (T,J,3)
        self.edges = edges
        self.fit_camera(self.pos)

    def geoms(self, t):
        n = self.pos.shape[0]
        a = resample_idx(t, self.n_out, n)
        self.set_frame_center(a)
        i0 = int(np.floor(a))
        i1 = min(i0 + 1, n - 1)
        w = a - i0
        p = self.pos[i0] * (1 - w) + self.pos[i1] * w  # (J,3)
        r = self.radius * 0.035
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(self.color / 255.0), 1.0], roughnessFactor=0.6)
        meshes = []
        sph = trimesh.creation.uv_sphere(radius=r, count=[8, 8])
        tf = np.tile(np.eye(4), (p.shape[0], 1, 1))
        tf[:, :3, 3] = p
        meshes.append(pyrender.Mesh.from_trimesh(sph, material=mat, poses=tf))
        for a_, b_ in self.edges:
            seg = np.stack([p[a_], p[b_]])
            if np.linalg.norm(seg[1] - seg[0]) < 1e-5:
                continue
            cyl = trimesh.creation.cylinder(radius=r * 0.45, segment=seg, sections=6)
            meshes.append(pyrender.Mesh.from_trimesh(cyl, material=mat))
        return meshes


class MeshPanel(Panel):
    def __init__(self, label, color, verts, faces):
        super().__init__(label, color)
        self.verts = verts  # (T,V,3)
        self.faces = faces
        self.fit_camera(self.verts)
        self.mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*(self.color / 255.0), 1.0], roughnessFactor=0.7, metallicFactor=0.0
        )

    def geoms(self, t):
        n = self.verts.shape[0]
        a = resample_idx(t, self.n_out, n)
        self.set_frame_center(a)
        i = min(int(round(a)), n - 1)
        tm = trimesh.Trimesh(self.verts[i], self.faces, process=False)
        return [pyrender.Mesh.from_trimesh(tm, material=self.mat, smooth=True)]


class RobotPanel(Panel):
    def __init__(self, label, color, bodies, frames):
        super().__init__(label, color)
        self.bodies = bodies
        self.frames = frames  # list of {body_pos (33,3), body_quat (33,4 wxyz)}
        # preload STL trimeshes
        self.stl_cache = {}
        for b in bodies:
            for m in b.get("meshes", []):
                f = m["file"]
                if f not in self.stl_cache:
                    p = G1_STL_DIR / f
                    self.stl_cache[f] = trimesh.load(str(p), force="mesh") if p.exists() else None
        seq = np.array([fr["body_pos"] for fr in frames], float)  # (T,nb,3)
        self.fit_camera(seq)
        self.mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*(self.color / 255.0), 1.0], roughnessFactor=0.5, metallicFactor=0.1
        )

    def geoms(self, t):
        n = len(self.frames)
        a = resample_idx(t, self.n_out, n)
        self.set_frame_center(a)
        i = min(int(round(a)), n - 1)
        fr = self.frames[i]
        bp = np.asarray(fr["body_pos"], float)
        bq = np.asarray(fr["body_quat"], float)
        meshes = []
        for bi, b in enumerate(self.bodies):
            world = np.eye(4)
            world[:3, :3] = quat_wxyz_to_mat(bq[bi])
            world[:3, 3] = bp[bi]
            for m in b.get("meshes", []):
                tm = self.stl_cache.get(m["file"])
                if tm is None:
                    continue
                local = np.eye(4)
                local[:3, :3] = quat_wxyz_to_mat(np.asarray(m.get("quat", [1, 0, 0, 0]), float))
                local[:3, 3] = np.asarray(m.get("pos", [0, 0, 0]), float)
                meshes.append(pyrender.Mesh.from_trimesh(tm, material=self.mat, smooth=False, poses=(world @ local)[None]))
        return meshes


# ----------------------------------------------------------------------------- compose
def label_strip(img: np.ndarray, text: str) -> np.ndarray:
    out = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), tuple(int(c) for c in BG))
    out.paste(Image.fromarray(img), (0, LABEL_H))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), text, font=font)
    d.text(((PANEL_W - (tb[2] - tb[0])) / 2, (LABEL_H - (tb[3] - tb[1])) / 2 - tb[1]), text, fill=(30, 35, 45), font=font)
    return np.asarray(out)


def build_panels(case):
    panels = []
    for r in case["reps"]:
        if r["type"] == "skeleton":
            p = SkeletonPanel(r["label"], hex2rgb(r["color"]), np.asarray(r["positions"], np.float32), r["edges"])
        elif r["type"] == "mesh":
            verts = np.fromfile(VERTS_DIR / r["verts_file"], np.float32).reshape(r["num_frames"], r["num_verts"], 3)
            faces = np.frombuffer(base64.b64decode(r["faces_b64"]), np.int32).reshape(-1, 3)
            p = MeshPanel(r["label"], hex2rgb(r["color"]), verts, faces)
        elif r["type"] == "robot":
            p = RobotPanel(r["label"], hex2rgb(r["color"]), r["bodies"], r["frames"])
        else:
            continue
        panels.append(p)
    return panels


def hex2rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def render_case(case_path: Path, fps: int, max_frames: int):
    case = json.loads(case_path.read_text())
    panels = build_panels(case)
    n_out = min(max(p_n_frames(p) for p in panels), max_frames)
    for p in panels:
        p.n_out = n_out

    renderer = pyrender.OffscreenRenderer(PANEL_W, PANEL_H)
    out_path = OUT_DIR / f"{case['id']}.mp4"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=6, macro_block_size=8)
    for t in range(n_out):
        strips = []
        for p in panels:
            try:
                img = p._render_scene(renderer, p.geoms(t))
            except Exception as e:  # noqa: BLE001
                img = np.tile(BG, (PANEL_H, PANEL_W, 1))
                print(f"  [warn] {p.label} frame {t}: {e}")
            strips.append(label_strip(img, p.label.split(" (")[0]))
        frame = np.concatenate(strips, axis=1)
        writer.append_data(frame)
        if t % 30 == 0:
            print(f"  {case['id']} frame {t}/{n_out}")
    writer.close()
    renderer.delete()
    size_mb = out_path.stat().st_size / 1e6
    print(f"[done] {out_path}  ({n_out} frames, {size_mb:.2f} MB)")
    return out_path


def p_n_frames(p) -> int:
    if isinstance(p, SkeletonPanel):
        return p.pos.shape[0]
    if isinstance(p, MeshPanel):
        return p.verts.shape[0]
    if isinstance(p, RobotPanel):
        return len(p.frames)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None, help="case ids; default all in data dir")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=180)
    args = ap.parse_args()

    if args.cases:
        paths = [DATA_DIR / f"case_{c}.json" for c in args.cases]
    else:
        paths = sorted(DATA_DIR.glob("case_*.json"))
    print(f"rendering {len(paths)} case(s) -> {OUT_DIR}")
    for cp in paths:
        render_case(cp, args.fps, args.max_frames)


if __name__ == "__main__":
    main()

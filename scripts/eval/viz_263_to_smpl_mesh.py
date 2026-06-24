#!/usr/bin/env python3
"""Headless SMPL *mesh* renderer for the 263->IK->SMPL eval outputs.

Reads the IK'd SMPL parameters (global_orient / body_pose / transl) saved by
``scripts/eval/hml263_to_smpl_ik.py`` and renders an actual SMPL body mesh
(pyrender + EGL), instead of matplotlib skeleton strips. The skinning matches
what the eval_dashboard Three.js viewer shows for the same params.

Usage:
    python3 scripts/eval/viz_263_to_smpl_mesh.py \
        --smplx-dir output/evaluation/mib_ms272_ikfix/gtctrl/smplx \
        --out-dir   output/evaluation/mib_ms272_ikfix/_viz_mesh \
        --ids 000824 013641 006132
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import trimesh
import pyrender
import imageio.v2 as imageio
import smplx

N_BODY = 23  # SMPL body_pose joints (69 = 23*3)
MESH_COLOR = (0.45, 0.62, 0.92, 1.0)   # soft blue
FLOOR_COLOR = (0.85, 0.85, 0.88, 1.0)


def _resolve_model_dir(model_dir: Path) -> Path:
    cands = []
    if model_dir.name == "body_models":
        cands.append(model_dir.with_name("body_models_nochumpy"))
    cands.append(model_dir)
    for c in cands:
        if (c / "smpl" / "SMPL_NEUTRAL.pkl").exists():
            return c
    return model_dir


def load_smpl(model_dir: str, device):
    md = _resolve_model_dir(Path(model_dir))
    model = smplx.create(str(md), model_type="smpl", gender="neutral",
                         ext="pkl", batch_size=1).to(device)
    model.eval()
    return model


@torch.no_grad()
def smpl_vertices(model, global_orient, body_pose_63, transl, device, bs=64):
    """global_orient (T,3), body_pose_63 (T,63), transl (T,3) -> verts (T,V,3)."""
    T = global_orient.shape[0]
    out = []
    for s in range(0, T, bs):
        e = min(s + bs, T)
        b = e - s
        body69 = np.zeros((b, 69), dtype=np.float32)
        body69[:, :63] = body_pose_63[s:e]
        res = model(
            betas=torch.zeros(b, 10, device=device),
            body_pose=torch.from_numpy(body69).to(device),
            global_orient=torch.from_numpy(global_orient[s:e].astype(np.float32)).to(device),
            transl=torch.from_numpy(transl[s:e].astype(np.float32)).to(device),
        )
        out.append(res.vertices.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _look_at(eye, center, up=(0, 1, 0)):
    eye = np.asarray(eye, np.float32); center = np.asarray(center, np.float32)
    up = np.asarray(up, np.float32)
    f = center - eye; f /= (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = s; m[:3, 1] = u; m[:3, 2] = -f; m[:3, 3] = eye
    return m


def make_renderer(w, h):
    return pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)


def render_frame(renderer, verts, faces, cam_pose, center, w, h):
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=[0.35, 0.35, 0.35])
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=MESH_COLOR, metallicFactor=0.1, roughnessFactor=0.75)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))

    # ground grid
    gsz = 3.0
    ground = trimesh.creation.box(extents=(gsz, 0.01, gsz))
    ground.apply_translation([center[0], -0.005, center[2]])
    gmat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=FLOOR_COLOR, metallicFactor=0.0, roughnessFactor=1.0)
    scene.add(pyrender.Mesh.from_trimesh(ground, material=gmat, smooth=False))

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=w / h)
    scene.add(cam, pose=cam_pose)
    # key + fill lights
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
              pose=_look_at([center[0] + 2, center[1] + 3, center[2] + 2], center))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.6),
              pose=_look_at([center[0] - 2, center[1] + 2, center[2] - 1], center))
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    return color


def viz_clip(model, npz_path, out_dir, device, w, h, nframes, gif_stride, fps):
    z = np.load(npz_path, allow_pickle=True)
    go, bp, tr = z["global_orient"], z["body_pose"], z["transl"]
    mpjpe = float(z["fit_mpjpe_mm"].mean()) if "fit_mpjpe_mm" in z.files else -1.0
    verts = smpl_vertices(model, go, bp, tr, device)        # (T,V,3)
    faces = model.faces.astype(np.int32)

    # floor align: drop lowest vertex to y=0
    verts[..., 1] -= verts[..., 1].min()

    sid = Path(npz_path).stem
    renderer = make_renderer(w, h)
    T = verts.shape[0]
    radius = 2.3   # camera distance (subject-following => keeps body large)

    def _render(i):
        # camera follows the subject: center on this frame's centroid
        c = np.array([verts[i, :, 0].mean(), 0.9, verts[i, :, 2].mean()], np.float32)
        eye = [c[0] + 0.85 * radius, c[1] + 0.5 * radius, c[2] + radius]
        cp = _look_at(eye, [c[0], c[1] * 0.92, c[2]])
        return render_frame(renderer, verts[i], faces, cp, c, w, h)

    # strip of nframes evenly spaced
    sel = np.linspace(0, T - 1, nframes).astype(int)
    strip = np.concatenate([_render(i) for i in sel], axis=1)
    strip_path = os.path.join(out_dir, f"{sid}_mesh_strip.png")
    imageio.imwrite(strip_path, strip)

    # gif over full clip
    frames = [_render(i) for i in range(0, T, gif_stride)]
    gif_path = os.path.join(out_dir, f"{sid}_mesh.gif")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    renderer.delete()
    print(f"[{sid}] T={T} mpjpe={mpjpe:.1f}mm -> {strip_path} ({len(frames)} gif frames)")
    return strip_path, gif_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit npz stems (e.g. 000824); default = first 3")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--nframes", type=int, default=6)
    ap.add_argument("--gif-stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_smpl(args.model_dir, device)

    if args.ids:
        paths = [os.path.join(args.smplx_dir, f"{i}.npz") for i in args.ids]
    else:
        paths = sorted(glob.glob(os.path.join(args.smplx_dir, "*.npz")))[:3]

    for p in paths:
        if not os.path.exists(p):
            print(f"[skip] missing {p}")
            continue
        viz_clip(model, p, args.out_dir, device,
                 args.width, args.height, args.nframes, args.gif_stride, args.fps)


if __name__ == "__main__":
    main()

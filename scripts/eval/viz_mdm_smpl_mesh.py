#!/usr/bin/env python3
"""Render MDM-generated motions as SMPL meshes (sanity check the repro pipeline).

Reads the IK output npz (``global_orient`` / ``body_pose`` / ``transl``) produced by
``scripts/eval/hml263_to_smpl_ik.py`` and renders each clip to an mp4 (+gif) with
a camera that follows the subject. Offscreen via pyrender EGL.

Usage:
    PYOPENGL_PLATFORM=egl python3 scripts/eval/viz_mdm_smpl_mesh.py \
        --npz-dir outputs/evaluation/mdm_h3d272_repro_1000s/mdm_smpl135 \
        --ids 000000,000019,000021 \
        --out-dir outputs/evaluation/mdm_h3d272_repro_1000s/viz_smpl_mesh
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
import pyrender
import imageio.v2 as imageio


def resolve_model_dir(override=None):
    cands = [
        override,
        os.environ.get("HFTRAINER_SMPL_MODEL_DIR"),
        "checkpoints/smpl_models",
        "ref_repo/MDM/body_models_nochumpy",
        "ref_repo/MDM/body_models",
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("no SMPL model dir found")


class SMPLMesh:
    def __init__(self, model_dir, device="cpu"):
        import smplx

        self.device = device
        self.model = smplx.create(
            model_dir, model_type="smpl", gender="neutral",
            batch_size=1, use_pca=False,
        ).to(device)
        self.faces = np.asarray(self.model.faces, dtype=np.int32)

    @torch.no_grad()
    def vertices(self, global_orient, body_pose63, transl, batch=128):
        n = len(global_orient)
        out = []
        for s in range(0, n, batch):
            e = min(s + batch, n)
            bp = np.zeros((e - s, 69), np.float32)
            bp[:, :63] = body_pose63[s:e]
            self.model.batch_size = e - s
            res = self.model(
                body_pose=torch.from_numpy(bp).to(self.device),
                global_orient=torch.from_numpy(np.asarray(global_orient[s:e], np.float32)).to(self.device),
                transl=torch.from_numpy(np.asarray(transl[s:e], np.float32)).to(self.device),
            )
            out.append(res.vertices.cpu().numpy())
        return np.concatenate(out, 0)  # (T, V, 3)


def render_clip(verts, faces, out_mp4, out_gif, fps=30, W=540, H=720, max_frames=240):
    T = min(len(verts), max_frames)
    verts = verts[:T]
    # camera-follow: per-frame xz centroid, fixed y, distance from global bbox
    cen = verts.mean(1)  # (T,3)
    ext = verts.reshape(-1, 3)
    span = (ext.max(0) - ext.min(0))
    half = float(max(span[1], span[0])) / 2 + 0.2
    dist = half / np.tan(np.deg2rad(45) / 2) + 0.6
    y_mid = float((ext[:, 1].max() + ext[:, 1].min()) / 2)

    r = pyrender.OffscreenRenderer(W, H)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45), aspectRatio=W / H)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.36, 0.55, 0.85, 1.0), metallicFactor=0.05, roughnessFactor=0.65,
    )
    frames = []
    for i in range(T):
        scene = pyrender.Scene(bg_color=[0.96, 0.965, 0.972, 1.0], ambient_light=[0.35, 0.35, 0.38])
        tm = trimesh.Trimesh(verts[i], faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))
        cx, cz = float(cen[i, 0]), float(cen[i, 2])
        # camera pose: located at (cx, y_mid, cz+dist) looking at (cx, y_mid, cz)
        pose = np.eye(4)
        pose[:3, 3] = [cx, y_mid, cz + dist]
        scene.add(cam, pose=pose)
        # 3-point lighting
        for (vec, inten) in [((-0.5, 1.0, 1.0), 4.0), ((0.8, 0.3, 0.6), 2.0), ((0.0, 0.8, -1.0), 2.2)]:
            lp = np.eye(4)
            lp[:3, 3] = [cx + vec[0] * 3, y_mid + vec[1] * 3, cz + vec[2] * 3]
            d = np.array([cx, y_mid, cz]) - lp[:3, 3]
            d = d / (np.linalg.norm(d) + 1e-9)
            # build a look-at rotation for the directional light
            up = np.array([0, 1.0, 0])
            zc = -d
            xc = np.cross(up, zc); xc /= np.linalg.norm(xc) + 1e-9
            yc = np.cross(zc, xc)
            lp[:3, :3] = np.stack([xc, yc, zc], 1)
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=inten), pose=lp)
        color, _ = r.render(scene)
        frames.append(color)
    r.delete()

    imageio.mimwrite(out_mp4, frames, fps=fps, quality=8, macro_block_size=1)
    # downscaled gif for inline markdown preview
    gif_frames = [f[::2, ::2] for f in frames]
    imageio.mimwrite(out_gif, gif_frames, fps=min(fps, 20), loop=0)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--ids", default="000000,000019,000021")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-frames", type=int, default=240)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = SMPLMesh(resolve_model_dir(args.model_dir), device=args.device)

    for sid in [x.strip() for x in args.ids.split(",") if x.strip()]:
        p = Path(args.npz_dir) / f"{sid}.npz"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        d = np.load(p, allow_pickle=True)
        fps = float(d["target_fps"]) if "target_fps" in d.files else 30.0
        v = mesh.vertices(d["global_orient"], d["body_pose"], d["transl"])
        mpjpe = float(d["fit_mpjpe_mm"].mean()) if "fit_mpjpe_mm" in d.files else -1
        n = render_clip(
            v, mesh.faces,
            str(out / f"{sid}.mp4"), str(out / f"{sid}.gif"),
            fps=int(round(fps)), max_frames=args.max_frames,
        )
        print(f"[ok] {sid}: T={len(v)} rendered={n} fps={fps:.0f} mpjpe={mpjpe:.1f}mm -> {out/f'{sid}.mp4'}")


if __name__ == "__main__":
    main()

"""Export a HyMotion-V2M inference result into a self-contained web viewer.

Renders the decoded **SMPL-H mesh** (not just a skeleton) next to the original
video.  Produces ``outputs/visualization/v2m_viewer/<name>/`` with:

  - ``mesh.bin``    : float32 [num_frames * num_vertices * 3] world-space verts
  - ``faces.bin``   : uint32  [num_faces * 3] triangle indices (shared by all frames)
  - ``meta.json``   : num_frames / num_vertices / num_faces / fps / floor info
  - ``video.mp4``   : the original input video (copied for side-by-side)
  - ``index.html``  : three.js mesh viewer (left video, right SMPL mesh)

Usage:
    python scripts/visualization/export_v2m_viewer.py \
        --motion outputs/inference/hymotion_v2m/.../motion.npz \
        --video  /path/to/original.mp4 \
        --name   000163 --fps 30
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
HTML_PATH = Path(__file__).resolve().parent / "v2m_viewer_template.html"


def compute_vertices(motion_path: str):
    """Decode SMPL-H mesh vertices for every frame from a motion.npz file."""
    import torch

    from hftrainer.models.motion.hymotion_v2m.vendor.hymotion.bodymodels.smpl_skeleton import (
        SMPLMesh,
    )

    data = np.load(motion_path)
    rot6d = torch.as_tensor(np.asarray(data["rot6d"]), dtype=torch.float32)
    transl = torch.as_tensor(np.asarray(data["transl"]), dtype=torch.float32)
    shapes = torch.as_tensor(np.asarray(data["shapes"]), dtype=torch.float32)

    # strip the leading batch dim (B==1)
    if rot6d.ndim == 4:
        rot6d = rot6d[0]
    if transl.ndim == 3:
        transl = transl[0]
    if shapes.ndim == 3:
        shapes = shapes[0]

    L, J = rot6d.shape[0], rot6d.shape[1]
    # shapes -> (L, n_betas): SMPLMesh.forward auto-repeats a (1, n) shape, but we
    # expand explicitly so trans/rot6d batch dims always line up.
    if shapes.shape[0] != L:
        shapes = shapes[:1].expand(L, shapes.shape[-1]).contiguous()

    mesh = SMPLMesh()
    mesh.eval()
    with torch.no_grad():
        # decode in chunks to keep memory bounded for long sequences
        verts = []
        chunk = 64
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            out = mesh({
                "rot6d": rot6d[s:e],
                "shapes": shapes[s:e],
                "trans": transl[s:e],
            })
            verts.append(out["vertices"].cpu())
        vertices = torch.cat(verts, dim=0).numpy().astype(np.float32)  # (L, V, 3)
    faces = np.asarray(mesh.faces).astype(np.uint32)  # (F, 3)
    return vertices, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="motion.npz from infer")
    ap.add_argument("--video", required=True, help="original input video (.mp4)")
    ap.add_argument("--bbox", default=None, help="tracking bbox npz (bbox_v2.npz)")
    ap.add_argument("--name", default="sample", help="output subdir name")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "outputs/visualization/v2m_viewer"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces = compute_vertices(args.motion)
    T, V, _ = vertices.shape
    F = faces.shape[0]

    # write binaries (little-endian, JS Float32Array / Uint32Array compatible)
    vertices.astype("<f4").tofile(out_dir / "mesh.bin")
    faces.astype("<u4").tofile(out_dir / "faces.bin")

    # floor is at y/z ~ 0 after grounding; expose bbox so the viewer can frame it
    vmin = vertices.reshape(-1, 3).min(axis=0).tolist()
    vmax = vertices.reshape(-1, 3).max(axis=0).tolist()

    meta = {
        "fps": float(args.fps),
        "num_frames": int(T),
        "num_vertices": int(V),
        "num_faces": int(F),
        "bbox_min": [round(float(x), 5) for x in vmin],
        "bbox_max": [round(float(x), 5) for x in vmax],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    shutil.copy(args.video, out_dir / "video.mp4")
    shutil.copy(HTML_PATH, out_dir / "index.html")

    # Optional: per-frame human tracking box, drawn over the video pane.
    if args.bbox and os.path.exists(args.bbox):
        bb = dict(np.load(args.bbox, allow_pickle=True))
        boxes = np.asarray(bb["bbox"], dtype=np.float32)[:, :4]  # (Tb, 4) x1y1x2y2
        se = bb.get("start_end", np.array([0, boxes.shape[0] - 1]))
        vw = vh = None
        try:
            import cv2

            cap = cv2.VideoCapture(str(args.video))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception:
            pass
        bbox_meta = {
            "fps": float(args.fps),
            "video_width": vw,
            "video_height": vh,
            "start_frame": int(np.asarray(se).reshape(-1)[0]),
            "end_frame": int(np.asarray(se).reshape(-1)[-1]),
            "boxes": np.round(boxes, 2).tolist(),
        }
        with open(out_dir / "bbox.json", "w") as f:
            json.dump(bbox_meta, f)
        print(f"  bbox: {boxes.shape[0]} boxes, video={vw}x{vh}")

    print(f"Viewer exported to: {out_dir}")
    print(f"  frames={T} vertices={V} faces={F} fps={args.fps}")
    print(f"  bbox_min={meta['bbox_min']} bbox_max={meta['bbox_max']}")
    print(f"  files: {sorted(os.listdir(out_dir))}")


if __name__ == "__main__":
    main()

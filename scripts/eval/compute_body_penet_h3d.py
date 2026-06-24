#!/usr/bin/env python3
"""Body self-penetration on the SMPL mesh (libigl winding-number proxy).

MBench's Body_Penetration counts self-colliding mesh triangles with a CUDA BVH
(``torch-mesh-isect``), which is not buildable in this env (no nvcc front-end /
CUDA-ABI mismatch). We use an equivalent geometry signal that needs no CUDA: for
a properly embedded closed surface the generalized winding number at any surface
vertex is 0.5; where two body sheets interpenetrate it jumps (>=1.5 or <0). So a
vertex is self-penetrating iff ``|wn - 0.5| > 0.5``. We report the per-frame
percentage of such vertices, averaged over frames and clips (lower = better).

Body_pose comes from each method's own joint rotations (motion_135 / GT-272);
global orient / translation / betas are irrelevant to self-intersection and set
to zero/neutral. Reuses the same manifest as ``compute_phys_h3d.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _ROOT)

_SMPL_DIR = os.path.join(_ROOT, "ref_repo/ViMoGen/data/body_models/smpl")
_FRAME_STEP = 2


def _rot6d_to_rotmat_rowmajor(d6):
    x = d6.reshape(*d6.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _body_axis_angle(path, mode):
    """(T,21,3) axis-angle for SMPL body joints 1..21."""
    from scipy.spatial.transform import Rotation as R
    if mode == "m135":
        d = np.load(path, allow_pickle=True)
        if "motion_135" not in d:
            return None
        m = np.asarray(d["motion_135"], np.float32)
        rot6d = m[:, 3:135].reshape(-1, 22, 6)[:, 1:22]
        rotmat = _rot6d_to_rotmat_rowmajor(rot6d)
    else:
        from hftrainer.datasets.motion.representation.humanml_repr import (
            recover_local_rotations_and_root,
        )
        m = np.load(path)
        if m.ndim != 2 or m.shape[1] != 272:
            return None
        rot, _ = recover_local_rotations_and_root(m)
        rotmat = np.asarray(rot, np.float32)[:, 1:22]
    T = rotmat.shape[0]
    return R.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 21, 3).astype(np.float32)


def _init_worker():
    global _SMPL, _FACES, _torch
    import torch
    import smplx
    _torch = torch
    _SMPL = smplx.SMPL(model_path=_SMPL_DIR, gender="neutral", batch_size=1)
    _SMPL.eval()
    _FACES = np.asarray(_SMPL.faces, dtype=np.int32)


def _verts_from_body_aa(aa):
    """aa (T,21,3) -> SMPL vertices (T,6890,3) (pose-only; orient/transl zeroed)."""
    T = aa.shape[0]
    body = np.zeros((T, 69), np.float32)
    body[:, :63] = aa.reshape(T, 63)                 # joints 1..21; hands(22,23)=0
    with _torch.no_grad():
        out = _SMPL(
            global_orient=_torch.zeros(T, 3),
            body_pose=_torch.from_numpy(body),
            betas=_torch.zeros(T, 10),
            transl=_torch.zeros(T, 3),
        )
    return out.vertices.numpy().astype(np.float64)


def _one(task):
    path, mode = task
    try:
        import igl
        aa = _body_axis_angle(path, mode)
        if aa is None or aa.shape[0] < 2:
            return None
        V = _verts_from_body_aa(aa)                   # (T,6890,3)
        F = _FACES
        eps = float(os.environ.get("BP_EPS", "0.001"))   # outward offset (m)
        pcts = []
        for t in range(0, V.shape[0], _FRAME_STEP):
            Vt = V[t]
            # outward vertex normals; query a point just OUTSIDE each local surface.
            # non-penetrating -> outside the whole body (wn~0); a vertex engulfed by
            # another body part -> inside it (wn~1). Surface singularity avoided.
            N = igl.per_vertex_normals(Vt, F)
            Q = Vt + eps * N
            wn = igl.fast_winding_number_for_meshes(Vt, F, Q)
            pcts.append(float((wn > 0.5).mean()) * 100.0)
        return float(np.mean(pcts)) if pcts else None
    except Exception:  # noqa: BLE001
        return None


def _list_files(src, mode, limit, seed):
    pat = ".npz" if mode == "m135" else ".npy"
    files = sorted(e.path for e in os.scandir(src) if e.name.endswith(pat))
    if limit and len(files) > limit:
        rng = np.random.RandomState(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), limit, False))]
    return files


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--m135-dir")
    g.add_argument("--gt272-dir")
    g.add_argument("--manifest")
    ap.add_argument("--tag", default="method")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.manifest:
        methods = []
        for ln in open(args.manifest):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            tag, mode, d = ln.split("\t")
            methods.append((tag, mode, d))
    elif args.m135_dir:
        methods = [(args.tag, "m135", args.m135_dir)]
    else:
        methods = [(args.tag, "gt272", args.gt272_dir)]

    import multiprocessing as mp
    out = {}
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for tag, mode, d in methods:
            if not os.path.isdir(d):
                print(f"[skip] {tag}: missing {d}", flush=True)
                continue
            tasks = [(f, mode) for f in _list_files(d, mode, args.limit, args.seed)]
            print(f"[bp:{tag}] mode={mode} n_files={len(tasks)}", flush=True)
            vals = [r for r in pool.imap_unordered(_one, tasks, chunksize=2)
                    if r is not None]
            mean = float(np.mean(vals)) if vals else 0.0
            out[tag] = {"n": len(vals), "BodyPenet": mean}
            print(f"[TABLE] {tag}  n={len(vals)}  BodyPenet%={mean:.3f}", flush=True)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=1)
        print(f"[done] -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()

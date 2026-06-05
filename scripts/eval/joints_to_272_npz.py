#!/usr/bin/env python3
"""Encode motion into native MotionStreamer-272 @30fps npz, via the UNIFIED
joints->MoMask-IK->272 chain, so every method (ours + baselines + GT reference)
shares the SAME rotation convention. This is the fair "Protocol A": 272-FID then
reflects motion (position) quality rather than SMPL-vs-IK rotation encoding gap.

Pipeline (common entry = 30fps joints)::

    <input> --> joints@30fps --process_file--> HumanML3D-263
            --decode_263_to_pose--> positions + R_local
            --encode_smpl_to_272--> (T,272)   # SAME encoder as motion135_to_272

Inputs (auto-detected per file or via --input-kind):
  * joints  : <id>.npy of (T,22,3) at --src-fps (resampled to 30)
  * m135    : <id>.npz with key motion_135 (30fps SMPL) -> FK -> joints@30
  * m272    : <id>.npy/.npz native 272 -> SMPL-FK joints@30 (GT reference, clean)

Output: ``<out>/<id>.npz`` with key ``motion_272`` (un-normalized, 30 fps).
"""
from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
for p in (str(REPO), str(REPO / "tools"), str(REPO / "scripts/eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

_FUNCS = {}


def _init_worker(kind="joints"):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from hftrainer.datasets.motion.representation.humanml_repr import (
        joints_to_humanml263, setup_process_globals)
    from convert_momask263_to_h3d272 import decode_263_to_pose
    from motionstreamer_272_encoder import encode_smpl_to_272
    setup_process_globals()
    _FUNCS["j2h"] = joints_to_humanml263
    _FUNCS["dec"] = decode_263_to_pose
    _FUNCS["enc"] = encode_smpl_to_272
    if kind == "m135":  # heavy FK stack only needed for our 135 preds
        from motionstreamer_272_encoder import _canonical_272_offsets
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
        _FUNCS["fk"] = motion135_to_fk
        _FUNCS["bo"] = torch.from_numpy(_canonical_272_offsets()).float()


def _resample(j, src_fps, dst_fps=30.0):
    j = np.asarray(j, dtype=np.float32)
    T = len(j)
    if abs(src_fps - dst_fps) < 1e-6 or T < 2:
        return j
    newT = max(2, int(round(T * dst_fps / src_fps)))
    xs = np.linspace(0.0, T - 1, newT)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, T - 1)
    w = (xs - x0)[:, None, None].astype(np.float32)
    return j[x0] * (1.0 - w) + j[x1] * w


def _load_joints30(fp: Path, kind: str, src_fps: float) -> np.ndarray:
    if kind == "joints":
        j = np.load(str(fp)).astype(np.float32)
        return _resample(j, src_fps, 30.0)
    if kind == "m135":
        import torch
        d = np.load(str(fp), allow_pickle=True)
        if isinstance(d, np.lib.npyio.NpzFile):
            m135 = np.asarray(d["motion_135"], dtype=np.float32)[:, :135]
        else:
            m135 = np.asarray(d, dtype=np.float32)[:, :135]
        world_pos, _wr, _tr, _lr = _FUNCS["fk"](
            torch.from_numpy(m135).float(), _FUNCS["bo"], rotation_space="local")
        return world_pos.detach().cpu().numpy().astype(np.float32)  # (T,22,3) @30fps
    if kind == "m272":
        # native 272 -> SMPL local rot -> FK joints @30 (clean GT joints)
        from hftrainer.datasets.motion.representation.humanml_repr import (
            recover_local_rotations_and_root, fk_smplh_joints, DEFAULT_PATHS)
        m = np.load(str(fp))
        m272 = m["motion_272"] if (str(fp).endswith(".npz")) else m
        rot, root = recover_local_rotations_and_root(np.asarray(m272, np.float32))
        return np.asarray(fk_smplh_joints(rot, root, DEFAULT_PATHS.resolve("smplh_model")),
                          dtype=np.float32)
    raise ValueError(kind)


def _process_one(args_tuple):
    fp, op, kind, src_fps = args_tuple
    if Path(op).exists():
        return ("skip", fp.stem)
    try:
        j30 = _load_joints30(fp, kind, src_fps)
        m263, _ = _FUNCS["j2h"](j30)
        pos, R_all, _ = _FUNCS["dec"](m263)
        m272 = _FUNCS["enc"](pos, R_all).astype(np.float32)
        if not np.isfinite(m272).all():
            return ("bad", fp.stem)
        np.savez(str(op), motion_272=m272)
        return ("ok", fp.stem)
    except Exception as e:  # noqa: BLE001
        return ("fail:%s:%s" % (type(e).__name__, e), fp.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-kind", choices=["joints", "m135", "m272"], default="joints")
    ap.add_argument("--src-fps", type=float, default=20.0)
    ap.add_argument("--ext", default=None, help="file extension to glob (default by kind)")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    indir = Path(args.in_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = args.ext or (".npz" if args.input_kind in ("m135",) else ".npy")

    if args.ids:
        ids = [s.strip() for s in Path(args.ids).read_text().splitlines() if s.strip()]
        files = [indir / f"{i}{ext}" for i in ids]
    else:
        files = sorted(indir.glob(f"*{ext}"))
    files = [f for f in files if f.exists()]
    if args.limit:
        files = files[: args.limit]

    tasks = [(f, out / f"{f.stem}.npz", args.input_kind, args.src_fps) for f in files]
    print(f"[+] {len(tasks)} files, kind={args.input_kind}, workers={args.workers}", flush=True)

    ok = bad = skip = fail = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.input_kind,)) as pool:
        for i, (status, sid) in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=8)):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "bad":
                bad += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  [fail] {sid}: {status}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} (ok={ok} skip={skip} bad={bad} fail={fail})", flush=True)

    print(f"[+] DONE ok={ok} skip={skip} bad={bad} fail={fail} -> {out}", flush=True)


if __name__ == "__main__":
    main()

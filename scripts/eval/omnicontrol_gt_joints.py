#!/usr/bin/env python3
"""Dump HumanML3D-native GT world joints (20fps, ``(T,22,3)``) for the shared
editing clip set, for use as OmniControl spatial-control hints (Table-6 ExpB).

Uses CondMDI's ``new_joint_vecs_abs_3d`` + ``recover_from_ric(abs_3d=True)`` so the
joints live in the exact HumanML3D canonical frame OmniControl was trained on
(xz-centred first frame, facing +z), matching OmniControl's ``Mean_raw``/``Std_raw``
normalisation.  Output ``<out>/<source_id>.npy``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

for _n, _v in {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "str": str, "unicode": str}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)

import torch  # noqa: E402

CONDMDI = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/CondMDI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=196)
    args = ap.parse_args()

    os.chdir(str(CONDMDI))
    sys.path.insert(0, str(CONDMDI))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    abs_dir = CONDMDI / "dataset/HumanML3D/new_joint_vecs_abs_3d"
    mean = np.load(str(CONDMDI / "dataset/HumanML3D/Mean_abs_3d.npy")).astype(np.float32)
    std = np.load(str(CONDMDI / "dataset/HumanML3D/Std_abs_3d.npy")).astype(np.float32)

    sp = Path(args.source_id_file)
    if not sp.is_absolute():
        sp = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer") / sp
    txt = sp.read_text()
    try:
        ids = [str(x) for x in json.loads(txt)]
    except Exception:  # noqa: BLE001
        ids = [s.strip() for s in txt.splitlines() if s.strip()]

    out = Path(args.out)
    if not out.is_absolute():
        out = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer") / out
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for sid in ids:
        p = abs_dir / f"{sid}.npy"
        if not p.exists():
            continue
        m = np.load(str(p)).astype(np.float32)
        L = min(len(m), args.max_frames)
        m = m[:L]
        # un-normalize is NOT needed: new_joint_vecs_abs_3d stores raw 263 abs_3d.
        joints = recover_from_ric(torch.from_numpy(m).float(), 22, abs_3d=True)  # (L,22,3)
        np.save(str(out / f"{sid}.npy"), joints.numpy().astype(np.float32))
        n += 1
    print(f"[gt_joints] wrote {n} clips -> {out}")


if __name__ == "__main__":
    main()

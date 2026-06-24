"""Convert HumanML3D MotionStreamer-272 GT clips -> our model's motion_135
(trans3 + 22x6D rot6d, 30 fps), so HumanML3D test clips can be used as the
SOURCE motion for M2M editing-task evaluation (in-between / keyframe / spatial /
trajectory) on the standard benchmark — making our numbers comparable to
CondMDI / OmniControl / MotionLab / UMO which all evaluate on HumanML3D.

Validated conversion (2026-06-02):
    rot, root = recover_local_rotations_and_root(m272)   # (T,22,3,3),(T,3)
    rot6d     = matrix_to_rotation_6d(rot, convention="ROW")   # NOT "column"!
    motion_135 = concat([root(3), rot6d.reshape(T,132)])
Round-trip motion135_to_272(motion_135) vs m272: MAE 0.002-0.004; model-FK joint
error vs stored 272 positions: 9-18 mm (mean) — essentially lossless (the small
residual is the canonical-skeleton FK approximation).

IMPORTANT: the model rot6d uses the "row" reading here because the model's
``motion135_to_fk`` decoder (differentiable_fk) is row-major; using "column"
gives ~520 mm garbage. Keep this convention.

Usage:
    # build source motion_135 NPZ for all HumanML3D test clips
    python3 scripts/eval/h3d_272_to_135.py \
        --gt-dir /dev/shm/ms272_data/motion_data \
        --split ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt \
        --out-dir data/eval/h3d_editing/source_npz
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

_HUMANML_REPR = None


def _load_humanml_repr():
    """Load the standalone representation module without importing hftrainer packages."""
    global _HUMANML_REPR
    if _HUMANML_REPR is None:
        path = os.path.join(REPO, "hftrainer/datasets/motion/representation/humanml_repr.py")
        spec = importlib.util.spec_from_file_location("_humanml_repr_direct", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _HUMANML_REPR = module
    return _HUMANML_REPR


def _matrix_to_row6d(rot: np.ndarray) -> np.ndarray:
    """Rotation matrices (...,3,3) -> project row convention 6D.

    This matches ``matrix_to_rotation_6d(..., convention="row")`` in
    ``hftrainer.motion.representation.rotation`` without importing hftrainer:
    take the first two matrix columns, then arrange them as
    [R00,R01,R10,R11,R20,R21].
    """
    rot = np.asarray(rot, dtype=np.float32)
    col = np.concatenate([rot[..., 0:3, 0], rot[..., 0:3, 1]], axis=-1)
    return col[..., [0, 3, 1, 4, 2, 5]]


def humanml272_to_motion135(m272: np.ndarray) -> np.ndarray:
    """(T,272) MotionStreamer GT -> (T,135) model motion (trans3 + 22x6D rot6d)."""
    recover_local_rotations_and_root = _load_humanml_repr().recover_local_rotations_and_root
    rot, root = recover_local_rotations_and_root(np.asarray(m272, dtype=np.float32))
    rot = np.asarray(rot, dtype=np.float32)            # (T,22,3,3) local rotmats
    root = np.asarray(root, dtype=np.float32)          # (T,3) pelvis world trans
    d6 = _matrix_to_row6d(rot)
    T = rot.shape[0]
    return np.concatenate([root, d6.reshape(T, 132)], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="/dev/shm/ms272_data/motion_data")
    ap.add_argument("--split", default=os.path.join(
        REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data/eval/h3d_editing/source_npz"))
    ap.add_argument("--max", type=int, default=0, help="0 = all test clips")
    ap.add_argument("--ids", nargs="*", default=None, help="Optional explicit clip ids to convert.")
    ap.add_argument("--id-file", default=None, help="Optional newline-separated clip ids to convert.")
    args = ap.parse_args()

    gt_dir = args.gt_dir
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")
    if args.ids:
        ids = [str(x).strip() for x in args.ids if str(x).strip()]
    elif args.id_file:
        with open(args.id_file) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    else:
        with open(args.split) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    if args.max > 0:
        ids = ids[:args.max]
    os.makedirs(args.out_dir, exist_ok=True)

    ok = skip = 0
    for i, cid in enumerate(ids):
        src = os.path.join(gt_dir, cid + ".npy")
        if not os.path.exists(src):
            skip += 1
            continue
        try:
            m272 = np.load(src).astype(np.float32)
            m135 = humanml272_to_motion135(m272)
            np.savez(os.path.join(args.out_dir, cid + ".npz"),
                     motion_135=m135, source_id=cid)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {cid}: {type(e).__name__}: {e}")
            skip += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ids)}  ok={ok} skip={skip}")
    print(f"[done] {ok} motion_135 NPZ -> {args.out_dir}  (skipped {skip})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert SMPL-85 (T, 85) to MotionStreamer's 272-dim representation.

Pipeline per file:
    1. Load smpl_85 = [pose_72, trans_3, beta_10] @ 30 fps.
    2. ``face_z_transform``: rotate first frame so root forward (-z axis) faces Z+,
       and rotate trans accordingly.  Mirrors MotionStreamer's ``face_z_transform.py``.
    3. SMPL-X forward kinematics -> joint positions (T, 22, 3) global, matching
       MotionStreamer's official ``infer_get_joints.py`` path.
    4. ``representation_272`` forward path: pack into (T, 272).  Mirrors
       MotionStreamer's ``representation_272.py``.

Outputs:
    <out_dir>/<id>.npy  shape (T, 272), in native units (NOT pre-standardized).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hftrainer.motion.representation.motion272 import (  # noqa: E402
    encode_smpl_to_272,
    face_z_transform_smpl85 as _face_z_transform_smpl85,
    fk_smpl85_joints,
    smpl85_to_local_rotmat,
)


# ---------------------------------------------------------------------------
# face_z_transform: align first-frame root orientation to face +Z
# ---------------------------------------------------------------------------

def face_z_transform_smpl85(smpl_85: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the public library API."""
    return _face_z_transform_smpl85(smpl_85)


# ---------------------------------------------------------------------------
# SMPL forward kinematics
# ---------------------------------------------------------------------------

def smpl_fk(smpl_85: np.ndarray, smpl_path: Path, device: torch.device,
            smpl_model=None, model_type: str = "smplx",
            fixed_batch_size: int = 0) -> np.ndarray:
    """Backward-compatible wrapper around ``fk_smpl85_joints``."""
    return fk_smpl85_joints(
        smpl_85,
        smpl_model_dir=str(smpl_path),
        model_type=model_type,
        device=str(device),
        batch_size=fixed_batch_size if fixed_batch_size > 0 else len(smpl_85),
        model=smpl_model,
        return_model=True,
    )


# ---------------------------------------------------------------------------
# representation_272 forward path
# ---------------------------------------------------------------------------

def rot_yaw(yaw: float) -> np.ndarray:
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def representation_272_forward(joints: np.ndarray, smpl_85_face_z: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around ``encode_smpl_to_272``."""
    return encode_smpl_to_272(joints, smpl85_to_local_rotmat(smpl_85_face_z)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smpl85_dir", required=True)
    p.add_argument("--out_dir_272", required=True)
    p.add_argument(
        "--smpl_path",
        default=str(REPO_ROOT / "checkpoints" / "smpl_models"),
    )
    p.add_argument(
        "--model_type",
        default="smplx",
        choices=["smpl", "smplx"],
        help=(
            "FK body model. MotionStreamer's official pipeline uses 'smplx' "
            "(infer_get_joints.py), so 'smplx' is the default to stay "
            "distribution-faithful with the evaluator."
        ),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    src = Path(args.smpl85_dir)
    dst = Path(args.out_dir_272)
    dst.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    files = [f for f in sorted(src.glob("*.npy")) if ".tmp." not in f.name]
    if args.max_samples is not None:
        files = files[: args.max_samples]
    print(f"[+] {len(files)} input smpl_85 files in {src}")

    # Quick first pass: read shapes to find max sequence length, so we can
    # build SMPL-X model exactly once with that batch size.
    max_T = 0
    for f in files:
        try:
            T = int(np.load(str(f), mmap_mode="r").shape[0])
            max_T = max(max_T, T)
        except Exception:
            pass
    print(f"[+] max seq length = {max_T}")

    smpl_model = None
    n_ok = n_err = 0
    for f in tqdm(files, ncols=80):
        out_file = dst / f.name
        if out_file.exists():
            n_ok += 1
            continue
        try:
            smpl_85 = np.load(str(f))
            if smpl_85.shape[1] != 85 or len(smpl_85) < 4:
                n_err += 1
                continue
            smpl_85_fz = face_z_transform_smpl85(smpl_85)
            joints, smpl_model = smpl_fk(
                smpl_85_fz, Path(args.smpl_path), device, smpl_model,
                model_type=args.model_type,
                fixed_batch_size=max_T,
            )
            m272 = representation_272_forward(joints, smpl_85_fz)
            np.save(str(out_file), m272)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [!] {f.name}: {e}", flush=True)

    print(f"[+] wrote {n_ok}/{len(files)} files to {dst} ({n_err} errors)")


if __name__ == "__main__":
    main()

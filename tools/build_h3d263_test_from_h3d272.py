#!/usr/bin/env python3
"""Build a HumanML3D-263 (20 fps) test set from humanml3d_272 (30 fps) data.

Pipeline per clip (thin CLI driver over
``hftrainer.datasets.motion.representation.humanml_repr.humanml272_to_humanml263``)::

    272 (T_30, 272)
      -> SMPL-H FK joints (--joints_from smpl_fk, default) -> (T_30, 22, 3)
      -> linear_resample_positions 30->20 -> joints (T_20, 22, 3)
      -> MoMask process_file -> m263 (T_20-1, 263)
      -> recover_from_ric -> new_joints (T_20-1, 22, 3)

The 263 features use HumanML3D's canonical layout, so they can be fed directly
to the official ``text_mot_match`` evaluator (after standardisation with the
official Mean/Std). The output directory mirrors HumanML3D's structure and the
input expected by ``scripts/eval/eval_momask_native_h3d263.py``::

    <out>/new_joint_vecs/<id>.npy   (T, 263)
    <out>/new_joints/<id>.npy        (T, 22, 3)
    <out>/Mean.npy /Std.npy          (263,)
    <out>/test.txt                   valid ids

Usage::

    python3 tools/build_h3d263_test_from_h3d272.py \
        --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --out work_dirs/h3d263_eval/h3d263_test_recon \
        --mean_std_dir ref_repo/MDM/t2m/text_mot_match/meta
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOMASK = _REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"
if _MOMASK.is_dir() and str(_MOMASK) not in sys.path:
    sys.path.insert(0, str(_MOMASK))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
if str(_REPO_ROOT) not in sys.path:  # allow `python3 tools/build_...py` to find hftrainer
    sys.path.insert(0, str(_REPO_ROOT))

from convert_momask263_to_h3d272 import linear_resample_positions  # noqa: E402

# Canonical conversion lives in hftrainer; this CLI is a thin driver over it.
import hftrainer.datasets.motion.representation.humanml_repr as _hml  # noqa: E402

_NJOINT = 22


# ======================= 272 -> global joints ===============================

# Legacy "stored positions" decode (kept for the fallback canonical-skeleton
# path); canonical implementation lives in the hftrainer module.
decode_272_to_global_positions = _hml.recover_272_stored_positions


# ======================= MoMask process_file globals ========================

def _setup_motion_process_globals(ref_first_frame_pos: np.ndarray):
    """Configure MoMask process_file globals (delegates to hftrainer)."""
    _hml.setup_process_globals(canonical_ref_joints=ref_first_frame_pos)


def m272_file_to_263(m272: np.ndarray, src_fps: float, dst_fps: float,
                     joints_from: str = "smpl_fk"):
    """272 -> (m263, new_joints). Thin wrapper over the hftrainer module."""
    return _hml.humanml272_to_humanml263(
        m272, src_fps=src_fps, dst_fps=dst_fps, joints_from=joints_from,
        ensure_globals=False,  # globals already set by main() via setup_process_globals
    )


# ============================== CLI =========================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_h3d272", required=True,
                   help="humanml3d_272 root (with motion_data/, split/test.txt).")
    p.add_argument("--out", required=True, help="output recon root.")
    p.add_argument("--split", default=None,
                   help="split file of ids (default: <src>/split/test.txt).")
    p.add_argument("--ref_id", default="000021",
                   help="id used to set the canonical target skeleton offsets "
                        "(only used as a fallback when --canonical_ref_joints is unset).")
    p.add_argument("--canonical_ref_joints",
                   default="ref_repo/TeSMo/dataset/HumanML3D/000021.npy",
                   help="Path to OFFICIAL HumanML3D canonical joints (T, >=22, 3). Its "
                        "first frame defines the uniform_skeleton target so the "
                        "reconstructed 263 matches the official skeleton scale (required "
                        "for the official Mean/Std + evaluator to be valid). If the file "
                        "is missing, falls back to the 272-decoded --ref_id skeleton "
                        "(NOT recommended: introduces a global skeleton-scale mismatch).")
    p.add_argument("--src_fps", type=float, default=30.0)
    p.add_argument("--dst_fps", type=float, default=20.0)
    p.add_argument("--mean_std_dir", default=None,
                   help="dir containing official mean.npy/std.npy (263). If None, "
                        "computed from the reconstructed set (NOT recommended for "
                        "the official evaluator).")
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--joints_from", choices=["smpl_fk", "positions"], default="smpl_fk",
                   help="Joint source: 'smpl_fk' (recover SMPL rotations + SMPL-H FK, "
                        "matches the official HumanML3D joint source -- recommended) or "
                        "'positions' (272 stored positions; legacy, ~30mm off official).")
    args = p.parse_args()

    src = Path(args.src_h3d272)
    motion_dir = src / "motion_data"
    out = Path(args.out)
    (out / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (out / "new_joints").mkdir(parents=True, exist_ok=True)

    split_file = Path(args.split) if args.split else (src / "split" / "test.txt")
    ids = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]
    if args.max_clips:
        ids = ids[:args.max_clips]
    print(f"[+] {len(ids)} ids from {split_file}  (joints_from={args.joints_from})")

    # canonical target skeleton (uniform_skeleton target). Prefer the OFFICIAL
    # HumanML3D canonical joints so the reconstructed 263 lands on the exact
    # skeleton scale the official Mean/Std + evaluator expect.
    ref_path = Path(args.canonical_ref_joints) if args.canonical_ref_joints else None
    if ref_path is not None and ref_path.exists():
        ref_joints_off = np.load(str(ref_path))[:, :_NJOINT, :].astype(np.float64)
        _setup_motion_process_globals(ref_joints_off[0])
        print(f"[+] canonical target skeleton from OFFICIAL joints: {ref_path}")
    else:
        ref_m272 = np.load(str(motion_dir / f"{args.ref_id}.npy"))
        ref_joints30 = decode_272_to_global_positions(ref_m272)
        ref_joints20 = linear_resample_positions(ref_joints30, args.src_fps, args.dst_fps)
        _setup_motion_process_globals(ref_joints20[0])
        print(f"[!] OFFICIAL canonical joints not found at {args.canonical_ref_joints}; "
              f"using 272-decoded {args.ref_id} skeleton (scale mismatch likely).")

    ok, fail = [], 0
    all_vecs = []
    for sid in tqdm(ids, ncols=80, desc="272->263"):
        f = motion_dir / f"{sid}.npy"
        if not f.exists():
            fail += 1
            continue
        try:
            m272 = np.load(str(f))
            if m272.ndim != 2 or m272.shape[1] != 272 or len(m272) < 8:
                fail += 1
                continue
            m263, new_joints = m272_file_to_263(m272, args.src_fps, args.dst_fps,
                                                joints_from=args.joints_from)
            if not np.isfinite(m263).all():
                fail += 1
                continue
            np.save(str(out / "new_joint_vecs" / f"{sid}.npy"), m263)
            np.save(str(out / "new_joints" / f"{sid}.npy"), new_joints)
            ok.append(sid)
            all_vecs.append(m263)
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {sid}: {e}")
            fail += 1

    (out / "test.txt").write_text("\n".join(ok) + "\n")
    print(f"[+] converted {len(ok)} clips, failed {fail}")

    # Mean/Std
    if args.mean_std_dir:
        msd = Path(args.mean_std_dir)
        mean_src = msd / "mean.npy" if (msd / "mean.npy").exists() else msd / "Mean.npy"
        std_src = msd / "std.npy" if (msd / "std.npy").exists() else msd / "Std.npy"
        shutil.copy(str(mean_src), str(out / "Mean.npy"))
        shutil.copy(str(std_src), str(out / "Std.npy"))
        print(f"[+] copied official Mean/Std from {msd}")
    else:
        cat = np.concatenate(all_vecs, axis=0)
        mean = cat.mean(axis=0)
        std = cat.std(axis=0)
        std[std < 1e-6] = 1e-6
        np.save(str(out / "Mean.npy"), mean.astype(np.float32))
        np.save(str(out / "Std.npy"), std.astype(np.float32))
        print("[!] computed Mean/Std from recon set (official evaluator expects "
              "official stats; pass --mean_std_dir for correct normalisation).")


if __name__ == "__main__":
    main()

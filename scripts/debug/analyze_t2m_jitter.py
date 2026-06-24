#!/usr/bin/env python3
"""Quantify per-method temporal jitter on the HumanML3D test set to locate
HY-Motion's reported jitter.

For GT / HY-Motion / Go-to-Zero (all raw 272), per sample we decode two joint
streams and the rotation channel, then report mean acceleration / jerk:

* stored positions   : recover_272_stored_positions(m272)  -> (T,22,3)
* FK joints (viewer) : recover local rot + root -> row6d 135 -> motion135_to_fk
* rotation channel   : 2nd-order diff of local rotation matrices (rad-ish)

If FK-joint jitter >> stored-position jitter for HY, the jitter lives in the
rotation channel (model rot6d or the 135<->272 round-trip), not in positions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _accel(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(x, n=2, axis=0), axis=-1).mean())


def _jerk(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hy_dir", default="outputs/evaluation/hymotion_h3d272/hy_272")
    ap.add_argument("--g2z_dir", default="outputs/evaluation/motionmillion_h3d272/mm_272_len150")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--bone_offsets", default="scripts/eval/assets/bone_offsets_canon272.npy")
    args = ap.parse_args()

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    try:
        from hftrainer.datasets.motion.representation.humanml_repr import (
            recover_272_stored_positions, recover_local_rotations_and_root,
        )
    except Exception:
        from hftrainer.models.motion.components.utils.humanml_repr import (
            recover_272_stored_positions, recover_local_rotations_and_root,
        )
    from hftrainer.pipelines.motion.differentiable_fk import (
        motion135_to_fk, rotmat_to_rot6d_row_major,
    )

    bo = torch.from_numpy(np.load(args.bone_offsets).astype(np.float32))
    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()
    hy_dir, g2z_dir = Path(args.hy_dir), Path(args.g2z_dir)

    def fk_joints(m272):
        rot, root = recover_local_rotations_and_root(m272)
        row6d = rotmat_to_rot6d_row_major(torch.from_numpy(np.asarray(rot, np.float32)))
        m135 = torch.cat([torch.from_numpy(np.asarray(root, np.float32)),
                          row6d.reshape(row6d.shape[0], 132)], dim=-1)
        wp, _, _, _ = motion135_to_fk(m135, bo, rotation_space="local")
        return wp.numpy().astype(np.float32), np.asarray(rot, np.float32)

    acc = {m: {"stored_acc": [], "stored_jerk": [], "fk_acc": [], "fk_jerk": [],
               "rot_acc": [], "root_jerk": [], "rel_jerk": []}
           for m in ("GT", "HY-Motion", "Go-to-Zero")}
    seen = set()
    for idx, (name, caption, gt, ml) in enumerate(pairs):
        if name in seen:
            continue
        hf = hy_dir / f"{idx:06d}.npy"
        gf = g2z_dir / f"{idx:06d}.npy"
        if not (hf.exists() and gf.exists()):
            continue
        seen.add(name)
        if len(seen) > args.n:
            break
        for label, m in (("GT", np.asarray(gt, np.float32)),
                         ("HY-Motion", np.load(hf).astype(np.float32)),
                         ("Go-to-Zero", np.load(gf).astype(np.float32))):
            sp = recover_272_stored_positions(m)
            fk, rot = fk_joints(m)
            acc[label]["stored_acc"].append(_accel(sp))
            acc[label]["stored_jerk"].append(_jerk(sp))
            acc[label]["fk_acc"].append(_accel(fk))
            acc[label]["fk_jerk"].append(_jerk(fk))
            acc[label]["rot_acc"].append(
                float(np.linalg.norm(np.diff(rot.reshape(rot.shape[0], -1), n=2, axis=0), axis=-1).mean()))
            _, root = recover_local_rotations_and_root(m)
            root = np.asarray(root, np.float32)
            acc[label]["root_jerk"].append(_jerk(root[:, None, :]))
            acc[label]["rel_jerk"].append(_jerk(fk - fk[:, :1]))  # pose relative to pelvis
            acc[label].setdefault("rootXZ_jerk", []).append(
                _jerk(root[:, None, [0, 2]]))
            acc[label].setdefault("rootY_jerk", []).append(
                _jerk(root[:, None, [1]]))

    print(f"\n=== T2M jitter ({len(seen)} samples; lower=smoother) ===")
    cols = ["fk_jerk", "root_jerk", "rootXZ_jerk", "rootY_jerk", "rel_jerk", "rot_acc"]
    hdr = "method".ljust(12) + "".join(c.rjust(12) for c in cols)
    print(hdr)
    for m in ("GT", "HY-Motion", "Go-to-Zero"):
        row = m.ljust(12) + "".join(f"{np.mean(acc[m][c]):>12.5f}" for c in cols)
        print(row)
    # ratios vs GT
    print("\n--- ratio vs GT (x worse) ---")
    for m in ("HY-Motion", "Go-to-Zero"):
        row = m.ljust(12) + "".join(
            f"{np.mean(acc[m][c]) / max(np.mean(acc['GT'][c]), 1e-9):>12.2f}" for c in cols)
        print(row)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prove whether the 135<->272 round-trip is jitter-neutral.

HY-Motion goes model-135 -> motion135_to_272 -> 272 (-> viewer 272->135->FK),
i.e. one extra encode round-trip vs Go-to-Zero (native 272). To decide whether
the observed root jitter is a model defect or a conversion artifact, we take GT
(native 272, known smooth) and pass it through the SAME extra round-trip:

    gt272 -> (rot,root) -> 135 -> motion135_to_272 -> rt272 -> (rot,root) -> FK

If the round-tripped GT root jitter stays ~= native GT, the conversion is proven
jitter-neutral and HY's jitter must originate from the model output itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _jerk(x):
    return float(np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--bone_offsets", default="scripts/eval/assets/bone_offsets_canon272.npy")
    args = ap.parse_args()

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    try:
        from hftrainer.datasets.motion.representation.humanml_repr import recover_local_rotations_and_root
    except Exception:
        from hftrainer.models.motion.components.utils.humanml_repr import recover_local_rotations_and_root
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk, rotmat_to_rot6d_row_major
    from hftrainer.motion.representation.convert import motion135_to_motion272

    bo = torch.from_numpy(np.load(args.bone_offsets).astype(np.float32))

    def to135(m272):
        rot, root = recover_local_rotations_and_root(m272)
        row6d = rotmat_to_rot6d_row_major(torch.from_numpy(np.asarray(rot, np.float32)))
        return torch.cat([torch.from_numpy(np.asarray(root, np.float32)),
                          row6d.reshape(row6d.shape[0], 132)], dim=-1)

    def fk_root_jerk(m272):
        rot, root = recover_local_rotations_and_root(m272)
        row6d = rotmat_to_rot6d_row_major(torch.from_numpy(np.asarray(rot, np.float32)))
        m135 = torch.cat([torch.from_numpy(np.asarray(root, np.float32)),
                          row6d.reshape(row6d.shape[0], 132)], dim=-1)
        wp, _, _, _ = motion135_to_fk(m135, bo, rotation_space="local")
        fk = wp.numpy().astype(np.float32)
        return _jerk(fk), _jerk(np.asarray(root, np.float32)[:, None, :])

    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()

    nat_fk, nat_root, rt_fk, rt_root = [], [], [], []
    seen = set()
    for idx, (name, cap, gt, ml) in enumerate(pairs):
        if name in seen:
            continue
        seen.add(name)
        if len(seen) > args.n:
            break
        gt = np.asarray(gt, np.float32)
        fk_j, root_j = fk_root_jerk(gt)
        nat_fk.append(fk_j); nat_root.append(root_j)
        # round-trip: gt272 -> 135 -> 272 -> measure
        m135 = to135(gt).numpy().astype(np.float32)
        rt272 = motion135_to_motion272(m135, rotation_space="local", skeleton="canon272")
        fk_j2, root_j2 = fk_root_jerk(np.asarray(rt272, np.float32))
        rt_fk.append(fk_j2); rt_root.append(root_j2)

    print(f"\n=== 135<->272 round-trip jitter test ({len(seen)} GT samples) ===")
    print(f"{'variant':<22}{'fk_jerk':>12}{'root_jerk':>12}")
    print(f"{'GT native (272)':<22}{np.mean(nat_fk):>12.5f}{np.mean(nat_root):>12.5f}")
    print(f"{'GT round-trip':<22}{np.mean(rt_fk):>12.5f}{np.mean(rt_root):>12.5f}")
    print(f"{'ratio (rt/native)':<22}{np.mean(rt_fk)/max(np.mean(nat_fk),1e-9):>12.2f}"
          f"{np.mean(rt_root)/max(np.mean(nat_root),1e-9):>12.2f}")


if __name__ == "__main__":
    main()

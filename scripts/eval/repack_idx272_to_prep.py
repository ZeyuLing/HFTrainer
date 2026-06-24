#!/usr/bin/env python3
"""Repack per-pair 272-dim T2M predictions (``<idx:06d>.npy`` keyed by the
MotionStreamer-272 evaluator pair order) into the ``t2m_compare`` viewer format:
``<canonical_id>.npz`` with a ROW-major ``motion_135`` field.

``canonical_id`` == the HumanML3D test ``name`` returned by
``MotionStreamer272Evaluator.load_test_pairs()`` (same id space the existing prep
dirs under ``ms272_tables_h3d_0607/prep`` use), so the repacked method shares the
viewer's common-id intersection with GT / MotionStreamer / etc.

For each ``name`` we keep the *first* pair (first caption) and also dump a
``captions.json`` mapping ``name -> caption`` (the actual prompt used to generate
that clip), so the viewer can show the true generation prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _recover():
    try:
        from hftrainer.datasets.motion.representation.humanml_repr import (
            recover_local_rotations_and_root,
        )
    except Exception:
        from hftrainer.models.motion.components.utils.humanml_repr import (
            recover_local_rotations_and_root,
        )
    from hftrainer.pipelines.motion.differentiable_fk import rotmat_to_rot6d_row_major

    return recover_local_rotations_and_root, rotmat_to_rot6d_row_major


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="dir of <idx:06d>.npy raw-272 preds")
    ap.add_argument("--out_dir", required=True, help="output prep dir for <name>.npz")
    args = ap.parse_args()

    import torch

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    recover_local_rotations_and_root, rotmat_to_rot6d_row_major = _recover()

    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    captions: dict[str, str] = {}
    ok = miss = skip = 0
    for idx, (name, caption, gt, ml) in enumerate(pairs):
        if name in seen:
            continue
        pf = pred_dir / f"{idx:06d}.npy"
        if not pf.exists():
            miss += 1
            continue
        seen.add(name)
        out_f = out_dir / f"{name}.npz"
        captions[name] = caption
        if out_f.exists():
            skip += 1
            continue
        m272 = np.load(pf).astype(np.float32)
        rot, root = recover_local_rotations_and_root(m272)  # (T,22,3,3), (T,3)
        row6d = rotmat_to_rot6d_row_major(torch.from_numpy(np.asarray(rot, np.float32)))
        m135 = np.concatenate(
            [np.asarray(root, np.float32), row6d.reshape(row6d.shape[0], 132).numpy()],
            axis=-1,
        ).astype(np.float32)
        np.savez(out_f, motion_135=m135)
        ok += 1
        if ok % 500 == 0:
            print(f"  {ok} written (miss={miss} skip={skip})", flush=True)

    json.dump(captions, open(out_dir / "captions.json", "w"))
    print(f"DONE ok={ok} skip={skip} miss={miss} unique_names={len(seen)} -> {out_dir}")


if __name__ == "__main__":
    main()

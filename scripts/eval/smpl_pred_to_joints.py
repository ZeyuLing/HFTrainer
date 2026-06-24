#!/usr/bin/env python3
"""Convert SMPL pred dicts (FlowMDM / DoubleTake pred.npz) to (T,22,3) world joints
via true SMPL-H forward kinematics, matching the KIMODO joints path so all foreign
methods reach the evaluator through the SAME joints->272 canonicalization.

Input dir: flattened <id>.npz each with global_orient(T,3), body_pose(T,63), transl(T,3).
Output dir: <id>.npy (T,22,3) @30fps.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)

from hftrainer.datasets.motion.representation.humanml_repr import fk_smplh_joints  # noqa: E402
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.npz"))[args.shard_index::args.num_shards]
    ok = fail = skip = 0
    for fp in files:
        op = out / f"{fp.stem}.npy"
        if args.skip_existing and op.exists():
            skip += 1; continue
        try:
            d = np.load(str(fp), allow_pickle=True)
            go = np.asarray(d["global_orient"], np.float32).reshape(-1, 1, 3)
            bp = np.asarray(d["body_pose"], np.float32).reshape(go.shape[0], 21, 3)
            transl = np.asarray(d["transl"], np.float32).reshape(-1, 3)
            aa = np.concatenate([go, bp], axis=1)  # (T,22,3)
            R = axis_angle_to_matrix(torch.from_numpy(aa)).numpy()  # (T,22,3,3)
            joints = fk_smplh_joints(R, transl)  # (T,22,3)
            joints = np.asarray(joints, np.float32)
            if not np.isfinite(joints).all():
                raise ValueError("non-finite joints")
            np.save(str(op), joints)
            ok += 1
            if ok % 200 == 0:
                print(f"[smpl2joints] ok={ok} skip={skip} fail={fail}", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[smpl2joints] FAIL {fp.stem}: {e}", flush=True)
    print(f"[smpl2joints] DONE ok={ok} skip={skip} fail={fail} -> {out}", flush=True)


if __name__ == "__main__":
    main()

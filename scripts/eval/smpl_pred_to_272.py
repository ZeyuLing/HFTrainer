#!/usr/bin/env python3
"""Faithful SMPL-pred -> MS272 via encode_smpl_to_272(true joints, true local rot).

This skips BOTH the 135 round-trip and the 263/IK round-trip; it feeds the
generator's own SMPL rotations + true SMPL-H FK joints straight into the SAME
canonicalizing encoder that produced the GT val_stream 272. Use for FlowMDM /
DoubleTake (SMPL output) so their 272-FID reflects motion quality, not encoding.

Input dir: flattened <id>.npz with global_orient(T,3), body_pose(T,63), transl(T,3).
Output: <out>/<id>.npz with key motion_272 (un-normalized, 30fps).
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
from hftrainer.motion.representation.motion272 import encode_smpl_to_272  # noqa: E402
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
    ap.add_argument("--zup-to-yup", action="store_true",
                    help="FlowMDM/DoubleTake (Globalvelandy) SMPL is in AMASS Z-up; "
                         "rotate root orient + transl into the Y-up frame expected by "
                         "encode_smpl_to_272. The harness _rot6d_features_to_smpl omits this.")
    ap.add_argument("--transpose-rot", action="store_true",
                    help="The VM harness used mmotion rotation_6d_to_matrix, which is the "
                         "TRANSPOSE of the official priormdm to_matrix convention. Undo it "
                         "by transposing every joint rotation back to the true orientation.")
    args = ap.parse_args()

    # Z-up (AMASS) -> Y-up: (x,y,z) -> (x, z, -y)
    Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)

    in_dir = Path(args.in_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.npz"))[args.shard_index::args.num_shards]
    ok = fail = skip = 0
    for fp in files:
        op = out / f"{fp.stem}.npz"
        if args.skip_existing and op.exists():
            skip += 1; continue
        try:
            d = np.load(str(fp), allow_pickle=True)
            go = np.asarray(d["global_orient"], np.float32).reshape(-1, 1, 3)
            bp = np.asarray(d["body_pose"], np.float32).reshape(go.shape[0], 21, 3)
            transl = np.asarray(d["transl"], np.float32).reshape(-1, 3)
            aa = np.concatenate([go, bp], axis=1)  # (T,22,3)
            R = axis_angle_to_matrix(torch.from_numpy(aa)).numpy().astype(np.float32)  # (T,22,3,3)
            if args.transpose_rot:
                R = np.swapaxes(R, -1, -2).astype(np.float32)
            if args.zup_to_yup:
                # harness transl=[traj_x, root_y, traj_y]; AMASS trans=[traj_x, traj_y, root_y].
                # Y-up transl = Rx @ AMASS_trans = [traj_x, root_y, -traj_y].
                transl = np.stack([transl[:, 0], transl[:, 1], -transl[:, 2]], axis=-1).astype(np.float32)
                R[:, 0] = np.einsum("ij,tjk->tik", Rx, R[:, 0]).astype(np.float32)
            joints = np.asarray(fk_smplh_joints(R, transl), np.float32)  # (T,22,3)
            m272 = encode_smpl_to_272(joints, R).astype(np.float32)
            if not np.isfinite(m272).all():
                raise ValueError("non-finite 272")
            np.savez(str(op), motion_272=m272)
            ok += 1
            if ok % 200 == 0:
                print(f"[smpl2272] ok={ok} skip={skip} fail={fail}", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[smpl2272] FAIL {fp.stem}: {e}", flush=True)
    print(f"[smpl2272] DONE ok={ok} skip={skip} fail={fail} -> {out}", flush=True)


if __name__ == "__main__":
    main()

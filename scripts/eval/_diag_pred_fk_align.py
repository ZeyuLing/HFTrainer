"""Diagnose coordinate/FK alignment of OUR pred 272-path vs the GT 272 forward.

For N GT test clips we feed the GT's OWN rotations (decoded from GT272) through
the EXACT prediction encoding path (``motion135_to_272`` -> ``differentiable_fk``
with the canon272 skeleton) and compare the resulting 272 to the ground-truth
272, block-by-block. Because the model is removed from the loop, any residual is
purely our FK / coordinate / joint-definition mismatch.

We also report the SMPL-H FK roundtrip (``reencode_272_via_fk``) used by the
``--also-refk`` real baseline, so the two FK paths can be compared head-to-head.
"""
import os
import sys

import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

GT_DIR = "/dev/shm/ms272_data/motion_data"

from motionstreamer_272_encoder import (  # noqa: E402
    motion135_to_272, reencode_272_via_fk, reencode_272_via_stored_positions,
)
from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_local_rotations_and_root,
)
from hftrainer.pipelines.motion.differentiable_fk import (  # noqa: E402
    rotmat_to_rot6d_row_major,
)

BLOCKS = {
    "root_vel[0:2]": (0, 2),
    "heading[2:8]": (2, 8),
    "pos[8:74]": (8, 74),
    "vel[74:140]": (74, 140),
    "rot[140:272]": (140, 272),
}


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def blocks(a, b):
    return {k: rel(a[:, s:e], b[:, s:e]) for k, (s, e) in BLOCKS.items()}


def pred_path_roundtrip(m272):
    """GT272 -> (root, local rotmat) -> ROW-major 135 -> motion135_to_272."""
    out = recover_local_rotations_and_root(m272)
    # out is (local_rotmat, root_pos) in some order; detect by shape
    a, b = out[0], out[1]
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 4:           # (T,22,3,3)
        rotmat, root = a, b
    else:
        rotmat, root = b, a
    T = rotmat.shape[0]
    rot6d = rotmat_to_rot6d_row_major(torch.from_numpy(rotmat).float()).reshape(T, 132)
    root = torch.from_numpy(root.reshape(T, 3)).float()
    m135 = torch.cat([root, rot6d], dim=1).numpy().astype(np.float32)
    return motion135_to_272(m135)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ids = sorted(f[:-4] for f in os.listdir(GT_DIR) if f.endswith(".npy"))
    rng = np.random.RandomState(0)
    sel = [ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False)]

    pred_b, fk_b, stored_b = [], [], []
    pred_o, fk_o, stored_o = [], [], []
    for cid in sel:
        m = np.load(os.path.join(GT_DIR, cid + ".npy"))
        if m.shape[0] < 5:
            continue
        try:
            rt_pred = pred_path_roundtrip(m)
            rt_fk = reencode_272_via_fk(m)
            rt_st = reencode_272_via_stored_positions(m)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {cid}] {e}")
            continue
        L = min(m.shape[0], rt_pred.shape[0], rt_fk.shape[0], rt_st.shape[0])
        pred_o.append(rel(m[:L], rt_pred[:L])); pred_b.append(blocks(m[:L], rt_pred[:L]))
        fk_o.append(rel(m[:L], rt_fk[:L])); fk_b.append(blocks(m[:L], rt_fk[:L]))
        stored_o.append(rel(m[:L], rt_st[:L])); stored_b.append(blocks(m[:L], rt_st[:L]))

    def agg(lst):
        return {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}

    print(f"\n== Alignment diagnostic over {len(pred_o)} GT clips ==")
    for name, o, b in [
        ("PRED-path (differentiable_fk + canon272)", pred_o, pred_b),
        ("SMPL-H FK (reencode_272_via_fk, =refk)", fk_o, fk_b),
        ("stored-positions (encoding math only)", stored_o, stored_b),
    ]:
        print(f"\n[{name}]  overall rel-err mean={np.mean(o):.3e} max={np.max(o):.3e}")
        for k, v in agg(b).items():
            print(f"    {k:18s} {v:.3e}")


if __name__ == "__main__":
    main()

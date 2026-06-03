"""Gate A: round-trip self-consistency of the SMPL->272 encoder.

For N random GT 272 clips:
  (1) stored-positions round-trip: decode (stored pos + local rot) -> encode
      -> compare to GT272.  Should be ~float precision (validates ENCODING math).
  (2) FK round-trip: decode (rot,root) -> SMPL-H FK -> encode -> compare.
      Quantifies the SMPL-H-rest-FK vs native-SMPL-X-betas domain gap.
"""
import os
import sys
import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

from motionstreamer_272_encoder import (  # noqa: E402
    reencode_272_via_stored_positions, reencode_272_via_fk,
)

GT_DIR = os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")


def rel_err(a, b):
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + 1e-12
    return num / den


def per_block_err(a, b):
    blocks = {
        "root_vel[0:2]": (0, 2),
        "heading[2:8]": (2, 8),
        "pos[8:74]": (8, 74),
        "vel[74:140]": (74, 140),
        "rot[140:272]": (140, 272),
    }
    out = {}
    for k, (s, e) in blocks.items():
        out[k] = rel_err(a[:, s:e], b[:, s:e])
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ids = sorted(f[:-4] for f in os.listdir(GT_DIR) if f.endswith(".npy"))
    rng = np.random.RandomState(0)
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    sel = [ids[i] for i in pick]

    stored_errs, fk_errs = [], []
    stored_blocks, fk_blocks = [], []
    for cid in sel:
        m = np.load(os.path.join(GT_DIR, cid + ".npy"))
        if m.shape[0] < 4:
            continue
        rt_stored = reencode_272_via_stored_positions(m)
        rt_fk = reencode_272_via_fk(m)
        stored_errs.append(rel_err(m, rt_stored))
        fk_errs.append(rel_err(m, rt_fk))
        stored_blocks.append(per_block_err(m, rt_stored))
        fk_blocks.append(per_block_err(m, rt_fk))

    def agg_blocks(lst):
        keys = lst[0].keys()
        return {k: float(np.mean([d[k] for d in lst])) for k in keys}

    print(f"== Gate A round-trip over {len(stored_errs)} GT clips ==")
    print(f"[stored-positions] overall rel-err: mean={np.mean(stored_errs):.3e} "
          f"max={np.max(stored_errs):.3e}")
    for k, v in agg_blocks(stored_blocks).items():
        print(f"    {k:18s} {v:.3e}")
    print(f"[FK SMPL-H]        overall rel-err: mean={np.mean(fk_errs):.3e} "
          f"max={np.max(fk_errs):.3e}")
    for k, v in agg_blocks(fk_blocks).items():
        print(f"    {k:18s} {v:.3e}")

    ok = np.max(stored_errs) < 1e-3
    print(f"\nGATE A (stored-positions round-trip < 1e-3): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()

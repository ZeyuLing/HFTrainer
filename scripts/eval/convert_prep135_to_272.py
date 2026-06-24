#!/usr/bin/env python3
"""Convert a prep dir of <id>.npz(motion_135) into <id>.npz(motion_272) using the
SAME motion135_to_272 FK chain the MS-272 evaluator applies to predictions.

Used to build the *identity* GT-272 reference for the VAE rFID (Table 5): the GT
motionhub clip travels smplx -> motion_vector -> smplx (no VAE) -> row135 -> 272,
exactly like the VAE reconstructions, so FID(recon_272, identity_272) isolates VAE
roundtrip error (instead of the smplx<->native-272 representation gap)."""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    args = ap.parse_args()
    os.chdir(REPO)
    from motionstreamer_272_encoder import motion135_to_272

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".npz"))
    files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard_idx]
    ok = fail = 0
    for i, f in enumerate(files):
        out = os.path.join(args.out_dir, f)
        if os.path.exists(out):
            ok += 1
            continue
        try:
            m135 = np.load(os.path.join(args.in_dir, f))["motion_135"]
            m272 = motion135_to_272(m135)
            np.savez(out, motion_272=np.asarray(m272, dtype=np.float32))
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            if fail <= 5:
                print(f"[fail] {f}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} ok={ok} fail={fail}", flush=True)
    print(f"[DONE] ok={ok} fail={fail} -> {args.out_dir}")


if __name__ == "__main__":
    main()

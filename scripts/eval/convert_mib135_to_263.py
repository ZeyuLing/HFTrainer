#!/usr/bin/env python3
"""Convert MIB (minimal in-between) ``motion_135`` predictions (30 fps, SMPL-22)
to official HumanML3D-263 features (20 fps) for the native MoMask/Guo evaluator.

Each input ``<src_id>.npz`` (key ``motion_135`` of shape ``(T30, 135)``) is routed
through :func:`motion198_to_humanml263` -- the SAME final stages used to build
the GT 263 set (SMPL-H FK -> resample 30->20 -> process_file IK) -- so the result
plugs straight into ``scripts/eval/eval_momask_native_h3d263.py --mode pred``.

This makes our in-betweening row directly comparable to UMO Table 5
(CondMDI / MotionLab / UMO), which is computed on the Guo-263 evaluator.

Usage::

    python3 scripts/eval/convert_mib135_to_263.py \
        --in-dir  output/evaluation/mib_h3d_full/_stdfid_repack/smpl_cfg20 \
        --out-dir output/evaluation/mib_h3d_full/_pred263/smpl_cfg20 \
        --workers 16
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ROT_SPACE = "local"
_SRC_FPS = 30.0
_DST_FPS = 20.0


def _worker(args):
    src_path, out_path = args
    try:
        from hftrainer.datasets.motion.representation.humanml_repr import (
            motion198_to_humanml263, setup_process_globals,
        )
        setup_process_globals()
        z = np.load(src_path)
        m135 = z["motion_135"].astype(np.float32)
        if m135.ndim != 2 or m135.shape[1] < 135 or m135.shape[0] < 4:
            return (os.path.basename(src_path), "bad_shape")
        m263, _ = motion198_to_humanml263(
            m135, rotation_space=_ROT_SPACE,
            src_fps=_SRC_FPS, dst_fps=_DST_FPS, ensure_globals=False,
        )
        if not np.isfinite(m263).all() or len(m263) < 40:
            return (os.path.basename(src_path), "short_or_nan(%d)" % len(m263))
        np.save(out_path, m263.astype(np.float32))
        return (os.path.basename(src_path), "ok")
    except Exception as e:  # noqa: BLE001
        return (os.path.basename(src_path), "ERR:%s:%s" % (type(e).__name__, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = sorted(glob.glob(str(in_dir / "*.npz")))
    if args.limit:
        srcs = srcs[: args.limit]
    jobs = []
    for s in srcs:
        sid = os.path.basename(s)[:-4]
        op = str(out_dir / f"{sid}.npy")
        if os.path.exists(op):
            continue
        jobs.append((s, op))
    print(f"[+] {len(jobs)} to convert (of {len(srcs)} src) -> {out_dir}", flush=True)

    import multiprocessing as mp
    ok = 0
    bad = []
    with mp.Pool(args.workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=8)):
            if status == "ok":
                ok += 1
            else:
                bad.append((name, status))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)} ok={ok} bad={len(bad)}", flush=True)
    print(f"[+] DONE ok={ok} bad={len(bad)} -> {out_dir}", flush=True)
    for name, st in bad[:20]:
        print("   bad:", name, st)


if __name__ == "__main__":
    main()

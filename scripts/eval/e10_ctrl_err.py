#!/usr/bin/env python3
"""Mean geodesic rotation Ctrl.Err (deg) over observed (mask==0) joints for an
E10 part-control eval npz dir, using the SAME definition as \\ours
(``hftrainer.evaluation.motion.m2m_eval_metrics.compute_rotation_ctrl_error``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)

from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    compute_rotation_ctrl_error,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    vals = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "motion_135" not in d or "gt_motion_135" not in d or "src_mask" not in d:
            continue
        r = compute_rotation_ctrl_error(
            np.asarray(d["motion_135"], dtype=np.float32),
            np.asarray(d["gt_motion_135"], dtype=np.float32),
            np.asarray(d["src_mask"], dtype=np.float32))
        if "rot_ctrl_err_deg" in r and np.isfinite(r["rot_ctrl_err_deg"]):
            vals.append(r["rot_ctrl_err_deg"])
    out = {
        "tag": args.tag,
        "npz_dir": args.npz_dir,
        "n": len(vals),
        "rot_ctrl_err_deg_mean": float(np.mean(vals)) if vals else None,
        "rot_ctrl_err_deg_std": float(np.std(vals)) if vals else None,
    }
    print(f"[{args.tag}] n={out['n']} Ctrl.Err(deg)={out['rot_ctrl_err_deg_mean']}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()

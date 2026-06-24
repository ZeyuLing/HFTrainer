#!/usr/bin/env python3
"""Print table-ready \\ours metric values from the collected per-setting JSONs in
<out_root>/_metrics/{setting}__{ric,new,fid}.json.

Units: positions are converted to cm; FID/R@k/Div from the 272 evaluator.
"""
from __future__ import annotations

import argparse
import json
import os


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir",
                    default="output/evaluation/paper_ours_ep590/_metrics")
    ap.add_argument("--settings", nargs="+", default=None)
    args = ap.parse_args()

    md = args.metrics_dir
    settings = args.settings or sorted({
        f.split("__")[0] for f in os.listdir(md) if "__" in f})

    def cm(x):
        return round(x * 100, 3) if isinstance(x, (int, float)) else None

    def r(x, n=3):
        return round(x, n) if isinstance(x, (int, float)) else None

    for s in settings:
        ric = _load(os.path.join(md, f"{s}__ric.json"))
        new = _load(os.path.join(md, f"{s}__new.json")).get(s, {})
        fid = _load(os.path.join(md, f"{s}__fid.json"))
        row = {
            "n": new.get("n") or ric.get("n"),
            "FID": r(fid.get("fid"), 2),
            "R@3": r((fid.get("r_precision_pred") or [None, None, None])[2]),
            "Div": r(fid.get("diversity_pred"), 2),
            "MPJPE_cm": cm(new.get("mpjpe_all")),
            "P_MPJPE_cm": cm(new.get("p_mpjpe")),
            "KPS_err_cm": cm(new.get("kps_err")),
            "KPS_fail@20": r(new.get("kps_fail@20cm")),
            "KPS_fail@50": r(new.get("kps_fail@50cm")),
            "RotCtrl_deg": r(new.get("rot_ctrl_err_deg"), 3),
            "Traj_err_m": r(new.get("trajectory_err_m"), 4),
            "Traj_fail@20": r(new.get("trajectory_fail@20cm")),
            "Traj_fail@50": r(new.get("trajectory_fail@50cm")),
            "Foot": r(new.get("foot_skating_ratio")),
            "Jitter": r(new.get("jitter_pos"), 1),
        }
        present = {k: v for k, v in row.items() if v is not None}
        print(f"{s:22s} {present}")


if __name__ == "__main__":
    main()

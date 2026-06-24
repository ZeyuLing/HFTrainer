#!/usr/bin/env python3
"""Recompute per-setting M2M metrics from saved \\ours NPZ dumps.

The paper \\ours jobs save NPZ with ``motion_135`` (pred), ``gt_motion_135`` and
``src_mask`` (T, 198; first 135 channels align with motion_135, 1=generate,
0=keep). This recomputes the FULL metric set -- including metrics added after a
job started (geodesic rotation Ctrl.Err, KPS Err / Fail@k, axis-aware Traj.Err /
Fail@cm) -- WITHOUT re-running generation.

Usage:
    python3 scripts/eval/collect_ours_posthoc_metrics.py \
        --base output/evaluation/paper_ours_ep590 \
        --settings E10_A_upper E3_adaptive E5_A_xz_dense \
        --out output/evaluation/paper_ours_ep590/_posthoc_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from hftrainer.evaluation.motion.m2m_eval_metrics import compute_all_metrics

BONE = "data/hymotion_m2m_data/bone_offsets_22.pt"


def _agg(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    return float(arr.mean()) if arr.size else None


def collect_dir(npz_files, bone_offsets, max_samples=None):
    per_metric = defaultdict(list)
    n = 0
    for f in npz_files:
        if max_samples and n >= max_samples:
            break
        try:
            d = np.load(f, allow_pickle=True)
            pred = d["motion_135"].astype(np.float32)
            gt = d["gt_motion_135"].astype(np.float32) if "gt_motion_135" in d else None
            mask = d["src_mask"][:, :135].astype(np.float32) if "src_mask" in d else None
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {os.path.basename(f)}: {exc}")
            continue
        m = compute_all_metrics(pred, gt, mask, bone_offsets, fps=30.0,
                                compute_fk=True)
        for k, v in m.items():
            per_metric[k].append(v)
        n += 1
    return {k: _agg(v) for k, v in per_metric.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/evaluation/paper_ours_ep590")
    ap.add_argument("--settings", nargs="+", default=None,
                    help="explicit setting subdir names; default = auto-discover")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bone_offsets = torch.load(BONE, map_location="cpu").numpy()

    if args.settings:
        settings = args.settings
    else:
        settings = sorted(
            d for d in os.listdir(args.base)
            if os.path.isdir(os.path.join(args.base, d)) and not d.startswith("_"))

    out = {}
    for s in settings:
        npz = sorted(glob.glob(os.path.join(args.base, s, "**", "*.npz"),
                               recursive=True))
        if not npz:
            print(f"[{s}] no npz")
            continue
        metrics, n = collect_dir(npz, bone_offsets, args.max_samples)
        out[s] = {"n": n, **metrics}
        keys = ["fid", "mpjpe_all", "p_mpjpe", "kps_err", "kps_fail@20cm",
                "kps_fail@50cm", "rot_ctrl_err_deg", "trajectory_err_m",
                "trajectory_fail@20cm", "trajectory_fail@50cm",
                "foot_skating_ratio", "jitter_pos"]
        shown = {k: (round(metrics[k], 4) if metrics.get(k) is not None else None)
                 for k in keys if k in metrics}
        print(f"[{s}] n={n} {shown}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()

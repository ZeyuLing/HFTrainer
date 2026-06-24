#!/usr/bin/env python3
"""Project \\ours / KIMODO body-part *rotation* control outputs into the Table-6
Experiment-B *position* space, locally (single GPU).

For each (tag, npz_dir, part) job it computes, into ``--metrics-dir``:
  * ``<tag>__new.json``  -- observed-joint position error (KPS/MPJPE), jitter,
    foot-skating  (paper_npz_observed_pos_mpjpe logic; caption-independent).
  * ``<tag>__fid.json``  -- 272 FK-matched FID / R@k / Diversity
    (eval_editing_272_fid logic; needs English captions). Skipped when fid=0.

The TMR evaluator is loaded ONCE and reused across all FID jobs.

Jobs are given as ``tag:npz_dir:part:max:fid`` tuples via repeated --job, or a
built-in default set (ours 11 parts + KIMODO coarse-from-editfix +
KIMODO fine-from-rot position-only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

from bodypart_pos_common import part_joints  # noqa: E402
from paper_npz_observed_pos_mpjpe import _eval_npz  # noqa: E402
from eval_editing_272_fid import eval_one  # noqa: E402
from eval_motionstreamer_272 import MEAN_STD, load_evaluator  # noqa: E402


def position_metrics(npz_dir, part, tag, max_samples):
    joints = part_joints(part)
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if max_samples:
        files = files[:max_samples]
    accum = {"obs_mpjpe_ric": [], "obs_mpjpe_world": [], "mpjpe_all_ric": [],
             "jitter": [], "foot_skating": []}
    n_fail = 0
    for f in files:
        try:
            r = _eval_npz(f, joints)
        except Exception as e:  # noqa: BLE001
            if n_fail == 0:
                print(f"  [warn] {tag} failed on {os.path.basename(f)}: "
                      f"{type(e).__name__}: {e}")
            n_fail += 1
            r = None
        if r:
            for k in accum:
                accum[k].append(r[k])
    if not accum["obs_mpjpe_ric"]:
        return None
    out = {"tag": tag, "part": part, "joints": joints, "npz_dir": npz_dir,
           "n": len(accum["obs_mpjpe_ric"]), "n_fail": n_fail}
    for k, v in accum.items():
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_std"] = float(np.std(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir",
                    default=os.path.join(REPO, "output/evaluation/bodypart_table6_pos/_metrics"))
    ap.add_argument("--job", action="append", default=[],
                    help="tag:npz_dir:part:max:fid (repeatable). If none, use defaults.")
    ap.add_argument("--only", default=None, help="substring filter on tag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)

    jobs = []  # (tag, npz_dir, part, max, fid)
    if args.job:
        for j in args.job:
            tag, d, part, mx, fid = j.split(":")
            jobs.append((tag, d, part, int(mx), int(fid)))
    else:
        OURS = os.path.join(REPO, "output/evaluation/paper_ours_ep590")
        KED = os.path.join(REPO, "output/evaluation/m2m_editfix_paper/"
                           "kimodo_caption_editfix_ep240/kimodo_caption_editfix_ep240")
        KROT = os.path.join(REPO, "output/evaluation/bodypart_table6_rot/kimodo")
        parts = ["A_upper", "B_lower", "C_spine_only", "D_arms_only", "E_legs_only",
                 "F_left_arm", "G_right_arm", "H_left_leg", "I_right_leg",
                 "J_feet_only", "K_no_feet"]
        for p in parts:
            d = os.path.join(OURS, f"E10_{p}", "smpl_caption_editfix_latest", f"E10_{p}", "npz")
            jobs.append((f"ours_{p}", d, p, 300, 1))
        # KIMODO coarse: editfix (English caption -> valid FID)
        for p in ["A_upper", "B_lower"]:
            d = os.path.join(KED, f"E10_{p}", "npz")
            jobs.append((f"kimodo_{p}", d, p, 0, 1))
        # KIMODO fine: rot tree (Chinese caption -> position metrics only)
        for p in ["C_spine_only", "D_arms_only", "E_legs_only", "F_left_arm",
                  "G_right_arm", "H_left_leg", "I_right_leg", "J_feet_only", "K_no_feet"]:
            d = os.path.join(KROT, p, "npz")
            jobs.append((f"kimodo_{p}", d, p, 0, 0))

    if args.only:
        jobs = [j for j in jobs if args.only in j[0]]

    need_fid = any(j[4] for j in jobs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textencoder = motionencoder = mean = std = None
    if need_fid:
        mean = np.load(os.path.join(MEAN_STD, "Mean.npy"))
        std = np.load(os.path.join(MEAN_STD, "Std.npy"))
        textencoder, motionencoder = load_evaluator(device)
        print("[+] TMR evaluator loaded", flush=True)

    for tag, d, part, mx, fid in jobs:
        if not os.path.isdir(d):
            print(f"[skip] {tag}: missing dir {d}", flush=True)
            continue
        # position metrics
        pm = position_metrics(d, part, tag, mx)
        if pm is not None:
            with open(os.path.join(args.metrics_dir, f"{tag}__new.json"), "w") as fh:
                json.dump(pm, fh, indent=2)
            print(f"[pos] {tag}: n={pm['n']} KPS_ric={pm['obs_mpjpe_ric_mean']*100:.2f}cm "
                  f"foot={pm['foot_skating_mean']:.3f} jitter={pm['jitter_mean']:.1f}",
                  flush=True)
        else:
            print(f"[pos] {tag}: no valid npz", flush=True)
        # FID
        if fid:
            res = eval_one(d, tag, device, args.seed, mx,
                           textencoder, motionencoder, mean, std)
            outp = os.path.join(args.metrics_dir, f"{tag}__fid.json")
            with open(outp, "w") as fh:
                json.dump(res if res is not None else {}, fh, indent=2)
            if res is not None:
                print(f"[fid] {tag}: FID={res['FID']:.4f} R@3={res['R@3']:.4f} "
                      f"Div={res['Diversity']:.3f}", flush=True)
            else:
                print(f"[fid] {tag}: FID skipped (insufficient n)", flush=True)


if __name__ == "__main__":
    main()

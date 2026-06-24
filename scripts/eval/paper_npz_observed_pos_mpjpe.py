#!/usr/bin/env python3
"""Observed-joint position error (MPJPE / KPS) for Table-6 *Experiment B*
(position-based fine-grained body-part control).

Given a flat dir of per-clip ``{id}.npz`` (each with ``motion_135`` = prediction
and ``gt_motion_135`` = reference) and a body-part ``--part`` key, this computes
the **position error of the observed joints** -- i.e. how well the joints whose
3D positions were given as control are reproduced -- plus FID-companion metrics
(jitter, foot-skating).  The same script scores every method (CondMDI,
OmniControl, \\ours, KIMODO) so the numbers are strictly comparable.

Two spaces are reported (both restricted to the part's observed joints):
  * ``obs_mpjpe_ric``  -- 272-ric space (per-frame xz-centred + heading-removed
    root-relative joint positions, ``[8:74]`` of the 272 vector).  This is the
    PRIMARY number, identical convention to ``paper_npz_ric_mpjpe.py`` so it is
    comparable to every other MPJPE in the paper.
  * ``obs_mpjpe_world`` -- raw FK world space (floor-aligned y-min=0), a global
    diagnostic that rewards absolute position accuracy.

Usage::

    python3 scripts/eval/paper_npz_observed_pos_mpjpe.py \
        --npz-dir <dir of {id}.npz> --part A_upper \
        --tag condmdi_A_upper --out-json .../condmdi_A_upper__new.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

from bodypart_pos_common import part_joints  # noqa: E402
from motionstreamer_272_encoder import motion135_to_272  # noqa: E402
import torch  # noqa: E402
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk  # noqa: E402
from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    compute_foot_ground_metrics,
)

_BONE22 = torch.load(
    os.path.join(_REPO_ROOT, "data/hymotion_m2m_data/bone_offsets_22.pt"),
    map_location="cpu",
).float()
_NJ = 22


def _world_positions(m135: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(m135[:, :135], dtype=np.float32))
    wp, _, _, _ = motion135_to_fk(t, _BONE22, rotation_space="local")
    pos = wp.numpy()
    pos[:, :, 1] -= pos[:, :, 1].min()
    return pos


def _ric(m272: np.ndarray) -> np.ndarray:
    T = m272.shape[0]
    return m272[:, 8:8 + 3 * _NJ].reshape(T, _NJ, 3)


def _jitter(pos: np.ndarray, fps: int = 30) -> float:
    if pos.shape[0] < 4:
        return 0.0
    acc = np.diff(pos, n=3, axis=0) * (fps ** 3)
    return float(np.linalg.norm(acc, axis=-1).mean())


def _eval_npz(path: str, joints: list[int]) -> dict | None:
    d = np.load(path, allow_pickle=True)
    if "motion_135" not in d or "gt_motion_135" not in d:
        return None
    pred135 = np.asarray(d["motion_135"], dtype=np.float32)
    gt135 = np.asarray(d["gt_motion_135"], dtype=np.float32)
    # 272-ric space (root-relative, heading-removed) -- primary
    mp = motion135_to_272(pred135)
    mg = motion135_to_272(gt135)
    T = min(len(mp), len(mg))
    Pp, Gg = _ric(mp[:T]), _ric(mg[:T])
    err_ric = np.linalg.norm(Pp - Gg, axis=-1)          # (T,22) m
    # world space (floor-aligned) -- diagnostic
    wp = _world_positions(pred135)
    wg = _world_positions(gt135)
    Tw = min(len(wp), len(wg))
    err_world = np.linalg.norm(wp[:Tw] - wg[:Tw], axis=-1)
    foot = compute_foot_ground_metrics(wp, fps=30.0)
    j = np.asarray(joints, dtype=np.int64)
    return {
        "obs_mpjpe_ric": float(err_ric[:, j].mean()),
        "obs_mpjpe_world": float(err_world[:, j].mean()),
        "mpjpe_all_ric": float(err_ric.mean()),
        "jitter": _jitter(Pp, 30),
        "foot_skating": float(foot["foot_skating_ratio"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    joints = part_joints(args.part)
    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if args.max_samples:
        files = files[: args.max_samples]
    accum = {"obs_mpjpe_ric": [], "obs_mpjpe_world": [], "mpjpe_all_ric": [],
             "jitter": [], "foot_skating": []}
    n_fail = 0
    for f in files:
        try:
            r = _eval_npz(f, joints)
        except Exception as e:  # noqa: BLE001
            if n_fail == 0:
                print(f"[warn] failed on {f}: {type(e).__name__}: {e}")
            n_fail += 1
            r = None
        if r:
            for k in accum:
                accum[k].append(r[k])
    if not accum["obs_mpjpe_ric"]:
        print(f"[{args.tag}] no valid npz in {args.npz_dir}")
        return
    out = {"tag": args.tag, "part": args.part, "joints": joints,
           "npz_dir": args.npz_dir, "n": len(accum["obs_mpjpe_ric"]),
           "n_fail": n_fail}
    for k, v in accum.items():
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_std"] = float(np.std(v))
    print(f"\n=== {args.tag}  part={args.part} ({len(joints)} obs joints) ===")
    print(f" n={out['n']}  obs-MPJPE[ric]={out['obs_mpjpe_ric_mean']*100:.2f}cm  "
          f"obs-MPJPE[world]={out['obs_mpjpe_world_mean']*100:.2f}cm  "
          f"all-MPJPE[ric]={out['mpjpe_all_ric_mean']*100:.2f}cm  "
          f"jitter={out['jitter_mean']:.1f}  foot={out['foot_skating_mean']:.3f}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate Minimal In-Betweening (E2 both_1f) metrics in the **UMO protocol**.

UMO (Sec. Evaluation Metrics) computes MPJPE in the HumanML3D-272 representation
space: per-frame xz-centered + heading-removed joint positions (``[8:74]`` of the
272 vector). That space removes the global trajectory drift that dominates raw
world-coordinate MPJPE (50+ cm), making it the apples-to-apples metric against
UMO's 8.55 cm. ``[P]-MPJPE`` is the same metric restricted to the ``[preserve]``
(condition) frames -> ~0 for our hard-imputation interface.

Reads the per-sample NPZ saved by ``eval_m2m_v2_all_tasks.py --save-npz``
(``motion_135`` pred, ``gt_motion_135``, ``src_mask``), converts both pred & GT
to 272 via the canonical SMPL-X-272 skeleton (``motion135_to_272``), and reports
mean(+/-std across reps) for MPJPE / [P]-MPJPE / jitter / foot-skating.

Usage:
  python3 scripts/eval/aggregate_mib_umo.py \
      --root output/evaluation/mib_h3d_full \
      --reps 0 1 2 3 4 --max-samples 0 --out docs/temp/mib_umo_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)       # for motionstreamer_272_encoder
sys.path.insert(0, _REPO_ROOT)      # for hftrainer (when run as __main__)
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


def _world_positions(m135: np.ndarray) -> np.ndarray:
    """FK to grounded world joint positions (T,22,3) for foot-skating."""
    t = torch.from_numpy(np.asarray(m135[:, :135], dtype=np.float32))
    wp, _, _, _ = motion135_to_fk(t, _BONE22, rotation_space="local")
    pos = wp.numpy()
    pos[:, :, 1] -= pos[:, :, 1].min()  # ground feet at y=0
    return pos

MODELS = {
    "kimodo": "kimodo_caption_editfix_ep240",
    "smpl": "smpl_caption_editfix_ep230",
    "kimodo_latest": "kimodo_caption_editfix_latest",
    "smpl_latest": "smpl_caption_editfix_latest",
}
SETTINGS = ["blank", "cfg20"]
_NJ = 22


def _ric(m272: np.ndarray) -> np.ndarray:
    """272 -> (T,22,3) heading-removed, per-frame root-relative joint positions."""
    T = m272.shape[0]
    return m272[:, 8:8 + 3 * _NJ].reshape(T, _NJ, 3)


def _jitter(pos: np.ndarray, fps: int = 30) -> float:
    """Mean jerk magnitude (3rd-order finite diff of positions), m/s^3 -> scaled."""
    if pos.shape[0] < 4:
        return 0.0
    acc = np.diff(pos, n=3, axis=0) * (fps ** 3)
    return float(np.linalg.norm(acc, axis=-1).mean())


def _eval_npz(path: str) -> dict | None:
    d = np.load(path, allow_pickle=True)
    if "motion_135" not in d or "gt_motion_135" not in d:
        return None
    mp = motion135_to_272(np.asarray(d["motion_135"], dtype=np.float32))
    mg = motion135_to_272(np.asarray(d["gt_motion_135"], dtype=np.float32))
    T = min(len(mp), len(mg))
    Pp, Gg = _ric(mp[:T]), _ric(mg[:T])
    err = np.linalg.norm(Pp - Gg, axis=-1)  # (T,22) metres
    sm = np.asarray(d["src_mask"])
    fm = (sm.max(-1) > 0.5)[:T]             # True=generated
    pres = ~fm
    foot = compute_foot_ground_metrics(_world_positions(d["motion_135"]), fps=30.0)
    out = {
        "mpjpe_full": float(err.mean()),
        "mpjpe_gen": float(err[fm].mean()) if fm.any() else float(err.mean()),
        "p_mpjpe": float(err[pres].mean()) if pres.any() else 0.0,
        "jitter": _jitter(Pp, 30),
        "foot_skating": float(foot["foot_skating_ratio"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/evaluation/mib_h3d_full")
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap samples per rep (0=all)")
    ap.add_argument("--out", default="docs/temp/mib_umo_metrics.json")
    args = ap.parse_args()

    results = {}
    for mk, md in MODELS.items():
        for st in SETTINGS:
            rep_means = []  # per-rep dict of metric->mean
            n_total = 0
            for rep in args.reps:
                npz_dir = os.path.join(
                    args.root, md, st, f"rep{rep}", md, "E2_both_1f", "npz")
                files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
                if args.max_samples:
                    files = files[: args.max_samples]
                if not files:
                    continue
                accum = {"mpjpe_full": [], "mpjpe_gen": [],
                         "p_mpjpe": [], "jitter": [], "foot_skating": []}
                for f in files:
                    try:
                        r = _eval_npz(f)
                    except Exception as e:
                        if not getattr(main, "_warned", False):
                            print(f"[warn] _eval_npz failed on {f}: "
                                  f"{type(e).__name__}: {e}", flush=True)
                            main._warned = True
                        r = None
                    if r:
                        for k in accum:
                            accum[k].append(r[k])
                if not accum["mpjpe_full"]:
                    continue
                rep_means.append({k: float(np.mean(v)) for k, v in accum.items()})
                n_total += len(accum["mpjpe_full"])
                print(f"[{mk}/{st}] rep{rep}: n={len(accum['mpjpe_full'])} "
                      f"MPJPE={np.mean(accum['mpjpe_full'])*100:.2f}cm "
                      f"[P]={np.mean(accum['p_mpjpe'])*100:.3f}cm", flush=True)
            if not rep_means:
                continue
            key = f"{mk}_{st}"
            agg = {}
            for m in ["mpjpe_full", "mpjpe_gen", "p_mpjpe", "jitter",
                      "foot_skating"]:
                vals = np.array([rm[m] for rm in rep_means])
                agg[m + "_mean"] = float(vals.mean())
                agg[m + "_std"] = float(vals.std())
            agg["n_reps"] = len(rep_means)
            agg["n_samples_total"] = n_total
            results[key] = agg
            print(f"=== {key}: MPJPE={agg['mpjpe_full_mean']*100:.2f}"
                  f"+/-{agg['mpjpe_full_std']*100:.2f}cm  "
                  f"[P]-MPJPE={agg['p_mpjpe_mean']*100:.3f}cm  "
                  f"jitter={agg['jitter_mean']:.1f}  "
                  f"(reps={agg['n_reps']}, N={n_total})", flush=True)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDONE -> {args.out}")
    print("\n=== SUMMARY (UMO protocol, 272-ric space, cm) ===")
    print(f"{'variant':16s} {'MPJPE':>10s} {'[P]-MPJPE':>10s} "
          f"{'jitter':>8s} {'foot':>8s}")
    for k, v in results.items():
        print(f"{k:16s} {v['mpjpe_full_mean']*100:8.2f}cm "
              f"{v['p_mpjpe_mean']*100:8.3f}cm {v['jitter_mean']:8.1f} "
              f"{v['foot_skating_mean']:8.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generic 272-ric-space MPJPE / [P]-MPJPE aggregator over a *flat* npz dir.

Reuses the exact metric definition validated for the paper MIB row
(``scripts/eval/aggregate_mib_umo.py``): MPJPE is computed in the HumanML3D-272
representation space (per-frame xz-centered + heading-removed root-relative joint
positions, ``[8:74]`` of the 272 vector), which removes global trajectory drift.
``[P]-MPJPE`` is the same metric on the preserved (observed) frames; ``mpjpe_gen``
is restricted to generated frames.

Unlike ``aggregate_mib_umo`` (which hard-codes the MIB rep{N}/E2_both_1f layout),
this script takes ANY directory of per-sample ``{idx}.npz`` (each with
``motion_135``, ``gt_motion_135``, ``src_mask``) so it can be reused for every
task/setting and every baseline that emits the same NPZ schema.

Usage:
    python3 scripts/eval/paper_npz_ric_mpjpe.py \
        --npz-dir output/.../E2_pre20/smpl_caption_editfix_latest/E2_pre20/npz \
        --tag ours_E2_pre20 --out-json /tmp/e2_pre20_ric.json
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
    fm = (sm.max(-1) > 0.5)[:T]             # True = generated
    pres = ~fm
    foot = compute_foot_ground_metrics(_world_positions(d["motion_135"]), fps=30.0)
    return {
        "mpjpe_full": float(err.mean()),
        "mpjpe_gen": float(err[fm].mean()) if fm.any() else float(err.mean()),
        "p_mpjpe": float(err[pres].mean()) if pres.any() else 0.0,
        "jitter": _jitter(Pp, 30),
        "foot_skating": float(foot["foot_skating_ratio"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if args.max_samples:
        files = files[: args.max_samples]
    accum = {"mpjpe_full": [], "mpjpe_gen": [], "p_mpjpe": [],
             "jitter": [], "foot_skating": []}
    n_fail = 0
    for f in files:
        try:
            r = _eval_npz(f)
        except Exception as e:  # noqa: BLE001
            if n_fail == 0:
                print(f"[warn] _eval_npz failed on {f}: {type(e).__name__}: {e}")
            n_fail += 1
            r = None
        if r:
            for k in accum:
                accum[k].append(r[k])
    if not accum["mpjpe_full"]:
        print(f"[{args.tag}] no valid npz in {args.npz_dir}")
        return
    out = {"tag": args.tag, "npz_dir": args.npz_dir,
           "n": len(accum["mpjpe_full"]), "n_fail": n_fail}
    for k, v in accum.items():
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_std"] = float(np.std(v))
    print(f"\n=== {args.tag}  (272-ric space) ===")
    print(f" n={out['n']}  MPJPE_full={out['mpjpe_full_mean']*100:.2f}cm  "
          f"MPJPE_gen={out['mpjpe_gen_mean']*100:.2f}cm  "
          f"[P]-MPJPE={out['p_mpjpe_mean']*100:.3f}cm  "
          f"jitter={out['jitter_mean']:.1f}  foot={out['foot_skating_mean']:.3f}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()

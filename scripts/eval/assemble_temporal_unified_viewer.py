#!/usr/bin/env python3
"""Assemble the m2m_eval_viewer directory tree for the Table-4 temporal-completion
comparison (GT + CondMDI / KIMODO / MotionLab / ours across pre20/post20/mid60).

The viewer expects ``EVAL_DIR/<model>/<TASK_setting>/npz/<idx>.npz`` and groups
panels across models by the case key ``<TASK_setting>__<idx>``. We key every model
by the eval_h3d_editing ``source_id`` so GT/mask/caption (shared) and each model's
pred land in ONE case:

  - baselines: temporal_unified/<m>/<proto>/eval_npz/<source_id>.npz  (already by id)
  - ours:      paper_ours_ep590/E2_<proto>/.../npz/{i:05d}.npz, where i indexes
               eval_h3d_editing in order -> relink to <source_id>.npz

Symlinks (no copies) keep it cheap on CephFS.
"""
from __future__ import annotations
import argparse, json, os, glob
import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
TU = os.path.join(REPO, "output/evaluation/temporal_unified")
OURS = os.path.join(REPO, "output/evaluation/paper_ours_ep590")

# which protocols each model produced
MODEL_PROTOS = {
    "ours":      ["pre20", "post20", "mid60"],
    "condmdi":   ["pre20", "mid60"],
    "motionlab": ["pre20", "post20", "mid60"],
    "kimodo":    ["pre20", "post20", "mid60"],
}

# baseline eval_npz subdir per model (KIMODO uses the faithful rotation-transfer
# variant eval_npz_rot; others use eval_npz).
EVAL_SUBDIR = {"kimodo": "eval_npz_rot", "condmdi": "eval_npz", "motionlab": "eval_npz"}


def src_for(model, proto, i, sid):
    if model == "ours":
        return os.path.join(
            OURS, f"E2_{proto}", "smpl_caption_editfix_latest",
            f"E2_{proto}", "npz", f"{i:05d}.npz")
    return os.path.join(TU, model, proto, EVAL_SUBDIR.get(model, "eval_npz"), f"{sid}.npz")


def _splice_to_gt(npz_path):
    """Load an eval npz and hard-keep GT on observed/condition frames so the
    VIEWER shows a clean GT condition for every method (matching \\ours, whose
    condition is verbatim GT). Only generated frames differ across methods."""
    z = np.load(npz_path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    m = np.asarray(z["src_mask"], dtype=np.float32)
    pred = np.asarray(z["motion_135"], dtype=np.float32).copy()
    gt = np.asarray(z["gt_motion_135"], dtype=np.float32)
    obs = m[:, :135].max(axis=-1) < 0.5          # True = observed/condition frame
    T = min(len(pred), len(gt), len(obs))
    o = obs[:T]
    pred[:T][o] = gt[:T][o]
    out["motion_135"] = pred
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(REPO, "output/evaluation/temporal_unified_viewer"))
    ap.add_argument("--n", type=int, default=60, help="number of common clips per protocol")
    ap.add_argument("--data-file", default=os.path.join(REPO, "data/eval/m2m_v2/eval_h3d_editing.json"))
    args = ap.parse_args()

    data_list = json.load(open(args.data_file))["data_list"]
    # idx -> source_id (ours npz index i maps to data_list[i])
    idx_sid = [str(it.get("source_id") or os.path.splitext(os.path.basename(it["motion_path"]))[0])
               for it in data_list]

    os.makedirs(args.out_dir, exist_ok=True)
    made = {m: 0 for m in MODEL_PROTOS}
    for model, protos in MODEL_PROTOS.items():
        for proto in protos:
            dst_dir = os.path.join(args.out_dir, model, f"E2_{proto}", "npz")
            os.makedirs(dst_dir, exist_ok=True)
            for i in range(args.n):
                sid = idx_sid[i]
                src = src_for(model, proto, i, sid)
                if not os.path.exists(src):
                    continue
                dst = os.path.join(dst_dir, f"{sid}.npz")
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                np.savez(dst, **_splice_to_gt(src))
                made[model] += 1
    print("[assemble] out-dir:", args.out_dir)
    for m, c in made.items():
        print(f"  {m}: {c} links across {MODEL_PROTOS[m]}")


if __name__ == "__main__":
    main()

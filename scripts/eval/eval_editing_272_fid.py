"""Distribution metrics (FID / R-Precision / MM-Dist / Diversity) for the M2M
*editing* tasks, on the MotionStreamer 272-dim TMR evaluator.

Unlike scripts/eval/eval_motionstreamer_272.py (which keys predictions by
HumanML3D clip-id and compares against the native GT272 test set), the editing
eval (scripts/eval/eval_m2m_v2_all_tasks.py) writes idx-keyed NPZ
(``{idx:05d}.npz``) where EACH file is self-contained:

    motion_135      (T,135)  -> prediction
    gt_motion_135   (T,135)  -> ground-truth (the edited clip's reference)
    caption         str
    src_mask, task_key, ...

So we compute a *self-contained* FID: both pred and GT go through the SAME
135 -> 272 conversion (motion135_to_272, canonical SMPL-X-272 skeleton), making
it an FK-matched (fair) FID — the conversion overhead cancels because it is
applied identically to both distributions.

Usage
-----
    python3 scripts/eval/eval_editing_272_fid.py \
        --pred-npz-dir output/evaluation/m2m_editfix_paper/kimodo_caption_editfix_ep240/E2_both_1f/npz \
        --tag kimodo_E2_both_1f --out-json /tmp/e2_fid.json

    # batch over every task_setting under a model dir:
    python3 scripts/eval/eval_editing_272_fid.py \
        --model-dir output/evaluation/m2m_editfix_paper/kimodo_caption_editfix_ep240 \
        --out-json output/evaluation/m2m_editfix_paper/kimodo_272fid.json
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

# Reuse the validated evaluator + metric helpers.
from eval_motionstreamer_272 import (  # noqa: E402
    MEAN_STD,
    UNIT_LENGTH,
    calculate_activation_statistics,
    calculate_frechet_distance,
    crop_and_norm,
    diversity_of,
    encode_items,
    load_evaluator,
)
from motionstreamer_272_encoder import motion135_to_272  # noqa: E402


def _load_npz_dir(npz_dir, max_samples=0):
    """Return list of (caption, pred_135, gt_135)."""
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if max_samples > 0:
        files = files[:max_samples]
    out = []
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [load-fail] {os.path.basename(fp)}: {e}")
            continue
        if "motion_135" not in d or "gt_motion_135" not in d:
            continue
        m = np.asarray(d["motion_135"], dtype=np.float32)
        g = np.asarray(d["gt_motion_135"], dtype=np.float32)
        cap = ""
        if "caption" in d:
            cap = str(d["caption"])
        out.append((cap, m, g))
    return out


def _build_272_items(records, which, mean, std, rng):
    """records: list of (cap, pred_135, gt_135). which in {pred, gt}."""
    items = []
    skipped = 0
    for cap, m135, g135 in records:
        if not cap:
            skipped += 1
            continue
        src = m135 if which == "pred" else g135
        if src.shape[0] < UNIT_LENGTH + 1:
            skipped += 1
            continue
        try:
            m272 = motion135_to_272(src)
        except Exception as e:  # noqa: BLE001
            print(f"  [272-fail] {e}")
            skipped += 1
            continue
        m, L = crop_and_norm(m272, mean, std, rng)
        if m is None:
            skipped += 1
            continue
        items.append((cap, m, L))
    return items, skipped


def eval_one(npz_dir, tag, device, seed, max_samples,
             textencoder, motionencoder, mean, std):
    records = _load_npz_dir(npz_dir, max_samples)
    if len(records) < 64:
        print(f"  [skip] {tag}: only {len(records)} npz (<64, FID unreliable)")
        return None
    # GT (reference) distribution — same 135->272 path as pred (FK-matched).
    gt_items, sg = _build_272_items(records, "gt", mean, std,
                                    np.random.RandomState(seed))
    pred_items, sp = _build_272_items(records, "pred", mean, std,
                                      np.random.RandomState(seed))
    if len(gt_items) < 32 or len(pred_items) < 32:
        print(f"  [skip] {tag}: gt={len(gt_items)} pred={len(pred_items)} (<32 batch)")
        return None
    gt = encode_items(gt_items, textencoder, motionencoder, device,
                      np.random.RandomState(seed))
    pred = encode_items(pred_items, textencoder, motionencoder, device,
                        np.random.RandomState(seed))
    gt_mu, gt_cov = calculate_activation_statistics(gt["em"])
    pmu, pcov = calculate_activation_statistics(pred["em"])
    fid = calculate_frechet_distance(gt_mu, gt_cov, pmu, pcov)
    gt_div = diversity_of(gt["em"], np.random.RandomState(seed + 100))
    pred_div = diversity_of(pred["em"], np.random.RandomState(seed + 100))
    res = {
        "tag": tag,
        "n_records": len(records),
        "n_gt": len(gt_items),
        "n_pred": len(pred_items),
        "FID": float(fid),
        "R@1": float(pred["R"][0]),
        "R@2": float(pred["R"][1]),
        "R@3": float(pred["R"][2]),
        "MM_Dist": float(pred["matching"]),
        "Diversity": float(pred_div),
        "gt_R@1": float(gt["R"][0]),
        "gt_MM_Dist": float(gt["matching"]),
        "gt_Diversity": float(gt_div),
    }
    print(f"\n=== {tag} (272 FK-matched FID) ===")
    print(f" n={len(records)} FID={fid:.4f}  R@1={res['R@1']:.4f} "
          f"R@2={res['R@2']:.4f} R@3={res['R@3']:.4f}  "
          f"MM-Dist={res['MM_Dist']:.4f}  Div={res['Diversity']:.4f}")
    print(f" (gt-self: R@1={res['gt_R@1']:.4f} MM-Dist={res['gt_MM_Dist']:.4f} "
          f"Div={res['gt_Diversity']:.4f})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz-dir", default=None,
                    help="single <task_setting>/npz dir")
    ap.add_argument("--model-dir", default=None,
                    help="a model dir; eval every <task_setting>/npz under it")
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    mean = np.load(os.path.join(MEAN_STD, "Mean.npy"))
    std = np.load(os.path.join(MEAN_STD, "Std.npy"))
    textencoder, motionencoder = load_evaluator(device)
    print("evaluator loaded")

    targets = []  # (tag, npz_dir)
    if args.pred_npz_dir:
        targets.append((args.tag, args.pred_npz_dir))
    if args.model_dir:
        for d in sorted(glob.glob(os.path.join(args.model_dir, "*", "npz"))):
            ts = os.path.basename(os.path.dirname(d))
            targets.append((ts, d))
    if not targets:
        ap.error("provide --pred-npz-dir or --model-dir")

    results = {}
    for tag, npz_dir in targets:
        if not os.path.isdir(npz_dir):
            print(f"  [missing] {npz_dir}")
            continue
        r = eval_one(npz_dir, tag, device, args.seed, args.max_samples,
                     textencoder, motionencoder, mean, std)
        if r is not None:
            results[tag] = r

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json} ({len(results)} task_settings)")


if __name__ == "__main__":
    main()

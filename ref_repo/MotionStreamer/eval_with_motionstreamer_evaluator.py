#!/usr/bin/env python3
"""Standalone MotionStreamer-evaluator evaluation script.

Takes a directory of pre-generated 272-dim motion NPZ/NPY files (one per
caption, indexed by sample id) plus the original 272-dim HumanML3D test set
and the paired text annotations, and computes FID / R-Precision / MM-Dist /
Diversity using MotionStreamer's TMR-based motion+text evaluator
(``Evaluator_272/epoch=99.ckpt``).

Usage::

    python ref_repo/MotionStreamer/MotionStreamer/eval_with_motionstreamer_evaluator.py \
        --pred_dir <path/to/pred_272dim>  \
        --evaluator_ckpt ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt \
        --data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --out_json work_dirs/ms_eval/<method>.json \
        [--gt_only]   # GT sanity check (use real motions as predictions)

Each prediction file is expected to be:
    motion_<id>.npy   shape (T, 272), already in the 272-dim native units
                       (not standardized).

The script standardizes both real and predicted motions with MotionStreamer's
``humanml3d_272/mean_std/{Mean,Std}.npy`` before encoding, matching the
official evaluation protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# This script lives at ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py
# but the importable Python tree (Evaluator_272/, utils/, etc.) lives one level
# deeper, inside the upstream clone at MotionStreamer/.  We add both to sys.path
# so the script can be invoked from either location.
THIS_DIR = Path(__file__).resolve().parent
MS_ROOT = THIS_DIR / "MotionStreamer" if (THIS_DIR / "MotionStreamer").exists() else THIS_DIR
sys.path.insert(0, str(MS_ROOT))
sys.path.insert(0, str(MS_ROOT / "Evaluator_272"))


def _import_evaluator_modules():
    from mld.models.architectures.temos.textencoder.distillbert_actor import (
        DistilbertActorAgnosticEncoder,
    )
    from mld.models.architectures.temos.motionencoder.actor import (
        ActorAgnosticEncoder,
    )

    return DistilbertActorAgnosticEncoder, ActorAgnosticEncoder


def load_evaluator(ckpt_path: Path, device: torch.device):
    DistilbertActorAgnosticEncoder, ActorAgnosticEncoder = _import_evaluator_modules()
    distilbert_path = os.environ.get("MOTIONSTREAMER_DISTILBERT_PATH", "distilbert-base-uncased")
    textenc = DistilbertActorAgnosticEncoder(
        distilbert_path, num_layers=4, latent_dim=256
    )
    motenc = ActorAgnosticEncoder(
        nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["state_dict"]
    text_sd = {
        k.replace("textencoder.", ""): v
        for k, v in sd.items()
        if k.startswith("textencoder.")
    }
    motion_sd = {
        k.replace("motionencoder.", ""): v
        for k, v in sd.items()
        if k.startswith("motionencoder.")
    }
    textenc.load_state_dict(text_sd, strict=True)
    motenc.load_state_dict(motion_sd, strict=True)
    textenc.eval().to(device)
    motenc.eval().to(device)
    return textenc, motenc


def euclidean_distance_matrix(a, b):
    d1 = -2 * a @ b.T
    d2 = (a ** 2).sum(axis=1, keepdims=True)
    d3 = (b ** 2).sum(axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0))


def calc_top_k(argmax, k):
    n = argmax.shape[0]
    gt = np.arange(n)[:, None].repeat(n, 1)
    correct = np.zeros(n, dtype=bool)
    out = np.zeros((n, k), dtype=bool)
    for i in range(k):
        correct = correct | (argmax[:, i] == gt[:, i])
        out[:, i] = correct
    return out


def r_precision(text_emb, motion_emb, top_k=3):
    d = euclidean_distance_matrix(text_emb, motion_emb)
    matching = d.trace()
    arg = np.argsort(d, axis=1)
    top = calc_top_k(arg, top_k)
    return top.sum(0), matching


def diversity(emb, n=300):
    n = min(n, len(emb))
    a = emb[np.random.choice(len(emb), n, replace=False)]
    b = emb[np.random.choice(len(emb), n, replace=False)]
    return float(np.linalg.norm(a - b, axis=1).mean())


def calc_frechet(mu1, c1, mu2, c2, eps=1e-6):
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(c1.shape[0]) * eps
        covmean = linalg.sqrtm((c1 + offset).dot(c2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(c1) + np.trace(c2) - 2 * np.trace(covmean))


def activation_stats(x):
    return x.mean(axis=0), np.cov(x, rowvar=False)


# ---------------------------------------------------------------------------
# Test-set caption pairing (mirrors humanml3d_272/dataset_eval_t2m.py)
# ---------------------------------------------------------------------------

def load_test_pairs(data_root: Path, max_motion_length=300, min_motion_length=60,
                    fps=30, unit_length=4):
    motion_dir = data_root / "motion_data"
    text_dir = data_root / "texts"
    split = (data_root / "split" / "test.txt").read_text().splitlines()
    pairs = []  # (name, caption, motion_arr, m_length)
    for name in split:
        name = name.strip()
        if not name:
            continue
        m_file = motion_dir / f"{name}.npy"
        t_file = text_dir / f"{name}.txt"
        if not (m_file.exists() and t_file.exists()):
            continue
        motion = np.load(m_file)
        if len(motion) < min_motion_length or len(motion) >= max_motion_length:
            continue
        for line in t_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 4:
                continue
            caption = parts[0]
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
            if f_tag == 0.0 and t_tag == 0.0:
                m = motion
            else:
                m = motion[int(f_tag * fps): int(t_tag * fps)]
                if len(m) < min_motion_length or len(m) >= max_motion_length:
                    continue
            ml = (len(m) // unit_length) * unit_length
            if ml < min_motion_length:
                continue
            pairs.append((name, caption, m[:ml], ml))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluator_ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--pred_dir", default=None,
                   help="Directory of <name>.npy 272-dim predicted motions, keyed by test-set id. "
                        "Ignored when --gt_only is set.")
    p.add_argument("--gt_only", action="store_true",
                   help="Sanity check: use real motions for both 'pred' and 'real'. "
                        "Expected output: FID ~ 0, R-P ~ real, MM-Dist ~ real.")
    p.add_argument("--out_json", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_motion_length", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32,
                   help="R-Precision/MM-Dist are computed within chunks of this size (32 in MotionStreamer paper).")
    p.add_argument("--n_repeats", type=int, default=20,
                   help="MotionStreamer paper averages metrics over 20 random shuffles of the test set.")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] device = {device}")

    mean = np.load(data_root / "mean_std" / "Mean.npy")
    std = np.load(data_root / "mean_std" / "Std.npy")

    print("[+] Loading test pairs ...")
    pairs = load_test_pairs(data_root, max_motion_length=args.max_motion_length)
    print(f"    pairs: {len(pairs)}")

    print("[+] Loading evaluator ...")
    textenc, motenc = load_evaluator(Path(args.evaluator_ckpt), device)

    # Prepare aligned tensors for real & pred
    captions, real_motions, pred_motions, lengths = [], [], [], []
    skipped_no_pred = 0
    for name, caption, gt, ml in pairs:
        if args.gt_only:
            pred = gt
        else:
            pred_file = pred_dir / f"{name}.npy"
            if not pred_file.exists():
                skipped_no_pred += 1
                continue
            pred = np.load(pred_file)
            pred_ml = (len(pred) // 4) * 4
            if pred_ml < 60:
                skipped_no_pred += 1
                continue
            pred = pred[:pred_ml]
        captions.append(caption)
        real_motions.append(gt)
        pred_motions.append(pred)
        lengths.append(min(ml, len(pred)))

    n = min(len(real_motions), len(pred_motions), len(lengths))
    print(f"[+] aligned samples: {n}  (skipped {skipped_no_pred} test pairs with no pred)")

    def standardize_pad(arrs):
        out = np.zeros((len(arrs), args.max_motion_length, 272), dtype=np.float32)
        for i, a in enumerate(arrs):
            t = min(len(a), args.max_motion_length)
            out[i, :t] = (a[:t] - mean) / std
        return out

    real_np = standardize_pad(real_motions[:n])
    pred_np = standardize_pad(pred_motions[:n])
    lens = np.array(lengths[:n], dtype=np.int64)

    print("[+] Encoding ...")
    real_emb = []
    pred_emb = []
    text_emb = []
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            real_b = torch.from_numpy(real_np[i:j]).to(device).float()
            pred_b = torch.from_numpy(pred_np[i:j]).to(device).float()
            len_b = torch.from_numpy(lens[i:j]).to(device).long()
            text_b = captions[i:j]
            real_emb.append(motenc(real_b, len_b).loc.cpu().numpy())
            pred_emb.append(motenc(pred_b, len_b).loc.cpu().numpy())
            text_emb.append(textenc(text_b).loc.cpu().numpy())

    real_emb = np.concatenate(real_emb, axis=0)
    pred_emb = np.concatenate(pred_emb, axis=0)
    text_emb = np.concatenate(text_emb, axis=0)

    # Metrics — average over multiple random shuffles, computing R-Precision /
    # MM-Dist within chunks of 32 (the MotionStreamer / TMR / T2M-GPT protocol).
    chunk = args.batch_size
    rp_real_list, rp_pred_list = [], []
    ms_real_list, ms_pred_list = [], []
    fid_list, div_real_list, div_pred_list = [], [], []

    rng = np.random.default_rng(args.seed)
    for rep in range(args.n_repeats):
        idx = rng.permutation(n)
        rp_real = np.zeros(3)
        rp_pred = np.zeros(3)
        ms_real = 0.0
        ms_pred = 0.0
        nb = 0
        for i in range(0, n // chunk * chunk, chunk):
            j = i + chunk
            sub = idx[i:j]
            r, m = r_precision(text_emb[sub], real_emb[sub], top_k=3)
            rp_real += r
            ms_real += m
            r, m = r_precision(text_emb[sub], pred_emb[sub], top_k=3)
            rp_pred += r
            ms_pred += m
            nb += chunk
        rp_real /= nb
        rp_pred /= nb
        ms_real /= nb
        ms_pred /= nb
        rp_real_list.append(rp_real)
        rp_pred_list.append(rp_pred)
        ms_real_list.append(ms_real)
        ms_pred_list.append(ms_pred)

        mu_r, c_r = activation_stats(real_emb[idx])
        mu_p, c_p = activation_stats(pred_emb[idx])
        fid_list.append(calc_frechet(mu_r, c_r, mu_p, c_p))
        div_real_list.append(diversity(real_emb))
        div_pred_list.append(diversity(pred_emb))

    rp_real = np.stack(rp_real_list).mean(0)
    rp_pred = np.stack(rp_pred_list).mean(0)
    ms_real = float(np.mean(ms_real_list))
    ms_pred = float(np.mean(ms_pred_list))
    fid = float(np.mean(fid_list))
    div_real = float(np.mean(div_real_list))
    div_pred = float(np.mean(div_pred_list))
    rp_real_std = np.stack(rp_real_list).std(0).tolist()
    rp_pred_std = np.stack(rp_pred_list).std(0).tolist()

    metrics = {
        "n_samples_used": int(nb),
        "n_repeats": args.n_repeats,
        "fid": fid,
        "fid_std": float(np.std(fid_list)),
        "diversity_real": div_real,
        "diversity_pred": div_pred,
        "r_precision_real": rp_real.tolist(),
        "r_precision_real_std": rp_real_std,
        "r_precision_pred": rp_pred.tolist(),
        "r_precision_pred_std": rp_pred_std,
        "matching_score_real": ms_real,
        "matching_score_pred": ms_pred,
        "config": {
            "evaluator_ckpt": str(args.evaluator_ckpt),
            "data_root": str(args.data_root),
            "pred_dir": str(args.pred_dir) if args.pred_dir else None,
            "gt_only": args.gt_only,
            "batch_size": args.batch_size,
        },
    }
    out_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Shared text-to-motion retrieval metrics (FID / R-Precision / MM-Dist / Diversity).

These are the canonical Guo et al. / T2M-GPT / MotionStreamer metric primitives,
operating purely on pre-computed motion/text *embeddings*. Both the HumanML3D-263
(MoMask) and the MotionStreamer-272 evaluators feed their own encoder outputs
through :func:`aggregate_t2m_metrics`, so the scoring protocol stays identical and
verifiable across feature spaces.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def euclidean_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d1 = -2 * a @ b.T
    d2 = (a ** 2).sum(axis=1, keepdims=True)
    d3 = (b ** 2).sum(axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0))


def calc_top_k(argmax: np.ndarray, k: int) -> np.ndarray:
    n = argmax.shape[0]
    gt = np.arange(n)[:, None].repeat(n, 1)
    correct = np.zeros(n, dtype=bool)
    out = np.zeros((n, k), dtype=bool)
    for i in range(k):
        correct = correct | (argmax[:, i] == gt[:, i])
        out[:, i] = correct
    return out


def r_precision(text_emb: np.ndarray, motion_emb: np.ndarray, top_k: int = 3):
    """Return (per-rank correct counts summed over the batch, matching score)."""
    d = euclidean_distance_matrix(text_emb, motion_emb)
    matching = d.trace()
    arg = np.argsort(d, axis=1)
    top = calc_top_k(arg, top_k)
    return top.sum(0), matching


def diversity(emb: np.ndarray, n: int = 300, rng: Optional[np.random.Generator] = None) -> float:
    n = min(n, len(emb))
    if n <= 0:
        return 0.0
    if rng is None:
        a = emb[np.random.choice(len(emb), n, replace=False)]
        b = emb[np.random.choice(len(emb), n, replace=False)]
    else:
        a = emb[rng.choice(len(emb), n, replace=False)]
        b = emb[rng.choice(len(emb), n, replace=False)]
    return float(np.linalg.norm(a - b, axis=1).mean())


def calc_frechet(mu1, c1, mu2, c2, eps: float = 1e-6) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(c1.shape[0]) * eps
        covmean = linalg.sqrtm((c1 + offset).dot(c2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(c1) + np.trace(c2) - 2 * np.trace(covmean))


def activation_stats(x: np.ndarray):
    return x.mean(axis=0), np.cov(x, rowvar=False)


def aggregate_t2m_metrics(
    text_emb: np.ndarray,
    real_emb: np.ndarray,
    pred_emb: np.ndarray,
    n_repeats: int = 20,
    chunk: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    """Average T2M metrics over ``n_repeats`` random shuffles of the sample pool.

    R-Precision / Matching-Score are computed within chunks of ``chunk`` (the
    MotionStreamer / TMR / T2M-GPT protocol); FID and Diversity use the full pool
    per shuffle. This mirrors
    ``ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py`` exactly.

    Args:
        text_emb: ``(N, D)`` text embeddings, paired row-wise with the motions.
        real_emb: ``(N, D)`` GT motion embeddings.
        pred_emb: ``(N, D)`` predicted motion embeddings.
        n_repeats: number of random shuffles to average over.
        chunk: per-chunk batch size for R-Precision / MM-Dist.
        seed: RNG seed for reproducible shuffles.
    """
    n = min(len(text_emb), len(real_emb), len(pred_emb))
    text_emb, real_emb, pred_emb = text_emb[:n], real_emb[:n], pred_emb[:n]

    rp_real_list, rp_pred_list = [], []
    ms_real_list, ms_pred_list = [], []
    fid_list, div_real_list, div_pred_list = [], [], []

    rng = np.random.default_rng(seed)
    nb = 0
    for _ in range(n_repeats):
        idx = rng.permutation(n)
        rp_real = np.zeros(3)
        rp_pred = np.zeros(3)
        ms_real = ms_pred = 0.0
        nb = 0
        for i in range(0, n // chunk * chunk, chunk):
            sub = idx[i : i + chunk]
            r, m = r_precision(text_emb[sub], real_emb[sub], top_k=3)
            rp_real += r
            ms_real += m
            r, m = r_precision(text_emb[sub], pred_emb[sub], top_k=3)
            rp_pred += r
            ms_pred += m
            nb += chunk
        if nb == 0:
            raise ValueError(f"Not enough samples ({n}) for chunk size {chunk}")
        rp_real_list.append(rp_real / nb)
        rp_pred_list.append(rp_pred / nb)
        ms_real_list.append(ms_real / nb)
        ms_pred_list.append(ms_pred / nb)

        mu_r, c_r = activation_stats(real_emb[idx])
        mu_p, c_p = activation_stats(pred_emb[idx])
        fid_list.append(calc_frechet(mu_r, c_r, mu_p, c_p))
        div_real_list.append(diversity(real_emb))
        div_pred_list.append(diversity(pred_emb))

    return {
        "n_samples_used": int(nb),
        "n_repeats": n_repeats,
        "fid": float(np.mean(fid_list)),
        "fid_std": float(np.std(fid_list)),
        "diversity_real": float(np.mean(div_real_list)),
        "diversity_pred": float(np.mean(div_pred_list)),
        "r_precision_real": np.stack(rp_real_list).mean(0).tolist(),
        "r_precision_real_std": np.stack(rp_real_list).std(0).tolist(),
        "r_precision_pred": np.stack(rp_pred_list).mean(0).tolist(),
        "r_precision_pred_std": np.stack(rp_pred_list).std(0).tolist(),
        "matching_score_real": float(np.mean(ms_real_list)),
        "matching_score_pred": float(np.mean(ms_pred_list)),
    }

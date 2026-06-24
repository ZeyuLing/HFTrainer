#!/usr/bin/env python3
"""Compute Mean/Std (38,) for the G1-native T2M motion representation.

Samples clips from the training list, encodes each to the 38-d representation
(see ``physflow/g1_repr.py``), and accumulates per-dimension mean/std over all
frames.  Writes ``Mean.npy`` / ``Std.npy`` into ``--out-dir`` (the layout
expected by ``HyMotionT2MBundle(mean_std_dir=...)``).

NOTE: recompute after the full 456k retarget re-run finishes -- stats computed
on partially-stale data will have a slightly off translation scale.

Usage::

    python3 scripts/embodied/compute_g1_motion_stats.py \
        --list data/annotation/train_g1_t2m.json \
        --out-dir data/g1_t2m_stats --num-samples 20000 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Allow ``import`` of the repo package when run as a script.
import sys
sys.path.insert(0, os.getcwd())
from hftrainer.models.motion.physflow.g1_repr import (  # noqa: E402
    G1_MOTION_DIM, encode_g1_motion,
)


def _load_encode(g1_dir, rel):
    try:
        data = dict(np.load(os.path.join(g1_dir, rel), allow_pickle=True))
        m = encode_g1_motion(data)  # (T, 38)
        if m.ndim != 2 or m.shape[1] != G1_MOTION_DIM or m.shape[0] < 2:
            return None
        # per-dim sum, sumsq, count
        return (m.sum(0), (m.astype(np.float64) ** 2).sum(0), m.shape[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='data/annotation/train_g1_t2m.json')
    ap.add_argument('--g1-dir', default='data/g1')
    ap.add_argument('--out-dir', default='data/g1_t2m_stats')
    ap.add_argument('--num-samples', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.list) as f:
        blob = json.load(f)
    items = blob['items'] if isinstance(blob, dict) else blob
    rels = [it['g1_path'] for it in items]
    random.seed(args.seed)
    if args.num_samples and args.num_samples < len(rels):
        rels = random.sample(rels, args.num_samples)
    print(f'[stats] computing over {len(rels)} clips, {args.workers} workers',
          flush=True)

    dsum = np.zeros(G1_MOTION_DIM, dtype=np.float64)
    dsq = np.zeros(G1_MOTION_DIM, dtype=np.float64)
    count = 0
    n_ok = n_bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_load_encode, args.g1_dir, r) for r in rels]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r is None:
                n_bad += 1
                continue
            s, sq, c = r
            dsum += s
            dsq += sq
            count += c
            n_ok += 1
            if (i + 1) % 2000 == 0:
                print(f'[stats]   {i+1}/{len(rels)} clips, frames={count}',
                      flush=True)

    if count == 0:
        raise RuntimeError('No valid clips encoded; check --g1-dir / --list')

    mean = (dsum / count).astype(np.float32)
    var = (dsq / count) - (dsum / count) ** 2
    std = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'Mean.npy'), mean)
    np.save(os.path.join(args.out_dir, 'Std.npy'), std)
    print(f'[stats] ok={n_ok} bad={n_bad} frames={count}', flush=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print('[stats] mean:', mean)
    print('[stats] std :', std)
    print(f'[stats] wrote -> {args.out_dir}/Mean.npy, Std.npy', flush=True)


if __name__ == '__main__':
    main()

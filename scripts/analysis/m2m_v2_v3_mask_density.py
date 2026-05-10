"""Compare the fraction of (frame, channel) cells that the model is asked to
generate (mask=1) under v2 vs v3 sampler, on identical (T=360, 198) shape.

Higher mean(mask) <=> more positions must be predicted.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, '.')

from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import sample_condition
from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import sample_condition_v3


T = 360
D = 198
N = 4000

# Channel groups: trans 0:3, rot6d 3:135, pos 135:198
TRANS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 135)
POS_SLICE = slice(135, 198)


def run(sampler_name, n=N, seed=0):
    rng = np.random.RandomState(seed)
    masks = np.empty((n, T, D), dtype=np.float32)
    edit_flags = np.empty(n, dtype=bool)
    for i in range(n):
        if sampler_name == 'v2':
            m, e = sample_condition(T, rng)
        else:
            m, e = sample_condition_v3(T, rng)
        masks[i] = m
        edit_flags[i] = e
    return masks, edit_flags


def report(name, masks, edit_flags):
    overall = float(masks.mean())
    trans = float(masks[..., TRANS_SLICE].mean())
    rot = float(masks[..., ROT_SLICE].mean())
    pos = float(masks[..., POS_SLICE].mean())

    # per-sample fraction
    per_sample = masks.reshape(masks.shape[0], -1).mean(axis=1)
    # frames with anything to generate
    frame_active = (masks.sum(axis=-1) > 0).mean()
    # full-generate frames (all 198 channels masked)
    frame_full = (masks.sum(axis=-1) == D).mean()
    # known frames
    frame_known = (masks.sum(axis=-1) == 0).mean()

    edit_pct = float(edit_flags.mean())

    print(f'\n=== {name} ===')
    print(f'  mean(mask) overall        : {overall*100:.2f}%   (= avg fraction of cells to generate)')
    print(f'  trans channels (3 dims)   : {trans*100:.2f}%')
    print(f'  rot6d  channels (132 dims): {rot*100:.2f}%')
    print(f'  pos    channels (63 dims) : {pos*100:.2f}%')
    print(f'  per-sample mean(mask) p50/p90 : {np.median(per_sample)*100:.2f}% / {np.percentile(per_sample,90)*100:.2f}%')
    print(f'  frames with any gen target  : {frame_active*100:.2f}%')
    print(f'  fully-generate frames       : {frame_full*100:.2f}%')
    print(f'  fully-known frames          : {frame_known*100:.2f}%')
    print(f'  edit_mode rate              : {edit_pct*100:.2f}%')
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f'[mask density compare] N={args.n}, T={T}, D={D}')

    m_v2, e_v2 = run('v2', n=args.n, seed=args.seed)
    g_v2 = report('v2 (current production until 2026-04-26 13:32)', m_v2, e_v2)

    m_v3, e_v3 = run('v3', n=args.n, seed=args.seed + 1)
    g_v3 = report('v3 (current after restart 2026-04-26 13:32)', m_v3, e_v3)

    delta = g_v3 - g_v2
    direction = 'MORE' if delta > 0 else 'LESS'
    print(f'\n=== verdict ===')
    print(f'  v2 fraction-to-generate : {g_v2*100:.2f}%')
    print(f'  v3 fraction-to-generate : {g_v3*100:.2f}%')
    print(f'  Δ (v3 - v2)             : {delta*100:+.2f} pp ({delta/g_v2*100:+.1f}% relative)')
    print(f'  v3 asks model to generate {direction} positions per sample on average.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build the v2 E3 keyframe-interpolation test set from the Private pool.

Why a v2 build (2026-04-25)
---------------------------
The original `eval_e3_keyframe.json` was sampled from training-side data
and labelled by source-folder. The v2 follows the same recipe as
`build_e2_inbetween_v2_data.py`:

  1. Sample exclusively from the held-out Private pool
     (`/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private`)
     so models have not seen these motions during training.
  2. Stratify across BOTH the action-type category and a 4-bucket
     pelvis-speed axis (static / slow / moderate / fast). Speed
     is computed as path_len_xz / duration_sec and buckets reflect the
     Private-pool empirical distribution (p25 / p50 / p75 cutoffs at
     0.05 / 0.10 / 0.20 m/s — well-calibrated for typical Private clips
     where most actions are upper-body driven and the pelvis travels
     little).
  3. Keep the same per-motion non-trivial filters as E2 (length window,
     non-T-pose start+end, head↔tail pose-delta or pelvis-path minimum).
  4. Aim for >= 220 unique motions shared across all six E3 settings
     (every-5f / every-10f / every-15f (legacy C) / every-30f (legacy A)
     / every-60f (legacy B) / adaptive-acceleration (legacy D)).

Captions
--------
This script writes `eval_e3_keyframe_v2.json` with the raw scan caption.
The companion `tools/rewrite_caption_file.py` then runs every item
through the rewriter service to produce
`eval_e3_keyframe_v2_rewritten.json` (12-20 word "A person ..." form).

Usage
-----
    python3 tools/build_e3_keyframe_v2_data.py
    python3 tools/build_e3_keyframe_v2_data.py --n-samples 240 --cap 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

WORKSPACE_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
SCAN = WORKSPACE_ROOT / 'data/eval/m2m_v2/_pelvis_pathlen_scan.json'
OUT_DIR = WORKSPACE_ROOT / 'data/eval/m2m_v2'
OUT_PATH = OUT_DIR / 'eval_e3_keyframe_v2.json'
OUT_REWRITTEN_PATH = OUT_DIR / 'eval_e3_keyframe_v2_rewritten.json'
PRIVATE_ROOT = Path(
    '/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private'
)

sys.path.insert(0, str(WORKSPACE_ROOT))
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_smplx_pose,
)

MIN_FRAMES = 120
MAX_FRAMES = 360

# Same calibration as E2: Private clips do not contain real T-poses, but
# we still want the head/tail to be clearly non-identity to avoid trivial
# rest-pose interpolation cases. The 0.10 cut also passes every E2-v2
# hand-picked item.
TPOSE_WINDOW = 3
TPOSE_DEV_MIN = 0.10
POSE_DELTA_MIN = 0.03
PATH_LEN_MIN = 0.3

# Speed buckets driven by the Private-scan empirical distribution
# (see tools/build_e3_keyframe_v2_data.py docstring for the percentile
# justification). Boundaries align with p25 / p50 / p75 of the full pool.
SPEED_BUCKETS = [
    # (label, lo m/s inclusive, hi m/s exclusive)
    ('static',   0.00,  0.05),
    ('slow',     0.05,  0.10),
    ('moderate', 0.10,  0.20),
    ('fast',     0.20,  10.0),
]
SPEED_LABELS = [b[0] for b in SPEED_BUCKETS]

DEFAULT_N_SAMPLES = 240
DEFAULT_MAX_PER_CELL = 8           # ≤8 motions per (category, speed) cell
DEFAULT_MAX_PER_CATEGORY = 30      # also cap per category to spread
DEFAULT_SEED = 42

IDENTITY_6D = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def canonical_path(p: str) -> str:
    p = str(p)
    marker = 'apdcephfs_cq11/share_1467498/'
    if marker in p and not p.startswith('/'):
        p = '/' + p[p.index(marker):]
    return p


def speed_bucket(speed_mps: float) -> str:
    for label, lo, hi in SPEED_BUCKETS:
        if lo <= speed_mps < hi:
            return label
    return 'fast'


def load_motion_rot6d(npz_path: Path) -> Optional[np.ndarray]:
    try:
        d = np.load(npz_path, allow_pickle=True)
        pk = 'poses' if 'poses' in d.files else 'body_pose'
        poses = d[pk].astype(np.float32)
        rot6d_flat = process_smplx_pose(
            poses, 'rotation_6d', 'smpl_22')
        T = rot6d_flat.shape[0]
        return rot6d_flat.reshape(T, 22, 6)
    except Exception:
        return None


def load_pelvis_trans(npz_path: Path) -> Optional[np.ndarray]:
    try:
        d = np.load(npz_path, allow_pickle=True)
        tk = 'trans' if 'trans' in d.files else 'transl'
        return np.asarray(d[tk], dtype=np.float32)
    except Exception:
        return None


def tpose_dev(body6d_window: np.ndarray) -> float:
    return float(np.linalg.norm(body6d_window - IDENTITY_6D, axis=-1).mean())


def pose_delta(head6d: np.ndarray, tail6d: np.ndarray) -> float:
    head_mean = head6d.mean(axis=0)
    tail_mean = tail6d.mean(axis=0)
    return float(np.linalg.norm(head_mean - tail_mean, axis=-1).mean())


def pelvis_path_len_xz(trans: np.ndarray) -> float:
    if trans is None or len(trans) < 2:
        return 0.0
    diffs = np.diff(trans[:, [0, 2]], axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum())


def evaluate(entry: Dict) -> Optional[Dict]:
    """Load NPZ, run filters, return enriched candidate dict or None."""
    path = Path(entry['path'])
    if not path.exists():
        return None
    rot6d = load_motion_rot6d(path)
    if rot6d is None:
        return None
    T = rot6d.shape[0]
    if not (MIN_FRAMES <= T <= MAX_FRAMES):
        return None
    body6d = rot6d[:, 1:]
    w = TPOSE_WINDOW
    head_dev = tpose_dev(body6d[:w])
    tail_dev = tpose_dev(body6d[-w:])
    if head_dev < TPOSE_DEV_MIN or tail_dev < TPOSE_DEV_MIN:
        return None
    pose_d = pose_delta(body6d[:w], body6d[-w:])
    trans = load_pelvis_trans(path)
    path_len = pelvis_path_len_xz(trans) if trans is not None else 0.0
    if pose_d < POSE_DELTA_MIN and path_len < PATH_LEN_MIN:
        return None
    fps = float(entry.get('fps', 30.0)) or 30.0
    duration_sec = T / fps
    speed_mps = float(path_len) / duration_sec if duration_sec > 0 else 0.0
    return {
        'motion_path': canonical_path(str(path)),
        'action_name': entry.get('action_name', ''),
        'caption_en': entry.get('caption_en', ''),
        'category': entry.get('category', 'other'),
        'source': entry.get('rel_dir', ''),
        'num_frames': int(T),
        'fps': fps,
        'duration_sec': round(duration_sec, 2),
        'pelvis_speed_mps': round(speed_mps, 4),
        'speed_bucket': speed_bucket(speed_mps),
        'head_tpose_dev': round(head_dev, 4),
        'tail_tpose_dev': round(tail_dev, 4),
        'pose_delta': round(pose_d, 4),
        'path_len_xz': round(path_len, 3),
    }


def prefilter(scan: List[Dict]) -> List[Dict]:
    out = []
    seen_action = set()
    for e in scan:
        n = int(e.get('num_frames', 0) or 0)
        if not (MIN_FRAMES <= n <= MAX_FRAMES):
            continue
        a = e.get('action_name', '')
        if a and a in seen_action:
            continue
        out.append(e)
        if a:
            seen_action.add(a)
    return out


def cell_score(e: Dict) -> float:
    """Cheap pre-ordering signal: prefer items with more pelvis travel
    (likely to pass path_len/pose_delta) and lower duplication risk."""
    return float(e.get('path_len_xz', 0.0) or 0.0)


def order_within_cell(items: List[Dict], rng: random.Random) -> List[Dict]:
    items = list(items)
    items.sort(key=lambda e: -cell_score(e))
    out, bucket = [], []
    for e in items:
        bucket.append(e)
        if len(bucket) >= 10:
            rng.shuffle(bucket)
            out.extend(bucket)
            bucket = []
    if bucket:
        rng.shuffle(bucket)
        out.extend(bucket)
    return out


def stratified_pick(
    scan: List[Dict],
    n_target: int,
    cap_per_cell: int,
    cap_per_cat: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)

    print(f'  prefilter: frames in [{MIN_FRAMES}, {MAX_FRAMES}], '
          f'1-per-action_name')
    flat = prefilter(scan)
    print(f'  after prefilter: {len(flat)} candidates')

    # Bucket by (category, speed_bucket)
    by_cell: Dict[tuple, List[Dict]] = defaultdict(list)
    for e in flat:
        cat = e.get('category', 'other')
        n = e.get('num_frames', 0) or 0
        pl = e.get('path_len_xz', 0.0) or 0.0
        dur = float(e.get('duration_sec') or n / 30.0)
        sp = pl / dur if dur else 0.0
        by_cell[(cat, speed_bucket(sp))].append(e)

    cell_keys = sorted(by_cell.keys())
    print(f'  populated (category, speed) cells: {len(cell_keys)} '
          f'(cap_per_cell={cap_per_cell}, cap_per_cat={cap_per_cat})')

    ordered = {k: order_within_cell(by_cell[k], rng) for k in cell_keys}

    picked: List[Dict] = []
    per_cell_kept: Dict[tuple, int] = defaultdict(int)
    per_cat_kept: Dict[str, int] = defaultdict(int)
    cell_cursor: Dict[tuple, int] = defaultdict(int)
    rejects = {'npz_filter': 0}

    while len(picked) < n_target:
        progressed = False
        for k in cell_keys:
            if len(picked) >= n_target:
                break
            cat, _spd = k
            if per_cell_kept[k] >= cap_per_cell:
                continue
            if per_cat_kept[cat] >= cap_per_cat:
                continue
            pool = ordered[k]
            while cell_cursor[k] < len(pool):
                e = pool[cell_cursor[k]]
                cell_cursor[k] += 1
                cand = evaluate(e)
                if cand is None:
                    rejects['npz_filter'] += 1
                    continue
                picked.append(cand)
                per_cell_kept[k] += 1
                per_cat_kept[cat] += 1
                progressed = True
                break
        if not progressed:
            break
        if len(picked) % 20 == 0:
            print(f'    picked={len(picked):3d} '
                  f'cells={len([k for k,v in per_cell_kept.items() if v>0])}')

    print(f'  NPZ-load rejects: {rejects["npz_filter"]}')
    return picked


def write_output(path: Path, samples: List[Dict], seed: int,
                 cap_per_cell: int, cap_per_cat: int) -> None:
    cat_counts: Dict[str, int] = defaultdict(int)
    speed_counts: Dict[str, int] = defaultdict(int)
    cell_counts: Dict[tuple, int] = defaultdict(int)
    for s in samples:
        cat_counts[s['category']] += 1
        speed_counts[s['speed_bucket']] += 1
        cell_counts[(s['category'], s['speed_bucket'])] += 1

    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_cat[s['category']].append(s)
    total = len(samples) or 1
    cat_detail: Dict[str, Dict] = {}
    for c in sorted(by_cat.keys()):
        rows = by_cat[c]
        actions = sorted({r.get('action_name', '') for r in rows
                          if r.get('action_name')})
        caps = []
        seen_cap = set()
        for r in rows:
            cap = (r.get('caption_en') or '').strip()
            if cap and cap not in seen_cap:
                caps.append(cap)
                seen_cap.add(cap)
            if len(caps) >= 5:
                break
        # speed mix within category
        sub_speed: Dict[str, int] = defaultdict(int)
        for r in rows:
            sub_speed[r['speed_bucket']] += 1
        cat_detail[c] = {
            'count': len(rows),
            'percent': round(100.0 * len(rows) / total, 1),
            'unique_actions': len(actions),
            'example_actions': actions[:5],
            'example_captions_en': caps,
            'speed_mix': dict(sub_speed),
        }

    speed_detail: Dict[str, Dict] = {}
    for sl in SPEED_LABELS:
        cnt = speed_counts.get(sl, 0)
        speed_detail[sl] = {
            'count': cnt,
            'percent': round(100.0 * cnt / total, 1),
        }

    pose_deltas = [s['pose_delta'] for s in samples]
    path_lens = [s['path_len_xz'] for s in samples]
    speeds = [s['pelvis_speed_mps'] for s in samples]
    frames = [s['num_frames'] for s in samples]
    frames_sorted = sorted(frames)
    meta = {
        'task_id': 'E3',
        'task_name': 'Keyframe Interpolation (v2, 6-setting ablation)',
        'version': 'v2_private_20260425',
        'source': str(PRIVATE_ROOT),
        'source_scan': str(SCAN.relative_to(WORKSPACE_ROOT)),
        'description': (
            'Private (Dongming) held-out pool, stratified across '
            'action-type categories AND a 4-bucket pelvis-speed axis '
            '(static / slow / moderate / fast at 0.05 / 0.10 / 0.20 m/s '
            'cutoffs). Same non-trivial filters as E2-v2 (length window, '
            'head+tail non-T-pose, head↔tail pose-delta or pelvis-path '
            'minimum, 1-sample-per-action-name). Shared across all six '
            'E3 settings (every-5f / every-10f / every-15f / every-30f '
            '/ every-60f / adaptive-acceleration).'
        ),
        'total_items': len(samples),
        'category_distribution': dict(cat_counts),
        'category_distribution_detail': cat_detail,
        'speed_distribution': dict(speed_counts),
        'speed_distribution_detail': speed_detail,
        'cell_distribution': {
            f'{c}|{s}': n for (c, s), n in sorted(cell_counts.items())
        },
        'speed_buckets_def': {
            label: {'min_mps': lo, 'max_mps': hi}
            for label, lo, hi in SPEED_BUCKETS
        },
        'frame_stats': {
            'min': frames_sorted[0] if frames_sorted else 0,
            'max': frames_sorted[-1] if frames_sorted else 0,
            'mean': round(sum(frames_sorted) / max(1, len(frames_sorted)), 1),
            'median': frames_sorted[len(frames_sorted) // 2]
                      if frames_sorted else 0,
        },
        'speed_stats_mps': {
            'min': round(min(speeds), 4) if speeds else 0,
            'max': round(max(speeds), 4) if speeds else 0,
            'mean': round(sum(speeds) / max(1, len(speeds)), 4)
                     if speeds else 0,
        },
        'seed': seed,
        'pose_delta_range': [round(min(pose_deltas), 4) if pose_deltas else 0,
                             round(max(pose_deltas), 4) if pose_deltas else 0],
        'path_len_range': [round(min(path_lens), 3) if path_lens else 0,
                           round(max(path_lens), 3) if path_lens else 0],
        'filters': {
            'min_frames': MIN_FRAMES,
            'max_frames': MAX_FRAMES,
            'tpose_window': TPOSE_WINDOW,
            'tpose_dev_min': TPOSE_DEV_MIN,
            'pose_delta_min': POSE_DELTA_MIN,
            'path_len_min': PATH_LEN_MIN,
            'cap_per_cell': cap_per_cell,
            'cap_per_category': cap_per_cat,
        },
    }
    data_list = []
    for i, s in enumerate(samples):
        data_list.append({
            'motion_path': s['motion_path'],
            'action_name': s['action_name'],
            'caption_en': s['caption_en'],
            'category': s['category'],
            'speed_bucket': s['speed_bucket'],
            'pelvis_speed_mps': s['pelvis_speed_mps'],
            'source': s['source'],
            'num_frames': s['num_frames'],
            'fps': s['fps'],
            'duration_sec': s['duration_sec'],
            '_sample_idx': i,
            '_pose_delta': s['pose_delta'],
            '_path_len_xz': s['path_len_xz'],
            '_head_tpose_dev': s['head_tpose_dev'],
            '_tail_tpose_dev': s['tail_tpose_dev'],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'meta': meta, 'data_list': data_list}, f,
                  ensure_ascii=False, indent=2)
    print(f'  wrote {path.relative_to(WORKSPACE_ROOT)} '
          f'(n={len(data_list)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument('--cap', type=int, default=DEFAULT_MAX_PER_CELL,
                    help='Max samples per (category, speed) cell.')
    ap.add_argument('--cap-cat', type=int, default=DEFAULT_MAX_PER_CATEGORY,
                    help='Max samples per category.')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    print(f'Loading {SCAN.relative_to(WORKSPACE_ROOT)}...')
    scan: List[Dict] = json.load(open(SCAN))
    print(f'  {len(scan)} Private motions in scan')

    print(f'\n== Stratified picking '
          f'(n_target={args.n_samples}, cap_per_cell={args.cap}, '
          f'cap_per_cat={args.cap_cat}) ==')
    picked = stratified_pick(
        scan, args.n_samples, args.cap, args.cap_cat, args.seed,
    )
    print(f'\n  selected {len(picked)} samples')

    cat_count: Dict[str, int] = defaultdict(int)
    spd_count: Dict[str, int] = defaultdict(int)
    for p in picked:
        cat_count[p['category']] += 1
        spd_count[p['speed_bucket']] += 1
    print('  category distribution:')
    for c in sorted(cat_count):
        print(f'    {c:18s} {cat_count[c]}')
    print('  speed-bucket distribution:')
    for s in SPEED_LABELS:
        print(f'    {s:10s} {spd_count.get(s, 0)}')

    print('\n== Writing output ==')
    write_output(OUT_PATH, picked, args.seed, args.cap, args.cap_cat)
    write_output(OUT_REWRITTEN_PATH, picked, args.seed, args.cap, args.cap_cat)
    print('\ndone.')


if __name__ == '__main__':
    main()

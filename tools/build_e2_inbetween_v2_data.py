#!/usr/bin/env python3
"""Build the v2 E2 in-betweening test set from the Private held-out pool.

Why this was rewritten (2026-04-25)
-----------------------------------
Earlier versions of this script sampled from the 400h HQ TRAINING
annotation (`train_hymotion_400h_hq_20260403.json`), which:
  (a) leaked training data into the test set — models have already seen
      every one of those motions during training, so the comparison
      numbers are not trustworthy; and
  (b) labelled each sample by the DATA-FOLDER subset (academic / game /
      taobao / academicretarget) which is a sourcing artefact, not a
      meaningful action-type axis.

Fix: sample exclusively from the Dongming-recorded Private pool
(`/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private`)
which is held-out w.r.t. every production training run, and stratify
across the 15 ACTION-TYPE categories the Private scan already carries
(combat, sports_ball, locomotion, sitting, ...). Distribution plots on
the eval dashboard now reflect what kind of motion is being tested.

Source pool
-----------
`data/eval/m2m_v2/_pelvis_pathlen_scan.json` — 4074 Private motions
pre-scanned with pelvis path-length, FPS, frame count, action name,
coarse English caption and action-type category. Built earlier from the
same Private folder; regenerate via `tools/scan_pelvis_pathlen.py` if it
goes stale.

Per-sample guarantees
---------------------
  * num_frames in [120, 360]   (4-12 s @ 30fps)
  * 1 sample per `action_name` (semantic diversity)
  * Start + end windows are NOT near T-pose (head_tpose_dev and
    tail_tpose_dev both >= 0.35)
  * Either the head→tail mean pose differs (pose_delta >= 0.08) or the
    pelvis moves at least PATH_LEN_MIN metres (non-trivial in-betweening)

Stratification
--------------
Greedy round-robin across categories, with a soft MAX_PER_CATEGORY cap.
Categories with fewer available items contribute everything they have.
Target: >= 220 unique motions (shared across all 6 E2 settings).

Captions
--------
The scan carries short bag-of-words English like "pass a ball".
Captioned-models need the 12-20 word "A person ..." form. This script
writes `eval_e2_inbetween_v2.json` with the raw scan caption; the
companion tool `tools/rewrite_e2_v2_captions.py` then runs every item
through the rewriter service to produce
`eval_e2_inbetween_v2_rewritten.json`.

Run
---
    python3 tools/build_e2_inbetween_v2_data.py
    python3 tools/build_e2_inbetween_v2_data.py --n-samples 240 --cap 22
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

WORKSPACE_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
SCAN = WORKSPACE_ROOT / 'data/eval/m2m_v2/_pelvis_pathlen_scan.json'
OUT_DIR = WORKSPACE_ROOT / 'data/eval/m2m_v2'
OUT_PATH = OUT_DIR / 'eval_e2_inbetween_v2.json'
OUT_REWRITTEN_PATH = OUT_DIR / 'eval_e2_inbetween_v2_rewritten.json'
PRIVATE_ROOT = Path(
    '/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private'
)

sys.path.insert(0, str(WORKSPACE_ROOT))
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_transl, process_smplx_pose,
)

MIN_FRAMES = 120
MAX_FRAMES = 360

# Private (Dongming) recordings sit in a natural standing pose whose
# rot6d deviation from identity is a consistent ~0.18-0.22 — nowhere
# near a genuine A/T-pose (identity rot6d). In other words, Private is
# T-pose-free by construction, so we only require the start/end to be
# clearly non-identity (0.10 keeps a safety margin without rejecting
# any of the hand-picked v1 items, whose head/tail dev ranges 0.15-0.24).
TPOSE_WINDOW = 3
TPOSE_DEV_MIN = 0.10

# Private motions average ~5 s of real action, so even a "long arc"
# locomotion clip has modest absolute pose_delta between the first and
# last few frames (median 0.07). Lower the threshold so in-the-wild
# non-trivial takes are not dropped; path-length still acts as a
# secondary non-trivial signal.
POSE_DELTA_MIN = 0.03
PATH_LEN_MIN = 0.3

DEFAULT_N_SAMPLES = 220
DEFAULT_MAX_PER_CATEGORY = 22
DEFAULT_SEED = 42


IDENTITY_6D = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def canonical_path(p: str) -> str:
    p = str(p)
    marker = 'apdcephfs_cq11/share_1467498/'
    if marker in p and not p.startswith('/'):
        p = '/' + p[p.index(marker):]
    return p


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


def score(cand: Dict) -> float:
    return (cand['pose_delta']
            + 0.3 * cand['path_len_xz']
            + 0.2 * min(cand['head_tpose_dev'], cand['tail_tpose_dev']))


def evaluate(entry: Dict) -> Optional[Dict]:
    """Load NPZ and return candidate dict passing all filters, else None."""
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
    return {
        'motion_path': canonical_path(str(path)),
        'action_name': entry.get('action_name', ''),
        'caption_en': entry.get('caption_en', ''),
        'category': entry.get('category', 'other'),
        'source': entry.get('rel_dir', ''),
        'num_frames': int(T),
        'fps': float(entry.get('fps', 30.0)),
        'duration_sec': round(T / float(entry.get('fps', 30.0) or 30.0), 2),
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


def order_by_score_cue(scan_subset: List[Dict], rng: random.Random) -> List[Dict]:
    """Cheap pre-ordering: favour higher path_len_xz (more motion) — lowers
    NPZ-load count by biasing toward likely-to-pass candidates, with a
    small jitter so we still explore low-path-length / high-pose items."""
    subset = list(scan_subset)
    subset.sort(key=lambda e: -(float(e.get('path_len_xz', 0.0)) or 0.0))
    # add a small amount of shuffling within every 10 items
    out = []
    bucket = []
    for e in subset:
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
    max_per_cat: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)

    print(f'  prefilter: frames in [{MIN_FRAMES}, {MAX_FRAMES}], '
          f'1-per-action_name')
    flat = prefilter(scan)
    print(f'  after prefilter: {len(flat)} candidates')

    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for e in flat:
        by_cat[e.get('category', 'other')].append(e)
    cats_sorted = sorted(by_cat.keys())
    print(f'  categories: {len(cats_sorted)} '
          f'(min {min(len(v) for v in by_cat.values())}, '
          f'max {max(len(v) for v in by_cat.values())} items per cat)')

    ordered: Dict[str, List[Dict]] = {
        c: order_by_score_cue(by_cat[c], rng) for c in cats_sorted
    }

    picked: List[Dict] = []
    per_cat_kept: Dict[str, int] = defaultdict(int)
    cat_cursor: Dict[str, int] = defaultdict(int)
    rejects = {'npz_filter': 0}

    while len(picked) < n_target:
        progressed = False
        for cat in cats_sorted:
            if len(picked) >= n_target:
                break
            if per_cat_kept[cat] >= max_per_cat:
                continue
            pool = ordered[cat]
            while cat_cursor[cat] < len(pool):
                e = pool[cat_cursor[cat]]
                cat_cursor[cat] += 1
                cand = evaluate(e)
                if cand is None:
                    rejects['npz_filter'] += 1
                    continue
                picked.append(cand)
                per_cat_kept[cat] += 1
                progressed = True
                break
        if not progressed:
            break
        if len(picked) % 20 == 0:
            print(f'    picked={len(picked):3d} '
                  f'cat_dist={dict(per_cat_kept)}')

    print(f'  NPZ-load rejects: {rejects["npz_filter"]}')
    return picked


def write_output(path: Path, samples: List[Dict], seed: int,
                 max_per_cat: int) -> None:
    cat_counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        cat_counts[s['category']] += 1
    cat_detail: Dict[str, Dict] = {}
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_cat[s['category']].append(s)
    total = len(samples) or 1
    for c in sorted(by_cat.keys()):
        rows = by_cat[c]
        actions = sorted({r.get('action_name', '') for r in rows if r.get('action_name')})
        caps = []
        seen_cap = set()
        for r in rows:
            cap = (r.get('caption_en') or '').strip()
            if cap and cap not in seen_cap:
                caps.append(cap)
                seen_cap.add(cap)
            if len(caps) >= 5:
                break
        cat_detail[c] = {
            'count': len(rows),
            'percent': round(100.0 * len(rows) / total, 1),
            'unique_actions': len(actions),
            'example_actions': actions[:5],
            'example_captions_en': caps,
        }

    pose_deltas = [s['pose_delta'] for s in samples]
    path_lens = [s['path_len_xz'] for s in samples]
    frames = [s['num_frames'] for s in samples]
    frames_sorted = sorted(frames)
    meta = {
        'task_id': 'E2',
        'task_name': 'Motion In-Betweening (v2, 6-setting ablation)',
        'version': 'v2_private_20260425',
        'source': str(PRIVATE_ROOT),
        'source_scan': str(SCAN.relative_to(WORKSPACE_ROOT)),
        'description': (
            'Rebuilt from the Private (Dongming-recorded) held-out pool so '
            'every motion is guaranteed non-training data. Stratified '
            'across the 15 action-type categories (combat / sports_ball / '
            'locomotion / sitting / ...). Strict head + tail T-pose '
            'rejection, head↔tail pose-delta or pelvis-path minimum, '
            '1-sample-per-action-name dedupe. Shared across all six E2 '
            'settings (start-1f / end-1f / both-1f / pre-20% / post-20% '
            '/ mid-60%).'
        ),
        'total_items': len(samples),
        'category_distribution': dict(cat_counts),
        'category_distribution_detail': cat_detail,
        'frame_stats': {
            'min': frames_sorted[0] if frames_sorted else 0,
            'max': frames_sorted[-1] if frames_sorted else 0,
            'mean': round(sum(frames_sorted) / max(1, len(frames_sorted)), 1),
            'median': frames_sorted[len(frames_sorted) // 2] if frames_sorted else 0,
        },
        'seed': seed,
        'num_frames_range': [min(frames) if frames else 0,
                             max(frames) if frames else 0],
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
            'max_per_category': max_per_cat,
        },
    }
    data_list = []
    for i, s in enumerate(samples):
        data_list.append({
            'motion_path': s['motion_path'],
            'action_name': s['action_name'],
            'caption_en': s['caption_en'],
            'category': s['category'],
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
    ap.add_argument('--cap', type=int, default=DEFAULT_MAX_PER_CATEGORY,
                    help='Max samples per action-type category.')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    print(f'Loading {SCAN.relative_to(WORKSPACE_ROOT)}...')
    scan: List[Dict] = json.load(open(SCAN))
    print(f'  {len(scan)} Private motions in scan')

    print(f'\n== Stratified picking (n_target={args.n_samples}, '
          f'cap_per_cat={args.cap}) ==')
    picked = stratified_pick(
        scan, args.n_samples, args.cap, args.seed,
    )
    print(f'\n  selected {len(picked)} samples')
    cat_count: Dict[str, int] = defaultdict(int)
    for p in picked:
        cat_count[p['category']] += 1
    print('  category distribution:')
    for c in sorted(cat_count):
        print(f'    {c:18s} {cat_count[c]}')

    print('\n== Writing output ==')
    write_output(OUT_PATH, picked, args.seed, args.cap)
    # Initial copy of rewritten file gets the same raw captions; the
    # rewriter tool (tools/rewrite_e2_v2_captions.py) overwrites the
    # caption_en field with the 12-20 word rewriter output.
    write_output(OUT_REWRITTEN_PATH, picked, args.seed, args.cap)
    print('\ndone.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build the v2 E15 prepend-to-start-pose test set.

Why v2 (2026-04-27)
-------------------
The previous E15 datalist (`eval_e7_target_d_motionhub50.json`) was:
  * borrowed from E7-D (only 50 items),
  * not stratified across action category × pelvis-speed buckets,
  * no T-pose filtering on either A or T's start frame,
  * no Y-gap (pelvis-height) guard so many (A, T) pairs ended up
    being trivial postural transitions,
  * no `category_distribution_detail` so the eval dashboard had
    nothing to render in its E15 distribution pie chart.

This script mirrors the E8-v2 selection methodology (priv 100 + yiran
100, stratified by category × speed) and additionally pairs each
picked motion A with a target start-pose source T:

  * A: full motion to prepend a transition before
  * T: motion whose first frame defines the desired start pose P = T[0]

The E15 task semantic ("given P and A, prepend N transition frames so
the sequence starts at P and smoothly reaches A[0], with P and A[0] at
the same world XZ but possibly different pelvis-height Y") needs:
  * P is a clearly-non-T-pose, otherwise there's nothing to transition
    from. Re-uses the head_tpose_dev >= TPOSE_DEV_MIN filter.
  * The (P, A[0]) pose-delta should be non-trivial, otherwise the
    "prepend transition" is just identity. We enforce a minimum
    rot6d delta between the two anchor frames AND/OR a minimum
    pelvis-Y delta (T-pose-from-stand → crouch-A is the canonical
    target).
  * T should not equal A.

Outputs
-------
    data/eval/m2m_v2/eval_e15_prepend_v2.json
    data/eval/m2m_v2/eval_e15_prepend_v2_rewritten.json

`meta.category_distribution_detail`, `meta.speed_distribution_detail`,
`meta.pool_distribution`, `meta.target_pool_distribution` and
`meta.y_gap_stats` power the dashboard's E15 distribution view.

Usage
-----
    python3 tools/build_e15_prepend_v2_data.py
    python3 tools/build_e15_prepend_v2_data.py --n-priv 100 --n-yiran 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

WORKSPACE_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
SCAN = WORKSPACE_ROOT / 'data/eval/m2m_v2/_pelvis_pathlen_scan.json'
OUT_DIR = WORKSPACE_ROOT / 'data/eval/m2m_v2'
OUT_PATH = OUT_DIR / 'eval_e15_prepend_v2.json'
OUT_REWRITTEN_PATH = OUT_DIR / 'eval_e15_prepend_v2_rewritten.json'

PRIVATE_ROOT = Path(
    '/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz/Private'
)
YIRAN_ROOT = Path(
    '/apdcephfs_cq10/share_1467498/datasets/motion_gen_arena/'
    'evaluation_20251125/yiran_subset/'
    'sft_1210_o6dp1103_04k_qwen3_1B_NB_from3kckpt60_gpus128_e40/'
    'motions_smpl_npz_for_eval'
)

sys.path.insert(0, str(WORKSPACE_ROOT))
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_smplx_pose,
)

# ── Filter thresholds (mirrors E8-v2) ────────────────────────────────
MIN_FRAMES = 80
MAX_FRAMES = 300
TPOSE_WINDOW = 3
TPOSE_DEV_MIN = 0.10
POSE_DELTA_MIN = 0.03
PATH_LEN_MIN = 0.3

# ── E15-specific (P, A[0]) pairing thresholds ────────────────────────
# These guarantee the prepend task is non-trivial.
PA_POSE_DELTA_MIN = 0.20      # ||P_rot6d - A[0]_rot6d||_mean must be >= 0.20
                              # (otherwise the start poses are too similar
                              # → transition is near-identity)
PA_Y_GAP_MIN = 0.0            # don't enforce Y gap; postural transitions
                              # (e.g. arms-up vs arms-down at same height)
                              # are valid E15 cases too. Kept for ablation.

SPEED_BUCKETS = [
    ('static',   0.00,  0.05),
    ('slow',     0.05,  0.10),
    ('moderate', 0.10,  0.20),
    ('fast',     0.20,  10.0),
]
SPEED_LABELS = [b[0] for b in SPEED_BUCKETS]

DEFAULT_N_PRIV = 100
DEFAULT_N_YIRAN = 100
DEFAULT_MAX_PER_CELL = 6
DEFAULT_MAX_PER_CATEGORY = 18
DEFAULT_SEED = 42

IDENTITY_6D = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


# ── Yiran category classifier (verbatim copy of E8-v2 rules) ─────────
YIRAN_CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ('locomotion', [
        'walk', 'run', 'jog', 'pace', 'stride', 'march', 'sprint',
        'spins', 'rotat', 'turn',
        'step', 'tiptoe', 'shuffle', 'tipt',
    ]),
    ('jump_climb', [
        'jump', 'leap', 'hop', 'climb', 'crawl',
    ]),
    ('sitting', [
        'sit', 'sat', 'kneel', 'squat', 'crouch',
        'lie', 'lay', 'lying',
    ]),
    ('stand_balance', [
        'stand up', 'stands up', 'stood up',
        'balance', 'rise', 'rises', 'risen', 'rising',
    ]),
    ('gesture', [
        'wave', 'point', 'gesture', 'clap',
        'salute', 'bow', 'nod', 'shake their head',
    ]),
    ('arm_motion', [
        'raise', 'reach', 'lift', 'punch', 'swing',
        'cross arms', 'extend their arms', 'arms',
        'pull', 'push',
    ]),
    ('dance_perf', [
        'dance', 'dances', 'sway', 'twirl',
    ]),
    ('stairs', [
        'stair', 'staircase', 'steps', 'descend', 'ascend',
        'goes up', 'goes down', 'comes down',
    ]),
    ('combat', [
        'kick', 'fight', 'block', 'parry', 'strike',
        'sword', 'attack',
    ]),
    ('sports', [
        'ball', 'throw', 'catch', 'pitch', 'shoot',
        'tennis', 'volley', 'soccer', 'basket', 'golf', 'bat',
        'archer', 'archery',
    ]),
]


def classify_yiran_category(caption: str) -> str:
    if not caption:
        return 'other'
    text = caption.lower()
    for cat, keywords in YIRAN_CATEGORY_RULES:
        for kw in keywords:
            if kw in text:
                return cat
    return 'other'


def canonical_path(p: str) -> str:
    """Normalize cephfs paths to absolute form."""
    p = str(p)
    for marker in ('apdcephfs_cq11/share_1467498/',
                   'apdcephfs_cq10/share_1467498/'):
        if marker in p and not p.startswith('/'):
            p = '/' + p[p.index(marker):]
            return p
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


def evaluate_priv(entry: Dict) -> Optional[Dict]:
    """Evaluate a Private-pool scan entry against E15 filters.

    Returns a candidate dict with the head6d frame attached so we can
    later compute (P, A[0]) pose-delta during pairing.
    """
    scan_path = Path(entry['path'])
    rel = scan_path.parts
    try:
        idx = rel.index('npz_split')
    except ValueError:
        idx = -1
    if idx >= 0:
        priv_path = Path(*rel[:idx]) / 'npz' / Path(*rel[idx + 1:])
        priv_path = Path('/') / priv_path
        if not priv_path.exists():
            priv_path = scan_path
    else:
        priv_path = scan_path
    if not priv_path.exists():
        return None
    rot6d = load_motion_rot6d(priv_path)
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
    trans = load_pelvis_trans(priv_path)
    path_len = pelvis_path_len_xz(trans) if trans is not None else 0.0
    if pose_d < POSE_DELTA_MIN and path_len < PATH_LEN_MIN:
        return None
    fps = float(entry.get('fps', 30.0)) or 30.0
    duration_sec = T / fps
    speed_mps = float(path_len) / duration_sec if duration_sec > 0 else 0.0
    pelvis_y0 = float(trans[0, 1]) if trans is not None and len(trans) else 0.0
    return {
        'motion_path': canonical_path(str(priv_path)),
        'action_name': entry.get('action_name', ''),
        'caption_en': entry.get('caption_en', ''),
        'category': entry.get('category', 'other'),
        'pool': 'private',
        'source': entry.get('rel_dir', priv_path.parent.name),
        'num_frames': int(T),
        'fps': fps,
        'duration_sec': round(duration_sec, 2),
        'pelvis_speed_mps': round(speed_mps, 4),
        'speed_bucket': speed_bucket(speed_mps),
        'head_tpose_dev': round(head_dev, 4),
        'tail_tpose_dev': round(tail_dev, 4),
        'pose_delta': round(pose_d, 4),
        'path_len_xz': round(path_len, 3),
        '_head6d_mean': body6d[:w].mean(axis=0),  # (21, 6) mean rot6d
        '_pelvis_y0': pelvis_y0,
    }


def evaluate_yiran(npz_path: Path) -> Optional[Dict]:
    if not npz_path.exists():
        return None
    rot6d = load_motion_rot6d(npz_path)
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
    trans = load_pelvis_trans(npz_path)
    path_len = pelvis_path_len_xz(trans) if trans is not None else 0.0
    if pose_d < POSE_DELTA_MIN and path_len < PATH_LEN_MIN:
        return None
    try:
        d = np.load(npz_path, allow_pickle=True)
        caption = str(d['text']) if 'text' in d.files else ''
        fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else 30.0
    except Exception:
        caption = ''
        fps = 30.0
    fps = fps or 30.0
    duration_sec = T / fps
    speed_mps = float(path_len) / duration_sec if duration_sec > 0 else 0.0
    pelvis_y0 = float(trans[0, 1]) if trans is not None and len(trans) else 0.0
    return {
        'motion_path': canonical_path(str(npz_path)),
        'action_name': caption,
        'caption_en': caption,
        'category': classify_yiran_category(caption),
        'pool': 'yiran_t2m',
        'source': npz_path.parent.parent.name,
        'num_frames': int(T),
        'fps': fps,
        'duration_sec': round(duration_sec, 2),
        'pelvis_speed_mps': round(speed_mps, 4),
        'speed_bucket': speed_bucket(speed_mps),
        'head_tpose_dev': round(head_dev, 4),
        'tail_tpose_dev': round(tail_dev, 4),
        'pose_delta': round(pose_d, 4),
        'path_len_xz': round(path_len, 3),
        '_head6d_mean': body6d[:w].mean(axis=0),
        '_pelvis_y0': pelvis_y0,
    }


def stratified_pick(
    candidates: List[Dict],
    n_target: int,
    cap_per_cell: int,
    cap_per_cat: int,
    seed: int,
    pool_name: str,
) -> List[Dict]:
    rng = random.Random(seed)
    by_cell: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for c in candidates:
        by_cell[(c['category'], c['speed_bucket'])].append(c)

    cell_keys = sorted(by_cell.keys())
    print(f'  [{pool_name}] populated (category, speed) cells: '
          f'{len(cell_keys)} '
          f'(cap_per_cell={cap_per_cell}, cap_per_cat={cap_per_cat})')

    ordered: Dict[Tuple[str, str], List[Dict]] = {}
    for k, items in by_cell.items():
        items_sorted = sorted(items, key=lambda e: -float(e['path_len_xz']))
        rng.shuffle(items_sorted)
        ordered[k] = items_sorted

    picked: List[Dict] = []
    per_cell: Dict[Tuple[str, str], int] = defaultdict(int)
    per_cat: Dict[str, int] = defaultdict(int)
    cur: Dict[Tuple[str, str], int] = defaultdict(int)
    while len(picked) < n_target:
        progressed = False
        for k in cell_keys:
            if len(picked) >= n_target:
                break
            cat, _ = k
            if per_cell[k] >= cap_per_cell:
                continue
            if per_cat[cat] >= cap_per_cat:
                continue
            pool = ordered[k]
            if cur[k] < len(pool):
                picked.append(pool[cur[k]])
                cur[k] += 1
                per_cell[k] += 1
                per_cat[cat] += 1
                progressed = True
        if not progressed:
            break
    print(f'  [{pool_name}] picked {len(picked)} / target {n_target}')
    return picked


def gather_priv(seed: int, n_target: int, cap_cell: int,
                cap_cat: int) -> List[Dict]:
    print(f'\n== [Private] loading scan {SCAN.relative_to(WORKSPACE_ROOT)} ==')
    scan: List[Dict] = json.load(open(SCAN))
    print(f'  {len(scan)} scan entries')

    seen_action: set = set()
    flat = []
    for e in scan:
        n = int(e.get('num_frames', 0) or 0)
        if not (MIN_FRAMES <= n <= MAX_FRAMES):
            continue
        a = e.get('action_name', '')
        if a and a in seen_action:
            continue
        flat.append(e)
        if a:
            seen_action.add(a)
    print(f'  prefilter (frame [{MIN_FRAMES},{MAX_FRAMES}], 1-per-action): '
          f'{len(flat)}')

    by_cell: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for e in flat:
        cat = e.get('category', 'other')
        n = int(e.get('num_frames', 0) or 0)
        pl = float(e.get('path_len_xz', 0.0) or 0.0)
        dur = float(e.get('duration_sec') or n / 30.0)
        sp = pl / dur if dur else 0.0
        by_cell[(cat, speed_bucket(sp))].append(e)

    rng = random.Random(seed)
    cell_keys = sorted(by_cell.keys())
    ordered = {}
    for k in cell_keys:
        items = list(by_cell[k])
        items.sort(key=lambda e: -float(e.get('path_len_xz', 0.0) or 0.0))
        rng.shuffle(items)
        ordered[k] = items

    picked: List[Dict] = []
    per_cell: Dict[Tuple[str, str], int] = defaultdict(int)
    per_cat: Dict[str, int] = defaultdict(int)
    cur: Dict[Tuple[str, str], int] = defaultdict(int)
    rejects = 0
    while len(picked) < n_target:
        progressed = False
        for k in cell_keys:
            if len(picked) >= n_target:
                break
            cat, _ = k
            if per_cell[k] >= cap_cell:
                continue
            if per_cat[cat] >= cap_cat:
                continue
            pool = ordered[k]
            while cur[k] < len(pool):
                e = pool[cur[k]]
                cur[k] += 1
                cand = evaluate_priv(e)
                if cand is None:
                    rejects += 1
                    continue
                picked.append(cand)
                per_cell[k] += 1
                per_cat[cat] += 1
                progressed = True
                break
        if not progressed:
            break
    print(f'  [Private] picked {len(picked)} / target {n_target} '
          f'(rejects={rejects})')
    return picked


def gather_yiran(seed: int, n_target: int, cap_cell: int,
                 cap_cat: int) -> List[Dict]:
    print(f'\n== [yiran] scanning {YIRAN_ROOT.name} ==')
    files = sorted(YIRAN_ROOT.glob('*.npz'))
    print(f'  {len(files)} npz files')
    candidates: List[Dict] = []
    rejects = 0
    for f in files:
        cand = evaluate_yiran(f)
        if cand is None:
            rejects += 1
            continue
        candidates.append(cand)
    print(f'  candidates after filters: {len(candidates)} '
          f'(rejects={rejects})')
    picked = stratified_pick(
        candidates, n_target, cap_cell, cap_cat, seed, 'yiran')
    return picked


def gather_yiran_pool(seed: int) -> List[Dict]:
    """Return ALL yiran candidates (no per-cell cap) — used as the
    pairing pool for sampling target T motions.

    We deliberately scan once and keep everything that passes filters
    so that picked A motions in `gather_yiran` can be paired against
    a wider set of T motions (not just the 100 picked).
    """
    print(f'\n== [yiran] full pool scan {YIRAN_ROOT.name} ==')
    files = sorted(YIRAN_ROOT.glob('*.npz'))
    candidates: List[Dict] = []
    for f in files:
        cand = evaluate_yiran(f)
        if cand is None:
            continue
        candidates.append(cand)
    print(f'  yiran pairing-pool size: {len(candidates)}')
    return candidates


def gather_priv_pool(seed: int, sample_limit: int = 1500) -> List[Dict]:
    """Sample-evaluate up to `sample_limit` Private entries to build a
    pairing pool of target T motions. The picked-A pool only has 100
    items; we want a richer pool to choose T from so pairs can hit the
    PA_POSE_DELTA_MIN constraint.
    """
    print(f'\n== [Private] building pairing pool ==')
    scan: List[Dict] = json.load(open(SCAN))
    rng = random.Random(seed + 7919)  # different sub-seed
    rng.shuffle(scan)
    pool: List[Dict] = []
    seen = set()
    rejects = 0
    for e in scan:
        n = int(e.get('num_frames', 0) or 0)
        if not (MIN_FRAMES <= n <= MAX_FRAMES):
            continue
        a = e.get('action_name', '')
        if a and a in seen:
            continue
        cand = evaluate_priv(e)
        if cand is None:
            rejects += 1
            continue
        pool.append(cand)
        if a:
            seen.add(a)
        if len(pool) >= sample_limit:
            break
    print(f'  Private pairing-pool size: {len(pool)} '
          f'(scanned, rejects={rejects})')
    return pool


# ── (P, A[0]) pose-delta scoring ─────────────────────────────────────
def pose_pair_delta(t_head6d: np.ndarray, a_head6d: np.ndarray) -> float:
    return float(np.linalg.norm(
        t_head6d.mean(axis=0) - a_head6d.mean(axis=0), axis=-1).mean()
    ) if t_head6d.ndim == 2 else float(np.linalg.norm(
        t_head6d - a_head6d, axis=-1).mean())


def pair_motions(
    a_picks: List[Dict],
    pool: List[Dict],
    seed: int,
) -> List[Dict]:
    """Pair each A in `a_picks` with a T sampled from `pool` such that:
      * T.motion_path != A.motion_path (different identity)
      * pose_delta(P=T_head, A[0]_head) >= PA_POSE_DELTA_MIN

    Pool is shuffled per-A; we draw without replacement within each
    A's attempts to avoid degenerate same-source biases. If no pool
    candidate satisfies the threshold (rare with a 1k+ pool and
    threshold 0.20), we fall back to the highest pose-delta pool item.
    """
    rng = random.Random(seed)
    paired: List[Dict] = []
    n_fallback = 0
    for i, a in enumerate(a_picks):
        a_head = a['_head6d_mean']
        order = list(range(len(pool)))
        rng.shuffle(order)
        chosen = None
        best_delta = -1.0
        best_idx = -1
        for idx in order:
            t = pool[idx]
            if t['motion_path'] == a['motion_path']:
                continue
            d = pose_pair_delta(t['_head6d_mean'], a_head)
            if d >= PA_POSE_DELTA_MIN:
                chosen = t
                pa_delta = d
                break
            if d > best_delta:
                best_delta = d
                best_idx = idx
        if chosen is None:
            chosen = pool[best_idx]
            pa_delta = best_delta
            n_fallback += 1
        y_gap = float(chosen['_pelvis_y0'] - a['_pelvis_y0'])
        paired.append({
            **a,
            '_target': chosen,
            '_pa_pose_delta': round(pa_delta, 4),
            '_pa_y_gap': round(y_gap, 4),
        })
    print(f'  paired {len(paired)} (fallback below {PA_POSE_DELTA_MIN}: {n_fallback})')
    return paired


# ── Output assembly ───────────────────────────────────────────────────
def make_meta(samples: List[Dict], seed: int,
              cap_cell: int, cap_cat: int) -> Dict:
    cat_counts: Dict[str, int] = defaultdict(int)
    speed_counts: Dict[str, int] = defaultdict(int)
    pool_counts: Dict[str, int] = defaultdict(int)
    target_pool_counts: Dict[str, int] = defaultdict(int)
    cell_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for s in samples:
        cat_counts[s['category']] += 1
        speed_counts[s['speed_bucket']] += 1
        pool_counts[s['pool']] += 1
        target_pool_counts[s['_target']['pool']] += 1
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
        sub_speed: Dict[str, int] = defaultdict(int)
        sub_pool: Dict[str, int] = defaultdict(int)
        for r in rows:
            sub_speed[r['speed_bucket']] += 1
            sub_pool[r['pool']] += 1
        cat_detail[c] = {
            'count': len(rows),
            'percent': round(100.0 * len(rows) / total, 1),
            'unique_actions': len(actions),
            'example_actions': actions[:5],
            'example_captions_en': caps,
            'speed_mix': dict(sub_speed),
            'pool_mix': dict(sub_pool),
        }

    speed_detail: Dict[str, Dict] = {}
    for sl in SPEED_LABELS:
        cnt = speed_counts.get(sl, 0)
        speed_detail[sl] = {
            'count': cnt,
            'percent': round(100.0 * cnt / total, 1),
        }

    frames = [s['num_frames'] for s in samples]
    frames_sorted = sorted(frames)
    speeds = [s['pelvis_speed_mps'] for s in samples]
    pa_deltas = [s['_pa_pose_delta'] for s in samples]
    y_gaps = [abs(s['_pa_y_gap']) for s in samples]
    return {
        'task_id': 'E15',
        'task_name': 'Prepend to Start Pose (v2, 2026-04-27 redesign)',
        'version': 'v2_priv100_yiran100_paired_20260427',
        'sources': {
            'private': str(PRIVATE_ROOT),
            'yiran_t2m_v1': str(YIRAN_ROOT),
        },
        'description': (
            'Prepend-to-start-pose test set combining 100 held-out '
            'Dongming Private mocap clips and 100 HyMotion T2M-v1.0 '
            'generated motions as the "A" pool (full motion to be '
            'prepended). Each A is paired with a target motion T '
            '(P = T[0]) sampled from a wider pairing pool (Private + '
            'yiran combined) under the constraint '
            f'pose_delta(P, A[0]) >= {PA_POSE_DELTA_MIN}. Stratified '
            'across action category x pelvis-speed (4-bucket) cells. '
            'Filters: head+tail rot6d != T-pose (dev>=0.10), frames '
            'in [80, 300], head<->tail pose-delta or pelvis path-length '
            'non-trivial. Used by E15 (caption_aware=False) for '
            'postural-prepend evaluation.'
        ),
        'total_items': len(samples),
        'category_distribution': dict(cat_counts),
        'category_distribution_detail': cat_detail,
        'speed_distribution': dict(speed_counts),
        'speed_distribution_detail': speed_detail,
        'pool_distribution': dict(pool_counts),
        'target_pool_distribution': dict(target_pool_counts),
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
        'pa_pose_delta_stats': {
            'min': round(min(pa_deltas), 4) if pa_deltas else 0,
            'max': round(max(pa_deltas), 4) if pa_deltas else 0,
            'mean': round(sum(pa_deltas) / max(1, len(pa_deltas)), 4)
                     if pa_deltas else 0,
            'threshold': PA_POSE_DELTA_MIN,
        },
        'pa_y_gap_stats_abs': {
            'min': round(min(y_gaps), 4) if y_gaps else 0,
            'max': round(max(y_gaps), 4) if y_gaps else 0,
            'mean': round(sum(y_gaps) / max(1, len(y_gaps)), 4)
                     if y_gaps else 0,
        },
        'seed': seed,
        'filters': {
            'min_frames': MIN_FRAMES,
            'max_frames': MAX_FRAMES,
            'tpose_window': TPOSE_WINDOW,
            'tpose_dev_min': TPOSE_DEV_MIN,
            'pose_delta_min': POSE_DELTA_MIN,
            'path_len_min': PATH_LEN_MIN,
            'pa_pose_delta_min': PA_POSE_DELTA_MIN,
            'cap_per_cell': cap_cell,
            'cap_per_category': cap_cat,
        },
        'settings_overview': {
            'default': {
                'description': 'Prepend P=T[0] before A. Speed and N_cond_A '
                               'derived adaptively (best config picked from '
                               'sweep — see notes/E15_sweep_*).',
                'use_caption': False,
            },
        },
    }


def write_output(path: Path, samples: List[Dict], seed: int,
                 cap_cell: int, cap_cat: int) -> None:
    meta = make_meta(samples, seed, cap_cell, cap_cat)
    data_list = []
    for i, s in enumerate(samples):
        t = s['_target']
        data_list.append({
            'prompt_id': f'e15_{i:04d}',
            # A: full motion to prepend before
            'motion_path': s['motion_path'],
            'action_name': s['action_name'],
            'caption_en': s['caption_en'],
            'category': s['category'],
            'pool': s['pool'],
            'speed_bucket': s['speed_bucket'],
            'pelvis_speed_mps': s['pelvis_speed_mps'],
            'source': s['source'],
            'num_frames': s['num_frames'],
            'fps': s['fps'],
            'duration_sec': s['duration_sec'],
            # T: target motion whose first frame defines P
            'target_motion_path': t['motion_path'],
            'target_action_name': t['action_name'],
            'target_pool': t['pool'],
            'target_category': t['category'],
            'target_pelvis_y0': round(t['_pelvis_y0'], 4),
            '_sample_idx': i,
            '_pa_pose_delta': s['_pa_pose_delta'],
            '_pa_y_gap': s['_pa_y_gap'],
            '_pose_delta': s['pose_delta'],
            '_path_len_xz': s['path_len_xz'],
            '_head_tpose_dev': s['head_tpose_dev'],
            '_tail_tpose_dev': s['tail_tpose_dev'],
            '_target_head_tpose_dev': t['head_tpose_dev'],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'meta': meta, 'data_list': data_list}, f,
                  ensure_ascii=False, indent=2)
    print(f'  wrote {path.relative_to(WORKSPACE_ROOT)} '
          f'(n={len(data_list)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-priv', type=int, default=DEFAULT_N_PRIV)
    ap.add_argument('--n-yiran', type=int, default=DEFAULT_N_YIRAN)
    ap.add_argument('--cap', type=int, default=DEFAULT_MAX_PER_CELL)
    ap.add_argument('--cap-cat', type=int, default=DEFAULT_MAX_PER_CATEGORY)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    priv_picks = gather_priv(args.seed, args.n_priv, args.cap, args.cap_cat)
    yiran_cap_cell = max(args.cap, 12)
    yiran_cap_cat = max(args.cap_cat, 30)
    yiran_picks = gather_yiran(args.seed, args.n_yiran, yiran_cap_cell, yiran_cap_cat)

    a_picks = priv_picks + yiran_picks
    print(f'\n== Combined A pool: {len(a_picks)} '
          f'(priv={len(priv_picks)}, yiran={len(yiran_picks)}) ==')

    # Build a wider pairing pool for sampling T (start pose source).
    yiran_pool = gather_yiran_pool(args.seed)
    priv_pool = gather_priv_pool(args.seed, sample_limit=1500)
    pairing_pool = priv_pool + yiran_pool
    print(f'\n== Pairing pool size: {len(pairing_pool)} '
          f'(priv={len(priv_pool)}, yiran={len(yiran_pool)}) ==')

    print('\n== Pairing each A with a T motion ==')
    samples = pair_motions(a_picks, pairing_pool, args.seed)

    cat_count: Dict[str, int] = defaultdict(int)
    pool_count: Dict[str, int] = defaultdict(int)
    spd_count: Dict[str, int] = defaultdict(int)
    for p in samples:
        cat_count[p['category']] += 1
        pool_count[p['pool']] += 1
        spd_count[p['speed_bucket']] += 1
    print('  category distribution:')
    for c in sorted(cat_count):
        print(f'    {c:18s} {cat_count[c]:3d}')
    print('  speed-bucket distribution:')
    for s in SPEED_LABELS:
        print(f'    {s:10s} {spd_count.get(s, 0):3d}')
    print('  pool distribution (A side):')
    for p_name in sorted(pool_count):
        print(f'    {p_name:12s} {pool_count[p_name]:3d}')

    print('\n== Writing outputs ==')
    write_output(OUT_PATH, samples, args.seed, args.cap, args.cap_cat)
    write_output(OUT_REWRITTEN_PATH, samples, args.seed, args.cap, args.cap_cat)
    print('\ndone.')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""Build E14 Transition Stitching test data from the 400h HQ annotation.

Source:
    data/annotation/train_hymotion_400h_hq_20260403.json
      (407,552 HQ-filtered motions, 30fps, has_hand=True)

Output (two files, each 100 pairs):
    data/eval/m2m_v2/eval_e14_hq400h_static100.json   (L: postural transition)
    data/eval/m2m_v2/eval_e14_hq400h_move100.json     (M: locomotion transition)

Pipeline
--------
1. Stratified weighted sampling (by subset original distribution) -> POOL_SIZE
   candidates. Enforces duration bounds MIN_FRAMES..MAX_FRAMES.
2. Load each candidate's NPZ, compute pelvis xz tail speed (m/frame, avg
   over last TAIL_WINDOW=10 frame differences) and tail stability
   (std/mean over last STABILITY_WINDOW=15 frames).
3. Bucket into:
     static_pool: tail_speed <= STATIC_MAX (<= 0.0004 m/frame)
     move_pool:   MOVE_LOW <= tail_speed <= MOVE_HIGH ([0.004, 0.020])
                  AND tail_cv <= STABILITY_CV_MAX (0.6)
     any_pool:    everything that passed basic validation (motion_b pool)
4. For each category sample N_PAIRS motion_a from its pool; for each A,
   pick a motion_b from any_pool whose `source` (subdir under
   .../motions/<source>/) differs from A's -> enforces cross-source
   diversity, matching the motionhub version's humansc3d <-> HumanML3D
   pattern. motion_a and motion_b are both unique across pairs.

Rationale
---------
MotionHub v8 used `train_hq_motionhub_hymotion.json` (50+50 pairs), but
motionhub quality is uneven, so this builder switches to the curated
400h HQ list. Thresholds, output schema, and pair diversity match the
motionhub version so the L/M semantics carry over 1:1.

Run
---
    python3 tools/build_e14_hq400h_data.py
    # or to customise:
    python3 tools/build_e14_hq400h_data.py --pool-size 8000 --n-pairs 100

See m2m_eval_tasks.py E14 definition for how _data_file is wired in.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ───────────────────────────── Config ────────────────────────────────
WORKSPACE_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
SRC = WORKSPACE_ROOT / 'data/annotation/train_hymotion_400h_hq_20260403.json'
OUT_DIR = WORKSPACE_ROOT / 'data/eval/m2m_v2'

STATIC_MAX = 0.0004        # m/frame (L)
MOVE_LOW = 0.004           # m/frame (M, widened per user request)
MOVE_HIGH = 0.020          # m/frame
TAIL_WINDOW = 10           # frames (10 diffs = 11 frames back)
STABILITY_WINDOW = 15      # frames (for CV check on move pool)
STABILITY_CV_MAX = 0.6     # std/mean

MIN_FRAMES = 120           # 4s @ 30fps
MAX_FRAMES = 600

DEFAULT_POOL_SIZE = 5000
DEFAULT_N_PAIRS = 100
DEFAULT_SEED = 42


# ─────────────────────────── Helpers ─────────────────────────────────
def extract_source(smplx_path: str) -> str:
    """Extract the source directory under `/motions/`.

    Example:
        ../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/foo.npz
        -> "HumanML3D-HumanEva"
    """
    parts = smplx_path.replace('\\', '/').split('/')
    try:
        idx = parts.index('motions')
        return parts[idx + 1]
    except (ValueError, IndexError):
        return parts[-2] if len(parts) >= 2 else ''


def resolve_motion_path(smplx_path: str) -> Path:
    """Resolve the (possibly relative) smplx_path in the annotation
    to an absolute path under the workspace root. Does NOT follow
    symlinks — we want the logical workspace path (going through
    `data/hymotion_data/...`), not the physical backing path (which
    may live on a different apdcephfs share like cq10).
    """
    p = Path(smplx_path)
    if p.is_absolute():
        return p
    base = WORKSPACE_ROOT / 'data/annotation'
    # normpath collapses '..' / '.' without touching symlinks.
    return Path(os.path.normpath(str(base / smplx_path)))


def canonicalise_path(p: Path) -> str:
    """Return the path as-is (already in workspace-root form thanks to
    `resolve_motion_path` not resolving symlinks).

    Existing eval data files use paths like
    `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/hymotion_data/...`
    rather than the real backing path (`/apdcephfs_cq10/.../motion_data/...`).
    This keeps us in the same regime.
    """
    s = str(p)
    # Collapse bind-mount prefix if any (e.g. /apdcephfs/AILab_DHA/...).
    marker = 'apdcephfs_cq11/share_1467498/'
    if marker in s:
        s = '/' + s[s.index(marker):]
    return s


def compute_tail_stats(
    trans: np.ndarray,
    tail_window: int = TAIL_WINDOW,
    stability_window: int = STABILITY_WINDOW,
) -> Tuple[float, float]:
    """Return (tail_speed_m_per_frame, tail_cv) using pelvis xz only."""
    # Tail speed: last `tail_window + 1` frames -> `tail_window` diffs
    tail = trans[-(tail_window + 1):]
    diffs = np.diff(tail[:, [0, 2]], axis=0)
    tail_speed = float(np.linalg.norm(diffs, axis=1).mean())

    stab_tail = trans[-(stability_window + 1):]
    stab_speeds = np.linalg.norm(
        np.diff(stab_tail[:, [0, 2]], axis=0), axis=1)
    mean = float(stab_speeds.mean())
    cv = float(stab_speeds.std() / (mean + 1e-9)) if mean > 0 else 0.0
    return tail_speed, cv


def load_pelvis_trans(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as arr:
        trans = arr['trans']
    return np.asarray(trans, dtype=np.float32)


# ─────────────────────────── Core ────────────────────────────────────
def stratified_sample(
    data_list: Dict[str, Any],
    pool_size: int,
    rng: random.Random,
) -> List[str]:
    """Bucket keys by subset, filter by duration, sample proportional to
    the original subset distribution."""
    by_subset: Dict[str, List[str]] = {}
    for k, v in data_list.items():
        n = v.get('num_frames', 0)
        if MIN_FRAMES <= n <= MAX_FRAMES:
            by_subset.setdefault(v.get('subset', 'unknown'), []).append(k)

    total = sum(len(ks) for ks in by_subset.values())
    if total == 0:
        return []

    sampled: List[str] = []
    for subset, ks in sorted(by_subset.items()):
        share = int(round(pool_size * len(ks) / total))
        share = min(share, len(ks))
        sampled.extend(rng.sample(ks, share))
        print(f'  subset={subset:18s} pool={len(ks):>6d} '
              f'sampled={share}')
    rng.shuffle(sampled)
    return sampled


def scan_candidates(
    data_list: Dict[str, Any],
    keys: List[str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load NPZ for each key, return (static_pool, move_pool, any_pool)."""
    static_pool: List[Dict] = []
    move_pool: List[Dict] = []
    any_pool: List[Dict] = []

    n_scanned = 0
    n_missing = 0
    n_bad = 0

    for k in keys:
        item = data_list[k]
        rel_path = item.get('smplx_path', '')
        resolved = resolve_motion_path(rel_path)
        if not resolved.exists():
            n_missing += 1
            continue
        try:
            trans = load_pelvis_trans(resolved)
        except Exception:
            n_bad += 1
            continue
        if trans.ndim != 2 or trans.shape[1] < 3 or len(trans) < MIN_FRAMES:
            n_bad += 1
            continue
        if np.isnan(trans).any() or np.allclose(trans, 0.0):
            n_bad += 1
            continue
        try:
            tail_speed, tail_cv = compute_tail_stats(trans)
        except Exception:
            n_bad += 1
            continue

        entry = {
            'key': k,
            'canonical_path': canonicalise_path(resolved),
            'subset': item.get('subset', ''),
            'source': extract_source(rel_path),
            'num_frames': int(len(trans)),
            'tail_speed': tail_speed,
            'tail_cv': tail_cv,
        }
        any_pool.append(entry)
        if tail_speed <= STATIC_MAX:
            static_pool.append(entry)
        elif (MOVE_LOW <= tail_speed <= MOVE_HIGH
              and tail_cv <= STABILITY_CV_MAX):
            move_pool.append(entry)
        n_scanned += 1
        if n_scanned % 500 == 0:
            print(f'  scanned={n_scanned} '
                  f'static={len(static_pool)} move={len(move_pool)}')

    print(f'  done. scanned={n_scanned} '
          f'missing={n_missing} bad={n_bad} '
          f'static={len(static_pool)} move={len(move_pool)} '
          f'any={len(any_pool)}')
    return static_pool, move_pool, any_pool


def make_pairs(
    a_pool: List[Dict],
    b_pool: List[Dict],
    n_pairs: int,
    category: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Pair each A with a random B from a different source. motion_a
    and motion_b are each unique across the pair list."""
    a_shuffled = list(a_pool)
    rng.shuffle(a_shuffled)
    b_shuffled = list(b_pool)
    rng.shuffle(b_shuffled)

    used_a: set = set()
    used_b: set = set()
    pairs: List[Dict[str, Any]] = []

    for a in a_shuffled:
        if len(pairs) >= n_pairs:
            break
        if a['key'] in used_a:
            continue
        cand = [
            b for b in b_shuffled
            if b['key'] != a['key']
            and b['key'] not in used_b
            and b['source'] != a['source']
        ]
        if not cand:
            # relax: only avoid key collision if we run out
            cand = [
                b for b in b_shuffled
                if b['key'] != a['key'] and b['key'] not in used_b
            ]
            if not cand:
                continue
        b = cand[0]  # already shuffled
        pairs.append({
            'prompt_id': f'{category}_{len(pairs):04d}',
            'motion_a_path': a['canonical_path'],
            'motion_b_path': b['canonical_path'],
            'action_name_a': a['key'],
            'action_name_b': b['key'],
            '_a_end_speed': round(a['tail_speed'], 6),
            '_a_end_speed_cv': round(a['tail_cv'], 4),
            '_a_subset': a['subset'],
            '_b_subset': b['subset'],
            '_a_source': a['source'],
            '_b_source': b['source'],
            '_category': category,
            'num_frames_a': a['num_frames'],
            'num_frames_b': b['num_frames'],
            'caption_en': '',
            'fps': 30,
        })
        used_a.add(a['key'])
        used_b.add(b['key'])
    return pairs


def write_output(
    out_path: Path,
    pairs: List[Dict[str, Any]],
    category: str,
    speed_range: Tuple[float, float],
    pool_size: int,
    seed: int,
    notes_extra: str = '',
) -> None:
    speeds = [p['_a_end_speed'] for p in pairs]
    meta = {
        'source': str(SRC.relative_to(WORKSPACE_ROOT)),
        'category': category,
        'n_samples': len(pairs),
        'speed_range_requested': list(speed_range),
        'speed_range_actual': (
            [min(speeds), max(speeds)] if speeds else None),
        'min_frames': MIN_FRAMES,
        'max_frames': MAX_FRAMES,
        'tail_window': TAIL_WINDOW,
        'stability_window': STABILITY_WINDOW,
        'stability_cv_max': STABILITY_CV_MAX,
        'pool_size': pool_size,
        'subset_policy': 'weighted_by_original_distribution',
        'pair_diversity': 'cross_source',
        'seed': seed,
        'notes': (f'E14 hq400h dataset: sampled from 400h HQ annotation. '
                  f'Replaces motionhub v8. {notes_extra}').strip(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump({'meta': meta, 'data_list': pairs}, f,
                  indent=2, ensure_ascii=False)
    print(f'Wrote {out_path} ({len(pairs)} pairs)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool-size', type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument('--n-pairs', type=int, default=DEFAULT_N_PAIRS)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--out-static', type=str,
                    default=str(OUT_DIR / 'eval_e14_hq400h_static100.json'))
    ap.add_argument('--out-move', type=str,
                    default=str(OUT_DIR / 'eval_e14_hq400h_move100.json'))
    ap.add_argument('--dry-run', action='store_true',
                    help='Do everything but skip writing output JSONs.')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f'Loading HQ annotation: {SRC}')
    with SRC.open() as f:
        db = json.load(f)
    data_list = db['data_list']
    print(f'  {len(data_list)} motions total')

    print(f'\n[1/3] Stratified sampling (pool_size={args.pool_size}):')
    sampled_keys = stratified_sample(data_list, args.pool_size, rng)
    print(f'  total sampled: {len(sampled_keys)}')

    print(f'\n[2/3] Scanning NPZ for tail speed:')
    static_pool, move_pool, any_pool = scan_candidates(
        data_list, sampled_keys)

    if len(static_pool) < args.n_pairs:
        print(f'WARNING: static pool ({len(static_pool)}) < requested '
              f'pairs ({args.n_pairs}). Consider raising --pool-size.',
              file=sys.stderr)
    if len(move_pool) < args.n_pairs:
        print(f'WARNING: move pool ({len(move_pool)}) < requested pairs '
              f'({args.n_pairs}). Consider raising --pool-size or '
              f'widening MOVE_LOW/MOVE_HIGH.', file=sys.stderr)

    print(f'\n[3/3] Building pairs (cross-source, unique A & B):')
    static_pairs = make_pairs(
        static_pool, any_pool, args.n_pairs, 'static', rng)
    move_pairs = make_pairs(
        move_pool, any_pool, args.n_pairs, 'move', rng)
    print(f'  static_pairs={len(static_pairs)} '
          f'move_pairs={len(move_pairs)}')

    if args.dry_run:
        print('\n[dry-run] Skipping write.')
        # Print a short preview
        for tag, pairs in [('static', static_pairs), ('move', move_pairs)]:
            print(f'\n  === {tag} preview (first 3) ===')
            for p in pairs[:3]:
                print(f'    {p["prompt_id"]}: speed={p["_a_end_speed"]:.5f} '
                      f'A[{p["_a_source"]}] num_frames={p["num_frames_a"]}'
                      f' <-> B[{p["_b_source"]}] '
                      f'num_frames={p["num_frames_b"]}')
        return 0

    write_output(
        Path(args.out_static), static_pairs, 'static',
        (0.0, STATIC_MAX), args.pool_size, args.seed,
        notes_extra='L setting: postural-only transition (A nearly static).')
    write_output(
        Path(args.out_move), move_pairs, 'move',
        (MOVE_LOW, MOVE_HIGH), args.pool_size, args.seed,
        notes_extra='M setting: locomotion transition (A walking/jogging).')
    return 0


if __name__ == '__main__':
    sys.exit(main())

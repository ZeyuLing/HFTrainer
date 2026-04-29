"""Rebuild the E9 (Motion Repair) evaluation datalist with stricter
"high-confidence real defect" sampling.

User requirements (2026-04-26):
1. Each defect type gets at least 20 cases (target 25, so the human can
   delete ~5 manually via the dashboard's manual-review UI).
2. Severity MUST equal `fail` (not borderline). Pulled live from the
   MotionQualityChecker, not from the static low_quality.json — the JSON
   only encodes per-checker pass/fail flags but not severity, and we want
   the final cut to reflect the current checker outputs.
3. Within each defect type, prioritise cases that are "highly likely to
   have real problems" → use mask-coverage as the proxy. We re-run the
   specific checker on each candidate and keep cases where the
   `invalid_mask` covers the largest fraction of (T × 22) cells.

The script outputs to `data/eval/m2m_v2/eval_e9_repair_v2.json`. The
existing v1 file (`eval_e9_repair.json`, 215 items) is left in place;
the dashboard pointer in `app.py:_DATALIST_FILES` is flipped over to v2
in a separate change so the new datalist takes effect.

Usage:
    python tools/build_e9_repair_v2.py \
        --target-per-type 25 --candidate-cap 60 \
        --device cuda --out data/eval/m2m_v2/eval_e9_repair_v2.json

The QC pass dominates runtime: ~0.5–1.5s per motion on a single GPU.
For 16 defect types × 60 candidates = 960 motions this is roughly
8–25 minutes wall time. Reduce `--candidate-cap` for a quicker dry run.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hftrainer.evaluation.quality_check_rules.motion_quality_checker import (  # noqa: E402
    MotionQualityChecker,
)

LOW_QUALITY_JSON = REPO_ROOT / 'data' / 'hymotion_m2m_refine_data' / 'data_quality_list' / 'low_quality.json'
DATA_DIR = REPO_ROOT / 'data' / 'hymotion_data'

# 16 defect types covered by the v1 dataset; `rotation_validity` is
# intentionally excluded — the source low_quality.json never marks any
# motion as failing rotation_validity, and the v1 dataset also did not
# include it.
DEFECT_TYPES = [
    'ankle_x', 'arm_penetration', 'candy_wrapper',
    'first_frame_rotation_velocity', 'foot_sliding', 'jitter',
    'joint_jump', 'joint_twist', 'knee_x', 'neck',
    'rotation_velocity', 'small_wobble',
    'spine', 'spine1', 'spine2', 'translation_velocity',
]


def translate_action(name: str) -> str:
    """Cheap fallback English caption used when no rewriter caption exists.

    The downstream rewriter pipeline (run before model inference) replaces
    these with proper "A person ..." captions so we just need a non-empty
    placeholder here.
    """
    cleaned = re.sub(r'_+', ' ', name).strip()
    return f'A person {cleaned}' if cleaned else 'A person performing a motion'


def load_npz_safe(p: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        npz = np.load(p, allow_pickle=True)
        out = dict(npz)
        if 'transl' in out and 'trans' not in out:
            out['trans'] = out['transl']
        return out
    except Exception:
        return None


def _dedup_key(rel_path: str) -> Tuple[str, str]:
    """Canonical identifier for a motion clip that collapses
    mirror/non-mirror duplicates of the same source.

    The motionhub data layout exposes most clips twice: once under
    ``<group>/<dir>/<file>.npz`` (original) and once under
    ``<group>/M_<dir>/<file>.npz`` (mirrored). The two are kinematically
    redundant — they mirror around the YZ plane and any defect detected
    on one nearly always re-appears on the other. Treating them as
    distinct test cases doubles the eval budget and (worse) lets the
    same source action appear multiple times in the dashboard.

    Key = (parent_dir_with_M_prefix_stripped, file_stem). Different time
    crops of the same source (``foo_originalframes_A_B`` vs
    ``foo_originalframes_C_D``) DO get different stems and stay
    separate — that is intentional, sequential clips are genuinely
    different motions.
    """
    parts = Path(rel_path).parts
    if not parts:
        return ('', '')
    stem = Path(parts[-1]).stem
    parent = parts[-2] if len(parts) >= 2 else ''
    parent_norm = parent[2:] if parent.startswith('M_') else parent
    # Include the higher-level group too so e.g. Game/Walk and Taobao/Walk
    # are not collapsed (paths only collide accidentally if the relative
    # group is shared).
    group = '/'.join(parts[:-2]) if len(parts) > 2 else ''
    return (f'{group}/{parent_norm}', stem)


def _load_excluded_keys(exclude_datalist: Optional[str]) -> Set[Tuple[str, str]]:
    if not exclude_datalist:
        return set()
    p = Path(exclude_datalist)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise SystemExit(f'exclude datalist not found: {p}')
    data = json.load(open(p))
    items = data.get('data_list') or []
    out: Set[Tuple[str, str]] = set()
    for it in items:
        motion_path = str(it.get('motion_path') or '').strip()
        if not motion_path:
            continue
        rel = motion_path
        for prefix in ('data/hymotion_data/', 'data/hymotion_data\\'):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
        if os.path.isabs(rel):
            try:
                rel = str(Path(rel).resolve().relative_to(DATA_DIR.resolve()))
            except Exception:
                pass
        out.add(_dedup_key(rel))
    return out


def collect_candidates(excluded_keys: Optional[Set[Tuple[str, str]]] = None) -> Dict[str, List[Dict]]:
    """Group low_quality items by defect type, keep all `failed_checks`
    occurrences (multi-label friendly: a motion failing both jitter and
    foot_sliding lands in BOTH pools).

    Mirror/non-mirror duplicates (``M_<dir>/<file>.npz`` vs
    ``<dir>/<file>.npz``) are collapsed up-front: only the variant
    with the largest ``failed_checks`` count survives — ties broken by
    lexicographic order so the choice is deterministic.
    """
    if not LOW_QUALITY_JSON.exists():
        raise SystemExit(f'low_quality.json not found: {LOW_QUALITY_JSON}')
    data = json.load(open(LOW_QUALITY_JSON))
    base_dir = data.get('data_dir', 'data/hymotion_data')

    # Pass 1: collapse mirror duplicates BEFORE expanding by defect type.
    # Tracking both raw entries lets us keep audit info on which variant
    # survived and which one was dropped.
    by_dedup: Dict[Tuple[str, str], Dict] = {}
    n_collapsed = 0
    for it in data['items']:
        failed = it.get('failed_checks') or []
        if not failed:
            continue
        rel = it['path']
        key = _dedup_key(rel)
        full = os.path.join(base_dir, rel)
        cand = {
            'rel_path': rel,
            'full_path': full,
            'failed_checks': failed,
            'borderline_checks': it.get('borderline_checks') or [],
            'all_checks': it.get('all_checks') or [],
        }
        prev = by_dedup.get(key)
        if prev is None:
            by_dedup[key] = cand
            continue
        # Same canonical motion → keep the variant with more failed_checks
        # (more conservative w.r.t. coverage), tie-break by path string.
        prev_score = (len(prev['failed_checks']),
                      len(prev['borderline_checks']),
                      prev['rel_path'])
        cur_score = (len(cand['failed_checks']),
                     len(cand['borderline_checks']),
                     cand['rel_path'])
        if cur_score > prev_score:
            by_dedup[key] = cand
        n_collapsed += 1
    print(f'[dedup] {n_collapsed} mirror/duplicate motion variants collapsed; '
          f'{len(by_dedup)} unique source clips remain.')

    # Pass 2: expand each unique source clip into one entry per failed
    # defect type (multi-label fan-out is preserved).
    excluded_keys = excluded_keys or set()
    out: Dict[str, List[Dict]] = defaultdict(list)
    for cand in by_dedup.values():
        if _dedup_key(cand['rel_path']) in excluded_keys:
            continue
        for d in cand['failed_checks']:
            if d in DEFECT_TYPES:
                out[d].append(cand)
    return dict(out)


def evaluate_candidate(
    checker_full: MotionQualityChecker,
    cand: Dict,
    defect_type: str,
    min_frames: int,
    max_frames: int,
) -> Optional[Dict]:
    """Run the single-defect checker on a candidate, returning a dict of
    metrics (mask_coverage, severity, T) when the candidate qualifies, or
    None when it should be dropped (file missing, frame count out of range,
    severity != fail, etc.)."""
    if not os.path.exists(cand['full_path']):
        return None
    npz = load_npz_safe(cand['full_path'])
    if npz is None or 'poses' not in npz:
        return None
    T = int(np.asarray(npz['poses']).shape[0])
    if T < min_frames or T > max_frames:
        return None
    fps = float(npz.get('mocap_framerate', 30))

    # Run the FULL checker once (expensive setup is amortised across all
    # 17 checkers; we're already paying the FK + body model cost).
    try:
        agg = checker_full.check(npz)
    except Exception:
        return None

    # AggregatedCheckResult is a regular class (not dict) — access via attrs.
    per = getattr(agg, 'all_results', None) or {}
    res = per.get(defect_type)  # CheckResult is a TypedDict, .get works.
    if res is None:
        return None
    severity = str(res.get('severity') or '').lower()
    if severity != 'fail':
        # User-required: only severity=fail is eligible.
        return None

    mask = res.get('invalid_mask')
    if mask is None:
        # Fallback: if mask is missing we can't measure coverage, drop.
        return None
    mask = np.asarray(mask)
    # Two shapes are common: (T,22) per-joint or (T,) per-frame. Normalise.
    if mask.ndim == 1:
        mask2d = np.broadcast_to(mask[:, None], (mask.shape[0], 22))
    elif mask.ndim == 2:
        mask2d = mask
    else:
        return None
    n_cells = max(int(mask2d.size), 1)
    coverage = float(mask2d.astype(bool).sum()) / n_cells
    coverage_frames = int(mask2d.any(axis=1).sum())
    return {
        'mask_coverage': coverage,
        'mask_frames': coverage_frames,
        'severity': severity,
        'num_frames': T,
        'fps': fps,
    }


def stratified_pick(
    by_defect: Dict[str, List[Dict]],
    target_per_type: int,
    candidate_cap: int,
    min_frames: int,
    max_frames: int,
    min_mask_frames: int,
    device: str,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Walk every defect type; for each, pre-rank candidates by a cheap
    multi-fail prior, evaluate up to `candidate_cap` of them with the
    checker, then keep the top `target_per_type` by mask coverage."""
    print(f'[init] loading MotionQualityChecker on {device} (≈4-8s)...')
    checker_full = MotionQualityChecker(device=device)
    print('[init] checker ready, starting per-type evaluation...')

    # Cross-defect dedup: a motion could be a top-coverage candidate for
    # `joint_twist` AND for `ankle_x` (the failed_checks list contains
    # both). Without this guard the same canonical clip would land in
    # multiple defect_type buckets, re-introducing the very duplication
    # the source-level dedup already removed.
    seen_keys: set = set()
    selected: List[Dict] = []
    stats: Dict[str, Dict] = {}
    # Process types in ascending pool-size order so rare defects (e.g.
    # spine/knee_x with only 30-ish candidates) get first pick before
    # the cross-defect dedup pass strips them out. Without this the
    # alphabetical default starves rare defect types because their
    # candidates are usually flagged by 3-5 defects at once and earlier
    # types claim them first.
    type_order = sorted(
        DEFECT_TYPES,
        key=lambda t: len(by_defect.get(t) or []),
    )
    for defect_type in type_order:
        pool = by_defect.get(defect_type) or []
        # Prior ranking: more failed checks = stronger anomaly. Tiebreak
        # by len(borderline_checks) so cases with extra warnings rise.
        pool_sorted = sorted(
            pool,
            key=lambda x: (-len(x['failed_checks']), -len(x['borderline_checks'])),
        )
        # Adaptive cap: types processed late in `type_order` lose ~80-90 %
        # of their candidates to cross-defect dedup, so they need a much
        # bigger head to still reach `target_per_type`. The first ~half
        # of types get the base cap; the rest scale up to 5× to give
        # joint_jump / foot_sliding / candy_wrapper a fighting chance.
        # A defect with a tiny global pool (e.g. knee_x with ~31 unique
        # clips in the whole DB) is naturally limited and is intentionally
        # left under target.
        idx_in_order = type_order.index(defect_type)
        if idx_in_order >= len(type_order) // 2:
            adaptive_cap = candidate_cap * 5
        else:
            adaptive_cap = candidate_cap
        head = pool_sorted[:adaptive_cap]
        evaluated: List[Tuple[Dict, Dict]] = []
        n_skipped = 0
        for ci, cand in enumerate(head):
            metrics = evaluate_candidate(
                checker_full, cand, defect_type,
                min_frames=min_frames, max_frames=max_frames,
            )
            if metrics is None:
                n_skipped += 1
                continue
            if metrics['mask_frames'] < min_mask_frames:
                # Don't waste a slot on a case where the checker barely
                # flagged anything — those tend to be ambiguous.
                n_skipped += 1
                continue
            evaluated.append((cand, metrics))
            if (ci + 1) % 10 == 0:
                print(f'  {defect_type}: {ci+1}/{len(head)} evaluated, '
                      f'{len(evaluated)} qualified, {n_skipped} skipped')
        # Final sort by mask coverage descending, then mask_frames.
        evaluated.sort(
            key=lambda x: (-x[1]['mask_coverage'], -x[1]['mask_frames']),
        )
        # Cross-defect dedup: walk in coverage order and skip any clip
        # whose canonical key has already been claimed by an earlier
        # defect type. Earlier defects in DEFECT_TYPES list win — order
        # is the alphabetical default, which is stable and reproducible.
        keep: List[Tuple[Dict, Dict]] = []
        n_dup = 0
        for cand, metrics in evaluated:
            key = _dedup_key(cand['rel_path'])
            if key in seen_keys:
                n_dup += 1
                continue
            keep.append((cand, metrics))
            seen_keys.add(key)
            if len(keep) >= target_per_type:
                break
        stats[defect_type] = {
            'pool_total': len(pool),
            'pool_after_prior': len(head),
            'qualified': len(evaluated),
            'cross_defect_dup_skipped': n_dup,
            'kept': len(keep),
        }
        for cand, metrics in keep:
            fname = os.path.basename(cand['full_path']).replace('.npz', '')
            name_clean = re.sub(r'_originalframes_\d+_\d+$', '', fname)
            name_clean = re.sub(r'_take_\d+$', '', name_clean)
            selected.append({
                'motion_path': cand['full_path'],
                'action_name': name_clean,
                'caption_en': translate_action(name_clean),
                'category': f'defect_{defect_type}',
                'defect_type': defect_type,
                'all_defects': cand['failed_checks'],
                'borderline_checks': cand['borderline_checks'],
                'num_frames': metrics['num_frames'],
                'fps': metrics['fps'],
                'duration_sec': round(metrics['num_frames'] / max(metrics['fps'], 1.0), 2),
                'source': 'low_quality_db_v2_severity_fail',
                'mask_coverage': round(metrics['mask_coverage'], 4),
                'mask_frames': metrics['mask_frames'],
            })
        print(f'[type {defect_type}] kept {len(keep)} / target {target_per_type} '
              f'(pool={len(pool)}, head={len(head)}, qualified={len(evaluated)})')
    return selected, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-per-type', type=int, default=25,
                    help='Target cases kept per defect type. '
                    'User contract is >=20; we keep an extra cushion '
                    'for manual deletion.')
    ap.add_argument('--candidate-cap', type=int, default=120,
                    help='Max # candidates per type to evaluate with QC. '
                    'Bumped from 60 to 120 (2026-04-27) so that even '
                    'after mirror-pair + cross-defect dedup we still '
                    'have enough qualified items to fill target_per_type '
                    'for each defect_type.')
    ap.add_argument('--min-frames', type=int, default=60)
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--min-mask-frames', type=int, default=2,
                    help='Reject items whose flagged frame count is below '
                         'this — too ambiguous to be a "real defect".')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='data/eval/m2m_v2/eval_e9_repair_v2.json')
    ap.add_argument(
        '--exclude-datalist',
        default='',
        help='Optional existing eval datalist whose motions should be excluded '
             'from the new selection (dedup-aware).',
    )
    args = ap.parse_args()

    excluded_keys = _load_excluded_keys(args.exclude_datalist)
    if excluded_keys:
        print(f'[exclude] loaded {len(excluded_keys)} prior motion keys from '
              f'{args.exclude_datalist}')

    by_defect = collect_candidates(excluded_keys)
    pool_summary = {k: len(v) for k, v in by_defect.items()}
    print('[stats] candidate pool by defect type:')
    for k in DEFECT_TYPES:
        print(f'  {k}: {pool_summary.get(k, 0)}')

    selected, stats = stratified_pick(
        by_defect,
        target_per_type=args.target_per_type,
        candidate_cap=args.candidate_cap,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        min_mask_frames=args.min_mask_frames,
        device=args.device,
    )

    # Distribution of final selection
    defect_dist = defaultdict(int)
    for it in selected:
        defect_dist[it['defect_type']] += 1
    out = {
        'meta': {
            'task_id': 'E9',
            'task_name': 'Motion Repair (v2 — severity=fail, mask-coverage ranked)',
            'description': (
                'Repair REAL defective motions from quality checker. '
                'v2 (2026-04-26) selection rules:\n'
                '  1. severity must == "fail" (live re-check, not stale flags);\n'
                '  2. each defect type targets >=20 cases ('
                f'this run: target_per_type={args.target_per_type});\n'
                '  3. within each type, items are ranked by mask-coverage '
                'descending so the highest-confidence real defects come first.'
            ),
            'total_items': len(selected),
            'source': str(LOW_QUALITY_JSON.relative_to(REPO_ROOT)),
            'defect_distribution': dict(defect_dist),
            'min_frames': args.min_frames,
            'max_frames': args.max_frames,
            'min_mask_frames': args.min_mask_frames,
            'target_per_type': args.target_per_type,
            'candidate_cap': args.candidate_cap,
            'excluded_prior_items': len(excluded_keys),
            'excluded_prior_datalist': args.exclude_datalist or None,
            'pool_summary': pool_summary,
            'selection_stats': stats,
        },
        'data_list': selected,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[done] wrote {out_path} — {len(selected)} items')
    print('[done] final defect distribution:')
    for k in DEFECT_TYPES:
        n = defect_dist.get(k, 0)
        flag = '  ' if n >= 20 else ' ⚠️'
        print(f'{flag} {k}: {n}')


if __name__ == '__main__':
    main()

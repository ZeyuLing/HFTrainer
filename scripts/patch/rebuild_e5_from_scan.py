#!/usr/bin/env python3
"""Rebuild eval_e5_trajectory.json from the pelvis-pathlen scan.

Problems with the previous E5 datalist (observed 2026-04-22):
  1. ~79% of samples had pelvis path_len_xz < 1m — actions that are
     stationary but got pulled in by category heuristics or keyword
     matches like "移动棋子" (chess piece move = literal 'move' keyword
     matches "移动", but the person is SITTING).
  2. action_name repetition: e.g. "移动棋子" appeared 3×, "坐着按住肩键
     并移动摇杆" appeared 5×. Same motion with different take indices
     tests nothing new.
  3. No true long-distance locomotion samples (max was 4.07m, mostly
     stationary sports animations).

Strategy:
  A. Require path_len_xz >= MIN_PATH (default 2.0m).
  B. Hard-exclude action_names containing "sitting / kneeling / sit-only
     motion" keywords — these creep in with high path_len because the
     arm swings translate pelvis_abs slightly when motion-captured.
  C. Dedupe by action_name — at most K samples (default 1) per exact
     action_name. Allows genuinely different same-name takes only if
     path_len differs substantially.
  D. Keep diverse categories (sports_ball, combat, locomotion, performance).
  E. Cap total to N samples (default 100), preserving the path_len
     descending order.

Run:  python3 tools/rebuild_e5_from_scan.py [--n 100] [--min-path 2.0]
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
DATA_DIR = ROOT / 'data' / 'eval' / 'm2m_v2'
SCAN_JSON = DATA_DIR / '_pelvis_pathlen_scan.json'
E5_ORIG = DATA_DIR / 'eval_e5_trajectory.json'
E5_REWRITTEN = DATA_DIR / 'eval_e5_trajectory_rewritten.json'

# Reject action_name whose Chinese content matches any of these markers.
# These capture scenarios where the mocap pelvis translates a lot
# (because cameras / plate slide slightly) but the actual action is
# stationary — hence OOD for "trajectory following".
STATIONARY_NAME_PATTERNS = [
    '坐', '跪', '蹲',
    '棋', '摇杆', '肩键',   # chess, gamepad — sitting & pressing buttons
    '操舵', '舵轮', '打舵',  # sailing: sitting and turning wheel
    '按键盘', '敲键盘',      # typing
    '书写', '写字',
    '擦桌', '擦地',
]

# Keep these categories (the ones where pelvis translation IS the signal).
PREFERRED_CATEGORIES = {
    'locomotion', 'sports_ball', 'sports_other', 'combat',
    'daily_stand', 'performance',
}


def is_stationary(action_name: str) -> bool:
    return any(p in action_name for p in STATIONARY_NAME_PATTERNS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100,
                   help='Target number of samples in the new datalist')
    p.add_argument('--min-path', type=float, default=2.0,
                   help='Minimum pelvis XZ path_len (meters)')
    p.add_argument('--max-per-action', type=int, default=1,
                   help='Max samples allowed per exact action_name')
    p.add_argument('--max-per-category', type=int, default=30,
                   help='Max samples per category (prevents sports_ball '
                        'from dominating since it has the highest path_len '
                        'clusters)')
    args = p.parse_args()

    if not SCAN_JSON.exists():
        raise SystemExit(f'Missing {SCAN_JSON}. Run tools/scan_pelvis_pathlen.py first.')

    all_entries = json.load(open(SCAN_JSON))
    # Already sorted by path_len_xz desc, but be safe:
    all_entries.sort(key=lambda e: -e['path_len_xz'])

    kept: List[Dict] = []
    seen_action: Dict[str, int] = Counter()
    seen_category: Dict[str, int] = Counter()
    rejects = Counter()

    for e in all_entries:
        if len(kept) >= args.n:
            break
        if e['path_len_xz'] < args.min_path:
            rejects['path_len_too_short'] += 1
            continue
        if is_stationary(e['action_name']):
            rejects['stationary_keyword'] += 1
            continue
        if e['category'] not in PREFERRED_CATEGORIES:
            rejects['category'] += 1
            continue
        if seen_action[e['action_name']] >= args.max_per_action:
            rejects['action_dedupe'] += 1
            continue
        if seen_category[e['category']] >= args.max_per_category:
            rejects['category_cap'] += 1
            continue
        kept.append(e)
        seen_action[e['action_name']] += 1
        seen_category[e['category']] += 1

    print(f'\nSelected {len(kept)} samples from {len(all_entries)} total:')
    print(f'  rejects: {dict(rejects)}')

    # Shape entries to match the existing datalist schema.
    data_list_items: List[Dict] = []
    for e in kept:
        data_list_items.append({
            'motion_path': e['path'],
            'action_name': e['action_name'],
            'caption_en': e['caption_en'],
            'category': e['category'],
            'num_frames': e['num_frames'],
            'fps': e['fps'],
            'duration_sec': e['duration_sec'],
            'source': e['rel_dir'],
            '_pelvis_path_len_xz': round(e['path_len_xz'], 3),  # keep provenance
        })

    # Distribution summary
    cats = Counter(it['category'] for it in data_list_items)
    print('\nCategory distribution:')
    for c, n in cats.most_common():
        print(f'  {c:18} {n}')
    pls = [it['_pelvis_path_len_xz'] for it in data_list_items]
    print(f'\npath_len_xz: min={min(pls):.2f}m median={sorted(pls)[len(pls)//2]:.2f}m '
          f'max={max(pls):.2f}m')
    print(f'  bins: >=3m: {sum(1 for x in pls if x >= 3)}  '
          f'>=4m: {sum(1 for x in pls if x >= 4)}  '
          f'>=5m: {sum(1 for x in pls if x >= 5)}')

    # Preserve original E5 metadata (task, description, settings) — only
    # replace `data_list`. Back up before writing.
    orig = json.load(open(E5_ORIG))
    backup = E5_ORIG.with_suffix(
        f'.json.bak_before_pathlen_rebuild_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy(E5_ORIG, backup)
    print(f'\nBacked up {E5_ORIG.name} → {backup.name}')

    orig['data_list'] = data_list_items
    with open(E5_ORIG, 'w') as f:
        json.dump(orig, f, ensure_ascii=False, indent=2)
    print(f'Wrote {E5_ORIG.name} (n={len(data_list_items)})')

    # Rewrite the _rewritten sister file if present by copying captions
    # from the existing rewritten file where available (key by motion_path).
    if E5_REWRITTEN.exists():
        rewritten_orig = json.load(open(E5_REWRITTEN))
        cap_map = {
            it.get('motion_path'): it
            for it in rewritten_orig.get('data_list', [])
        }
        new_rewritten_list = []
        missing = 0
        for it in data_list_items:
            src = cap_map.get(it['motion_path'])
            if src is None:
                missing += 1
                # Fall back to the non-rewritten version (still better than nothing).
                new_rewritten_list.append(dict(it))
            else:
                new_rewritten_list.append({**it, **{
                    k: v for k, v in src.items()
                    if k in ('caption_en', 'caption_rewritten', 'prompt_en',
                             'caption', 'caption_zh')
                }})

        rew_backup = E5_REWRITTEN.with_suffix(
            f'.json.bak_before_pathlen_rebuild_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        shutil.copy(E5_REWRITTEN, rew_backup)
        rewritten_orig['data_list'] = new_rewritten_list
        with open(E5_REWRITTEN, 'w') as f:
            json.dump(rewritten_orig, f, ensure_ascii=False, indent=2)
        print(f'Backed up {E5_REWRITTEN.name} → {rew_backup.name}')
        print(f'Wrote {E5_REWRITTEN.name} (n={len(new_rewritten_list)}, '
              f'missing rewritten caption for {missing} items → copied from base)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Rebuild eval_e2_inbetween.json using the pelvis-pathlen scan plus a
pose-difference score computed on-the-fly.

The previous E2 datalist suffered from the same issues as E5:
  - 120 samples but 113 unique action_names (≤7 duplicates)
  - Some action_names appeared 4× (e.g. "开帆船双手握舵轮向右打舵",
    "单膝跪地行一个吻手礼") testing nearly-identical motions
  - Heavy skew toward stationary manipulation (sitting chess / sailing)

E2 is motion in-betweening — the score should reward:
  1. Genuine action diversity (1 sample per action_name)
  2. Sufficient motion (not all constant T-pose padding). Use
     path_len_xz >= 0.5m as a weak motion signal (E2 doesn't require
     long locomotion, but truly-static clips are boring)
  3. Category spread — keep the distribution even across categories
     so different action types get tested.

Run: python3 tools/rebuild_e2_from_scan.py [--n 120]
"""
from __future__ import annotations
import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
DATA_DIR = ROOT / 'data' / 'eval' / 'm2m_v2'
SCAN_JSON = DATA_DIR / '_pelvis_pathlen_scan.json'
E2_ORIG = DATA_DIR / 'eval_e2_inbetween.json'
E2_REWRITTEN = DATA_DIR / 'eval_e2_inbetween_rewritten.json'

# Keep every category — E2 doesn't require locomotion; sitting/gesture
# actions ARE legitimate test cases.
PREFERRED_CATEGORIES = None  # None = keep all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=120)
    p.add_argument('--min-path', type=float, default=0.5,
                   help='Reject near-static clips (path_len_xz < threshold)')
    p.add_argument('--max-per-action', type=int, default=1)
    p.add_argument('--max-per-category', type=int, default=12,
                   help='Spread across action categories')
    args = p.parse_args()

    all_entries = json.load(open(SCAN_JSON))
    # Sort: primarily by path_len_xz (more motion = more interesting
    # in-betweening target), descending.
    all_entries.sort(key=lambda e: -e['path_len_xz'])

    kept: List[Dict] = []
    seen_action: Dict[str, int] = Counter()
    seen_cat: Dict[str, int] = Counter()
    rejects = Counter()

    for e in all_entries:
        if len(kept) >= args.n:
            break
        if e['num_frames'] < 60 or e['num_frames'] > 600:
            rejects['length'] += 1
            continue
        if e['path_len_xz'] < args.min_path:
            rejects['path_len_too_short'] += 1
            continue
        if seen_action[e['action_name']] >= args.max_per_action:
            rejects['action_dedupe'] += 1
            continue
        if seen_cat[e['category']] >= args.max_per_category:
            rejects['category_cap'] += 1
            continue
        kept.append(e)
        seen_action[e['action_name']] += 1
        seen_cat[e['category']] += 1

    # If we haven't hit the quota (e.g. too strict), relax category_cap
    # once and retry.
    if len(kept) < args.n:
        for e in all_entries:
            if len(kept) >= args.n:
                break
            if e['num_frames'] < 60 or e['num_frames'] > 600:
                continue
            if e['path_len_xz'] < args.min_path:
                continue
            if seen_action[e['action_name']] >= args.max_per_action:
                continue
            # Skip the per-category cap this pass
            if e in kept:
                continue
            # Check if this exact entry was already kept
            if any(k['path'] == e['path'] for k in kept):
                continue
            kept.append(e)
            seen_action[e['action_name']] += 1
            seen_cat[e['category']] += 1

    print(f'\nSelected {len(kept)} samples from {len(all_entries)} total:')
    print(f'  rejects: {dict(rejects)}')

    # Reshape to the datalist schema
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
            '_pelvis_path_len_xz': round(e['path_len_xz'], 3),
        })

    cats = Counter(it['category'] for it in data_list_items)
    print('\nCategory distribution:')
    for c, n in cats.most_common():
        print(f'  {c:18} {n}')
    unique_actions = len(set(it['action_name'] for it in data_list_items))
    print(f'\nUnique action_names: {unique_actions}/{len(data_list_items)}')
    pls = [it['_pelvis_path_len_xz'] for it in data_list_items]
    print(f'path_len_xz: min={min(pls):.2f}m median={sorted(pls)[len(pls)//2]:.2f}m '
          f'max={max(pls):.2f}m')

    # Back up + write
    orig = json.load(open(E2_ORIG))
    backup = E2_ORIG.with_suffix(
        f'.json.bak_before_pathlen_rebuild_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy(E2_ORIG, backup)
    print(f'\nBacked up {E2_ORIG.name} → {backup.name}')
    orig['data_list'] = data_list_items
    with open(E2_ORIG, 'w') as f:
        json.dump(orig, f, ensure_ascii=False, indent=2)
    print(f'Wrote {E2_ORIG.name} (n={len(data_list_items)})')

    if E2_REWRITTEN.exists():
        rewritten_orig = json.load(open(E2_REWRITTEN))
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
                new_rewritten_list.append(dict(it))
            else:
                new_rewritten_list.append({**it, **{
                    k: v for k, v in src.items()
                    if k in ('caption_en', 'caption_rewritten', 'prompt_en',
                             'caption', 'caption_zh')
                }})
        rew_backup = E2_REWRITTEN.with_suffix(
            f'.json.bak_before_pathlen_rebuild_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        shutil.copy(E2_REWRITTEN, rew_backup)
        rewritten_orig['data_list'] = new_rewritten_list
        with open(E2_REWRITTEN, 'w') as f:
            json.dump(rewritten_orig, f, ensure_ascii=False, indent=2)
        print(f'Backed up {E2_REWRITTEN.name} → {rew_backup.name}')
        print(f'Wrote {E2_REWRITTEN.name} (n={len(new_rewritten_list)}, '
              f'missing rewritten caption for {missing})')


if __name__ == '__main__':
    main()

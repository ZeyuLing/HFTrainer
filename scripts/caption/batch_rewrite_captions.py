#!/usr/bin/env python3
"""Batch-rewrite captions for E2/E5 (or any eval task) datalist.

Runs the `/api/rewrite_caption` endpoint for every item in
`eval_e{2,5}_*.json` and writes the result into the companion
`eval_e{2,5}_*_rewritten.json` file, updating (or adding) the
`caption_en` field with the English short caption from the rewriter.

After E2/E5 rebuild on 2026-04-22, the rewritten JSONs had 98-110 items
falling back to the Chinese action_name (because the new motion_path
wasn't in the old cache). Captioned models (`caption_local`,
`caption_global`) train on English rewrites, so showing them Chinese
causes an OOD dip.

Usage:
    python3 tools/batch_rewrite_captions.py --tasks E2 E5
    python3 tools/batch_rewrite_captions.py --tasks E13 --force
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
DATA_DIR = ROOT / 'data' / 'eval' / 'm2m_v2'
REWRITE_URL = 'http://127.0.0.1:8081/api/rewrite_caption'

DATALIST_MAP = {
    'E1':  ('eval_e1_t2m.json',              'eval_e1_t2m_rewritten.json'),
    'E2':  ('eval_e2_inbetween.json',        'eval_e2_inbetween_rewritten.json'),
    'E3':  ('eval_e3_keyframe.json',         'eval_e3_keyframe_rewritten.json'),
    'E5':  ('eval_e5_trajectory.json',       'eval_e5_trajectory_rewritten.json'),
    'E6':  ('eval_e6_foot_ground.json',      'eval_e6_foot_ground_rewritten.json'),
    'E8':  ('eval_e8_loop.json',             'eval_e8_loop_rewritten.json'),
    'E13': ('eval_e13_multi_prompt.json',    'eval_e13_multi_prompt_rewritten.json'),
}


def rewrite_one(caption_zh: str, tries: int = 2) -> str:
    """Call rewriter API. Returns English rewritten caption or empty
    string on failure."""
    for attempt in range(tries):
        try:
            r = requests.post(
                REWRITE_URL,
                json={'caption': caption_zh},
                timeout=30,
            )
            if r.status_code == 200:
                j = r.json()
                return j.get('rewritten', '').strip()
            print(f'    [http {r.status_code}] {r.text[:200]}')
        except Exception as exc:
            print(f'    [try {attempt+1}] {exc!r}')
            time.sleep(2)
    return ''


def process_task(task_id: str, force: bool = False, limit: int = 0):
    base_name, rew_name = DATALIST_MAP[task_id]
    base_path = DATA_DIR / base_name
    rew_path = DATA_DIR / rew_name

    if not base_path.exists():
        print(f'  {task_id}: missing {base_name}, skip')
        return
    base = json.load(open(base_path))
    base_dl = base.get('data_list', base)

    # Index existing rewritten entries by motion_path so we can re-use
    # already-rewritten captions unless --force is given.
    existing_rew: dict = {}
    if rew_path.exists():
        rew_json = json.load(open(rew_path))
        for it in rew_json.get('data_list', []):
            mp = it.get('motion_path')
            # Only treat it as already-rewritten if the caption_en looks
            # like English (ASCII-heavy) and differs from the Chinese
            # action_name — otherwise it's the fallback placeholder.
            cen = (it.get('caption_en') or '').strip()
            an = (it.get('action_name') or '').strip()
            if mp and cen and cen != an:
                ascii_ratio = sum(1 for c in cen if ord(c) < 128) / max(1, len(cen))
                if ascii_ratio > 0.7:
                    existing_rew[mp] = it

    # Produce new rewritten datalist item-by-item
    new_items = []
    n_cached, n_rewritten, n_failed = 0, 0, 0
    for i, it in enumerate(base_dl):
        if limit and i >= limit:
            new_items.append(it)
            continue
        mp = it.get('motion_path')
        an = it.get('action_name', '') or it.get('caption', '')
        if not force and mp in existing_rew:
            new_items.append({**it, **existing_rew[mp]})
            n_cached += 1
            continue

        print(f'  [{i+1}/{len(base_dl)}] {an[:60]}')
        en = rewrite_one(an)
        if en:
            n_rewritten += 1
            new_items.append({
                **it,
                'caption_en': en,
                'caption_zh': an,
            })
        else:
            n_failed += 1
            # Fall back to Chinese so the datalist still has an entry.
            new_items.append({**it, 'caption_en': an, 'caption_zh': an})

    # Save
    out = base.copy() if isinstance(base, dict) else {'data_list': None}
    out['data_list'] = new_items
    # Back up first
    if rew_path.exists():
        bak = rew_path.with_suffix(f'.json.bak_before_rewrite_batch_{int(time.time())}')
        import shutil
        shutil.copy(rew_path, bak)
        print(f'  backed up → {bak.name}')
    with open(rew_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'  {task_id}: cached={n_cached} rewritten={n_rewritten} failed={n_failed} → {rew_name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['E2', 'E5'],
                    help='Task IDs to rewrite')
    ap.add_argument('--force', action='store_true',
                    help='Re-rewrite even if an English caption already exists')
    ap.add_argument('--limit', type=int, default=0,
                    help='Only rewrite the first N items (for smoke testing)')
    args = ap.parse_args()

    for tid in args.tasks:
        if tid not in DATALIST_MAP:
            print(f'Skip unknown task {tid}')
            continue
        print(f'\n=== {tid} ===')
        process_task(tid, force=args.force, limit=args.limit)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Augment E2-v2 datalist JSON meta with `category_distribution_detail`.

The dashboard (`motion_annot_web/eval_dashboard/templates/task_detail.html`)
renders the test-case distribution chart only when `meta.category_distribution_detail`
is present. That field carries per-category `count / percent / unique_actions /
example_actions / example_captions_en`. Our original E2-v2 build script only
emitted the simpler `category_distribution` map, so the chart stayed blank.

This script patches both files in place:
    data/eval/m2m_v2/eval_e2_inbetween_v2.json
    data/eval/m2m_v2/eval_e2_inbetween_v2_rewritten.json

The detail block is derived from the items' own `category` + `action_name`
(or `motion_path` stem) + `caption_en` fields, so no re-selection of cases
is performed. Safe to rerun — overwrites only the augmentation fields.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
FILES = [
    ROOT / 'data' / 'eval' / 'm2m_v2' / 'eval_e2_inbetween_v2.json',
    ROOT / 'data' / 'eval' / 'm2m_v2' / 'eval_e2_inbetween_v2_rewritten.json',
]


def build_detail(items):
    by_cat = defaultdict(list)
    for it in items:
        cat = it.get('category') or 'unknown'
        by_cat[cat].append(it)

    total = sum(len(v) for v in by_cat.values()) or 1
    detail = {}
    for cat in sorted(by_cat.keys()):
        rows = by_cat[cat]
        actions = []
        for it in rows:
            a = it.get('action_name') or Path(it.get('motion_path', '')).stem
            if a:
                actions.append(a)
        uniq = sorted(set(actions))
        caps = []
        seen = set()
        for it in rows:
            c = (it.get('caption_en') or '').strip()
            if c and c not in seen:
                caps.append(c)
                seen.add(c)
            if len(caps) >= 5:
                break
        detail[cat] = {
            'count': len(rows),
            'percent': round(100.0 * len(rows) / total, 1),
            'unique_actions': len(uniq),
            'example_actions': uniq[:5],
            'example_captions_en': caps[:5],
        }
    return detail


def main():
    for fp in FILES:
        if not fp.exists():
            print(f'skip (missing): {fp}')
            continue
        d = json.loads(fp.read_text(encoding='utf-8'))
        items = d.get('data_list', [])
        detail = build_detail(items)
        meta = d.setdefault('meta', {})
        meta['category_distribution_detail'] = detail
        # frame_stats to mirror other tasks' dashboards
        frames = [int(it.get('num_frames') or 0) for it in items
                  if it.get('num_frames')]
        if frames:
            frames_sorted = sorted(frames)
            n = len(frames_sorted)
            meta['frame_stats'] = {
                'min': frames_sorted[0],
                'max': frames_sorted[-1],
                'mean': round(sum(frames_sorted) / n, 1),
                'median': frames_sorted[n // 2],
            }
        fp.write_text(
            json.dumps(d, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        cats = sorted(detail.keys())
        print(f'patched {fp.name}: categories={cats}, '
              f'counts={[detail[c]["count"] for c in cats]}')


if __name__ == '__main__':
    main()

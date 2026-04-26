#!/usr/bin/env python3
"""Scan the Private motion library and compute pelvis XZ path_len for
every NPZ. Output a JSON sorted by path_len descending.

Purpose: feeding E5 (trajectory following) and related eval case
selection with a real-displacement-based ranking, since the current
keyword/category-based rules mis-include stationary actions like
"移动棋子" (sitting chess move) or "开帆船...打舵" (sitting sailing).

This only needs pelvis translation (data['trans']), which is fast
to load — ~1-2ms per file vs ~50ms for full FK pose.

Output: data/eval/m2m_v2/_pelvis_pathlen_scan.json
        [{"path": ..., "action_name": ..., "num_frames": T, "fps": 30,
          "path_len_xz": 12.34, "chord_xz": 8.21, "category": ...}, ...]
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(ROOT))
# Reuse the same categorization / finger-only filter as the eval builder
# so the pool matches what the other eval scripts would accept.
from tools.build_m2m_v2_eval_data import (
    categorize_action, translate_action, is_finger_only_gesture,
)

SRC_BASE = Path('/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private')
OUT_PATH = ROOT / 'data' / 'eval' / 'm2m_v2' / '_pelvis_pathlen_scan.json'


def scan_one(path: Path) -> Optional[Dict]:
    """Fast pelvis-trans scan of a single NPZ."""
    try:
        d = np.load(str(path), allow_pickle=True)
    except Exception:
        return None
    files = set(d.files)
    tk = 'trans' if 'trans' in files else ('transl' if 'transl' in files else None)
    pk = 'poses' if 'poses' in files else ('body_pose' if 'body_pose' in files else None)
    if tk is None or pk is None:
        return None
    try:
        trans = np.asarray(d[tk], dtype=np.float32)
        T = int(d[pk].shape[0])
    except Exception:
        return None
    if trans.ndim != 2 or trans.shape[0] < 2 or trans.shape[1] < 3:
        return None
    # pelvis XZ path
    xz = trans[:T, [0, 2]]
    path_len = float(np.linalg.norm(np.diff(xz, axis=0), axis=-1).sum())
    chord = float(np.linalg.norm(xz[-1] - xz[0]))
    fps = float(d.get('mocap_framerate', 30)) if 'mocap_framerate' in files else 30.0
    return {
        'path': str(path),
        'num_frames': T,
        'fps': fps,
        'path_len_xz': path_len,
        'chord_xz': chord,
    }


def action_name_from_filename(fname: str) -> str:
    """Strip common suffixes to get the raw action name (mirror
    build_m2m_v2_eval_data.scan_all_motions)."""
    name = fname.replace('.npz', '')
    name = re.sub(r'_originalframes_\d+_\d+$', '', name)
    name = re.sub(r'_take_\d+$', '', name)
    return name


def main():
    dirs = sorted([d for d in SRC_BASE.iterdir() if d.is_dir()])
    print(f'Scanning {len(dirs)} subdirs under {SRC_BASE}...')

    all_entries: List[Dict] = []
    total = 0
    skipped = 0
    for d in dirs:
        npz_files = sorted(d.glob('*.npz'))
        for p in npz_files:
            total += 1
            entry = scan_one(p)
            if entry is None:
                skipped += 1
                continue
            an = action_name_from_filename(p.name)
            if is_finger_only_gesture(an, categorize_action(an)):
                continue
            entry['action_name'] = an
            entry['caption_en'] = translate_action(an)
            entry['category'] = categorize_action(an)
            entry['rel_dir'] = d.name
            entry['filename'] = p.name
            entry['duration_sec'] = round(entry['num_frames'] / entry['fps'], 2)
            all_entries.append(entry)
        if total % 500 == 0 or p == npz_files[-1] if npz_files else False:
            print(f'  scanned {total}/{4218} so far, kept {len(all_entries)}')

    all_entries.sort(key=lambda x: -x['path_len_xz'])
    print(f'\nDone: total={total}, skipped={skipped}, kept={len(all_entries)}')

    # Quick histogram
    pl = np.array([e['path_len_xz'] for e in all_entries])
    print(f'path_len_xz: min={pl.min():.2f}m median={np.median(pl):.2f}m '
          f'p90={np.percentile(pl, 90):.2f}m max={pl.max():.2f}m')
    for t in (1, 2, 3, 5, 8, 12):
        print(f'  >={t}m: {int((pl >= t).sum())}')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f'\nWrote {OUT_PATH} ({len(all_entries)} entries)')


if __name__ == '__main__':
    main()

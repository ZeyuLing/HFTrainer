#!/usr/bin/env python3
"""Build the (caption, G1-npz) training list for G1-native T2M fine-tuning.

Joins the robot-suitable ``high_quality.json`` clip set (already retargeted into
``data/g1/<rel>.npz``) with the HYMotion caption annotation
(``train_hymotion_400h.json``) so each clip carries:

    g1_path     -- relative path under ``--g1-dir`` (== high_quality ``path``)
    caption_rel -- caption json path relative to ``--data-dir`` (from the
                   annotation when available; the dataset additionally falls
                   back to a priority list of caption dirs at load time)
    num_frames  -- from the annotation (None -> dataset reads npz length)
    subset      -- academic / 3d / game / taobao / ...

The annotation is large (~300 MB) and CephFS is slow, so the parsed
``hq_set`` / ``cap_map`` are cached under ``/dev/shm`` for fast re-runs.

Usage::

    python3 scripts/embodied/build_g1_t2m_train_list.py \
        --out data/annotation/train_g1_t2m.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time

_HYMOTION_PREFIXES = (
    '../hymotion_data/', 'data/hymotion_data/', './hymotion_data/', 'hymotion_data/',
)

# Caption directories tried (in order) when the annotation has no caption_path.
# Mirrors what exists under each <subset>/<date>/ folder.
CAPTION_DIR_PRIORITY = [
    'human_checked_augmented_caption',
    'human_checked_caption',
    'improved_simple_caption',
    'augmented_caption',
    'raw_caption',
]


def norm_rel(p):
    if not p:
        return None
    for pre in _HYMOTION_PREFIXES:
        if p.startswith(pre):
            return p[len(pre):]
    return p


def load_cached(path, builder, force=False):
    cache = os.path.join('/dev/shm', 'g1t2m_' + path.replace('/', '_') + '.pkl')
    if not force and os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    obj = builder()
    try:
        with open(cache, 'wb') as f:
            pickle.dump(obj, f)
    except Exception:
        pass
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--high-quality',
                    default='data/hymotion_m2m_refine_data/data_quality_list/high_quality.json')
    ap.add_argument('--anno', default='data/annotation/train_hymotion_400h.json')
    ap.add_argument('--g1-dir', default='data/g1')
    ap.add_argument('--data-dir', default='data/hymotion_data')
    ap.add_argument('--out', default='data/annotation/train_g1_t2m.json')
    ap.add_argument('--verify-samples', type=int, default=20,
                    help='sample-verify g1 npz + caption existence on N items')
    ap.add_argument('--force-cache', action='store_true')
    args = ap.parse_args()

    t0 = time.time()

    def _load_hq():
        with open(args.high_quality) as f:
            hq = json.load(f)
        return {'data_dir': hq.get('data_dir', args.data_dir),
                'paths': [it['path'] for it in hq['items']]}

    def _load_cap():
        with open(args.anno) as f:
            ann = json.load(f)
        dl = ann['data_list']
        cap = {}
        for v in dl.values():
            rel = norm_rel(v.get('smplx_path'))
            if rel is None:
                continue
            cap[rel] = {
                'caption_rel': norm_rel(
                    v.get('hierarchical_caption_path') or v.get('caption_path')),
                'num_frames': v.get('num_frames'),
                'subset': v.get('subset'),
            }
        return cap

    print('[build] loading high_quality ...', flush=True)
    hq = load_cached(args.high_quality, _load_hq, force=args.force_cache)
    print(f'[build]   {len(hq["paths"])} robot-suitable clips '
          f'({time.time()-t0:.0f}s)', flush=True)

    print('[build] loading caption annotation ...', flush=True)
    cap_map = load_cached(args.anno, _load_cap, force=args.force_cache)
    print(f'[build]   {len(cap_map)} annotated motions ({time.time()-t0:.0f}s)',
          flush=True)

    items = []
    n_with_cap = 0
    for rel in hq['paths']:
        info = cap_map.get(rel)
        caption_rel = info['caption_rel'] if info else None
        if caption_rel:
            n_with_cap += 1
        items.append({
            'g1_path': rel,
            'caption_rel': caption_rel,
            'num_frames': info['num_frames'] if info else None,
            'subset': (info['subset'] if info else None) or rel.split('/')[0].lower(),
        })

    out = {
        'meta_info': {
            'dataset': 'g1_t2m (robot-suitable HYMotion retargeted to Unitree G1)',
            'version': 'v1',
            'g1_dir': args.g1_dir,
            'data_dir': args.data_dir,
            'caption_dir_priority': CAPTION_DIR_PRIORITY,
            'n_total': len(items),
            'n_with_annotation_caption': n_with_cap,
        },
        'items': items,
    }

    # ---- sample verification ----
    if args.verify_samples > 0:
        random.seed(0)
        samp = random.sample(items, min(args.verify_samples, len(items)))
        ok_g1 = ok_cap = 0
        for it in samp:
            g1 = os.path.join(args.g1_dir, it['g1_path'])
            if os.path.exists(g1):
                ok_g1 += 1
            cap_found = False
            if it['caption_rel'] and os.path.exists(
                    os.path.join(args.data_dir, it['caption_rel'])):
                cap_found = True
            else:
                for cd in CAPTION_DIR_PRIORITY:
                    cand = os.path.join(
                        args.data_dir,
                        it['g1_path'].replace('/motions/', f'/{cd}/').replace('.npz', '.json'))
                    if os.path.exists(cand):
                        cap_found = True
                        break
            ok_cap += int(cap_found)
        print(f'[build] sample verify (n={len(samp)}): '
              f'g1_exist={ok_g1}/{len(samp)} caption_resolvable={ok_cap}/{len(samp)}',
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'[build] wrote {len(items)} items '
          f'({n_with_cap} w/ annotation caption) -> {args.out} '
          f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()

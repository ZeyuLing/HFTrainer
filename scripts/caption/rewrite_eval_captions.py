#!/usr/bin/env python3
"""Rewrite all eval caption fields via the Qwen rewriter service.

For every datalist under ``data/eval/m2m_v2/`` that contains caption-like
fields (``caption``, ``caption_en``, ``text_caption``, ``text``, ``text_a``,
``text_b``), this script:

1. Calls the rewriter once per *unique* text (disk cache in
   ``data/eval/m2m_v2/_rewriter_cache.json`` keyed by raw text).
2. Writes ``{orig}_rewritten.json`` next to each input datalist with each
   caption field replaced by its rewritten version. The original is kept
   under ``{field}_original`` on the same item for traceability.

Usage
-----
    python3 scripts/rewrite_eval_captions.py                 # all datalists
    python3 scripts/rewrite_eval_captions.py --force         # ignore cache

Notes
-----
- Rewriter is the Qwen3-30B-A3B-GRPO endpoint at
  ``http://11.216.46.236:8080/v1`` (see
  ``motion_annot_web/completion_apps/rewriter_client.py``).
- On rewriter error the script falls back to the original text (logged).
- Concurrency via a small thread pool — the service handles parallel calls.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the existing rewriter client (avoids duplicating the prompt template)
sys.path.insert(0, str(PROJECT_ROOT / 'motion_annot_web' / 'completion_apps'))
from rewriter_client import RewriterClient  # noqa: E402

DATALIST_DIR = PROJECT_ROOT / 'data' / 'eval' / 'm2m_v2'
CACHE_PATH = DATALIST_DIR / '_rewriter_cache.json'

# Fields treated as "caption-like" — will all be rewritten in place.
CAPTION_FIELDS = ('caption', 'caption_en', 'text_caption', 'text',
                  'text_a', 'text_b')


def _load_cache(force: bool) -> Dict[str, str]:
    if force or not CACHE_PATH.is_file():
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, str]):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(CACHE_PATH)


def _gather_unique_texts(datalists: List[Path]) -> List[str]:
    unique: set = set()
    for p in datalists:
        items = json.load(open(p)).get('data_list', [])
        for it in items:
            for k in CAPTION_FIELDS:
                v = it.get(k, '')
                if isinstance(v, str) and v.strip():
                    unique.add(v.strip())
    return sorted(unique)


def _rewrite_batch(
    texts: List[str],
    cache: Dict[str, str],
    client: RewriterClient,
    workers: int = 8,
    checkpoint_every: int = 50,
) -> Dict[str, str]:
    """Rewrite all ``texts`` with concurrency; update cache in place.

    Returns the cache (also mutated in place).
    """
    todo = [t for t in texts if t not in cache]
    if not todo:
        print(f'  all {len(texts)} already cached')
        return cache

    print(f'  rewriting {len(todo)} new / {len(texts)} total '
          f'(workers={workers})')

    def _one(text: str) -> Tuple[str, str]:
        try:
            caption, _ = client.rewrite_one(text)
            caption = (caption or '').strip() or text
            return text, caption
        except Exception as e:
            print(f'    [warn] {text[:40]!r}: {e}')
            return text, text  # fallback to original

    done = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, t) for t in todo]
        for fut in concurrent.futures.as_completed(futures):
            orig, rewritten = fut.result()
            cache[orig] = rewritten
            done += 1
            if done % checkpoint_every == 0 or done == len(todo):
                rate = done / max(1e-3, time.time() - t0)
                eta = (len(todo) - done) / max(1e-3, rate)
                print(f'    [{done}/{len(todo)}] {rate:.1f}/s ETA {eta:.0f}s')
                _save_cache(cache)
    _save_cache(cache)
    return cache


def _emit_rewritten_datalist(path: Path, cache: Dict[str, str]) -> Tuple[int, int]:
    """Write ``{path_stem}_rewritten.json`` beside ``path``. Returns (n_items, n_changed)."""
    data = json.load(open(path))
    items = data.get('data_list', [])
    n_changed = 0
    for it in items:
        for k in CAPTION_FIELDS:
            v = it.get(k, '')
            if not isinstance(v, str) or not v.strip():
                continue
            orig = v.strip()
            rewritten = cache.get(orig, orig)
            if rewritten != orig:
                it[f'{k}_original'] = orig
                it[k] = rewritten
                n_changed += 1
    # Update meta
    meta = data.setdefault('meta', {})
    meta['rewritten_with'] = 'Qwen3-30B-A3B-GRPO'
    meta['rewritten_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    out_path = path.with_name(path.stem + '_rewritten.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(items), n_changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true',
                   help='Ignore on-disk cache and re-rewrite everything.')
    p.add_argument('--workers', type=int, default=8,
                   help='Concurrent rewriter calls.')
    p.add_argument('--pattern', default='eval_e*.json',
                   help='Glob pattern for input datalists (relative to DATALIST_DIR).')
    args = p.parse_args()

    datalists = sorted(
        f for f in DATALIST_DIR.glob(args.pattern)
        if not f.name.endswith('_rewritten.json')
        and not f.name.startswith('_')
    )
    if not datalists:
        print(f'No datalists matched {args.pattern}')
        return
    print(f'Found {len(datalists)} datalists')

    cache = _load_cache(args.force)
    print(f'Cache starts with {len(cache)} entries')

    unique = _gather_unique_texts(datalists)
    print(f'Unique caption-like strings across datalists: {len(unique)}')

    client = RewriterClient()
    cache = _rewrite_batch(unique, cache, client, workers=args.workers)

    print()
    print('Emitting rewritten datalists:')
    total_items = total_changed = 0
    for p_ in datalists:
        n_items, n_changed = _emit_rewritten_datalist(p_, cache)
        total_items += n_items
        total_changed += n_changed
        rel = p_.relative_to(DATALIST_DIR)
        out_rel = rel.with_name(p_.stem + '_rewritten.json')
        print(f'  {rel} -> {out_rel}  (items={n_items} changed={n_changed})')
    print()
    print(f'Done. {total_changed} caption-field values updated '
          f'across {total_items} items in {len(datalists)} datalists.')


if __name__ == '__main__':
    main()

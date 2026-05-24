#!/usr/bin/env python3
"""Merge caption embedding cache shards into one atomic cache file."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch


def _load_cache(path: Path) -> tuple[dict, dict]:
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('cache'), dict):
        raise ValueError(f'Invalid cache payload: {path}')
    meta = payload.get('meta', {})
    if not isinstance(meta, dict):
        meta = {}
    return payload['cache'], meta


def _atomic_save(cache: dict, meta: dict, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + '.tmp')
    torch.save({'cache': cache, 'meta': meta}, str(tmp))
    os.replace(str(tmp), str(out_file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True,
                        help='Shard cache files to merge.')
    parser.add_argument('--base', default=None,
                        help='Optional existing cache to include first.')
    parser.add_argument('--out', required=True,
                        help='Output merged cache path.')
    args = parser.parse_args()

    merged = {}
    sources = []
    llm_type = None
    max_length_llm = None

    if args.base and Path(args.base).is_file():
        cache, meta = _load_cache(Path(args.base))
        merged.update(cache)
        sources.append(str(args.base))
        llm_type = meta.get('llm_type', llm_type)
        max_length_llm = meta.get('max_length_llm', max_length_llm)
        print(f'base {args.base}: {len(cache)} entries')

    for raw_path in args.input:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        cache, meta = _load_cache(path)
        merged.update(cache)
        sources.append(str(path))
        llm_type = meta.get('llm_type', llm_type)
        max_length_llm = meta.get('max_length_llm', max_length_llm)
        print(f'shard {path}: {len(cache)} entries')

    meta = {
        'model': f'HYTextModel ({llm_type or "qwen3"} + clipl)',
        'llm_type': llm_type or 'qwen3',
        'max_length_llm': max_length_llm or 512,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_entries': len(merged),
        'sources': sources,
    }
    _atomic_save(merged, meta, Path(args.out))
    print(f'merged {len(merged)} entries -> {args.out}')


if __name__ == '__main__':
    main()

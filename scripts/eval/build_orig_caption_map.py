#!/usr/bin/env python3
"""Pre-extract ORIGINAL captions (hierarchical pool[0]) for a motionhub-style
annotation into a flat {name: caption} JSON, reading caption files in parallel
to dodge CephFS small-file latency.  Reused by TM2T / LoM generation so they do
not re-read thousands of caption JSONs on every launch.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_caption_from_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    pool = []
    if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
        for group in ("macro", "meso", "micro"):
            for item in data[group]:
                if isinstance(item, str) and item.strip():
                    pool.append(item.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption", "short caption"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    pool.append(val.strip())
                    break
    return pool[0] if pool else None


def _iter_entries(raw):
    data = raw["data_list"] if isinstance(raw, dict) and "data_list" in raw else raw
    if isinstance(data, dict):
        for name, entry in data.items():
            yield str(name), entry
    else:
        for i, entry in enumerate(data):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anno-file", required=True)
    p.add_argument("--data-dir", default="data/motionhub")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=64)
    args = p.parse_args()

    raw = json.loads(Path(args.anno_file).read_text())
    jobs = []
    for name, entry in _iter_entries(raw):
        c_rel = entry.get("hierarchical_caption_path")
        if c_rel:
            jobs.append((name, Path(args.data_dir) / c_rel))
    print(f"entries with caption path: {len(jobs)}", flush=True)

    out = {}

    def work(item):
        name, path = item
        return name, _load_caption_from_json(path)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, cap) in enumerate(ex.map(work, jobs)):
            if cap:
                out[name] = cap
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)} (ok={len(out)})", flush=True)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {len(out)} captions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

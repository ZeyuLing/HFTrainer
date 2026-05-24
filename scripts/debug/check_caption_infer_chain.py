#!/usr/bin/env python3
"""Quick caption-inference sanity checks for M2M eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _items(obj):
    if isinstance(obj, dict):
        data = obj.get("data_list", obj.get("data", obj.get("items", [])))
    else:
        data = obj
    return data.values() if isinstance(data, dict) else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval/m2m_v2")
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "eval_e1_rewritten.json",
            "eval_e2_rewritten.json",
            "eval_e3_rewritten.json",
            "eval_e4_rewritten.json",
            "eval_e13_rewritten.json",
        ],
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    cache_path = eval_dir / "caption_embeddings" / "cache.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = payload.get("cache", {})
    print("CACHE_PATH", cache_path)
    print("CACHE_META", payload.get("meta", {}))
    print("CACHE_N", len(cache))

    for name in args.files:
        path = eval_dir / name
        if not path.exists():
            print(name, "MISSING")
            continue
        obj = json.load(open(path))
        records = list(_items(obj))
        caps = []
        for item in records:
            if not isinstance(item, dict):
                continue
            cap = (
                item.get("caption")
                or item.get("caption_en")
                or item.get("text_caption")
                or ""
            ).strip()
            if cap:
                caps.append(cap)
        miss = [cap for cap in caps if cap not in cache]
        print(
            name,
            "items",
            len(records),
            "caps",
            len(caps),
            "uniq",
            len(set(caps)),
            "miss",
            len(miss),
            "miss_uniq",
            len(set(miss)),
        )
        if caps:
            print("  sample_caption:", caps[0][:160])
        if miss:
            print("  sample_miss:", miss[0][:160])


if __name__ == "__main__":
    main()

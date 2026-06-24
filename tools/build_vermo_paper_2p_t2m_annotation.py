#!/usr/bin/env python3
"""Build the deterministic VerMo paper annotation for MotionHub 2P T2M.

The InterGen/InterMask native evaluator scripts score the first ``N`` records in
``test_motionhub_2p.json`` after dropping entries without a usable union
caption.  This tool mirrors that key selection and tags every kept record with
``overfit_task=t2m`` so VerMo's multitask dataset cannot randomly choose another
caption-conditioned task.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional


def _first_hierarchical_caption(data: Dict[str, Any], data_root: str) -> Optional[str]:
    rel_path = data.get("hierarchical_caption_path")
    if not rel_path:
        return None
    path = os.path.join(data_root, rel_path)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    texts = []
    for level in ("macro", "meso", "micro"):
        value = captions.get(level, [])
        if isinstance(value, list):
            texts.extend(text for text in value if isinstance(text, str) and text.strip())
        elif isinstance(value, str) and value.strip():
            texts.append(value.strip())
    if not texts:
        for key in ("action", "category", "description"):
            value = captions.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
                break
    return texts[0] if texts else None


def build(args: argparse.Namespace) -> None:
    with open(args.input, "r", encoding="utf-8") as f:
        source = json.load(f)

    out_data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    source_items = list(source["data_list"].items())
    if args.limit > 0:
        source_items = source_items[: args.limit]

    for source_key, source_item in source_items:
        if args.require_caption and not _first_hierarchical_caption(source_item, args.data_dir):
            continue
        item = deepcopy(source_item)
        item["overfit_task"] = "t2m"
        item["overfit_source_key"] = source_key
        item["overfit_source_annotation"] = args.input
        out_data[source_key] = item

    output = {
        "meta_info": {
            **source.get("meta_info", {}),
            "dataset": "VerMo paper MotionHub 2P T2M deterministic subset",
            "source_annotation": args.input,
            "source_limit": args.limit,
            "require_caption": args.require_caption,
            "num_cases": len(out_data),
            "task": "t2m",
            "caption_policy": "first hierarchical caption, matching InterGen/InterMask eval",
        },
        "data_list": out_data,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output + ".tmp", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(args.output + ".tmp", args.output)
    print(json.dumps({
        "input": args.input,
        "output": args.output,
        "source_records": len(source["data_list"]),
        "kept_records": len(out_data),
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/annotation/test_motionhub_2p.json")
    parser.add_argument(
        "--output",
        default="data/annotation/vermo_paper_test_t2m_2p_first384_20260614.json",
    )
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--require-caption", action="store_true", default=True)
    parser.add_argument("--allow-missing-caption", dest="require_caption", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

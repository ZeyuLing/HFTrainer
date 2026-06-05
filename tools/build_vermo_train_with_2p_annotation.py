#!/usr/bin/env python3
"""Merge VerMo single-person/audio training data with 2-person training data."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict
from copy import deepcopy
from typing import Any, Dict


def _load_annotation(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "data_list" not in data or "meta_info" not in data:
        raise ValueError(f"Invalid annotation format: {path}")
    return data


def build(args: argparse.Namespace) -> None:
    base = _load_annotation(args.base)
    multi = _load_annotation(args.multi)

    data_list: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    source_counts = Counter()
    multi_counts = Counter()

    for key, item in base["data_list"].items():
        out_key = key
        if out_key in data_list:
            raise KeyError(f"Duplicate key in base annotation: {out_key}")
        data_list[out_key] = item
        source_counts["base"] += 1
        multi_counts["single"] += int(not isinstance(item.get("smplx_path"), list))
        multi_counts["multi"] += int(isinstance(item.get("smplx_path"), list))

    for key, item in multi["data_list"].items():
        item = deepcopy(item)
        item.setdefault("vermo_source_annotation", args.multi)
        out_key = f"motionclip2p_{key}"
        if out_key in data_list:
            raise KeyError(f"Duplicate merged key: {out_key}")
        data_list[out_key] = item
        source_counts["motionclip2p"] += 1
        multi_counts["single"] += int(not isinstance(item.get("smplx_path"), list))
        multi_counts["multi"] += int(isinstance(item.get("smplx_path"), list))

    output = {
        "meta_info": {
            "dataset": "train_hq_motionhub_hymotion plus MotionCLIP 2P train",
            "base_annotation": args.base,
            "multi_annotation": args.multi,
            "num_cases": len(data_list),
            "source_counts": dict(source_counts),
            "person_kind_counts": dict(multi_counts),
            "base_meta_info": base.get("meta_info", {}),
            "multi_meta_info": multi.get("meta_info", {}),
        },
        "data_list": data_list,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"Wrote {len(data_list)} cases to {args.output}")
    print("Source counts:", dict(source_counts))
    print("Person-kind counts:", dict(multi_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="data/annotation/train_hq_motionhub_hymotion.json",
    )
    parser.add_argument(
        "--multi",
        default="data/annotation/train_motionclip_2p.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/annotation/"
            "train_hq_motionhub_hymotion_motionclip2p_20260604.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

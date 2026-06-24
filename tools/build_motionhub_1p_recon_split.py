#!/usr/bin/env python3
"""Build a valid one-person MotionHub reconstruction split.

The top-level ``test_motionhub_1p.json`` in this workspace still points true
multi-person subsets at legacy ``smplx_55`` paths. Current assets live under
``smplx_55_1p`` in each subset's own ``test_1p.json``. This script keeps the
top-level entries whose motion files exist and replaces true-multi subsets with
their current subset-level one-person test records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


TRUE_MULTI_SUBSETS = {"interx", "interhuman", "hi4d", "chi3d"}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def path_exists(data_dir: Path, path: Any) -> bool:
    return isinstance(path, str) and (data_dir / path).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/motionhub/test_motionhub_1p.json")
    parser.add_argument("--output", default="data/annotation/vermo_recon_motionhub_1p_test_20260606.json")
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--replace-subsets", nargs="*", default=sorted(TRUE_MULTI_SUBSETS))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    replace_subsets = set(args.replace_subsets)
    src = load_json(args.input)
    out_data: dict[str, dict[str, Any]] = {}
    subset_counts: Counter = Counter()
    skipped: Counter = Counter()

    for key, record in src["data_list"].items():
        subset = str(record.get("subset", ""))
        if subset in replace_subsets:
            skipped["legacy_replaced_subset_entry"] += 1
            continue
        if not path_exists(data_dir, record.get("smplx_path")):
            skipped["missing_top_level_file"] += 1
            continue
        item = dict(record)
        item["num_person"] = 1
        out_data[key] = item
        subset_counts[subset] += 1

    for subset in sorted(replace_subsets):
        split_file = data_dir / subset / "test_1p.json"
        if not split_file.exists():
            skipped[f"missing_{subset}_test_1p"] += 1
            continue
        sub_src = load_json(split_file)
        for key, record in sub_src.get("data_list", {}).items():
            if not path_exists(data_dir, record.get("smplx_path")):
                skipped["missing_subset_file"] += 1
                continue
            item = dict(record)
            item["num_person"] = 1
            out_data[key] = item
            subset_counts[str(item.get("subset", subset))] += 1

    payload = {
        "meta_info": {
            "dataset": "motionhub",
            "version": "vermo_recon_1p_valid_paths",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_annotation": args.input,
            "replace_subsets": sorted(replace_subsets),
            "subset_counts": dict(subset_counts),
            "skipped": dict(skipped),
        },
        "data_list": out_data,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["meta_info"], indent=2, ensure_ascii=False))
    print(f"[build-motionhub-1p-recon] wrote {args.output} with {len(out_data)} samples")


if __name__ == "__main__":
    main()

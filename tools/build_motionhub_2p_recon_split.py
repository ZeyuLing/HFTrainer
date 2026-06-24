#!/usr/bin/env python3
"""Build a true two-person MotionHub reconstruction split.

Current MotionHub stores real interaction test records in each subset's
``test_2p.json`` and points them at ``smplx_55_2p``. The legacy top-level
``test_motionhub_1p.json`` stores outdated ``smplx_55`` paths for these subsets,
so this script uses subset ``test_2p.json`` files by default and keeps a legacy
P1/P2 regrouping mode only for audits.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
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


def person_id_from_path(path: str) -> str | None:
    base = os.path.basename(path).lower()
    if base == "p1.npz":
        return "P1"
    if base == "p2.npz":
        return "P2"
    return None


def group_key(path: str) -> str:
    return os.path.dirname(path)


def build_from_subset_test_2p(
    data_dir: Path,
    subset_allow: set[str],
) -> tuple[dict[str, dict[str, Any]], Counter, Counter]:
    out_data: dict[str, dict[str, Any]] = {}
    subset_counts: Counter = Counter()
    skipped: Counter = Counter()

    for subset in sorted(subset_allow):
        split_file = data_dir / subset / "test_2p.json"
        if not split_file.exists():
            skipped[f"missing_{subset}_test_2p"] += 1
            continue
        src = load_json(split_file)
        for key, record in src.get("data_list", {}).items():
            paths = record.get("smplx_path")
            if not isinstance(paths, list) or len(paths) != 2:
                skipped["not_two_person_path"] += 1
                continue
            if not all((data_dir / path).exists() for path in paths):
                skipped["missing_file"] += 1
                continue
            item = dict(record)
            item["num_person"] = 2
            out_data[key] = item
            subset_counts[str(item.get("subset", subset))] += 1
    return out_data, subset_counts, skipped


def build_from_legacy_1p_pairs(
    input_file: Path,
    data_dir: Path,
    subset_allow: set[str],
) -> tuple[dict[str, dict[str, Any]], Counter, Counter, int]:
    src = load_json(input_file)
    groups: dict[str, dict[str, tuple[str, dict[str, Any]]]] = defaultdict(dict)
    skipped: Counter = Counter()

    for key, record in src["data_list"].items():
        subset = record.get("subset")
        if subset not in subset_allow:
            skipped["subset"] += 1
            continue
        path = record.get("smplx_path")
        if not isinstance(path, str):
            skipped["non_str_path"] += 1
            continue
        person = person_id_from_path(path)
        if person is None:
            skipped["non_p1p2"] += 1
            continue
        groups[group_key(path)][person] = (key, record)

    out_data: dict[str, dict[str, Any]] = {}
    subset_counts: Counter = Counter()
    incomplete = []
    for gkey in sorted(groups):
        persons = groups[gkey]
        if "P1" not in persons or "P2" not in persons:
            incomplete.append(gkey)
            continue
        key1, rec1 = persons["P1"]
        key2, rec2 = persons["P2"]
        if rec1.get("num_frames") != rec2.get("num_frames") or rec1.get("fps") != rec2.get("fps"):
            skipped["length_or_fps_mismatch"] += 1
            continue
        paths = [rec1["smplx_path"], rec2["smplx_path"]]
        if not all((data_dir / path).exists() for path in paths):
            skipped["missing_file"] += 1
            continue

        subset = rec1.get("subset")
        out_key = f"{subset}_{gkey.replace('/', '_').replace(' ', '_')}_2p"
        record = dict(rec1)
        record["smplx_path"] = paths
        record["hierarchical_caption_path"] = None
        record["sep_hierarchical_caption_path"] = [
            rec1.get("hierarchical_caption_path"),
            rec2.get("hierarchical_caption_path"),
        ]
        record["person_caption_paths"] = record["sep_hierarchical_caption_path"]
        record["source_1p_keys"] = [key1, key2]
        record["num_person"] = 2
        out_data[out_key] = record
        subset_counts[str(subset)] += 1
    return out_data, subset_counts, skipped, len(incomplete)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/motionhub/test_motionhub_1p.json")
    parser.add_argument("--output", default="data/annotation/vermo_recon_motionhub_2p_test_20260606.json")
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--source", choices=["subset_test_2p", "legacy_1p_pairs"], default="subset_test_2p")
    parser.add_argument("--subsets", nargs="*", default=sorted(TRUE_MULTI_SUBSETS))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    subset_allow = set(args.subsets)
    incomplete = 0
    if args.source == "subset_test_2p":
        out_data, subset_counts, skipped = build_from_subset_test_2p(data_dir, subset_allow)
    else:
        out_data, subset_counts, skipped, incomplete = build_from_legacy_1p_pairs(
            Path(args.input), data_dir, subset_allow
        )

    payload = {
        "meta_info": {
            "dataset": "motionhub",
            "version": "vermo_recon_2p_from_test_motionhub_1p",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_annotation": args.input,
            "source": args.source,
            "subsets": sorted(subset_allow),
            "subset_counts": dict(subset_counts),
            "skipped": dict(skipped),
            "incomplete_groups": incomplete,
        },
        "data_list": out_data,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["meta_info"], indent=2, ensure_ascii=False))
    print(f"[build-motionhub-2p-recon] wrote {args.output} with {len(out_data)} samples")


if __name__ == "__main__":
    main()

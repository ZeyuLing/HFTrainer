#!/usr/bin/env python3
"""Shard flat MotionHub modality directories and update root split paths.

Some source datasets place more than 10k files directly under a modality
directory, which is fragile for public hosting.  This script moves direct files
under:

    {group_dir}/{modality}/{shard}/{filename}

and updates matching path fields in train/test annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def row_container(obj: Any) -> Any:
    return obj.get("data_list", obj) if isinstance(obj, dict) else obj


def iter_rows(obj: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    data = row_container(obj)
    if isinstance(data, dict):
        for key, row in data.items():
            if isinstance(row, dict):
                yield str(key), row
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                yield str(row.get("id", idx)), row


def shard_for_file(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem[:6] if len(stem) > 6 else stem


def move_direct_files(
    subset_root: Path,
    group_pattern: str,
    modalities: list[str],
    write: bool,
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    moved_summary: dict[str, dict[str, int]] = {}
    path_map: dict[str, str] = {}
    groups = sorted(path for path in subset_root.glob(group_pattern) if path.is_dir())
    for group in groups:
        for modality in modalities:
            modality_dir = group / modality
            if not modality_dir.exists():
                continue
            with os.scandir(modality_dir) as scan:
                direct_files = sorted(
                    Path(entry.path) for entry in scan if entry.is_file(follow_symlinks=False)
                )
            shard_counts: dict[str, int] = defaultdict(int)
            moved = 0
            for src in direct_files:
                shard = shard_for_file(src)
                dst = modality_dir / shard / src.name
                old_rel = Path(subset_root.name, src.relative_to(subset_root)).as_posix()
                new_rel = Path(subset_root.name, dst.relative_to(subset_root)).as_posix()
                path_map[old_rel] = new_rel
                shard_counts[shard] += 1
                if write:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        src.unlink()
                    else:
                        shutil.move(str(src), str(dst))
                moved += 1
            key = Path(group.name, modality).as_posix()
            moved_summary[key] = {
                "moved_direct_files": moved,
                "num_new_shards": len(shard_counts),
                "max_files_in_new_shard": max(shard_counts.values()) if shard_counts else 0,
            }
    return moved_summary, path_map


def update_splits(
    subset_root: Path,
    path_map: dict[str, str],
    path_keys: list[str],
    splits: list[str],
    write: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in splits:
        path = subset_root / f"{split}.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = 0
        updates = defaultdict(int)
        for _, row in iter_rows(obj):
            rows += 1
            for key in path_keys:
                value = row.get(key)
                if isinstance(value, str) and value in path_map:
                    row[key] = path_map[value]
                    updates[key] += 1
        if write and updates:
            path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result[split] = {
            "rows": rows,
            "updated_paths": dict(updates),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--group-pattern", default="*")
    parser.add_argument("--modalities", required=True, help="Comma-separated modality directory names.")
    parser.add_argument(
        "--path-keys",
        default="smplx_path,smplh_path,audio_path,speech_script_path,hierarchical_caption_path",
    )
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--report", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    path_keys = [item.strip() for item in args.path_keys.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    moved_summary, path_map = move_direct_files(
        subset_root=subset_root,
        group_pattern=args.group_pattern,
        modalities=modalities,
        write=args.write,
    )
    split_updates = update_splits(
        subset_root=subset_root,
        path_map=path_map,
        path_keys=path_keys,
        splits=splits,
        write=args.write,
    )
    report = {
        "subset_root": str(subset_root),
        "group_pattern": args.group_pattern,
        "modalities": modalities,
        "path_keys": path_keys,
        "write": bool(args.write),
        "moved_summary": moved_summary,
        "num_path_map_entries": len(path_map),
        "split_updates": split_updates,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

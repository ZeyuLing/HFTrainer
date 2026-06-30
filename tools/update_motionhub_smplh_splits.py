#!/usr/bin/env python3
"""Point MotionHub root split annotations at converted SMPL-H files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def row_container(obj: Any) -> Any:
    data = obj.get("data_list", obj) if isinstance(obj, dict) else obj
    return data


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


def smplh_relpath(subset_root: Path, rel: str, output_motion_dir: str) -> str:
    parts = list(Path(str(rel)).parts)
    if parts and parts[0] == subset_root.name:
        parts = parts[1:]
    replaced = False
    for idx, part in enumerate(parts):
        if part in {"smplx_55", "smplh_52"}:
            parts[idx] = output_motion_dir
            replaced = True
            break
    if not replaced:
        raise ValueError(f"cannot find motion dir in {rel}")
    return Path(subset_root.name, *parts).as_posix()


def resolve_from_data_root(data_root: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    return data_root / path


def process_split(
    data_root: Path,
    subset_root: Path,
    split: str,
    output_motion_dir: str,
    write: bool,
    drop_missing: bool,
) -> Dict[str, Any]:
    path = subset_root / f"{split}.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    data = row_container(obj)
    rows = 0
    changed = 0
    missing = []
    drop_keys = []
    for key, row in iter_rows(obj):
        rows += 1
        old = row.get("smplx_path") or row.get("motion_path")
        if not old:
            continue
        new = smplh_relpath(subset_root, str(old), output_motion_dir)
        if not resolve_from_data_root(data_root, new).exists():
            missing.append(new)
            if drop_missing:
                drop_keys.append(key)
            continue
        if row.get("smplx_path") != new:
            changed += 1
        row["smplx_path"] = new
        row["smplh_path"] = new
        row["smpl_type"] = "smplh"
        row["motion_representation"] = output_motion_dir
    if drop_missing and drop_keys:
        if isinstance(data, dict):
            for key in drop_keys:
                data.pop(key, None)
        elif isinstance(data, list):
            drop_set = {int(key) for key in drop_keys if str(key).isdigit()}
            data[:] = [row for idx, row in enumerate(data) if idx not in drop_set]
    if write:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "split": split,
        "path": str(path),
        "rows": rows,
        "changed": changed,
        "missing": missing[:20],
        "missing_count": len(missing),
        "dropped_missing_count": len(drop_keys) if drop_missing else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/motionhub")
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--output-motion-dir", default="smplh_52")
    parser.add_argument("--report", required=True)
    parser.add_argument("--drop-missing", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    subset_root = Path(args.subset_root)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    results = [
        process_split(
            data_root,
            subset_root,
            split,
            args.output_motion_dir,
            args.write,
            args.drop_missing,
        )
        for split in splits
    ]
    report = {
        "data_root": str(data_root),
        "subset_root": str(subset_root),
        "output_motion_dir": args.output_motion_dir,
        "write": bool(args.write),
        "drop_missing": bool(args.drop_missing),
        "splits": results,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

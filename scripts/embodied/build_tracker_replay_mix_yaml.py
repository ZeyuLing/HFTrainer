#!/usr/bin/env python3
"""Build a weighted ProtoMotions motion yaml for tracker replay experiments.

ProtoMotions ``MotionLib`` accepts a yaml file in the form:

    motions:
      - file: relative/path.motion
        weight: 1.0

This utility creates that file from one or more source directories/yamls/files.
It is intentionally small: the goal is to make replay-mix tracker experiments
reproducible without changing the RL training code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import yaml


def _iter_yaml_motion_files(path: Path) -> Iterable[tuple[Path, float]]:
    data = yaml.safe_load(path.read_text()) or {}
    for item in data.get("motions", []):
        rel = Path(item["file"])
        src = (path.parent / rel).resolve()
        yield src, float(item.get("weight", 1.0))


def _iter_motion_files(path: Path) -> Iterable[tuple[Path, float]]:
    if path.is_dir():
        for item in sorted(path.rglob("*.motion")):
            yield item.resolve(), 1.0
    elif path.suffix == ".yaml":
        yield from _iter_yaml_motion_files(path)
    elif path.suffix == ".motion":
        yield path.resolve(), 1.0
    else:
        raise ValueError(f"Unsupported source: {path}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_to_yaml(path: Path, out_path: Path) -> str:
    try:
        return os.path.relpath(path, out_path.parent)
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("PATH", "WEIGHT"),
        required=True,
        help="Motion source directory/yaml/file and per-motion sampling weight.",
    )
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Deterministically shuffle each source before applying --limit-per-source.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Compute md5 hashes and skip duplicate motion files. Off by default because large replay sets live on network storage.",
    )
    args = parser.parse_args()

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    seen: set[str] = set()
    for source_value, weight_value in args.source:
        source = Path(source_value).resolve()
        weight = float(weight_value)
        candidates = list(_iter_motion_files(source))
        if args.shuffle_seed is not None:
            random.Random(args.shuffle_seed + len(entries)).shuffle(candidates)
        count = 0
        for motion_path, source_weight in candidates:
            if not motion_path.is_file():
                continue
            digest = _md5(motion_path) if args.dedupe else None
            if args.dedupe:
                if digest in seen:
                    continue
                seen.add(str(digest))
            final_weight = weight * source_weight
            entries.append({"file": _rel_to_yaml(motion_path, out_path), "weight": final_weight})
            manifest.append(
                {
                    "source": str(source),
                    "motion_path": str(motion_path),
                    "md5": digest,
                    "weight": final_weight,
                }
            )
            count += 1
            if args.limit_per_source > 0 and count >= args.limit_per_source:
                break

    if not entries:
        raise SystemExit("No motion entries found.")

    out_path.write_text(yaml.safe_dump({"motions": entries}, sort_keys=False))
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"motions": manifest}, indent=2))
    print(f"[replay-mix] wrote {out_path} entries={len(entries)}")
    print(f"[replay-mix] wrote {manifest_path}")


if __name__ == "__main__":
    main()

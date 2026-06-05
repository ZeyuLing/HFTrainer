#!/usr/bin/env python3
"""Build an AMASS full benchmark yaml by concatenating train/val/test splits."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_SPLITS = ("train", "validation", "test")


def load_motions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    motions = data.get("motions")
    if not isinstance(motions, list):
        raise ValueError(f"{path} does not contain a motions list")
    return motions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=Path("ref_repo/ProtoMotions/data/yaml_files"),
        help="Directory containing amass_smpl_train/validation/test.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ref_repo/ProtoMotions/data/yaml_files/amass_smpl_full.yaml"),
        help="Output yaml path.",
    )
    args = parser.parse_args()

    full: list[dict] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    counts: dict[str, int] = {}

    for split in DEFAULT_SPLITS:
        path = args.yaml_dir / f"amass_smpl_{split}.yaml"
        motions = load_motions(path)
        counts[split] = len(motions)
        for motion in motions:
            motion_copy = deepcopy(motion)
            file_name = str(motion_copy.get("file", ""))
            if file_name in seen:
                duplicates.append(file_name)
            seen.add(file_name)
            motion_copy["idx"] = len(full)
            full.append(motion_copy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"motions": full}, f, sort_keys=False)

    print(f"wrote {args.output}")
    print("counts " + " ".join(f"{split}={counts[split]}" for split in DEFAULT_SPLITS))
    print(f"total={len(full)} unique_files={len(seen)} duplicates={len(duplicates)}")
    if duplicates:
        print("duplicate_examples=" + ",".join(duplicates[:10]))


if __name__ == "__main__":
    main()

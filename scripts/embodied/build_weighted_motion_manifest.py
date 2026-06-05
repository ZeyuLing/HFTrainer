#!/usr/bin/env python3
"""Build a ProtoMotions YAML motion manifest with group-level sampling weights.

ProtoMotions MotionLib supports YAML entries with a per-motion ``weight``.  This
helper lets us keep replay/native data at a fixed sampling mass while adding
adversarial or jump-specific motions without silently overwhelming the original
tracker distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def stable_path(path: Path) -> Path:
    """Return an absolute path without resolving symlinked mount aliases."""
    path = path.expanduser()
    return path if path.is_absolute() else path.absolute()


def parse_group(spec: str) -> tuple[str, Path, float]:
    parts = spec.split("::")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Group must be formatted as tag::/path/to/motions::total_weight"
        )
    tag, path, weight = parts
    tag = tag.strip()
    if not tag:
        raise argparse.ArgumentTypeError("Group tag must be non-empty")
    try:
        total_weight = float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid group weight: {weight}") from exc
    if total_weight <= 0:
        raise argparse.ArgumentTypeError("Group weight must be positive")
    return tag, stable_path(Path(path)), total_weight


def motion_entries(path: Path) -> list[tuple[Path, float]]:
    if path.is_file():
        if path.suffix == ".motion":
            return [(stable_path(path), 1.0)]
        if path.suffix == ".yaml":
            data = yaml.safe_load(path.read_text()) or {}
            entries: list[tuple[Path, float]] = []
            for item in data.get("motions", []):
                motion_path = Path(item["file"])
                if not motion_path.is_absolute():
                    motion_path = path.parent / motion_path
                entries.append((stable_path(motion_path), float(item.get("weight", 1.0))))
            return entries
        raise ValueError(f"Unsupported motion manifest source: {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return [(stable_path(p), 1.0) for p in sorted(path.rglob("*.motion"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        help="Sampling group as tag::motion_dir_or_file::total_weight. Repeat for each group.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    entries = []
    summary = {"groups": [], "num_motions": 0, "total_manifest_weight": 0.0}

    for tag, path, group_weight in args.group:
        files = motion_entries(path)
        if args.max_per_group > 0 and len(files) > args.max_per_group:
            files = sorted(rng.sample(files, args.max_per_group))
        if not files:
            raise ValueError(f"No .motion files found for group {tag}: {path}")
        source_weight_sum = sum(max(source_weight, 0.0) for _, source_weight in files)
        if source_weight_sum <= 0:
            raise ValueError(f"Non-positive source weights for group {tag}: {path}")
        for file, source_weight in files:
            final_weight = group_weight * max(source_weight, 0.0) / source_weight_sum
            entries.append(
                {
                    "file": str(file),
                    "weight": final_weight,
                }
            )
        summary["groups"].append(
            {
                "tag": tag,
                "path": str(path),
                "num_motions": len(files),
                "group_weight": group_weight,
                "source_weight_sum": source_weight_sum,
            }
        )
        summary["num_motions"] += len(files)
        summary["total_manifest_weight"] += group_weight

    payload = {"motions": entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids an extra PyYAML dependency.
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report PhysFlow Table-2 canonical rollout storage paths.

Formal per-case rollouts live under:

    outputs/evaluation/physflow/table2_tracker/<method>/<test_dataset>/<representation>/<case>.npz

The Table-2 runner roots under table2_tracker/unified_protocol_v1/runs are
debug/aggregation roots, not the stable artifact interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("lafan1_g1", "amass_test_g1", "wild_g1_clean")
DEFAULT_METHODS = ("any2track", "humanoid_gpt", "protomotions", "sonic", "beyondmimic")
DEFAULT_REPRESENTATIONS = ("g1_body30", "g1_qpos30")


def _task_root(root: Path) -> Path:
    root = Path(root)
    return root if root.name == "table2_tracker" else root / "table2_tracker"


def _count_npz(path: Path) -> int:
    return sum(1 for p in path.glob("*.npz") if p.is_file()) if path.is_dir() else 0


def _row(root: Path, split: str, representation: str, method: str) -> dict[str, Any]:
    path = _task_root(root) / method / split / representation
    return {
        "split": split,
        "representation": representation,
        "method": method,
        "path": str(path),
        "exists": path.is_dir(),
        "num_cases": _count_npz(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/evaluation/physflow"))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--representations", nargs="+", default=list(DEFAULT_REPRESENTATIONS))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    rows = [
        _row(args.root, split, representation, method)
        for method in args.methods
        for split in args.splits
        for representation in args.representations
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    print("method\tsplit\trepresentation\tnum_cases\texists\tpath")
    for row in rows:
        print(
            f"{row['method']}\t{row['split']}\t{row['representation']}\t"
            f"{row['num_cases']}\t{str(row['exists']).lower()}\t{row['path']}"
        )


if __name__ == "__main__":
    main()

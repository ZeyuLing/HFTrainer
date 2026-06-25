#!/usr/bin/env python3
"""Aggregate unified Table-2 released-baseline shard outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("amass_test_fixed600", "lafan1_fixed600", "wild_clean_fixed600")
METHODS = ("any2track", "humanoid_gpt")


def _load(path: Path) -> Any:
    last_error: json.JSONDecodeError | None = None
    for _ in range(20):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(0.25)
    raise last_error  # type: ignore[misc]


def _expected_shards(protocol_root: Path, split: str) -> list[int]:
    manifest_dir = protocol_root / "manifests" / split
    return sorted(int(p.stem.replace("shard_", "")) for p in manifest_dir.glob("shard_*.json"))


def _nonempty_shards(protocol_root: Path, split: str) -> list[int]:
    out = []
    for shard in _expected_shards(protocol_root, split):
        path = protocol_root / "manifests" / split / f"shard_{shard}.json"
        try:
            data = _load(path)
        except json.JSONDecodeError as exc:
            print(f"[aggregate] warning: skipping invalid manifest {path}: {exc}", file=sys.stderr)
            continue
        if data:
            out.append(shard)
    return out


def _missing_any2track(protocol_root: Path, split: str) -> list[int]:
    out_dir = protocol_root / "runs" / "any2track" / split
    missing = []
    for shard in _nonempty_shards(protocol_root, split):
        path = out_dir / f"eval_shard_{shard}.json"
        if not path.is_file():
            missing.append(shard)
    return missing


def _missing_hgpt(protocol_root: Path, split: str) -> list[int]:
    out_dir = protocol_root / "runs" / "humanoid_gpt" / split
    missing = []
    for shard in _nonempty_shards(protocol_root, split):
        path = out_dir / f"shard_{shard}" / "summary.json"
        if not path.is_file():
            missing.append(shard)
    return missing


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1"))
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma/space separated split names to aggregate.")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    splits = tuple(s for s in args.splits.replace(",", " ").split() if s)

    missing: dict[str, dict[str, list[int]]] = {}
    for split in splits:
        a2t_missing = _missing_any2track(args.protocol_root, split)
        hgpt_missing = _missing_hgpt(args.protocol_root, split)
        if a2t_missing or hgpt_missing:
            missing[split] = {}
            if a2t_missing:
                missing[split]["any2track"] = a2t_missing
            if hgpt_missing:
                missing[split]["humanoid_gpt"] = hgpt_missing

    if missing and not args.allow_missing:
        raise SystemExit(f"Missing non-empty shard outputs: {json.dumps(missing, indent=2)}")

    for split in splits:
        a2t_root = args.protocol_root / "runs" / "any2track" / split
        if a2t_root.is_dir() and not _missing_any2track(args.protocol_root, split):
            _run([sys.executable, "scripts/embodied/aggregate_opentrack_eval.py", "--eval-root", str(a2t_root)])
        hgpt_root = args.protocol_root / "runs" / "humanoid_gpt" / split
        if hgpt_root.is_dir() and not _missing_hgpt(args.protocol_root, split):
            _run([sys.executable, "scripts/embodied/aggregate_hgpt_eval.py", "--eval-root", str(hgpt_root), "--complete-thresh", "0.95"])

    table: dict[str, Any] = {"protocol_root": str(args.protocol_root), "summaries": {}, "missing": missing}
    for method in METHODS:
        table["summaries"][method] = {}
        for split in splits:
            summary_path = args.protocol_root / "runs" / method / split / "summary.json"
            if summary_path.is_file():
                table["summaries"][method][split] = _load(summary_path).get("summary", {})
    out_path = args.protocol_root / "table2_unified_summary.json"
    out_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps(table["summaries"], indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()

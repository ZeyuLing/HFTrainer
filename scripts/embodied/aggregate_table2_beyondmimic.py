#!/usr/bin/env python3
"""Aggregate BeyondMimic per-motion Table-2 specialist outputs."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SPLITS = ("lafan1_fixed600", "amass_test_fixed600", "wild_clean_fixed600")


def _load_json_retry(path: Path) -> Any:
    last_error: json.JSONDecodeError | None = None
    for _ in range(40):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(0.25)
    raise last_error  # type: ignore[misc]


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _expected_shards(protocol_root: Path, split: str) -> list[int]:
    manifest_dir = protocol_root / "manifests" / split
    return sorted(int(p.stem.replace("shard_", "")) for p in manifest_dir.glob("shard_*.json"))


def _nonempty_shards(protocol_root: Path, split: str) -> list[int]:
    out = []
    for shard in _expected_shards(protocol_root, split):
        data = _load_json_retry(protocol_root / "manifests" / split / f"shard_{shard}.json")
        if data:
            out.append(shard)
    return out


def _missing(protocol_root: Path, split: str) -> list[int]:
    out = []
    for shard in _nonempty_shards(protocol_root, split):
        path = protocol_root / "runs" / "beyondmimic" / split / f"shard_{shard}" / "summary.json"
        if not path.is_file():
            out.append(shard)
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "num_motions": float(len(rows)),
        "success_rate": mean("success"),
        "paper_success_rate": mean("paper_success"),
        "strict_success_rate": mean("strict_success"),
        "completion": mean("completion"),
        "root_err_mean": mean("root_err_mean"),
        "root_height_err_mean": mean("root_height_err_mean"),
        "raw_global_mpjpe_mm": mean("raw_global_mpjpe_mm"),
        "mpjpe_mm": mean("mpjpe_mm"),
        "local_mpjpe_mm": mean("local_mpjpe_mm"),
        "mpjve_mps": mean("mpjve_mps"),
        "local_mpjve_mps": mean("local_mpjve_mps"),
        "mpjae_mps2": mean("mpjae_mps2"),
        "local_mpjae_mps2": mean("local_mpjae_mps2"),
        "joint_err_mean": mean("joint_err_mean"),
        "max_joint_err_mean": mean("max_joint_err_mean"),
        "max_joint_err_max": mean("max_joint_err_max"),
        "min_height": mean("min_height"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1"))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    splits = tuple(s for s in args.splits.replace(",", " ").split() if s)

    missing = {split: _missing(args.protocol_root, split) for split in splits}
    missing = {k: v for k, v in missing.items() if v}
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing BeyondMimic shard outputs: {json.dumps(missing, indent=2)}")

    summaries: dict[str, Any] = {}
    motions_by_split: dict[str, Any] = {}
    for split in splits:
        rows: list[dict[str, Any]] = []
        motions: dict[str, Any] = {}
        for shard in _nonempty_shards(args.protocol_root, split):
            path = args.protocol_root / "runs" / "beyondmimic" / split / f"shard_{shard}" / "summary.json"
            if not path.is_file():
                continue
            data = _load_json_retry(path)
            for name, row in data.get("motions", {}).items():
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("shard", shard)
                    row.setdefault("motion", name)
                    rows.append(row)
                    motions[name] = row
        summaries[split] = _summarize(rows)
        motions_by_split[split] = motions

    payload = {
        "protocol_root": str(args.protocol_root),
        "method": "beyondmimic",
        "summaries": summaries,
        "motions": motions_by_split,
        "missing": missing,
    }
    out_path = args.protocol_root / "table2_unified_beyondmimic_summary.json"
    _write_json_atomic(out_path, payload)
    print(json.dumps({"summaries": summaries, "missing": missing}, indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()

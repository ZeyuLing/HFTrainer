#!/usr/bin/env python3
"""Aggregate ProtoMotions outputs from the unified Table-2 protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SPLITS = ("amass_test_fixed600", "lafan1_fixed600", "wild_clean_fixed600")


def _expected_shards(protocol_root: Path, split: str) -> list[int]:
    manifest_dir = protocol_root / "manifests" / split
    return sorted(int(p.stem.replace("shard_", "")) for p in manifest_dir.glob("shard_*.json"))


def _nonempty_shards(protocol_root: Path, split: str) -> list[int]:
    out = []
    for shard in _expected_shards(protocol_root, split):
        data = json.loads((protocol_root / "manifests" / split / f"shard_{shard}.json").read_text())
        if data:
            out.append(shard)
    return out


def _missing_logs(protocol_root: Path, split: str, method: str) -> list[int]:
    eval_dir = protocol_root / "runs" / "protomotions" / split / f"eval_{method}"
    missing = []
    for shard in _nonempty_shards(protocol_root, split):
        log = eval_dir / f"shard_{shard}.log"
        if not log.is_file() or "EVALUATION RESULTS" not in log.read_text(errors="replace"):
            missing.append(shard)
    return missing


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1"))
    ap.add_argument("--method", default="protomotions_g1_bones")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma/space separated split names to aggregate.")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    splits = tuple(s for s in args.splits.replace(",", " ").split() if s)

    missing = {split: _missing_logs(args.protocol_root, split, args.method) for split in splits}
    missing = {k: v for k, v in missing.items() if v}
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing ProtoMotions shard outputs: {json.dumps(missing, indent=2)}")

    summaries = {}
    for split in splits:
        if missing.get(split):
            continue
        eval_root = args.protocol_root / "runs" / "protomotions" / split
        motion_base = args.protocol_root / "proto_motions" / split
        if not eval_root.is_dir():
            continue
        expected_shards = _expected_shards(args.protocol_root, split)
        num_shards = max(expected_shards) + 1 if expected_shards else 0
        shard_template = f"{split}_g1_shard_{{shard}}.pt"
        _run([
            sys.executable,
            "scripts/embodied/aggregate_proto_eval_logs.py",
            "--eval-root",
            str(eval_root),
            "--motion-base",
            str(motion_base),
            "--num-shards",
            str(num_shards),
            "--shard-file-template",
            shard_template,
        ])
        _run([
            sys.executable,
            "scripts/embodied/aggregate_proto_predicted_motion_metrics.py",
            "--eval-root",
            str(eval_root),
            "--motion-base",
            str(motion_base),
            "--num-shards",
            str(num_shards),
            "--shard-file-template",
            shard_template,
            "--methods",
            args.method,
        ])
        summary_path = eval_root / "summary.json"
        pred_path = eval_root / "predicted_metrics.json"
        summaries[split] = {
            "full_eval": json.loads(summary_path.read_text()).get("results", {}).get(args.method, {}) if summary_path.is_file() else {},
            "predicted": json.loads(pred_path.read_text()).get("results", {}).get(args.method, {}).get("summary", {}) if pred_path.is_file() else {},
        }

    out = {"protocol_root": str(args.protocol_root), "method": args.method, "missing": missing, "summaries": summaries}
    out_path = args.protocol_root / "table2_unified_proto_summary.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()

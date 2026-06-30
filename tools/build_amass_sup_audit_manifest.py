#!/usr/bin/env python3
"""Build an AMASS_SUP per-source visual audit manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_split(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    data = obj.get("data_list", obj) if isinstance(obj, dict) else obj
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a data_list dict")
    return data


def source_subset(row: Dict[str, Any]) -> str:
    rel = row.get("smplx_path") or row.get("smplh_path")
    if not rel:
        return "unknown"
    parts = Path(str(rel)).parts
    return parts[2] if len(parts) > 2 else "unknown"


def subject_group(row: Dict[str, Any]) -> str:
    rel = row.get("smplx_path") or row.get("smplh_path")
    parts = Path(str(rel)).parts
    return parts[3] if len(parts) > 3 else "unknown"


def read_caption(data_root: Path, row: Dict[str, Any]) -> str:
    rel = row.get("hierarchical_caption_path")
    if not rel:
        return ""
    path = data_root / str(rel)
    if not path.exists():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    lines = []
    for key in ("category", "action", "complexity", "macro", "meso", "micro"):
        value = obj.get(key)
        if isinstance(value, list):
            for idx, text in enumerate(value, 1):
                if text:
                    lines.append(f"{key}[{idx}]: {text}")
        elif value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def motion_debug(path: Path) -> Dict[str, Any]:
    try:
        data = np.load(path, allow_pickle=True)
        poses = np.asarray(data["poses"])
        transl = np.asarray(data.get("transl", data.get("trans")))
        if poses.shape[1] == 156:
            smpl_type = "smplh"
        elif poses.shape[1] == 165:
            smpl_type = "smplx"
        else:
            smpl_type = "smpl"
        return {
            "path": str(path),
            "num_frames": int(poses.shape[0]),
            "pose_dim": int(poses.shape[1]),
            "smpl_type": smpl_type,
            "transl_y_mean": float(np.mean(transl[:, 1])),
            "transl_y_min": float(np.min(transl[:, 1])),
            "transl_y_max": float(np.max(transl[:, 1])),
        }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def stratified_train_sample(
    items: list[tuple[str, Dict[str, Any]]],
    n: int,
    seed: int,
) -> list[tuple[str, Dict[str, Any]]]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for item in items:
        buckets[subject_group(item[1])].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[0])
        rng.shuffle(bucket)
    selected: list[tuple[str, Dict[str, Any]]] = []
    keys = sorted(buckets)
    while len(selected) < n and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/motionhub")
    parser.add_argument("--subset-root", default="data/motionhub/amass_sup")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-per-subset", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    subset_root = Path(args.subset_root)
    split_rows: dict[str, list[tuple[str, Dict[str, Any]]]] = {}
    by_source: dict[str, dict[str, list[tuple[str, Dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    for split in ("test", "train"):
        rows = load_split(subset_root / f"{split}.json")
        for key, row in rows.items():
            if row.get("invalid"):
                continue
            by_source[source_subset(row)][split].append((key, row))
        split_rows[split] = list(rows.items())

    cases = []
    report = {
        "data_root": str(data_root),
        "subset_root": str(subset_root),
        "target_per_subset": args.target_per_subset,
        "seed": args.seed,
        "sources": {},
    }
    idx = 0
    for source in sorted(by_source):
        test_items = sorted(by_source[source].get("test", []), key=lambda item: item[0])
        train_items = sorted(by_source[source].get("train", []), key=lambda item: item[0])
        needed_train = max(0, args.target_per_subset - len(test_items))
        sampled_train = stratified_train_sample(
            train_items,
            needed_train,
            seed=args.seed + sum(ord(ch) for ch in source),
        )
        selected = [("test", item) for item in test_items] + [("train_sample", item) for item in sampled_train]
        debug_values = []
        subject_counts: Counter[str] = Counter()
        for split, (key, row) in selected:
            idx += 1
            motion_rel = row.get("smplx_path") or row.get("smplh_path")
            motion_path = data_root / str(motion_rel)
            debug = motion_debug(motion_path)
            debug_values.append(debug)
            subject_counts[subject_group(row)] += 1
            caption = read_caption(data_root, row)
            cases.append({
                "key": f"amass_sup_{source}_{idx:04d}_{key}",
                "dataset": "AMASS_SUP coordinate audit",
                "genre": source,
                "split": split,
                "duration": row.get("duration", 0),
                "fps": row.get("fps", 30),
                "caption": (
                    f"AMASS_SUP source={source} split={split} id={key}\n"
                    f"motion: {motion_path}\n"
                    f"stored_transl_y_mean={debug.get('transl_y_mean', 'n/a')}\n"
                    f"{caption}"
                ).strip(),
                "motions": [{
                    "id": "gt",
                    "label": f"AMASS_SUP · {source}",
                    "kind": "reference",
                    "smpl_path": str(motion_path),
                    "smpl_type": row.get("smpl_type") or debug.get("smpl_type") or "smplh",
                    "fps": row.get("fps", 30),
                    "debug": debug,
                }],
            })
        y_means = [item["transl_y_mean"] for item in debug_values if "transl_y_mean" in item]
        report["sources"][source] = {
            "train_total": len(train_items),
            "test_total": len(test_items),
            "manifest_cases": len(selected),
            "manifest_test_cases": len(test_items),
            "manifest_train_sample_cases": len(sampled_train),
            "subject_counts": dict(sorted(subject_counts.items())),
            "transl_y_mean_min": min(y_means) if y_means else None,
            "transl_y_mean_max": max(y_means) if y_means else None,
            "transl_y_mean_avg": float(np.mean(y_means)) if y_means else None,
            "load_errors": [item for item in debug_values if "error" in item][:10],
        }

    payload = {
        "meta": {
            "dataset": "amass_sup",
            "purpose": "SMPL-H coordinate audit; frontend must not canonicalize",
            "sources": sorted(by_source),
        },
        "cases": cases,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["total_cases"] = len(cases)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

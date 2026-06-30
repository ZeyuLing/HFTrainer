#!/usr/bin/env python3
"""Rebuild a compact, balanced PerMo train/test split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def load_split(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("data_list"), dict):
        raise ValueError(f"{path} must contain a data_list dict")
    return obj


def write_split(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_permo_row(row: Dict[str, Any]) -> tuple[str, str, str]:
    rel = row.get("smplh_path") or row.get("smplx_path")
    if not rel:
        raise ValueError(f"row has no smpl path: {row}")
    parts = Path(str(rel)).parts
    if len(parts) < 6:
        raise ValueError(f"unexpected PerMo path: {rel}")
    return parts[2], parts[3], parts[4]


def ordered_rows(train_obj: Dict[str, Any], test_obj: Dict[str, Any]) -> OrderedDict[str, Dict[str, Any]]:
    rows: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for source in (train_obj["data_list"], test_obj["data_list"]):
        for key, row in source.items():
            rows[key] = copy.deepcopy(row)
    return OrderedDict(sorted(rows.items()))


def proportional_targets(counts: Counter[str], total: int) -> dict[str, int]:
    raw = {key: value * total / sum(counts.values()) for key, value in counts.items()}
    targets = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(targets.values())
    for key, _ in sorted(
        ((key, raw[key] - targets[key]) for key in targets),
        key=lambda item: item[1],
        reverse=True,
    )[:remainder]:
        targets[key] += 1
    return dict(sorted(targets.items()))


def stable_cost(key: str) -> float:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def solve_balanced_test(
    rows: OrderedDict[str, Dict[str, Any]],
    test_size: int,
) -> tuple[set[str], Dict[str, Any]]:
    items = [
        (key, row, *parse_permo_row(row))
        for key, row in rows.items()
    ]
    categories = sorted({item[2] for item in items})
    labels = sorted({(item[2], item[3]) for item in items})
    actors = sorted({item[4] for item in items})
    category_actors = sorted({(item[2], item[4]) for item in items})

    category_totals = Counter(item[2] for item in items)
    label_totals = Counter((item[2], item[3]) for item in items)
    category_targets = proportional_targets(category_totals, test_size)
    actor_lo = test_size // len(actors)
    actor_hi = math.ceil(test_size / len(actors))

    row_values: list[list[int]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    constraint_names: list[Any] = []

    n = len(items)
    row_values.append([1] * n)
    lower_bounds.append(test_size)
    upper_bounds.append(test_size)
    constraint_names.append(("total", test_size))

    for category in categories:
        row_values.append([1 if item[2] == category else 0 for item in items])
        lower_bounds.append(category_targets[category])
        upper_bounds.append(category_targets[category])
        constraint_names.append(("category", category, category_targets[category]))

    for actor in actors:
        row_values.append([1 if item[4] == actor else 0 for item in items])
        lower_bounds.append(actor_lo)
        upper_bounds.append(actor_hi)
        constraint_names.append(("actor", actor, actor_lo, actor_hi))

    for category, label in labels:
        expected = category_targets[category] * label_totals[(category, label)] / category_totals[category]
        low = max(1, math.floor(expected))
        high = min(category_targets[category], max(low, math.ceil(expected)) + 1)
        row_values.append([1 if (item[2], item[3]) == (category, label) else 0 for item in items])
        lower_bounds.append(low)
        upper_bounds.append(high)
        constraint_names.append(("label", category, label, low, high))

    for category, actor in category_actors:
        high = math.ceil(category_targets[category] / len(actors)) + 1
        row_values.append([1 if (item[2], item[4]) == (category, actor) else 0 for item in items])
        lower_bounds.append(1)
        upper_bounds.append(high)
        constraint_names.append(("category_actor", category, actor, 1, high))

    matrix = lil_matrix((len(row_values), n), dtype=float)
    for row_idx, values in enumerate(row_values):
        for col_idx, value in enumerate(values):
            if value:
                matrix[row_idx, col_idx] = value

    objective = np.array([stable_cost(item[0]) for item in items])
    result = milp(
        c=objective,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), np.array(lower_bounds), np.array(upper_bounds)),
        options={"time_limit": 120, "mip_rel_gap": 0},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    selected = {items[idx][0] for idx, value in enumerate(result.x) if value > 0.5}
    if len(selected) != test_size:
        raise RuntimeError(f"expected {test_size} selected rows, got {len(selected)}")
    return selected, {
        "categories": categories,
        "labels": [f"{category}/{label}" for category, label in labels],
        "actors": actors,
        "category_targets": category_targets,
        "actor_bounds": [actor_lo, actor_hi],
        "num_constraints": len(constraint_names),
        "objective": float(result.fun),
    }


def split_summary(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    categories: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    actors: Counter[str] = Counter()
    category_actors: Counter[str] = Counter()
    frames = 0
    duration = 0.0
    for row in rows.values():
        category, label, actor = parse_permo_row(row)
        categories[category] += 1
        labels[f"{category}/{label}"] += 1
        actors[actor] += 1
        category_actors[f"{category}/{actor}"] += 1
        frames += int(row.get("num_frames") or 0)
        duration += float(row.get("duration") or 0.0)
    return {
        "rows": len(rows),
        "frames": frames,
        "duration_hours": round(duration / 3600.0, 6),
        "category": dict(sorted(categories.items())),
        "actor": dict(sorted(actors.items())),
        "num_labels": len(labels),
        "num_category_actors": len(category_actors),
        "label": dict(sorted(labels.items())),
        "category_actor": dict(sorted(category_actors.items())),
    }


def add_split_meta(obj: Dict[str, Any], split_name: str, policy: str) -> None:
    meta = obj.setdefault("meta_info", {})
    meta["dataset"] = f"permo {split_name} subset"
    meta["version"] = "v1"
    meta["split_policy"] = policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", default="data/motionhub/permo")
    parser.add_argument("--test-size", type=int, default=67)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    train_obj = load_split(subset_root / "train.json")
    test_obj = load_split(subset_root / "test.json")
    rows = ordered_rows(train_obj, test_obj)
    selected_test_keys, solver_report = solve_balanced_test(rows, args.test_size)

    new_train_rows = OrderedDict((key, row) for key, row in rows.items() if key not in selected_test_keys)
    new_test_rows = OrderedDict((key, row) for key, row in rows.items() if key in selected_test_keys)

    policy = "balanced_category_actor_label_v1"
    new_train_obj = copy.deepcopy(train_obj)
    new_test_obj = copy.deepcopy(test_obj)
    new_train_obj["data_list"] = new_train_rows
    new_test_obj["data_list"] = new_test_rows
    add_split_meta(new_train_obj, "train", policy)
    add_split_meta(new_test_obj, "test", policy)

    report: Dict[str, Any] = {
        "subset_root": str(subset_root),
        "test_size": args.test_size,
        "write": bool(args.write),
        "policy": policy,
        "solver": solver_report,
        "summaries": {
            "all": split_summary(rows),
            "train": split_summary(new_train_rows),
            "test": split_summary(new_test_rows),
        },
        "overlap": {
            "train_test_exact": len(set(new_train_rows) & set(new_test_rows)),
            "all_preserved": len(new_train_rows) + len(new_test_rows) == len(rows),
        },
    }

    if args.write:
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.json", "test.json"):
            shutil.copy2(subset_root / name, backup_dir / name)
        write_split(subset_root / "train.json", new_train_obj)
        write_split(subset_root / "test.json", new_test_obj)
        report["backup_dir"] = str(backup_dir)
        report["sha256"] = {
            "train.json": sha256(subset_root / "train.json"),
            "test.json": sha256(subset_root / "test.json"),
        }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

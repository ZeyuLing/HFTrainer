#!/usr/bin/env python3
"""Normalize BEAT S2G splits and derive a compact MotionHub test split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


PATH_KEYS = ("smplx_path", "smplh_path", "audio_path", "speech_script_path", "hierarchical_caption_path")
OPTIONAL_PATH_KEYS = ("audio_path", "speech_script_path", "hierarchical_caption_path")


def load_split(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("data_list"), dict):
        raise ValueError(f"{path} must be a dict split with a data_list dict")
    return obj


def load_first_existing_split(subset_root: Path, names: Iterable[str]) -> tuple[str, Dict[str, Any]]:
    for name in names:
        path = subset_root / name
        if path.exists():
            return name, load_split(path)
    raise FileNotFoundError(f"none of these split files exist in {subset_root}: {list(names)}")


def write_split(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_from_row(row: Dict[str, Any]) -> str:
    for key in ("smplh_path", "smplx_path", "audio_path", "speech_script_path", "hierarchical_caption_path"):
        value = row.get(key)
        if value:
            return Path(str(value)).stem
    raise ValueError(f"row has no known path: {row}")


def group_prefix(row: Dict[str, Any], n: int) -> str:
    parts = stem_from_row(row).split("_")
    return "_".join(parts[:n]) if len(parts) >= n else "_".join(parts)


def language_of(row: Dict[str, Any]) -> str:
    return str(row.get("language") or "unknown").lower()


def existing_files(data_root: Path, subset_root: Path) -> set[str]:
    existing: set[str] = set()
    for group_dir in sorted(path for path in subset_root.glob("beat_*_v2.0.0") if path.is_dir()):
        for modality in ("smplh_52", "wav16k_slices", "speech_script", "hierarchical_caption"):
            root = group_dir / modality
            if not root.exists():
                continue
            for dirpath, _, files in os.walk(root):
                rel_dir = Path(dirpath).relative_to(data_root)
                for filename in files:
                    if filename.startswith("."):
                        continue
                    existing.add((rel_dir / filename).as_posix())
    return existing


def normalized_paths(row: Dict[str, Any], output_motion_dir: str) -> Dict[str, str]:
    source = row.get("smplh_path") or row.get("smplx_path")
    if not source:
        raise ValueError(f"row has no smpl path: {row}")
    parts = Path(str(source)).parts
    if len(parts) < 3:
        raise ValueError(f"unexpected BEAT path: {source}")
    subset_name, group_name = parts[0], parts[1]
    stem = stem_from_row(row)
    speaker = stem.split("_", 1)[0]
    prefix = Path(subset_name) / group_name
    paths = {
        "smplh_path": (prefix / output_motion_dir / speaker / f"{stem}.npz").as_posix(),
        "smplx_path": (prefix / output_motion_dir / speaker / f"{stem}.npz").as_posix(),
    }
    if row.get("audio_path"):
        paths["audio_path"] = (prefix / "wav16k_slices" / speaker / f"{stem}.wav").as_posix()
    if row.get("speech_script_path"):
        paths["speech_script_path"] = (prefix / "speech_script" / speaker / f"{stem}.txt").as_posix()
    if row.get("hierarchical_caption_path"):
        paths["hierarchical_caption_path"] = (prefix / "hierarchical_caption" / speaker / f"{stem}.json").as_posix()
    return paths


def normalize_rows(
    rows: Dict[str, Dict[str, Any]],
    existing: set[str],
    output_motion_dir: str,
) -> Tuple[OrderedDict[str, Dict[str, Any]], Dict[str, Any]]:
    normalized: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    missing_required: list[dict[str, str]] = []
    missing_optional: Counter[str] = Counter()
    removed_motion501 = 0
    old_smplx_refs = 0
    changed_paths: Counter[str] = Counter()
    for key, row in rows.items():
        new_row = copy.deepcopy(row)
        if "smplx_55" in str(new_row.get("smplx_path", "")):
            old_smplx_refs += 1
        paths = normalized_paths(new_row, output_motion_dir)
        required = paths["smplh_path"]
        if required not in existing:
            missing_required.append({"id": key, "path": required})
            continue
        for path_key in PATH_KEYS:
            old_value = new_row.get(path_key)
            if path_key in paths:
                if old_value != paths[path_key]:
                    changed_paths[path_key] += 1
                new_row[path_key] = paths[path_key]
            elif path_key in new_row:
                new_row.pop(path_key, None)
                changed_paths[path_key] += 1
        for path_key in OPTIONAL_PATH_KEYS:
            value = new_row.get(path_key)
            if value and value not in existing:
                new_row.pop(path_key, None)
                missing_optional[path_key] += 1
        if "motion501_path" in new_row:
            new_row.pop("motion501_path", None)
            removed_motion501 += 1
        new_row["smpl_type"] = "smplh"
        new_row["motion_representation"] = output_motion_dir
        normalized[key] = new_row
    if missing_required:
        examples = missing_required[:10]
        raise RuntimeError(f"missing required SMPL-H files: {len(missing_required)} examples={examples}")
    return normalized, {
        "rows": len(normalized),
        "old_smplx55_refs": old_smplx_refs,
        "changed_paths": dict(changed_paths),
        "removed_motion501": removed_motion501,
        "missing_optional_removed": dict(missing_optional),
    }


def balanced_select(
    rows: OrderedDict[str, Dict[str, Any]],
    targets: Dict[str, int],
    group_prefix_len: int,
) -> Tuple[OrderedDict[str, Dict[str, Any]], Dict[str, Any]]:
    by_language: dict[str, OrderedDict[str, Dict[str, Any]]] = defaultdict(OrderedDict)
    for key, row in rows.items():
        by_language[language_of(row)][key] = row

    selected: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    summary: Dict[str, Any] = {}
    for language, target in targets.items():
        candidates = by_language.get(language, OrderedDict())
        if len(candidates) <= target:
            for key, row in candidates.items():
                selected[key] = row
            summary[language] = {
                "available": len(candidates),
                "selected": len(candidates),
                "target": target,
                "strategy": "all_available",
            }
            continue

        grouped: dict[str, deque[tuple[str, Dict[str, Any]]]] = OrderedDict()
        for key, row in candidates.items():
            grouped.setdefault(group_prefix(row, group_prefix_len), deque()).append((key, row))

        selected_for_lang: list[tuple[str, Dict[str, Any]]] = []
        group_queue: deque[str] = deque(grouped.keys())
        while group_queue and len(selected_for_lang) < target:
            group = group_queue.popleft()
            item_queue = grouped[group]
            selected_for_lang.append(item_queue.popleft())
            if item_queue:
                group_queue.append(group)
        for key, row in selected_for_lang:
            selected[key] = row
        summary[language] = {
            "available": len(candidates),
            "selected": len(selected_for_lang),
            "target": target,
            "strategy": f"round_robin_prefix{group_prefix_len}",
            "selected_groups": len({group_prefix(row, group_prefix_len) for _, row in selected_for_lang}),
        }
    return selected, summary


def split_summary(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    language = Counter(language_of(row) for row in rows.values())
    with_audio = Counter(language_of(row) for row in rows.values() if row.get("audio_path"))
    with_script = Counter(language_of(row) for row in rows.values() if row.get("speech_script_path"))
    with_caption = Counter(language_of(row) for row in rows.values() if row.get("hierarchical_caption_path"))
    frames = sum(int(row.get("num_frames") or 0) for row in rows.values())
    duration = sum(float(row.get("duration") or 0.0) for row in rows.values())
    return {
        "rows": len(rows),
        "frames": frames,
        "duration_hours": round(duration / 3600.0, 6),
        "language": dict(sorted(language.items())),
        "with_audio": dict(sorted(with_audio.items())),
        "with_script": dict(sorted(with_script.items())),
        "with_caption": dict(sorted(with_caption.items())),
    }


def subset_by_language(rows: OrderedDict[str, Dict[str, Any]], language: str) -> OrderedDict[str, Dict[str, Any]]:
    return OrderedDict((key, row) for key, row in rows.items() if language_of(row) == language)


def parse_targets(value: str) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, count = item.split(":", 1)
        targets[key.strip().lower()] = int(count)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/motionhub")
    parser.add_argument("--subset-root", default="data/motionhub/beat_v2.0.0")
    parser.add_argument("--output-motion-dir", default="smplh_52")
    parser.add_argument("--test-targets", default="chinese:100,english:100,japanese:100,spanish:69")
    parser.add_argument("--group-prefix-len", type=int, default=5)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    subset_root = Path(args.subset_root)
    existing = existing_files(data_root, subset_root)
    targets = parse_targets(args.test_targets)

    train_source_name, train_s2g = load_first_existing_split(
        subset_root, ("train_official.json", "train_s2g.json")
    )
    test_source_name, test_s2g = load_first_existing_split(
        subset_root, ("test_official.json", "test_s2g.json")
    )
    train_s2g_rows, train_norm_report = normalize_rows(
        train_s2g["data_list"], existing, args.output_motion_dir
    )
    test_s2g_rows, test_norm_report = normalize_rows(
        test_s2g["data_list"], existing, args.output_motion_dir
    )
    compact_test_rows, selection_report = balanced_select(
        test_s2g_rows, targets, args.group_prefix_len
    )

    train_obj = copy.deepcopy(train_s2g)
    train_obj["data_list"] = train_s2g_rows
    test_full_obj = copy.deepcopy(test_s2g)
    test_full_obj["data_list"] = test_s2g_rows
    compact_test_obj = copy.deepcopy(test_s2g)
    compact_test_obj["data_list"] = compact_test_rows

    output_objects: dict[str, Dict[str, Any]] = {
        "train_official.json": train_obj,
        "test_official.json": test_full_obj,
        "train.json": train_obj,
        "test.json": compact_test_obj,
    }
    for language in targets:
        group_root = subset_root / f"beat_{language}_v2.0.0"
        if not group_root.exists():
            continue
        language_train = copy.deepcopy(train_obj)
        language_train["data_list"] = subset_by_language(train_s2g_rows, language)
        language_test_full = copy.deepcopy(test_full_obj)
        language_test_full["data_list"] = subset_by_language(test_s2g_rows, language)
        language_test_compact = copy.deepcopy(compact_test_obj)
        language_test_compact["data_list"] = subset_by_language(compact_test_rows, language)
        prefix = group_root.relative_to(subset_root).as_posix()
        output_objects[f"{prefix}/train_official.json"] = language_train
        output_objects[f"{prefix}/test_official.json"] = language_test_full
        output_objects[f"{prefix}/train.json"] = language_train
        output_objects[f"{prefix}/test.json"] = language_test_compact

    report: Dict[str, Any] = {
        "data_root": str(data_root),
        "subset_root": str(subset_root),
        "source_splits": {
            "train": train_source_name,
            "test": test_source_name,
        },
        "output_motion_dir": args.output_motion_dir,
        "write": bool(args.write),
        "test_targets": targets,
        "group_prefix_len": args.group_prefix_len,
        "indexed_existing_files": len(existing),
        "normalize": {
            "train_s2g": train_norm_report,
            "test_s2g": test_norm_report,
        },
        "selection": selection_report,
        "summaries": {
            name: split_summary(obj["data_list"]) for name, obj in output_objects.items()
        },
        "overlap": {},
    }

    train_keys = set(train_s2g_rows)
    compact_test_keys = set(compact_test_rows)
    full_test_keys = set(test_s2g_rows)
    report["overlap"] = {
        "train_vs_compact_test_exact": len(train_keys & compact_test_keys),
        "train_vs_full_test_exact": len(train_keys & full_test_keys),
        "compact_test_subset_of_test_s2g": compact_test_keys.issubset(full_test_keys),
    }

    if args.write:
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_names = set(output_objects) | {"train_s2g.json", "test_s2g.json"}
        for name in sorted(backup_names):
            src = subset_root / name
            if src.exists():
                dst = backup_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for name, obj in output_objects.items():
            write_split(subset_root / name, obj)
        for stale_name in ("train_s2g.json", "test_s2g.json"):
            stale_path = subset_root / stale_name
            if stale_path.exists():
                stale_path.unlink()
        report["backup_dir"] = str(backup_dir)
        report["removed_stale_splits"] = ["train_s2g.json", "test_s2g.json"]
        report["sha256"] = {name: sha256(subset_root / name) for name in output_objects}

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

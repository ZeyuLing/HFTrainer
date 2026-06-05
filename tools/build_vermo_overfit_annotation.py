#!/usr/bin/env python3
"""Build a small deterministic all-task VerMo overfit annotation."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict
from copy import deepcopy
from typing import Dict, Iterable, Iterator, List, Tuple


TASK_SOURCES = {
    "pretrain": "text",
    "t2m": "text",
    "m2t": "text",
    "n2tm": "text",
    "pred": "text",
    "inbetween": "text",
    "m2d": "dance",
    "d2m": "dance",
    "t2md": "dance",
    "g2md": "dance",
    "n2md": "dance",
    "m2d_ar": "dance",
    "d2m_ar": "dance",
    "s2g": "speech",
    "g2s": "speech",
    "t2sg": "speech",
    "n2sg": "speech",
    "ss2sg": "speech",
    "s2g_ar": "speech",
}

SOURCE_FILES = {
    "text": [
        "data/annotation/test_motionhub_t2m.json",
        "data/annotation/test_motionhub_2p.json",
    ],
    "dance": ["data/annotation/test_motionhub_m2d.json"],
    "speech": ["data/annotation/test_motionhub_s2g.json"],
}


def _load_entries(paths: Iterable[str]) -> Iterator[Tuple[str, str, Dict]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data["data_list"].items():
            yield path, key, value


def _path_exists(data_dir: str, value) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return all(_path_exists(data_dir, item) for item in value)
    return os.path.exists(os.path.join(data_dir, value))


def _valid_entry(data_dir: str, group: str, item: Dict, max_duration: float) -> bool:
    if float(item.get("duration", 9999.0)) > max_duration:
        return False
    if not _path_exists(data_dir, item.get("smplx_path")):
        return False
    caption_path = item.get("hierarchical_caption_path", item.get("caption_path"))
    if group in {"text", "dance", "speech"} and not _path_exists(data_dir, caption_path):
        return False
    if group == "dance":
        return (
            _path_exists(data_dir, item.get("music_path"))
            and item.get("genre") is not None
        )
    if group == "speech":
        return (
            _path_exists(data_dir, item.get("audio_path"))
            and _path_exists(data_dir, item.get("speech_script_path"))
        )
    return True


def _select_entries(
    data_dir: str,
    group: str,
    samples_per_task: int,
    max_duration: float,
) -> List[Tuple[str, str, Dict]]:
    selected = []
    for path, key, value in _load_entries(SOURCE_FILES[group]):
        if _valid_entry(data_dir, group, value, max_duration):
            selected.append((path, key, value))
            if len(selected) >= samples_per_task:
                break
    if len(selected) < samples_per_task:
        raise RuntimeError(
            f"Need {samples_per_task} valid {group} entries, got {len(selected)}"
        )
    return selected


def build(args: argparse.Namespace) -> None:
    selected_by_group = {
        group: _select_entries(
            args.data_dir,
            group,
            args.samples_per_task,
            args.max_duration,
        )
        for group in sorted(set(TASK_SOURCES.values()))
    }

    data_list = OrderedDict()
    task_counts = Counter()
    source_counts = Counter()
    for task, group in TASK_SOURCES.items():
        selected_entries = selected_by_group[group]
        if task == "pretrain" and args.single_pretrain_source:
            selected_entries = [selected_entries[0] for _ in selected_entries]

        for sample_idx, (source_path, source_key, source_item) in enumerate(
            selected_entries
        ):
            item = deepcopy(source_item)
            item["overfit_task"] = task
            item["overfit_source_key"] = source_key
            item["overfit_source_annotation"] = source_path
            if task == "pretrain" and args.single_pretrain_source:
                item["overfit_duplicate_source_idx"] = 0
            case_key = f"case_{len(data_list):03d}_{task}_{sample_idx:02d}"
            data_list[case_key] = item
            task_counts[task] += 1
            source_counts[group] += 1

    output = {
        "meta_info": {
            "dataset": "VerMo all-task deterministic overfit",
            "samples_per_task": args.samples_per_task,
            "num_cases": len(data_list),
            "task_counts": dict(task_counts),
            "source_counts": dict(source_counts),
            "source_files": SOURCE_FILES,
            "path_policy": "paths are relative to data/motionhub",
            "max_duration": args.max_duration,
            "single_pretrain_source": args.single_pretrain_source,
        },
        "data_list": data_list,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(data_list)} cases to {args.output}")
    print("Task counts:", dict(task_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/annotation/vermo_overfit_alltasks_190_20260603.json",
    )
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument(
        "--single-pretrain-source",
        action="store_true",
        help=(
            "Repeat one pretrain source for all pretrain slots. The pretrain "
            "task is unconditional, so multiple unique pretrain targets share "
            "the same prompt and cannot all be exact-match reconstructed."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

#!/usr/bin/env python3
"""Build a small VerMo overfit annotation that keeps the main data pipeline."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, List, Tuple


TASK_SOURCES = OrderedDict(
    [
        ("t2m", "text"),
        ("m2t", "text"),
        ("n2tm", "text"),
        ("pred", "text"),
        ("inbetween", "text"),
        ("m2d", "dance"),
        ("d2m", "dance"),
        ("t2md", "dance"),
        ("g2md", "dance"),
        ("n2md", "dance"),
        ("m2d_ar", "dance"),
        ("d2m_ar", "dance"),
        ("s2g", "speech"),
        ("g2s", "speech"),
        ("t2sg", "speech"),
        ("n2sg", "speech"),
        ("ss2sg", "speech"),
        ("s2g_ar", "speech"),
    ]
)

TEXT_TRUE_MULTI_TASKS = {"t2m", "m2t", "n2tm", "pred", "inbetween"}
RANDOM_SOURCE_TASKS = {"n2tm", "n2md", "n2sg"}


def _load_entries(paths: Iterable[str]) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data["data_list"].items():
            yield path, key, value


def _path_exists(data_dir: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return all(_path_exists(data_dir, item) for item in value)
    return os.path.exists(os.path.join(data_dir, value))


def _caption_path(item: Dict[str, Any]) -> Any:
    return item.get("hierarchical_caption_path", item.get("caption_path"))


def _valid_entry(
    data_dir: str,
    group: str,
    item: Dict[str, Any],
    max_duration: float,
    require_single: bool,
    require_multi: bool,
) -> bool:
    if float(item.get("duration", 9999.0)) > max_duration:
        return False

    if group == "text" and (item.get("music_path") or item.get("audio_path")):
        return False
    if group == "dance" and (item.get("music_path") is None or item.get("genre") is None):
        return False
    if group == "speech" and (
        item.get("audio_path") is None or item.get("speech_script_path") is None
    ):
        return False

    smplx_path = item.get("smplx_path")
    if require_single and not isinstance(smplx_path, str):
        return False
    if require_multi and not isinstance(smplx_path, list):
        return False
    if not _path_exists(data_dir, smplx_path):
        return False

    caption_path = _caption_path(item)
    if group in {"text", "dance", "speech", "true_multi"} and not _path_exists(
        data_dir, caption_path
    ):
        return False
    if group == "dance":
        return _path_exists(data_dir, item.get("music_path"))
    if group == "speech":
        return _path_exists(data_dir, item.get("audio_path")) and _path_exists(
            data_dir, item.get("speech_script_path")
        )
    return True


def _select_entries(
    data_dir: str,
    paths: Iterable[str],
    group: str,
    count: int,
    max_duration: float,
    require_single: bool = False,
    require_multi: bool = False,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    selected = []
    for path, key, value in _load_entries(paths):
        if _valid_entry(
            data_dir,
            group,
            value,
            max_duration,
            require_single=require_single,
            require_multi=require_multi,
        ):
            selected.append((path, key, value))
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise RuntimeError(f"Need {count} valid {group} entries, got {len(selected)}")
    return selected


def _sample_count_for_task(args: argparse.Namespace, task: str) -> int:
    if task in RANDOM_SOURCE_TASKS:
        return args.random_task_samples
    return args.samples_per_task


def build(args: argparse.Namespace) -> None:
    max_regular = max(args.samples_per_task, args.random_task_samples)
    text_single = _select_entries(
        args.data_dir,
        args.base_annotations,
        "text",
        max_regular,
        args.max_duration,
        require_single=True,
    )
    dance_single = _select_entries(
        args.data_dir,
        args.base_annotations,
        "dance",
        max_regular,
        args.max_duration,
        require_single=True,
    )
    speech_single = _select_entries(
        args.data_dir,
        args.base_annotations,
        "speech",
        max_regular,
        args.max_duration,
        require_single=True,
    )
    true_multi = _select_entries(
        args.data_dir,
        args.true_multi_annotations,
        "true_multi",
        args.true_multi_per_text_task,
        args.max_duration,
        require_multi=True,
    )

    single_pool = {
        "text": text_single,
        "dance": dance_single,
        "speech": speech_single,
    }

    data_list: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    task_counts = Counter()
    source_counts = Counter()
    kind_counts = Counter()

    for task, group in TASK_SOURCES.items():
        total = _sample_count_for_task(args, task)
        true_count = (
            min(args.true_multi_per_text_task, total)
            if task in TEXT_TRUE_MULTI_TASKS
            else 0
        )
        single_count = total - true_count

        for i, (source_path, source_key, source_item) in enumerate(
            single_pool[group][:single_count]
        ):
            item = deepcopy(source_item)
            item["overfit_task"] = task
            item["overfit_source_key"] = source_key
            item["overfit_source_annotation"] = source_path
            item["overfit_multi_kind"] = "single_or_online_pseudo_candidate"
            case_key = f"case_{len(data_list):03d}_{task}_single_{i:02d}"
            data_list[case_key] = item
            task_counts[task] += 1
            source_counts[group] += 1
            kind_counts["single_or_online_pseudo_candidate"] += 1

        for i, (source_path, source_key, source_item) in enumerate(
            true_multi[:true_count]
        ):
            item = deepcopy(source_item)
            item["overfit_task"] = task
            item["overfit_source_key"] = source_key
            item["overfit_source_annotation"] = source_path
            item["overfit_multi_kind"] = "true_multi"
            case_key = f"case_{len(data_list):03d}_{task}_true_multi_{i:02d}"
            data_list[case_key] = item
            task_counts[task] += 1
            source_counts["true_multi"] += 1
            kind_counts["true_multi"] += 1

    output = {
        "meta_info": {
            "dataset": "VerMo main-pipeline overfit",
            "num_cases": len(data_list),
            "samples_per_task": args.samples_per_task,
            "random_task_samples": args.random_task_samples,
            "true_multi_per_text_task": args.true_multi_per_text_task,
            "max_duration": args.max_duration,
            "task_counts": dict(task_counts),
            "source_counts": dict(source_counts),
            "multi_kind_counts": dict(kind_counts),
            "base_annotations": args.base_annotations,
            "true_multi_annotations": args.true_multi_annotations,
            "tasks": list(TASK_SOURCES.keys()),
            "pipeline_policy": (
                "No static pseudo multi-person data is materialized here. "
                "The config inherits the main ComposeMultiPerson transform, "
                "including compose_prob and skip_with_audio."
            ),
            "random_task_policy": (
                "Random-source tasks use fewer samples to avoid conflicting "
                "identical prompts under the main random optional-modal policy."
            ),
        },
        "data_list": data_list,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data_list)} cases to {args.output}")
    print("Task counts:", dict(task_counts))
    print("Multi-kind counts:", dict(kind_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/annotation/vermo_overfit_mainpipeline_18tasks_93_20260604.json",
    )
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument(
        "--base-annotations",
        nargs="+",
        default=["data/annotation/train_hq_motionhub_hymotion.json"],
    )
    parser.add_argument(
        "--true-multi-annotations",
        nargs="+",
        default=["data/annotation/train_motionclip_2p.json"],
    )
    parser.add_argument("--samples-per-task", type=int, default=6)
    parser.add_argument("--random-task-samples", type=int, default=1)
    parser.add_argument("--true-multi-per-text-task", type=int, default=2)
    parser.add_argument("--max-duration", type=float, default=12.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

#!/usr/bin/env python3
"""Build deterministic VerMo overfit data with single, true-multi, and pseudo-multi cases."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


TASK_SOURCES = {
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

TASK_ORDER = list(TASK_SOURCES)
TRUE_MULTI_TASKS = {"t2m", "m2t", "n2tm", "pred", "inbetween"}
PSEUDO_MULTI_TASKS = {"t2m", "m2t", "n2tm", "pred", "inbetween"}

SOURCE_FILES = {
    "text": ["data/annotation/test_motionhub_t2m.json"],
    "dance": ["data/annotation/test_motionhub_m2d.json"],
    "speech": ["data/annotation/test_motionhub_s2g.json"],
    "true_multi": [
        "data/annotation/test_motionhub_2p.json",
        "data/annotation/train_motionclip_2p.json",
    ],
}


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


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


def _raw_smpl_npz(data_dir: str, rel_path: str) -> bool:
    path = os.path.join(data_dir, rel_path)
    try:
        with np.load(path, allow_pickle=True) as data:
            return "poses" in data and "trans" in data
    except Exception:
        return False


def _motion_fps_and_frames(
    data_dir: str, smplx_path: Any
) -> Tuple[Optional[float], Optional[int]]:
    paths = smplx_path if isinstance(smplx_path, list) else [smplx_path]
    fps_values = []
    frame_values = []
    for rel_path in paths:
        if not isinstance(rel_path, str):
            continue
        path = os.path.join(data_dir, rel_path)
        try:
            with np.load(path, allow_pickle=True) as data:
                if "motion_135" in data:
                    frame_values.append(int(data["motion_135"].shape[0]))
                elif "poses" in data:
                    frame_values.append(int(data["poses"].shape[0]))
                elif "trans" in data:
                    frame_values.append(int(data["trans"].shape[0]))
                if "mocap_framerate" in data:
                    fps_values.append(float(np.asarray(data["mocap_framerate"]).item()))
        except Exception:
            continue
    fps = fps_values[0] if fps_values else None
    frames = min(frame_values) if frame_values else None
    return fps, frames


def _apply_fixed_crop_metadata(
    data_dir: str,
    item: Dict[str, Any],
    crop_duration: float,
) -> None:
    if crop_duration <= 0:
        return
    fps, frames = _motion_fps_and_frames(data_dir, item.get("smplx_path"))
    if fps is None or frames is None or fps <= 0 or frames <= 0:
        return
    crop_frames = min(frames, max(1, int(round(float(crop_duration) * fps))))
    duration = crop_frames / fps
    item["_motion_audio_crop_start"] = 0.0
    item["_motion_audio_crop_start_frame"] = 0
    item["_motion_audio_crop_duration"] = duration
    item["_motion_audio_crop_num_frames"] = int(crop_frames)
    item["duration"] = duration


def _valid_entry(
    data_dir: str,
    group: str,
    item: Dict[str, Any],
    max_duration: float,
    require_single_raw_smpl: bool = False,
) -> bool:
    if float(item.get("duration", 9999.0)) > max_duration:
        return False
    smplx_path = item.get("smplx_path")
    if not _path_exists(data_dir, smplx_path):
        return False
    if require_single_raw_smpl and (
        not isinstance(smplx_path, str) or not _raw_smpl_npz(data_dir, smplx_path)
    ):
        return False
    caption_path = _caption_path(item)
    if group in {"text", "dance", "speech", "true_multi"} and not _path_exists(
        data_dir, caption_path
    ):
        return False
    if group == "true_multi":
        return isinstance(smplx_path, list) and len(smplx_path) >= 2
    if group == "dance":
        return _path_exists(data_dir, item.get("music_path")) and item.get("genre") is not None
    if group == "speech":
        return _path_exists(data_dir, item.get("audio_path")) and _path_exists(
            data_dir, item.get("speech_script_path")
        )
    return True


def _select_entries(
    data_dir: str,
    group: str,
    count: int,
    max_duration: float,
    require_single_raw_smpl: bool = False,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    selected = []
    for path, key, value in _load_entries(SOURCE_FILES[group]):
        if _valid_entry(data_dir, group, value, max_duration, require_single_raw_smpl):
            selected.append((path, key, value))
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise RuntimeError(f"Need {count} valid {group} entries, got {len(selected)}")
    return selected


def _first_caption_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if all(k in data and isinstance(data[k], list) for k in ("macro", "meso", "micro")):
        for key in ("macro", "meso", "micro"):
            for text in data[key]:
                if isinstance(text, str) and text.strip():
                    return text.strip()
    for item in data.get("result", []):
        if not isinstance(item, dict):
            continue
        for key in (
            "short_caption_rewritten",
            "short caption_rewritten",
            "short_caption",
            "short caption",
        ):
            value = item.get(key)
            if isinstance(value, list):
                for text in value:
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            elif isinstance(value, str) and value.strip():
                return value.strip()
    return "a person performs a motion"


def _copy_npz_with_offset(src_abs: str, dst_abs: str, offset_x: float) -> None:
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    with np.load(src_abs, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    for key in ("trans", "transl"):
        if key in arrays:
            arr = np.asarray(arrays[key]).copy()
            if arr.ndim >= 2 and arr.shape[-1] >= 1:
                arr[..., 0] += np.float32(offset_x)
                arrays[key] = arr
    np.savez_compressed(dst_abs, **arrays)


def _make_pseudo_multi(
    data_dir: str,
    pseudo_root: str,
    case_key: str,
    source_item_a: Dict[str, Any],
    source_item_b: Dict[str, Any],
) -> Tuple[List[str], str]:
    src_rel_a = source_item_a["smplx_path"]
    src_rel_b = source_item_b["smplx_path"]
    assert isinstance(src_rel_a, str), src_rel_a
    assert isinstance(src_rel_b, str), src_rel_b
    src_abs_a = os.path.join(data_dir, src_rel_a)
    src_abs_b = os.path.join(data_dir, src_rel_b)
    stem = sanitize(case_key)
    rel_dir = os.path.join(pseudo_root, stem).replace(os.sep, "/")
    p1_rel = f"{rel_dir}/P1.npz"
    p2_rel = f"{rel_dir}/P2.npz"
    _copy_npz_with_offset(src_abs_a, os.path.join(data_dir, p1_rel), -1.25)
    _copy_npz_with_offset(src_abs_b, os.path.join(data_dir, p2_rel), 1.25)

    cap_abs_a = os.path.join(data_dir, _caption_path(source_item_a))
    cap_abs_b = os.path.join(data_dir, _caption_path(source_item_b))
    caption_a = _first_caption_from_file(cap_abs_a)
    caption_b = _first_caption_from_file(cap_abs_b)
    person_captions = [caption_a, caption_b]
    pseudo_caption = (
        f"Person 1 caption: {caption_a}\n"
        f"Person 2 caption: {caption_b}"
    )
    cap_rel = f"{rel_dir}/caption.json"
    cap_out = os.path.join(data_dir, cap_rel)
    os.makedirs(os.path.dirname(cap_out), exist_ok=True)
    with open(cap_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "macro": [pseudo_caption],
                "meso": [pseudo_caption],
                "micro": [pseudo_caption],
                "person_captions": person_captions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return [p1_rel, p2_rel], cap_rel


def _choose_pseudo_partner(
    data_dir: str,
    pool: List[Tuple[str, str, Dict[str, Any]]],
    anchor_idx: int,
    min_offset: int,
) -> Tuple[str, str, Dict[str, Any]]:
    """Choose a second source with the same fps and a different motion."""
    _, anchor_key, anchor_item = pool[anchor_idx]
    anchor_fps, _ = _motion_fps_and_frames(data_dir, anchor_item["smplx_path"])
    order = list(range(anchor_idx + min_offset, len(pool))) + list(range(0, len(pool)))
    for j in order:
        if j == anchor_idx:
            continue
        path, key, item = pool[j]
        if key == anchor_key or item.get("smplx_path") == anchor_item.get("smplx_path"):
            continue
        fps, _ = _motion_fps_and_frames(data_dir, item["smplx_path"])
        if anchor_fps is None or fps is None or abs(float(anchor_fps) - float(fps)) < 1e-6:
            return path, key, item
    raise RuntimeError(f"Could not find pseudo-multi partner for source {anchor_key}")


def _append_case(
    data_dir: str,
    data_list: OrderedDict,
    task: str,
    sample_idx: int,
    source_path: str,
    source_key: str,
    source_item: Dict[str, Any],
    multi_kind: str,
    crop_duration: float,
    task_counts: Counter,
    kind_counts: Counter,
) -> None:
    item = deepcopy(source_item)
    item["overfit_task"] = task
    item["overfit_source_key"] = source_key
    item["overfit_source_annotation"] = source_path
    item["overfit_multi_kind"] = multi_kind
    if isinstance(item.get("smplx_path"), list):
        item["num_person"] = len(item["smplx_path"])
    else:
        item["num_person"] = 1
    _apply_fixed_crop_metadata(data_dir, item, crop_duration)
    case_key = f"case_{len(data_list):03d}_{task}_{multi_kind}_{sample_idx:02d}"
    data_list[case_key] = item
    task_counts[task] += 1
    kind_counts[multi_kind] += 1


def _crop_duration_for_case(args: argparse.Namespace, sample_idx: int) -> float:
    """Keep random-generation prompts identifiable after Duration's .1f formatting."""
    if args.unique_duration_step <= 0:
        return args.crop_duration
    offset = max(0, args.samples_per_task - 1 - sample_idx)
    return max(args.min_crop_duration, args.crop_duration - offset * args.unique_duration_step)


def _validate_prompt_unique(data_list: OrderedDict[str, Dict[str, Any]]) -> None:
    """Guard against duplicate n2tm prompts with different targets in overfit data."""
    seen: Dict[Tuple[str, int, str], str] = {}
    duplicates = []
    for key, item in data_list.items():
        task = item.get("overfit_task")
        if task != "n2tm":
            continue
        prompt_key = (
            task,
            int(item.get("num_person", 1) or 1),
            f"{float(item.get('duration', 0.0) or 0.0):.1f}",
        )
        if prompt_key in seen:
            duplicates.append((prompt_key, seen[prompt_key], key))
        else:
            seen[prompt_key] = key
    if duplicates:
        formatted = "; ".join(
            f"{prompt_key}: {first} / {second}"
            for prompt_key, first, second in duplicates
        )
        raise RuntimeError(f"Duplicate n2tm prompt keys after formatting: {formatted}")


def build(args: argparse.Namespace) -> None:
    single_need = max(args.samples_per_task, args.single_per_task)
    pseudo_need = max(args.samples_per_task, args.pseudo_per_task)
    selected_single = {
        group: _select_entries(args.data_dir, group, single_need, args.max_duration)
        for group in ("text", "dance", "speech")
    }
    selected_pseudo = {
        "text": _select_entries(
            args.data_dir,
            "text",
            pseudo_need,
            args.max_duration,
            require_single_raw_smpl=True,
        )
    }
    selected_true = _select_entries(
        args.data_dir,
        "true_multi",
        args.true_multi_per_task,
        args.max_duration,
    )

    data_list: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    task_counts: Counter = Counter()
    kind_counts: Counter = Counter()

    for task in TASK_ORDER:
        group = TASK_SOURCES[task]
        true_count = args.true_multi_per_task if task in TRUE_MULTI_TASKS else 0
        pseudo_count = args.pseudo_per_task if task in PSEUDO_MULTI_TASKS else 0
        single_count = args.samples_per_task - true_count - pseudo_count
        if single_count < 0:
            raise ValueError(
                f"samples_per_task={args.samples_per_task} is smaller than "
                f"true+pseudo count for task={task}"
            )

        for i, (src_path, src_key, src_item) in enumerate(selected_single[group][:single_count]):
            _append_case(
                args.data_dir,
                data_list,
                task,
                i,
                src_path,
                src_key,
                src_item,
                "single",
                _crop_duration_for_case(args, i),
                task_counts,
                kind_counts,
            )

        for i, (src_path, src_key, src_item) in enumerate(selected_true[:true_count]):
            sample_idx = single_count + i
            _append_case(
                args.data_dir,
                data_list,
                task,
                sample_idx,
                src_path,
                src_key,
                src_item,
                "true_multi",
                _crop_duration_for_case(args, sample_idx),
                task_counts,
                kind_counts,
            )

        if pseudo_count <= 0:
            continue

        pseudo_pool = selected_pseudo[group]
        for i, (src_path, src_key, src_item) in enumerate(pseudo_pool[:pseudo_count]):
            src_path_b, src_key_b, src_item_b = _choose_pseudo_partner(
                args.data_dir,
                pseudo_pool,
                i,
                pseudo_count,
            )
            case_key = f"{task}_{i}_{src_key}_with_{src_key_b}"
            item = deepcopy(src_item)
            smplx_paths, caption_path = _make_pseudo_multi(
                args.data_dir,
                args.pseudo_root,
                case_key,
                src_item,
                src_item_b,
            )
            item["smplx_path"] = smplx_paths
            item["hierarchical_caption_path"] = caption_path
            item["caption_path"] = caption_path
            item["num_person"] = 2
            item["overfit_pseudo_person_sources"] = [
                {
                    "source_annotation": src_path,
                    "source_key": src_key,
                    "caption": _first_caption_from_file(
                        os.path.join(args.data_dir, _caption_path(src_item))
                    ),
                },
                {
                    "source_annotation": src_path_b,
                    "source_key": src_key_b,
                    "caption": _first_caption_from_file(
                        os.path.join(args.data_dir, _caption_path(src_item_b))
                    ),
                },
            ]
            sample_idx = single_count + true_count + i
            _append_case(
                args.data_dir,
                data_list,
                task,
                sample_idx,
                src_path,
                src_key,
                item,
                "pseudo_multi",
                _crop_duration_for_case(args, sample_idx),
                task_counts,
                kind_counts,
            )

    output = {
        "meta_info": {
            "dataset": "VerMo deterministic overfit with true and pseudo multi-person cases",
            "samples_per_task": args.samples_per_task,
            "num_cases": len(data_list),
            "task_counts": dict(task_counts),
            "multi_kind_counts": dict(kind_counts),
            "source_files": SOURCE_FILES,
            "path_policy": "paths are relative to data/motionhub",
            "max_duration": args.max_duration,
            "crop_duration": args.crop_duration,
            "min_crop_duration": args.min_crop_duration,
            "unique_duration_step": args.unique_duration_step,
            "pseudo_root": args.pseudo_root,
            "true_multi_tasks": sorted(TRUE_MULTI_TASKS),
            "pseudo_multi_tasks": sorted(PSEUDO_MULTI_TASKS),
            "pseudo_multi_policy": (
                "Only non-audio text/motion tasks get pseudo multi-person cases. "
                "Audio/music/speech tasks remain single-person unless true multi-person data exists."
            ),
            "tasks": TASK_ORDER,
        },
        "data_list": data_list,
    }

    _validate_prompt_unique(data_list)

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
        default="data/annotation/vermo_overfit_multimode_180_short2s_promptuniq_textpseudo_noaudio_20260604.json",
    )
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--pseudo-root", default="vermo_pseudo_multi_textpseudo_noaudio_20260604")
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--single-per-task", type=int, default=10)
    parser.add_argument("--pseudo-per-task", type=int, default=3)
    parser.add_argument("--true-multi-per-task", type=int, default=3)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--crop-duration", type=float, default=2.0)
    parser.add_argument("--min-crop-duration", type=float, default=1.1)
    parser.add_argument("--unique-duration-step", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

#!/usr/bin/env python3
"""Build the public MotionHub dataset README from local split annotations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import orjson
except ImportError:  # pragma: no cover - optional speedup.
    orjson = None


SPLITS = ("train", "val", "test")
TEXT_GRANULARITIES = ("macro", "meso", "micro")


def load_json(path: Path) -> Any:
    if orjson is not None:
        return orjson.loads(path.read_bytes())
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_rows(obj: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    data = obj.get("data_list", obj) if isinstance(obj, dict) else obj
    if isinstance(data, dict):
        for key, row in data.items():
            if isinstance(row, dict):
                yield str(key), row
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                yield str(row.get("id", idx)), row
    else:
        raise TypeError(f"unsupported annotation format: {type(data)!r}")


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def is_present(value: Any) -> bool:
    return value not in (None, "", [])


def load_motion_frame_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        data = np_load(path)
    except Exception:
        return None
    if isinstance(data, dict):
        for key in ("transl", "global_orient", "body_pose", "poses", "motion"):
            if key in data:
                return int(len(data[key]))
        if data:
            return int(len(next(iter(data.values()))))
        return None
    return int(data.shape[0])


def np_load(path: Path) -> Any:
    import numpy as np

    loaded = np.load(path, allow_pickle=False)
    if hasattr(loaded, "files"):
        return {key: loaded[key] for key in loaded.files}
    return loaded


def count_texts_from_caption(caption_obj: Any) -> Counter:
    counts: Counter = Counter()
    if not isinstance(caption_obj, dict):
        return counts
    for key in TEXT_GRANULARITIES:
        value = caption_obj.get(key)
        if isinstance(value, str) and value.strip():
            counts[key] += 1
        elif isinstance(value, list):
            counts[key] += sum(1 for item in value if isinstance(item, str) and item.strip())
    for key in ("action", "category", "complexity"):
        value = caption_obj.get(key)
        if isinstance(value, str) and value.strip():
            counts[key] += 1
    return counts


def read_caption_ref(args: Tuple[Path, int]) -> Tuple[Counter, int]:
    cap_path, ref_count = args
    if not cap_path.exists():
        return Counter(), ref_count
    counts = count_texts_from_caption(load_json(cap_path))
    multiplied = Counter()
    for key, value in counts.items():
        multiplied[key] = value * ref_count
    return multiplied, 0


def count_exact_caption_refs(data_root: Path, ref_counts: Counter, workers: int) -> Tuple[Counter, int]:
    text_counts: Counter = Counter()
    missing_refs = 0
    refs_by_parent: Dict[Path, List[Tuple[str, int]]] = defaultdict(list)
    for rel_path, count in ref_counts.items():
        refs_by_parent[(data_root / rel_path).parent].append((rel_path, count))

    for parent, refs in refs_by_parent.items():
        if not parent.exists():
            missing_refs += sum(count for _, count in refs)
            continue
        try:
            existing_names = {path.name for path in parent.iterdir()}
        except OSError:
            missing_refs += sum(count for _, count in refs)
            continue
        tasks_list = []
        for rel_path, count in refs:
            if rel_path.rsplit("/", 1)[-1] not in existing_names:
                missing_refs += count
            else:
                tasks_list.append((data_root / rel_path, count))
        if not tasks_list:
            continue
        if workers <= 1 or len(tasks_list) < 512:
            iterator = map(read_caption_ref, tasks_list)
            for counts, missing in iterator:
                text_counts.update(counts)
                missing_refs += missing
            continue
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for counts, missing in executor.map(read_caption_ref, tasks_list):
                text_counts.update(counts)
                missing_refs += missing
    return text_counts, missing_refs


def format_trainable_tasks(row: Dict[str, Any]) -> str:
    field_counts: Counter = row["field_counts"]
    text_counts: Counter = row["text_counts"]
    total_text = sum(text_counts.get(key, 0) for key in TEXT_GRANULARITIES)
    parts: List[str] = []
    if field_counts["hierarchical_caption_path"]:
        if total_text:
            parts.append(f"text-to-motion: {fmt_int(total_text)} prompts")
            parts.append(f"motion-to-text: {fmt_int(field_counts['hierarchical_caption_path'])} motions / {fmt_int(total_text)} refs")
        else:
            parts.append(f"text-to-motion: {fmt_int(field_counts['hierarchical_caption_path'])} caption refs")
            parts.append(f"motion-to-text: {fmt_int(field_counts['hierarchical_caption_path'])} motions")
    if field_counts["music_path"]:
        parts.append(f"music-to-dance: {fmt_int(field_counts['music_path'])} pairs")
    if field_counts["audio_path"]:
        parts.append(f"speech/audio-to-gesture: {fmt_int(field_counts['audio_path'])} pairs")
    if field_counts["speech_script_path"]:
        parts.append(f"script-to-gesture: {fmt_int(field_counts['speech_script_path'])} scripts")
    if field_counts["interactor_key"]:
        parts.append(f"interaction text-to-motion: {fmt_int(field_counts['interactor_key'])} pairs")
    return "<br>".join(parts) if parts else "-"


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_hours(seconds: float) -> str:
    hours = seconds / 3600.0
    if hours >= 1000:
        return f"{hours:,.0f}"
    if hours >= 100:
        return f"{hours:,.1f}"
    return f"{hours:,.2f}"


def split_counts(split_sample_counts: Dict[str, int]) -> str:
    parts = [f"{name}:{fmt_int(split_sample_counts.get(name, 0))}" for name in SPLITS if split_sample_counts.get(name, 0)]
    return "<br>".join(parts) if parts else "-"


def format_text_counts(texts: Counter) -> str:
    parts = [f"{key}: {fmt_int(texts.get(key, 0))}" for key in TEXT_GRANULARITIES]
    if texts.get("speech_script", 0):
        parts.append(f"speech: {fmt_int(texts['speech_script'])}")
    return "<br>".join(parts)


def task_totals_for_row(row: Dict[str, Any]) -> Dict[str, int]:
    field_counts: Counter = row["field_counts"]
    text_counts: Counter = row["text_counts"]
    total_text = sum(text_counts.get(key, 0) for key in TEXT_GRANULARITIES)
    totals: Dict[str, int] = {}
    if field_counts["hierarchical_caption_path"]:
        totals["text-to-motion prompts"] = total_text or field_counts["hierarchical_caption_path"]
        totals["motion-to-text references"] = total_text or field_counts["hierarchical_caption_path"]
    if field_counts["music_path"]:
        totals["music-to-dance pairs"] = field_counts["music_path"]
    if field_counts["audio_path"]:
        totals["speech/audio-to-gesture pairs"] = field_counts["audio_path"]
    if field_counts["speech_script_path"]:
        totals["script-to-gesture scripts"] = field_counts["speech_script_path"]
    if field_counts["interactor_key"]:
        totals["interaction text-to-motion pairs"] = field_counts["interactor_key"]
    return totals


def summarize_subset(
    subset_dir: Path,
    data_root: Path,
    exact_captions: bool,
    caption_workers: int,
    check_motion_files: bool,
) -> Dict[str, Any]:
    split_sample_counts: Dict[str, int] = {}
    split_annotation_frame_counts: Dict[str, int] = {}
    split_motion_frame_counts: Dict[str, int] = {}
    split_seconds: Dict[str, float] = {}
    field_counts: Counter = Counter()
    text_counts: Counter = Counter()
    caption_ref_counts: Counter = Counter()
    caption_parent_exists: Dict[str, bool] = {}
    missing_caption_refs = 0
    missing_motion_refs = 0
    missing_music_refs = 0
    skipped_invalid_rows = 0
    motion_cache: Dict[str, Optional[int]] = {}
    music_cache: Dict[str, bool] = {}
    unique_genres: set[str] = set()
    has_splits = False

    for split in SPLITS:
        path = subset_dir / f"{split}.json"
        if not path.exists():
            continue
        has_splits = True
        obj = load_json(path)
        for _, row in iter_rows(obj):
            if row.get("invalid") is True:
                skipped_invalid_rows += 1
                continue
            split_sample_counts[split] = split_sample_counts.get(split, 0) + 1
            frames = row.get("num_frames")
            fps = row.get("fps", 30)
            duration = row.get("duration")
            if frames is None and duration is not None:
                frames = int(round(float(duration) * float(fps or 30)))
            frames = int(frames or 0)
            split_annotation_frame_counts[split] = split_annotation_frame_counts.get(split, 0) + frames

            motion_frames = None
            if check_motion_files and row.get("smplx_path"):
                rel_motion_path = str(row["smplx_path"])
                if rel_motion_path not in motion_cache:
                    motion_cache[rel_motion_path] = load_motion_frame_count(data_root / rel_motion_path)
                motion_frames = motion_cache[rel_motion_path]
                if motion_frames is None:
                    missing_motion_refs += 1
            if check_motion_files and row.get("music_path"):
                for rel_music_path in [str(x) for x in as_list(row.get("music_path")) if x]:
                    if rel_music_path not in music_cache:
                        music_cache[rel_music_path] = (data_root / rel_music_path).exists()
                    if not music_cache[rel_music_path]:
                        missing_music_refs += 1
            split_motion_frame_counts[split] = split_motion_frame_counts.get(split, 0) + int(motion_frames if motion_frames is not None else frames)
            split_seconds[split] = split_seconds.get(split, 0.0) + float(motion_frames if motion_frames is not None else frames) / float(fps or 30)

            for key, value in row.items():
                if is_present(value):
                    field_counts[key] += 1
            if is_present(row.get("genre")):
                unique_genres.add(str(row["genre"]))

            if row.get("hierarchical_caption_path"):
                paths = [str(x) for x in as_list(row.get("hierarchical_caption_path")) if x]
                if exact_captions:
                    for rel_path in paths:
                        parent_key = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
                        parent_exists = caption_parent_exists.get(parent_key)
                        if parent_exists is None:
                            parent_exists = (data_root / parent_key).exists()
                            caption_parent_exists[parent_key] = parent_exists
                        if parent_exists:
                            caption_ref_counts[rel_path] += 1
                        else:
                            missing_caption_refs += 1
                else:
                    for key in TEXT_GRANULARITIES:
                        text_counts[key] += len(paths)
            if row.get("speech_script_path"):
                text_counts["speech_script"] += len([x for x in as_list(row.get("speech_script_path")) if x])

    if exact_captions and caption_ref_counts:
        exact_counts, exact_missing_caption_refs = count_exact_caption_refs(data_root, caption_ref_counts, caption_workers)
        missing_caption_refs += exact_missing_caption_refs
        text_counts.update(exact_counts)

    return {
        "name": subset_dir.name,
        "has_splits": has_splits,
        "samples": sum(split_sample_counts.values()),
        "split_samples": split_sample_counts,
        "frames": sum(split_motion_frame_counts.values()),
        "seconds": sum(split_seconds.values()),
        "annotation_frames": sum(split_annotation_frame_counts.values()),
        "motion_files_checked": len(motion_cache),
        "music_files_checked": len(music_cache),
        "missing_motion_refs": missing_motion_refs,
        "missing_music_refs": missing_music_refs,
        "skipped_invalid_rows": skipped_invalid_rows,
        "field_counts": field_counts,
        "text_counts": text_counts,
        "missing_caption_refs": missing_caption_refs,
        "unique_genres": sorted(unique_genres),
        "caption_mode": "exact" if exact_captions else "fast",
    }


def make_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| Dataset | Splits | Clips | Motion files | Frames | Hours | "
        "Music files | Invalid skipped | Trainable tasks | Text counts | Missing refs |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"
    )
    lines = [header]
    for row in rows:
        texts = row["text_counts"]
        motion_missing = row.get("missing_motion_refs", 0)
        music_missing = row.get("missing_music_refs", 0)
        caption_missing = row.get("missing_caption_refs", 0) if row.get("caption_mode") == "exact" else "not checked"
        missing = (
            f"motion: {fmt_int(motion_missing)}<br>"
            f"music: {fmt_int(music_missing)}<br>"
            f"caption: {fmt_int(caption_missing) if isinstance(caption_missing, int) else caption_missing}"
        )
        lines.append(
            "| {name} | {splits} | {clips} | {motion_files} | {frames} | {hours} | {music_files} | {invalid} | {tasks} | {texts} | {missing} |".format(
                name=row["name"],
                splits=split_counts(row["split_samples"]),
                clips=fmt_int(row["samples"]),
                motion_files=fmt_int(row.get("motion_files_checked", 0)),
                frames=fmt_int(row["frames"]),
                hours=fmt_hours(row["seconds"]),
                music_files=fmt_int(row.get("music_files_checked", 0)),
                invalid=fmt_int(row.get("skipped_invalid_rows", 0)),
                tasks=format_trainable_tasks(row),
                texts=format_text_counts(texts),
                missing=missing,
            )
        )
    return "\n".join(lines)


def make_readme(
    data_root: Path,
    rows: List[Dict[str, Any]],
    skipped: List[str],
    exact_captions: bool,
    published: set[str],
    include_staged: bool,
) -> str:
    total_samples = sum(r["samples"] for r in rows)
    total_frames = sum(r["frames"] for r in rows)
    total_seconds = sum(r["seconds"] for r in rows)
    total_annotation_frames = sum(r["annotation_frames"] for r in rows)
    total_motion_files = sum(r.get("motion_files_checked", 0) for r in rows)
    total_music_files = sum(r.get("music_files_checked", 0) for r in rows)
    total_missing_motion_refs = sum(r.get("missing_motion_refs", 0) for r in rows)
    total_missing_music_refs = sum(r.get("missing_music_refs", 0) for r in rows)
    total_skipped_invalid_rows = sum(r.get("skipped_invalid_rows", 0) for r in rows)
    total_texts = Counter()
    total_missing_caption_refs = 0
    exact_subsets = []
    task_totals: Counter = Counter()
    for row in rows:
        total_texts.update(row["text_counts"])
        total_missing_caption_refs += row.get("missing_caption_refs", 0)
        if row.get("caption_mode") == "exact":
            exact_subsets.append(row["name"])
        task_totals.update(task_totals_for_row(row))

    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- other",
        "tags:",
        "- motion",
        "- smplx",
        "- motionhub",
        "- text-to-motion",
        "- music-to-dance",
        "pretty_name: MotionHub",
        "---",
        "",
        "# MotionHub",
        "",
        "MotionHub is organized as a collection of motion subsets with MotionHub-style `train.json` / `test.json` annotations.",
        "",
        "This README lists only subsets that have already been uploaded to this Hugging Face dataset after visual inspection and data-quality review. New subsets should be added here one by one when they are published.",
        "",
        "## Summary",
        "",
        f"- Generated on: `{date.today().isoformat()}`",
        f"- Published subsets included: `{len(rows)}`",
        f"- Total clips: `{fmt_int(total_samples)}`",
        f"- Total motion files checked: `{fmt_int(total_motion_files)}`",
        f"- Total music files checked: `{fmt_int(total_music_files)}`",
        f"- Total motion frames: `{fmt_int(total_frames)}`",
        f"- Total motion duration: `{fmt_hours(total_seconds)}` hours",
        f"- Annotation/motion frame difference: `{fmt_int(total_frames - total_annotation_frames)}`",
        f"- Invalid rows skipped: `{fmt_int(total_skipped_invalid_rows)}`",
        f"- Caption counting mode: `exact caption JSON parsing`",
        f"- Exact caption subsets: `{', '.join(sorted(exact_subsets)) if exact_subsets else 'none'}`",
        f"- Missing motion references: `{fmt_int(total_missing_motion_refs)}`",
        f"- Missing music references: `{fmt_int(total_missing_music_refs)}`",
        f"- Missing hierarchical caption references: `{fmt_int(total_missing_caption_refs)}`",
        f"- Published subsets: `{', '.join(sorted(published)) if published else 'none'}`",
        "",
        "## Dataset Table",
        "",
        make_table(rows),
        "",
        "## Trainable Task Totals",
        "",
    ]
    for task, count in sorted(task_totals.items()):
        lines.append(f"- `{task}`: {fmt_int(count)}")
    lines += [
        "",
        "## Text Granularity Totals",
        "",
        f"- `macro`: {fmt_int(total_texts.get('macro', 0))}",
        f"- `meso`: {fmt_int(total_texts.get('meso', 0))}",
        f"- `micro`: {fmt_int(total_texts.get('micro', 0))}",
        f"- `speech_script`: {fmt_int(total_texts.get('speech_script', 0))}",
        "",
        "## Counting Rules",
        "",
        "- Annotation frame counts are read from `num_frames` when available; otherwise duration and fps are used.",
        "- For published rows, motion volume is cross-checked against referenced `smplx_path` files; frame totals in the public table use motion-file lengths when available.",
        "- Hours use motion frames divided by each annotation row's fps.",
        "- `macro`, `meso`, and `micro` count non-empty strings parsed from hierarchical-caption JSON files.",
        "- `Missing caption refs` counts annotation references whose hierarchical-caption JSON could not be found.",
        "- `speech_script` counts non-empty `speech_script_path` entries.",
        "- Trainable tasks are counted only from explicit cross-modal supervision fields: captions enable text-to-motion and motion-to-text, music enables music-to-dance, speech audio/transcripts enable gesture tasks, and `interactor_key` plus captions mark interaction text-to-motion.",
        "- `smplx_path` is treated as the motion asset used to verify sample availability and frame counts; it is not counted as an independent trainable task.",
        "",
        "## Maintenance",
        "",
        "Regenerate this file from the repository root with:",
        "",
        "```bash",
        "python3 tools/build_motionhub_readme.py --data-root data/motionhub --output data/motionhub/README.md",
        "```",
        "",
        "Unpublished local subsets are intentionally omitted from the public README. For internal inventory previews, add `--include-staged`; do not upload that preview as the public dataset card.",
    ]
    if include_staged and skipped:
        lines += [
            "",
            "Directories without `train.json`, `val.json`, or `test.json` are not included in the table:",
            "",
            "- " + ", ".join(f"`{x}`" for x in skipped),
        ]
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/motionhub")
    parser.add_argument("--output", default="data/motionhub/README.md")
    parser.add_argument("--exact-captions", action="store_true", help="Force exact caption JSON parsing for every subset.")
    parser.add_argument("--fast-captions", action="store_true", help="Count caption references without opening caption JSON files.")
    parser.add_argument("--include-staged", action="store_true", help="Include unpublished local subsets for internal inventory preview.")
    parser.add_argument(
        "--exact-subsets",
        default=None,
        help="Comma-separated subset directories to count exactly. Defaults to published subsets.",
    )
    parser.add_argument("--caption-workers", type=int, default=min(32, (os.cpu_count() or 8) * 2))
    parser.add_argument(
        "--published-subsets",
        default="aist",
        help="Comma-separated subset directories already published in the public dataset repo.",
    )
    parser.add_argument("--skip-motion-file-check", action="store_true", help="Use annotation frame counts without loading motion files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    published = {x.strip() for x in args.published_subsets.split(",") if x.strip()}
    exact_subsets = (
        {x.strip() for x in args.exact_subsets.split(",") if x.strip()}
        if args.exact_subsets is not None
        else set(published)
    )
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for subset in sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not args.include_staged and subset.name not in published:
            continue
        print(f"[subset] {subset.name}", file=sys.stderr, flush=True)
        exact_for_subset = False
        if args.exact_captions:
            exact_for_subset = True
        elif not args.fast_captions:
            exact_for_subset = subset.name in exact_subsets
        row = summarize_subset(
            subset,
            data_root,
            exact_captions=exact_for_subset,
            caption_workers=max(1, args.caption_workers),
            check_motion_files=exact_for_subset and not args.skip_motion_file_check,
        )
        if row["has_splits"]:
            rows.append(row)
            print(
                f"[done] {subset.name} samples={row['samples']} frames={row['frames']} "
                f"missing_caption_refs={row.get('missing_caption_refs', 0)}",
                file=sys.stderr,
                flush=True,
            )
        else:
            skipped.append(subset.name)
    readme = make_readme(
        data_root,
        rows,
        skipped,
        exact_captions=not args.fast_captions,
        published=published,
        include_staged=args.include_staged,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(readme, encoding="utf-8")
    print(f"[write] {out_path} subsets={len(rows)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit generated result directories and optionally quarantine safe junk.

The script is intentionally conservative.  It never deletes data.  With
--apply, it moves quarantine candidates to .trash/result_cleanup_<timestamp>/,
preserving the original relative path so recovery is straightforward.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_TOP_KEEP = {
    "evaluation",
    "inference",
    "visualization",
    "conversion",
    "diagnostics",
    "tmp",
    "_archive",
    "logs",
    "data_processing",
}

CANONICAL_EVAL_TASKS = {
    "t2m",
    "m2m",
    "semantic_edit",
    "repair",
    "control",
    "interaction_t2m",
    "retarget",
    "embodied_tracking",
    "physics_eval",
    "babel",
    "babel_seq",
}

JUNK_NAME_RE = re.compile(
    r"(^|[_\-.])(tmp|temp|debug|smoke|scratch|dryrun|dry|quickcheck|wrong|invalid|broken|failed|nan|cache)([_\-.]|$)",
    re.IGNORECASE,
)

LEGACY_HINT_RE = re.compile(
    r"(rerun|fix\d*|pathfix|rootfix|fpsfix|pyfix|pipfix|depfix|statsfix|vaefix|rootalign)",
    re.IGNORECASE,
)


@dataclass
class Entry:
    path: str
    kind: str
    mtime: str
    age_hours: float
    immediate_children: int
    has_checkpoint: bool
    active: bool
    disposition: str
    reason: str
    moved_to: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--out-jsonl", default="", help="Write full manifest JSONL.")
    parser.add_argument("--out-csv", default="", help="Write full manifest CSV.")
    parser.add_argument("--max-depth", type=int, default=3, help="Scan depth below outputs/work_dirs.")
    parser.add_argument(
        "--protect-recent-hours",
        type=float,
        default=24.0,
        help="Never quarantine paths modified within this many hours.",
    )
    parser.add_argument(
        "--candidate-min-age-hours",
        type=float,
        default=24.0,
        help="Only mark junk-like paths older than this as quarantine candidates.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move quarantine candidates into .trash instead of only reporting.",
    )
    parser.add_argument(
        "--quarantine-root",
        default="",
        help="Destination root for --apply. Defaults to .trash/result_cleanup_<timestamp>.",
    )
    return parser.parse_args()


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def now_ts() -> datetime:
    return datetime.now(timezone.utc)


def iter_paths(base: Path, max_depth: int) -> list[Path]:
    if not base.exists():
        return []
    out: list[Path] = []
    base_depth = len(base.parts)
    for cur, dirs, files in os.walk(base):
        cur_path = Path(cur)
        depth = len(cur_path.parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if depth > 0:
            out.append(cur_path)
        if depth == 0:
            out.extend(cur_path / f for f in files)
        if depth >= max_depth:
            dirs[:] = []
    return sorted(out)


def get_ps_text() -> str:
    try:
        result = subprocess.run(
            ["ps", "-eo", "args"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return result.stdout


def is_active(path: Path, root: Path, ps_text: str) -> bool:
    rel = relpath(path, root)
    abs_path = str(path)
    alt_abs = abs_path.replace("/apdcephfs/AILab_DHA/", "/")
    candidates = {rel, f"./{rel}", abs_path, alt_abs}
    return any(item and item in ps_text for item in candidates)


def has_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.name.startswith("checkpoint-"):
                return True
    except OSError:
        return False
    return False


def immediate_children(path: Path) -> int:
    if not path.is_dir():
        return 0
    try:
        return sum(1 for _ in path.iterdir())
    except OSError:
        return 0


def classify(path: Path, root: Path, age_hours: float, active: bool, checkpoint: bool, protect_recent: float, candidate_min_age: float) -> tuple[str, str]:
    rel = relpath(path, root)
    parts = rel.split("/")
    name = path.name

    if active:
        return "protect", "referenced_by_live_process"
    if age_hours < protect_recent:
        return "protect", f"recent_mtime_lt_{protect_recent:g}h"

    if rel in {"outputs/tmp", "outputs/evaluation", "outputs/diagnostics", "outputs/visualization", "outputs/inference"}:
        return "keep", "container_directory"

    if rel.startswith("outputs/tmp/"):
        return "quarantine_candidate", "outputs_tmp_is_disposable"
    if rel in {"outputs/.numba_cache", "outputs/debug"}:
        return "quarantine_candidate", "cache_or_legacy_debug_output"

    if rel.startswith("outputs/evaluation/"):
        eval_parts = parts[2:]
        if eval_parts:
            task = eval_parts[0]
            if task.startswith("_tmp") or task == "_tmp":
                return "quarantine_candidate", "evaluation_tmp"
            if task not in CANONICAL_EVAL_TASKS and len(eval_parts) == 1:
                if JUNK_NAME_RE.search(task) or LEGACY_HINT_RE.search(task):
                    return "review", "legacy_top_level_eval_junk_name"
                return "review", "legacy_top_level_eval_not_canonical"
        if JUNK_NAME_RE.search(name):
            return "quarantine_candidate", "junk_like_name_under_evaluation"
        if age_hours >= max(candidate_min_age, 24.0) and LEGACY_HINT_RE.search(name):
            return "review", "legacy_fix_or_rerun_name_under_evaluation"
        return "keep", "evaluation_candidate_or_canonical"

    if rel.startswith("outputs/diagnostics/"):
        if age_hours >= 24 * 7:
            return "review", "diagnostic_older_than_7d"
        return "keep", "diagnostic_recent"

    if rel.startswith("outputs/") and len(parts) == 2:
        if parts[1] not in OUTPUT_TOP_KEEP:
            return "review", "unexpected_outputs_top_level"
        return "keep", "outputs_top_level"

    if rel.startswith("work_dirs/"):
        if path.is_file():
            if JUNK_NAME_RE.search(name) and age_hours >= candidate_min_age:
                return "quarantine_candidate", "loose_junk_like_work_dirs_file"
            return "review", "loose_file_in_work_dirs_root"
        if checkpoint:
            if "wrong" in name.lower() or "archived_wrong" in name.lower():
                return "quarantine_candidate", "wrong_training_source_checkpoint_dir"
            return "review", "training_dir_with_checkpoints"
        if JUNK_NAME_RE.search(name) and age_hours >= candidate_min_age:
            return "quarantine_candidate", "junk_like_work_dir_without_top_checkpoint"
        if "OLD" in name or "old" in name:
            return "quarantine_candidate", "old_named_work_dir"
        return "review", "work_dir_needs_owner_decision"

    return "review", "unclassified"


def build_manifest(args: argparse.Namespace) -> tuple[list[Entry], Path]:
    root = Path(args.root).resolve()
    scan_paths = iter_paths(root / "outputs", args.max_depth) + iter_paths(root / "work_dirs", 1)
    ps_text = get_ps_text()
    current = now_ts()
    entries: list[Entry] = []

    for path in scan_paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        mtime_dt = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        age_hours = max(0.0, (current - mtime_dt).total_seconds() / 3600.0)
        active = is_active(path, root, ps_text)
        checkpoint = has_checkpoint(path)
        disposition, reason = classify(
            path,
            root,
            age_hours,
            active,
            checkpoint,
            args.protect_recent_hours,
            args.candidate_min_age_hours,
        )
        entries.append(
            Entry(
                path=relpath(path, root),
                kind="dir" if path.is_dir() else "file",
                mtime=mtime_dt.astimezone().isoformat(timespec="seconds"),
                age_hours=round(age_hours, 2),
                immediate_children=immediate_children(path),
                has_checkpoint=checkpoint,
                active=active,
                disposition=disposition,
                reason=reason,
            )
        )

    return entries, root


def prune_nested_candidates(entries: list[Entry]) -> list[Entry]:
    selected = [e for e in entries if e.disposition == "quarantine_candidate"]
    selected_paths = sorted(e.path for e in selected)
    blocked: set[str] = set()
    for path in selected_paths:
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        while parent:
            if parent in selected_paths:
                blocked.add(path)
                break
            parent = parent.rsplit("/", 1)[0] if "/" in parent else ""
    return [e for e in selected if e.path not in blocked]


def apply_quarantine(entries: list[Entry], root: Path, quarantine_root: Path) -> None:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    for entry in prune_nested_candidates(entries):
        src = root / entry.path
        if not src.exists():
            continue
        dst = quarantine_root / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            suffix = datetime.now().strftime("%H%M%S")
            dst = dst.with_name(f"{dst.name}.{suffix}")
        shutil.move(str(src), str(dst))
        entry.moved_to = relpath(dst, root)


def write_outputs(entries: list[Entry], args: argparse.Namespace, root: Path) -> None:
    if args.out_jsonl:
        out = root / args.out_jsonl
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
    if args.out_csv:
        out = root / args.out_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(entries[0]).keys()) if entries else list(Entry.__dataclass_fields__.keys()))
            writer.writeheader()
            for entry in entries:
                writer.writerow(asdict(entry))


def print_summary(entries: list[Entry]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (entry.disposition, entry.reason)
        counts[key] = counts.get(key, 0) + 1
    print("Summary:")
    for (disposition, reason), count in sorted(counts.items()):
        print(f"  {disposition:22s} {count:5d}  {reason}")
    print()
    print("Quarantine candidates:")
    for entry in prune_nested_candidates(entries)[:200]:
        print(f"  {entry.path}  [{entry.reason}]")


def main() -> int:
    args = parse_args()
    entries, root = build_manifest(args)
    quarantine_root = Path(args.quarantine_root)
    if args.apply:
        if not quarantine_root:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_root = root / ".trash" / f"result_cleanup_{stamp}"
        elif not quarantine_root.is_absolute():
            quarantine_root = root / quarantine_root
        apply_quarantine(entries, root, quarantine_root)
    write_outputs(entries, args, root)
    print_summary(entries)
    if args.apply:
        print(f"\nMoved candidates to: {relpath(quarantine_root, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

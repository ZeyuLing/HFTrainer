#!/usr/bin/env python3
"""Upload a local folder tree to a Hugging Face repo in resumable batches."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import time
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import CommitOperationAdd, HfApi


def chunks(items: List[Path], size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def is_ignored(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def collect_files(folder: Path, ignore_patterns: list[str]) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        rel_dir = Path(dirpath).relative_to(folder).as_posix()
        if rel_dir == ".":
            rel_dir = ""
        keep_dirs = []
        for dirname in dirnames:
            rel = f"{rel_dir}/{dirname}".strip("/")
            if is_ignored(rel, ignore_patterns) or is_ignored(f"{rel}/", ignore_patterns):
                continue
            keep_dirs.append(dirname)
        dirnames[:] = keep_dirs
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(folder).as_posix()
            if is_ignored(rel, ignore_patterns):
                continue
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(folder).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--path-prefix", default="", help="Optional path prefix inside the repo.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-between-batches", type=float, default=0.0)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    path_prefix = args.path_prefix.strip("/")
    api = HfApi()
    api.update_repo_settings(repo_id=args.repo_id, repo_type=args.repo_type, private=bool(args.private))

    files = collect_files(folder, args.ignore_pattern)
    total_batches = (len(files) + args.batch_size - 1) // args.batch_size
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "folder": str(folder),
        "path_prefix": path_prefix,
        "batch_size": args.batch_size,
        "num_files": len(files),
        "total_batches": total_batches,
        "completed_batches": [],
        "skipped_batches": [],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "folder": str(folder),
                "num_files": len(files),
                "total_batches": total_batches,
                "batch_size": args.batch_size,
            },
            indent=2,
        ),
        flush=True,
    )

    for batch_idx, batch in enumerate(chunks(files, args.batch_size), start=1):
        if batch_idx < args.start_batch:
            print(f"[skip] batch {batch_idx}/{total_batches}", flush=True)
            report["skipped_batches"].append(batch_idx)
            continue
        operations = []
        for path in batch:
            rel = path.relative_to(folder).as_posix()
            path_in_repo = f"{path_prefix}/{rel}" if path_prefix else rel
            operations.append(CommitOperationAdd(path_or_fileobj=str(path), path_in_repo=path_in_repo))

        for attempt in range(1, args.max_retries + 1):
            try:
                print(
                    f"[commit] batch {batch_idx}/{total_batches} ops={len(operations)} attempt={attempt}",
                    flush=True,
                )
                info = api.create_commit(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    operations=operations,
                    commit_message=f"Upload MotionHub batch {batch_idx}/{total_batches}",
                    num_threads=args.num_threads,
                )
                print(f"[commit] done batch {batch_idx}/{total_batches} oid={info.oid}", flush=True)
                report["completed_batches"].append(
                    {
                        "batch": batch_idx,
                        "ops": len(operations),
                        "oid": info.oid,
                        "first": operations[0].path_in_repo if operations else None,
                        "last": operations[-1].path_in_repo if operations else None,
                    }
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                break
            except Exception as exc:  # noqa: BLE001 - upload retries need to catch API/network errors.
                msg = str(exc)
                if "No files have been modified" in msg or "empty commit" in msg.lower():
                    print(f"[commit] no changes batch {batch_idx}/{total_batches}", flush=True)
                    report["completed_batches"].append(
                        {"batch": batch_idx, "ops": len(operations), "oid": None, "no_changes": True}
                    )
                    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    break
                print(f"[retry] batch {batch_idx}/{total_batches} attempt={attempt} error={type(exc).__name__}: {exc}", flush=True)
                if attempt >= args.max_retries:
                    raise
                time.sleep(args.retry_sleep * attempt)
        if args.sleep_between_batches > 0 and batch_idx < total_batches:
            time.sleep(args.sleep_between_batches)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()

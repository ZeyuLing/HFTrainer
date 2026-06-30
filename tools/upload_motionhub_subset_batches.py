#!/usr/bin/env python3
"""Upload one MotionHub subset to Hugging Face with explicit repo paths.

``HfApi.upload_large_folder`` does not support ``path_in_repo`` in the version
available in this environment.  This script uses batched commits instead, so
each local file is mapped explicitly to:

    {subset}/{motion_dir}/*.npz
    {subset}/stats.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.utils import HfHubHTTPError


def chunks(items: List, size: int) -> Iterable[List]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def subset_add_ops(
    subset_root: Path,
    subset_name: str,
    motion_dir: str,
    skip_ops: int = 0,
    recursive: bool = False,
    include_root_json: bool = False,
) -> List[CommitOperationAdd]:
    if recursive:
        files = sorted(
            path
            for path in subset_root.rglob("*.npz")
            if motion_dir in path.relative_to(subset_root).parts
        )
    else:
        files = sorted((subset_root / motion_dir).glob("*.npz"))
    if not files:
        suffix = f"**/{motion_dir}" if recursive else motion_dir
        raise FileNotFoundError(subset_root / suffix)
    stats = subset_root / "stats.json"
    if not stats.exists():
        raise FileNotFoundError(stats)
    root_json = []
    if include_root_json:
        root_json = sorted(path for path in subset_root.glob("*.json") if path.name != "stats.json")
    paths = list(files) + root_json + [stats]
    if skip_ops > 0:
        paths = paths[skip_ops:]
    ops: List[CommitOperationAdd] = []
    for path in paths:
        if path == stats:
            path_in_repo = f"{subset_name}/stats.json"
        elif path in root_json:
            path_in_repo = f"{subset_name}/{path.name}"
        else:
            path_in_repo = f"{subset_name}/{path.relative_to(subset_root).as_posix()}"
        ops.append(CommitOperationAdd(path_or_fileobj=str(path), path_in_repo=path_in_repo))
    return ops


def delete_remote_motion_dir_ops(
    api: HfApi,
    repo_id: str,
    subset_name: str,
    motion_dir: str,
) -> List[CommitOperationDelete]:
    remote_dir = f"{subset_name}/{motion_dir}"
    try:
        if "/" in motion_dir:
            entries = list(
                api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=remote_dir,
                    recursive=True,
                )
            )
        else:
            entries = [
                entry
                for entry in api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=subset_name,
                    recursive=True,
                )
                if motion_dir in Path(getattr(entry, "path", "")).parts
            ]
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return []
        raise
    paths = [entry.path for entry in entries if getattr(entry, "path", "")]
    return [CommitOperationDelete(path_in_repo=path) for path in sorted(paths)]


def cleanup_root_ops(api: HfApi, repo_id: str) -> List[CommitOperationDelete]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    root_bad = [path for path in files if path == "stats.json" or path.startswith("smplx_55/")]
    return [CommitOperationDelete(path_in_repo=path) for path in sorted(root_bad)]


def commit_batches(
    api: HfApi,
    repo_id: str,
    ops: List,
    batch_size: int,
    message_prefix: str,
    start_batch: int,
    num_threads: int,
    sleep_between_batches: float,
) -> None:
    total_batches = (len(ops) + batch_size - 1) // batch_size
    for batch_idx, batch in enumerate(chunks(ops, batch_size), start=1):
        if batch_idx < start_batch:
            print(f"[skip] {message_prefix} batch {batch_idx}/{total_batches}", flush=True)
            continue
        print(
            f"[commit] {message_prefix} batch {batch_idx}/{total_batches} ops={len(batch)}",
            flush=True,
        )
        info = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=batch,
            commit_message=f"{message_prefix} ({batch_idx}/{total_batches})",
            num_threads=num_threads,
        )
        print(f"[commit] done {info.oid}", flush=True)
        if sleep_between_batches > 0 and batch_idx < total_batches:
            print(f"[sleep] {sleep_between_batches:.1f}s", flush=True)
            time.sleep(sleep_between_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ZeyuLing/MotionHub")
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--motion-dir", default="smplx_55")
    parser.add_argument("--recursive", action="store_true", help="Upload **/{motion_dir}/*.npz recursively.")
    parser.add_argument("--include-root-json", action="store_true", help="Upload root *.json annotations too.")
    parser.add_argument(
        "--skip-upload-ops",
        type=int,
        default=0,
        help="Skip this many add operations before constructing upload ops. Useful for resumable uploads.",
    )
    parser.add_argument(
        "--delete-remote-motion-dir",
        action="append",
        default=[],
        help="Remote subset motion directory to delete after uploading. May repeat.",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--sleep-between-batches", type=float, default=0.0)
    parser.add_argument("--cleanup-root-artifacts", action="store_true")
    parser.add_argument("--cleanup-batch-size", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset", private=False)

    if args.cleanup_root_artifacts:
        delete_ops = cleanup_root_ops(api, args.repo_id)
        print(f"[cleanup] root artifacts ops={len(delete_ops)}", flush=True)
        if delete_ops:
            commit_batches(
                api,
                args.repo_id,
                delete_ops,
                args.cleanup_batch_size,
                "Remove mistaken root MotionHub SMPL-X upload",
                1,
                args.num_threads,
                args.sleep_between_batches,
            )

    subset_root = Path(args.subset_root)
    add_ops = subset_add_ops(
        subset_root,
        args.subset_name,
        args.motion_dir,
        skip_ops=args.skip_upload_ops,
        recursive=args.recursive,
        include_root_json=args.include_root_json,
    )
    print(
        f"[upload] subset={args.subset_name} motion_dir={args.motion_dir} "
        f"skip_ops={args.skip_upload_ops} ops={len(add_ops)}",
        flush=True,
    )
    commit_batches(
        api,
        args.repo_id,
        add_ops,
        args.batch_size,
        f"Update {args.subset_name} {args.motion_dir} and stats",
        args.start_batch,
        args.num_threads,
        args.sleep_between_batches,
    )

    for remote_motion_dir in args.delete_remote_motion_dir:
        delete_ops = delete_remote_motion_dir_ops(
            api,
            args.repo_id,
            args.subset_name,
            remote_motion_dir,
        )
        print(
            f"[delete] subset={args.subset_name} remote_dir={remote_motion_dir} ops={len(delete_ops)}",
            flush=True,
        )
        if delete_ops:
            commit_batches(
                api,
                args.repo_id,
                delete_ops,
                args.cleanup_batch_size,
                f"Remove {args.subset_name} {remote_motion_dir}",
                1,
                args.num_threads,
                args.sleep_between_batches,
            )


if __name__ == "__main__":
    main()

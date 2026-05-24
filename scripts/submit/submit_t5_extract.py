#!/usr/bin/env python3
"""Submit T5 extraction shards to Taiji elastic GPU pool.

Submits N parallel single-GPU jobs, each processing a slice of unique caption
files. Uses elastic (preemptible) GPUs for cost efficiency.

Usage:
    python scripts/submit/submit_t5_extract.py --num-shards 64
    python scripts/submit/submit_t5_extract.py --num-shards 64 --start-shard 32  # resume from shard 32
    python scripts/submit/submit_t5_extract.py --num-shards 64 --dry-run        # print commands only
"""
import argparse
import os
import sys
import time

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from tools.taiji_submit import submit


def main():
    parser = argparse.ArgumentParser(
        description="Submit T5 feature extraction shards to Taiji"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=64,
        help="Total number of shards (parallel jobs)",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Starting shard index (for resuming partial submissions)",
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=None,
        help="Ending shard index (exclusive). Defaults to num-shards.",
    )
    parser.add_argument(
        "--anno",
        type=str,
        default="data/annotation/train_hq_motionhub_hymotion.json",
        help="Annotation file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/t5_feature",
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/Wan2.1-VACE-1.3B-diffusers",
        help="Model path (tokenizer + text_encoder)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for T5 encoding per job",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting",
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        default="t5ext",
        help="Task name prefix",
    )
    args = parser.parse_args()

    end_shard = args.end_shard if args.end_shard is not None else args.num_shards

    print(f"=== T5 Feature Extraction Submission ===")
    print(f"Shards: {args.start_shard} to {end_shard - 1} (of {args.num_shards} total)")
    print(f"Annotation: {args.anno}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max seq length: {args.max_seq_length}")
    print()

    submitted = []
    failed = []

    for shard_id in range(args.start_shard, end_shard):
        task_flag = f"{args.task_prefix}_s{shard_id:03d}"

        # Build the extraction command
        start_cmd = (
            f"cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ && "
            f"python3 scripts/data/extract_t5_features.py "
            f"--anno {args.anno} "
            f"--data-dir data/motionhub "
            f"--output-dir {args.output_dir} "
            f"--model-path {args.model_path} "
            f"--shard-id {shard_id} "
            f"--num-shards {args.num_shards} "
            f"--batch-size {args.batch_size} "
            f"--max-seq-length {args.max_seq_length}"
        )

        if args.dry_run:
            print(f"[DRY RUN] Shard {shard_id:03d}: {task_flag}")
            print(f"  cmd: {start_cmd}")
            print()
            continue

        try:
            print(f"Submitting shard {shard_id:03d}/{args.num_shards}...")
            submit(
                task_flag=task_flag,
                config_path="__UNUSED__",
                host_num=1,
                elastic=True,
                start_cmd_override=start_cmd,
                host_gpu_num=1,
            )
            submitted.append(task_flag)
            # Small delay to avoid API rate limiting
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR submitting shard {shard_id}: {e}")
            failed.append((shard_id, str(e)))

    print()
    print(f"=== Summary ===")
    print(f"Submitted: {len(submitted)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for shard_id, err in failed:
            print(f"  Shard {shard_id}: {err}")

    print()
    print("Monitor all shards:")
    print(f"  taiji_client trl | grep {args.task_prefix}")
    print()
    print("Stop all shards:")
    print(f"  for i in $(seq {args.start_shard} {end_shard - 1}); do")
    print(f'    taiji_client stop {args.task_prefix}_s$(printf "%03d" $i)')
    print(f"  done")


if __name__ == "__main__":
    main()

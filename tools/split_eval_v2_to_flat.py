#!/usr/bin/env python3
"""Split nested eval_v2 JSON into per-(model, task, setting) flat JSONs
suitable for motion_annot_web/eval_dashboard/data_importer.py.

Input:   a directory containing eval_v2_*.json files (one per model run)
Output:  <out_dir>/<model>__<task_id>_<setting>.json

Usage:
  python3 tools/split_eval_v2_to_flat.py \
      --in-dir work_dirs/caption_rewritten_20260421 \
      --out-dir work_dirs/caption_rewritten_20260421/import_jsons \
      --timestamp "2026-04-21 12:15:00"
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def split_one(nested_path: Path, out_dir: Path, timestamp: str) -> int:
    with open(nested_path) as f:
        nested = json.load(f)

    written = 0
    for model_name, model_block in nested.items():
        if not isinstance(model_block, dict) or "tasks" not in model_block:
            continue
        checkpoint = model_block.get("checkpoint", "")
        rotation_space = model_block.get("rotation_space", "local")
        tasks = model_block.get("tasks", {})
        has_caption = "caption" in model_name.lower()

        for task_key, entry in tasks.items():
            # task_key example: "E2_A", "E4_C_rhand_lfoot", "E10_A_upper"
            tid = entry.get("task_id")
            setting = entry.get("setting")
            if not tid or setting is None:
                continue
            agg = entry.get("aggregated", {})
            samples = entry.get("per_sample", [])
            num = entry.get("num_samples", len(samples))

            flat = {
                "model": model_name,
                "checkpoint": checkpoint,
                "rotation_space": rotation_space,
                "has_caption": has_caption,
                "timestamp": timestamp,
                "task_id": tid,
                "setting": setting,
                "num_prompts": num,
                "aggregated": agg,
                "per_sample": samples,
            }
            out_name = f"{model_name}__{tid}_{setting}.json"
            out_path = out_dir / out_name
            with open(out_path, "w") as g:
                json.dump(flat, g, ensure_ascii=False, indent=2)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True,
                        help="Directory containing eval_v2_*.json files (recursively)")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write per-task flat JSONs")
    parser.add_argument("--timestamp", default=None,
                        help="Timestamp to stamp into flat JSONs (default: now)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = args.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    nested_files = sorted(in_dir.rglob("eval_v2_*.json"))
    print(f"Found {len(nested_files)} nested eval_v2 JSONs under {in_dir}")
    total = 0
    for p in nested_files:
        n = split_one(p, out_dir, ts)
        print(f"  {p.relative_to(in_dir)} -> {n} flat JSONs")
        total += n
    print(f"\nWrote {total} flat per-(model,task,setting) JSONs to {out_dir}")


if __name__ == "__main__":
    main()

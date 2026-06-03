#!/usr/bin/env python3
"""Build the PhysFlow online-adversarial TEXT corpus from HumanML3D captions.

The PhysFlow "training data" is purely *text*: at train time KIMODO-G1 generates
the driving motion on the fly. This script therefore collects a large, diverse
text corpus and carves a FROZEN held-out eval split (never used for training the
loop), so generator/tracker improvements can be judged without leakage.

Source: data/hymotion_data/Academic/20250916/raw_caption/HumanML3D-<DATASET>/*.json
  Each json has result[].`short caption`; we take the first non-empty short
  caption per clip file. Duration parsed from the `origintime_<a>_<b>` filename
  suffix (clamped), else default.

Outputs (PromptSpec JSONL, schema matches physflow_kimodo_g1_runner.PromptSpec):
  - <out_dir>/physflow_text_train.jsonl   (split="train")
  - <out_dir>/physflow_text_eval.jsonl    (split="test", frozen, stratified)

The eval split is sampled with a FIXED seed and stratified across sub-datasets,
then the train split is everything else. Re-running with the same seed reproduces
the exact same eval set.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
CAP_DIR = ROOT / "data/hymotion_data/Academic/20250916/raw_caption"

_TIME_RE = re.compile(r"origintime_([0-9.]+)_([0-9.]+)")


def parse_duration(name: str, lo: float, hi: float, default: float) -> float:
    m = _TIME_RE.search(name)
    if not m:
        return default
    try:
        a, b = float(m.group(1)), float(m.group(2))
        d = b - a
        if d <= 0:
            return default
        return max(lo, min(hi, d))
    except Exception:
        return default


def first_caption(p: Path) -> str | None:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    for r in d.get("result") or []:
        c = (r.get("short caption") or "").strip()
        if c:
            return c
    return None


def discover_humanml3d_dirs() -> list[Path]:
    return sorted(d for d in CAP_DIR.glob("HumanML3D-*") if d.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap on train prompts (0 = no cap, use all)")
    ap.add_argument("--eval-size", type=int, default=300,
                    help="frozen held-out eval prompts (stratified across datasets)")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=6.0)
    ap.add_argument("--default-dur", type=float, default=4.0)
    ap.add_argument("--max-cap-len", type=int, default=120)
    ap.add_argument("--out-dir",
                    default=str(ROOT / "configs/experiments/physflow_kimodo_g1"))
    args = ap.parse_args()
    random.seed(args.seed)

    dirs = discover_humanml3d_dirs()
    if not dirs:
        raise SystemExit(f"No HumanML3D-* dirs under {CAP_DIR}")

    # Collect (source, filename, caption, duration), dedup captions globally.
    per_ds: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    total_files = 0
    for d in dirs:
        src = d.name.replace("HumanML3D-", "")
        files = sorted(d.glob("*.json"))
        total_files += len(files)
        for p in files:
            cap = first_caption(p)
            if not cap or len(cap) > args.max_cap_len:
                continue
            key = cap.lower()
            if key in seen:
                continue
            seen.add(key)
            per_ds[src].append({
                "prompt": cap,
                "source": src,
                "duration_sec": round(
                    parse_duration(p.name, args.min_dur, args.max_dur,
                                   args.default_dur), 2),
            })

    for src in per_ds:
        random.shuffle(per_ds[src])

    # Stratified frozen eval: round-robin across datasets until eval-size.
    eval_items: list[dict] = []
    srcs = sorted(per_ds.keys())
    while len(eval_items) < args.eval_size and any(per_ds[s] for s in srcs):
        for s in srcs:
            if per_ds[s]:
                eval_items.append(per_ds[s].pop())
                if len(eval_items) >= args.eval_size:
                    break

    # Train = everything left.
    train_items: list[dict] = []
    for s in srcs:
        train_items.extend(per_ds[s])
    random.shuffle(train_items)
    if args.max_train and len(train_items) > args.max_train:
        train_items = train_items[:args.max_train]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(items: list[dict], split: str, prefix: str, path: Path) -> None:
        with path.open("w") as f:
            for i, it in enumerate(items):
                spec = {
                    "id": f"{prefix}_{i:06d}",
                    "prompt": it["prompt"],
                    "category": "humanml3d",
                    "difficulty": 0,
                    "duration_sec": it["duration_sec"],
                    "split": split,
                    "source": it["source"],
                    "tags": [],
                }
                f.write(json.dumps(spec, ensure_ascii=False) + "\n")

    train_path = out_dir / "physflow_text_train.jsonl"
    eval_path = out_dir / "physflow_text_eval.jsonl"
    write(train_items, "train", "tr", train_path)
    write(eval_items, "test", "ev", eval_path)

    print(f"scanned {total_files} caption files across {len(dirs)} HumanML3D datasets")
    print(f"unique captions kept: {len(seen)}")
    print(f"TRAIN: {len(train_items)} -> {train_path}")
    print(f"EVAL (frozen, seed={args.seed}): {len(eval_items)} -> {eval_path}")
    by_src = defaultdict(int)
    for it in train_items:
        by_src[it["source"]] += 1
    print("train per-source:", dict(sorted(by_src.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()

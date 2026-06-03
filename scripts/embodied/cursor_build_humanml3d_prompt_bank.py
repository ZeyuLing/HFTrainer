#!/usr/bin/env python3
"""Build a KIMODO-G1 prompt bank (JSONL of PromptSpec) from HumanML3D captions.

Source: data/hymotion_data/Academic/20250916/raw_caption/HumanML3D-<DATASET>/*.json
Each json has result[].`short caption`. We take the first short caption per file
and sample N across datasets for diversity. Duration is parsed from the
`origintime_<a>_<b>` suffix in the filename (clamped), else defaults.

Output JSONL line schema matches PromptSpec in physflow_kimodo_g1_runner.py:
  {id, prompt, category, difficulty, duration_sec, split, source, tags}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
CAP_DIR = ROOT / "data/hymotion_data/Academic/20250916/raw_caption"

# Prefer datasets with cleaner, locomotion-rich content; sample across them.
DATASETS = [
    "HumanML3D-CMU",
    "HumanML3D-ACCAD",
    "HumanML3D-BMLmovi",
    "HumanML3D-KIT",
    "HumanML3D-Eyes_Japan_Dataset",
    "HumanML3D-BioMotionLab_NTroje",
    "HumanML3D-EKUT",
]

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
    res = d.get("result") or []
    for r in res:
        c = (r.get("short caption") or "").strip()
        if c:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=6.0)
    ap.add_argument("--default-dur", type=float, default=4.0)
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--output",
        default=str(ROOT / "configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl"),
    )
    args = ap.parse_args()
    random.seed(args.seed)

    # Gather candidate files per dataset, sample round-robin for diversity.
    per_ds: dict[str, list[Path]] = {}
    for ds in DATASETS:
        d = CAP_DIR / ds
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.json"))
        random.shuffle(files)
        per_ds[ds] = files

    selected: list[tuple[str, Path, str]] = []
    seen_prompts: set[str] = set()
    idx = 0
    # round-robin until we hit n
    while len(selected) < args.n and any(per_ds.values()):
        for ds in list(per_ds.keys()):
            if not per_ds[ds]:
                continue
            p = per_ds[ds].pop()
            cap = first_caption(p)
            if not cap:
                continue
            # skip near-duplicate / overly long captions
            key = cap.lower()
            if key in seen_prompts:
                continue
            if len(cap) > 120:
                continue
            seen_prompts.add(key)
            selected.append((ds, p, cap))
            if len(selected) >= args.n:
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, (ds, p, cap) in enumerate(selected):
            dur = parse_duration(p.name, args.min_dur, args.max_dur, args.default_dur)
            spec = {
                "id": f"hml_{i:03d}",
                "prompt": cap,
                "category": "humanml3d",
                "difficulty": 0,
                "duration_sec": round(dur, 2),
                "split": args.split,
                "source": ds.replace("HumanML3D-", ""),
                "tags": [],
            }
            f.write(json.dumps(spec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} prompts -> {out}")
    # quick preview
    for _, _, cap in selected[:8]:
        print("  -", cap)


if __name__ == "__main__":
    main()

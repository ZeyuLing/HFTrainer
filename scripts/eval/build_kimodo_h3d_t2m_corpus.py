#!/usr/bin/env python3
"""Build the HumanML3D T2M caption corpus used by KIMODO cache extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_first_full_caption(txt_path: Path) -> str | None:
    if not txt_path.exists():
        return None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        caption = parts[0].strip()
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            f_tag = t_tag = 0.0
        if caption and f_tag == 0.0 and t_tag == 0.0:
            return caption
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.humanml3d_272)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ids = [line.strip() for line in (root / "split" / "test.txt").read_text().splitlines() if line.strip()]
    kept = []
    for sid in ids:
        motion_path = root / "motion_data" / f"{sid}.npy"
        if not motion_path.exists():
            continue
        length = int(np.load(str(motion_path), mmap_mode="r").shape[0])
        if length < args.min_len or length >= args.max_len:
            continue
        caption = _read_first_full_caption(root / "texts" / f"{sid}.txt")
        if not caption:
            continue
        kept.append({"id": sid, "split": "test", "prompt": caption, "length": length})
        if args.max_samples and len(kept) >= args.max_samples:
            break

    with out.open("w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {len(kept)} prompts -> {out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build PhysFlow prompt specs from an official HumanML3D split.

This is for quantitative evaluation, not hand-picked visualization.  It reads a
HumanML3D split file of motion ids plus the standard ``texts/<id>.txt`` caption
files and writes PromptSpec JSONL consumable by the PhysFlow/KIMODO pipeline.

Two common protocols are supported:
  * all_full: every full-clip caption is one prompt-motion pair.
  * first_full: one deterministic full-clip caption per motion id.

The script does not filter prompt semantics.  For a robot/no-scene subset, run
``filter_physflow_scene_prompts.py`` on the generated JSONL and report the drop
statistics separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
DEFAULT_HML272 = ROOT / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"


def _parse_tag(value: str) -> float:
    value = value.strip().lower()
    if value == "nan":
        return 0.0
    return float(value)


def load_full_captions(text_path: Path) -> list[str]:
    captions: list[str] = []
    for line in text_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            cap = parts[0].strip()
            if cap:
                captions.append(cap)
            continue
        cap = parts[0].strip()
        try:
            f_tag = _parse_tag(parts[-2])
            to_tag = _parse_tag(parts[-1])
        except ValueError:
            continue
        if cap and f_tag == 0.0 and to_tag == 0.0:
            captions.append(cap)
    return captions


def motion_duration_sec(motion_path: Path, fps: float, fallback: float) -> float:
    if not motion_path.is_file():
        return fallback
    try:
        arr = np.load(motion_path, mmap_mode="r")
        if len(arr) > 0:
            return round(float(len(arr)) / fps, 3)
    except Exception:
        pass
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-file",
        default=str(ROOT / "ref_repo/TeSMo/dataset/HumanML3D/test.txt"),
        help="HumanML3D split file containing motion ids, one per line.",
    )
    ap.add_argument(
        "--texts-dir",
        default=str(DEFAULT_HML272 / "texts"),
        help="Directory containing HumanML3D texts/<id>.txt files.",
    )
    ap.add_argument(
        "--motion-dir",
        default=str(DEFAULT_HML272 / "motion_data"),
        help="Optional motion_data directory for duration estimates.",
    )
    ap.add_argument(
        "--out",
        default=str(
            ROOT
            / "configs/experiments/physflow_kimodo_g1/"
            / "physflow_bench_hml3d_official_test_allcaptions.jsonl"
        ),
    )
    ap.add_argument("--caption-mode", choices=["all_full", "first_full"], default="all_full")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fallback-duration-sec", type=float, default=4.0)
    ap.add_argument("--skip-mirror", action="store_true", help="Skip ids prefixed with M.")
    ap.add_argument("--id-prefix", default="hml3dtest")
    ap.add_argument("--source", default="HumanML3D-official-test")
    args = ap.parse_args()

    split_file = Path(args.split_file)
    texts_dir = Path(args.texts_dir)
    motion_dir = Path(args.motion_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if args.skip_mirror:
        ids = [mid for mid in ids if not mid.startswith("M")]

    rows: list[dict] = []
    missing_text = 0
    no_full_caption = 0
    for mid in ids:
        text_path = texts_dir / f"{mid}.txt"
        if not text_path.is_file():
            missing_text += 1
            continue
        captions = load_full_captions(text_path)
        if not captions:
            no_full_caption += 1
            continue
        selected = captions if args.caption_mode == "all_full" else captions[:1]
        duration = motion_duration_sec(
            motion_dir / f"{mid}.npy",
            fps=args.fps,
            fallback=args.fallback_duration_sec,
        )
        for caption_idx, caption in enumerate(selected):
            rows.append(
                {
                    "id": f"{args.id_prefix}_{mid}_c{caption_idx:02d}",
                    "prompt": caption,
                    "category": "humanml3d_official_test",
                    "difficulty": 0,
                    "duration_sec": duration,
                    "split": "test",
                    "source": args.source,
                    "hml3d_id": mid,
                    "caption_index": caption_idx,
                    "num_full_captions": len(captions),
                    "tags": ["official_humanml3d_test"],
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"split ids: {len(ids)}")
    print(f"rows: {len(rows)}")
    print(f"missing text files: {missing_text}")
    print(f"ids without full captions: {no_full_caption}")
    print(f"caption mode: {args.caption_mode}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

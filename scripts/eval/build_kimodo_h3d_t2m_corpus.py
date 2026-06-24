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


def _iter_anno_entries(raw):
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
    else:
        for idx, entry in enumerate(data):
            sid = entry.get("motion_id") or entry.get("sample_id") or entry.get("id") or idx
            yield str(sid), entry


def _load_from_annotation(
    anno_file: Path,
    caption_json: Path | None,
    min_len: int,
    max_len_exclusive: int,
    max_samples: int,
) -> list[dict]:
    raw = json.loads(anno_file.read_text())
    captions = {}
    if caption_json and caption_json.exists():
        captions = json.loads(caption_json.read_text())
    kept = []
    for sid, entry in _iter_anno_entries(raw):
        sid = str(sid)
        length = entry.get("num_frames")
        if length is None:
            fps = float(entry.get("fps", 30) or 30)
            duration = float(entry.get("duration", 0) or 0)
            length = int(round(duration * fps)) if duration > 0 else 0
        length = int(length)
        if length < min_len or length >= max_len_exclusive:
            continue
        caption = captions.get(sid)
        if not caption:
            caption = entry.get("caption") or entry.get("text") or entry.get("prompt")
        if not caption:
            continue
        kept.append({"id": sid, "split": "test", "prompt": str(caption), "length": length})
        if max_samples and len(kept) >= max_samples:
            break
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--out", required=True)
    parser.add_argument("--anno-file", default=None)
    parser.add_argument("--caption-json", default=None)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.humanml3d_272)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.anno_file:
        kept = _load_from_annotation(
            Path(args.anno_file),
            Path(args.caption_json) if args.caption_json else None,
            args.min_len,
            args.max_len,
            args.max_samples,
        )
    else:
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

#!/usr/bin/env python3
"""Diagnose likely swapped two-subclip BABEL captions with MotionStreamer embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/eval") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/eval"))

MIN_MOTION_LENGTH = 60
MAX_MOTION_LENGTH = 300
UNIT_LENGTH = 4


def _stub_unused_clip() -> None:
    if "clip" in sys.modules:
        return
    clip_stub = types.ModuleType("clip")
    clip_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("CLIP is not used by BABEL swap diagnostics")
    )
    clip_stub.tokenize = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("CLIP is not used by BABEL swap diagnostics")
    )
    sys.modules["clip"] = clip_stub


def _eval_len(raw_len: int) -> int:
    n = min(int(raw_len), MAX_MOTION_LENGTH)
    n = (n // UNIT_LENGTH) * UNIT_LENGTH
    return n if n >= MIN_MOTION_LENGTH else 0


def _parse_stream_text(path: Path, total_frames: int) -> list[dict[str, Any]] | None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split("*")
    if len(parts) != 2:
        return None
    first = parts[0].split("#")
    second = parts[1].split("#")
    if len(first) < 1 or len(second) < 5:
        return None
    try:
        boundary = int(float(second[-1]))
    except ValueError:
        return None
    boundary = max(0, min(boundary, total_frames))
    return [
        {"index": 0, "caption": first[0].strip(), "start": 0, "end": boundary},
        {"index": 1, "caption": second[0].strip(), "start": boundary, "end": total_frames},
    ]


def _load_motion_len(path: Path) -> int:
    arr = np.load(path, mmap_mode="r")
    return int(arr.shape[0])


def _load_eval_segment(path: Path, seg: dict[str, Any], stride: int) -> tuple[np.ndarray, int] | None:
    motion = np.load(path, mmap_mode="r")
    start = int(seg["start"])
    end = int(seg["end"])
    raw = np.asarray(motion[start:end:stride], dtype=np.float32)
    ml = _eval_len(len(raw))
    if ml <= 0:
        return None
    return raw[:ml].copy(), int(ml)


def _swap_stats(rows: list[dict[str, Any]], prefix: str, margin: float) -> dict[str, Any]:
    key = f"{prefix}_min_improvement"
    flagged = [r for r in rows if r[f"{prefix}_cross_better"] and float(r[key]) >= margin]
    return {
        "margin": float(margin),
        "count": int(len(flagged)),
        "rate_over_checked_pairs": float(len(flagged) / len(rows)) if rows else 0.0,
        "seq_ids": [r["sid"] for r in flagged[:50]],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-dir", default=str(REPO / "data/babel_272_stream/val_stream"))
    ap.add_argument("--text-dir", default=str(REPO / "data/babel_272_stream/val_stream_text"))
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "outputs/evaluation/babel_stream/val/swap_diagnosis_20260624"),
    )
    ap.add_argument("--device", default="")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top-examples", type=int, default=50)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _stub_unused_clip()

    import torch
    from babel_caption import rewrite_caption
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    from hftrainer.evaluation.evaluators.t2m_metrics import euclidean_distance_matrix

    motion_dir = Path(args.motion_dir)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stride = 2

    counts = {
        "motion_files": 0,
        "matching_text_files": 0,
        "extra_text_files": int(len(list(text_dir.glob("*.txt")))),
        "missing_text": 0,
        "bad_text": 0,
        "total_subclips": 0,
        "valid_subclips": 0,
        "valid_two_subclip_pairs": 0,
        "too_short_subclips": 0,
    }
    pair_records: list[dict[str, Any]] = []
    motions: list[np.ndarray] = []
    lengths: list[int] = []
    raw_texts: list[str] = []
    rewrite_texts: list[str] = []

    motion_files = sorted(motion_dir.glob("*.npy"))
    for file_i, motion_path in enumerate(motion_files, 1):
        sid = motion_path.stem
        counts["motion_files"] += 1
        text_path = text_dir / f"{sid}.txt"
        if not text_path.exists():
            counts["missing_text"] += 1
            continue
        counts["matching_text_files"] += 1
        total = _load_motion_len(motion_path)
        segs = _parse_stream_text(text_path, total)
        if not segs:
            counts["bad_text"] += 1
            continue
        counts["total_subclips"] += len(segs)
        loaded: list[tuple[np.ndarray, int] | None] = []
        for seg in segs:
            item = _load_eval_segment(motion_path, seg, stride)
            loaded.append(item)
            if item is None:
                counts["too_short_subclips"] += 1
            else:
                counts["valid_subclips"] += 1
        if len(segs) == 2 and all(x is not None for x in loaded):
            counts["valid_two_subclip_pairs"] += 1
            base = len(motions)
            for seg, item in zip(segs, loaded):
                assert item is not None
                motion, length = item
                motions.append(motion)
                lengths.append(length)
                cap = str(seg["caption"])
                raw_texts.append(cap)
                rewrite_texts.append(rewrite_caption(cap))
            pair_records.append(
                {
                    "sid": sid,
                    "idx0": base,
                    "idx1": base + 1,
                    "caption0": str(segs[0]["caption"]),
                    "caption1": str(segs[1]["caption"]),
                    "start0": int(segs[0]["start"]),
                    "end0": int(segs[0]["end"]),
                    "start1": int(segs[1]["start"]),
                    "end1": int(segs[1]["end"]),
                    "eval_len0": int(lengths[base]),
                    "eval_len1": int(lengths[base + 1]),
                }
            )
        if file_i % 250 == 0:
            print(
                f"[load] {file_i}/{len(motion_files)} files "
                f"subclips={counts['total_subclips']} valid={counts['valid_subclips']} "
                f"pairs={counts['valid_two_subclip_pairs']}",
                flush=True,
            )

    print(f"[encode] pairs={len(pair_records)} segments={len(motions)} device={device}", flush=True)
    ev = MotionStreamer272Evaluator(device=device)
    motion_emb = ev.encode_motion(motions, lengths, batch_size=args.batch_size)
    raw_emb = ev.encode_text(raw_texts, batch_size=args.batch_size)
    rewrite_emb = ev.encode_text(rewrite_texts, batch_size=args.batch_size)
    raw_dist = euclidean_distance_matrix(raw_emb, motion_emb)
    rewrite_dist = euclidean_distance_matrix(rewrite_emb, motion_emb)

    rows: list[dict[str, Any]] = []
    for rec in pair_records:
        i0 = int(rec["idx0"])
        i1 = int(rec["idx1"])
        row = {k: v for k, v in rec.items() if not k.startswith("idx")}
        for prefix, dist in (("raw", raw_dist), ("rewrite", rewrite_dist)):
            d00 = float(dist[i0, i0])
            d11 = float(dist[i1, i1])
            d01 = float(dist[i0, i1])
            d10 = float(dist[i1, i0])
            imp0 = d00 - d01
            imp1 = d11 - d10
            row.update(
                {
                    f"{prefix}_d_cap0_motion0": d00,
                    f"{prefix}_d_cap1_motion1": d11,
                    f"{prefix}_d_cap0_motion1": d01,
                    f"{prefix}_d_cap1_motion0": d10,
                    f"{prefix}_improve_cap0_if_swapped": imp0,
                    f"{prefix}_improve_cap1_if_swapped": imp1,
                    f"{prefix}_sum_improvement": imp0 + imp1,
                    f"{prefix}_min_improvement": min(imp0, imp1),
                    f"{prefix}_cross_better": bool(imp0 > 0 and imp1 > 0),
                    f"{prefix}_strict_margin1": bool(imp0 >= 1.0 and imp1 >= 1.0),
                    f"{prefix}_strict_margin2": bool(imp0 >= 2.0 and imp1 >= 2.0),
                }
            )
        rows.append(row)

    rows.sort(key=lambda r: (float(r["raw_min_improvement"]), float(r["raw_sum_improvement"])), reverse=True)
    summary = {
        "config": {
            "motion_dir": str(motion_dir),
            "text_dir": str(text_dir),
            "frame_step": int(stride),
            "fps_policy": "30fps evaluation by temporal decimation motion[start:end:2]",
            "device": str(device),
            "batch_size": int(args.batch_size),
            "criterion": (
                "cross_better means cap0-motion1 and cap1-motion0 are both closer than "
                "the original cap0-motion0 and cap1-motion1 pairings. strict_margin1/2 "
                "require both per-caption improvements to be at least 1/2 embedding-distance units."
            ),
        },
        "counts": counts,
        "raw_caption_swap_stats": {
            "margin_gt_0": _swap_stats(rows, "raw", 0.0),
            "margin_ge_1": _swap_stats(rows, "raw", 1.0),
            "margin_ge_2": _swap_stats(rows, "raw", 2.0),
        },
        "rewrite_caption_swap_stats": {
            "margin_gt_0": _swap_stats(rows, "rewrite", 0.0),
            "margin_ge_1": _swap_stats(rows, "rewrite", 1.0),
            "margin_ge_2": _swap_stats(rows, "rewrite", 2.0),
        },
        "top_raw_suspicious": rows[: args.top_examples],
    }
    (out_dir / "summary_30fps.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(out_dir / "pair_swap_scores_30fps.csv", rows)
    print(f"[done] wrote {out_dir} 30fps", flush=True)
    print(json.dumps(summary["counts"], indent=2), flush=True)
    print(json.dumps(summary["raw_caption_swap_stats"], indent=2), flush=True)


if __name__ == "__main__":
    main()

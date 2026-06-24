#!/usr/bin/env python3
"""Evaluate MotionStreamer BABEL stream before/after 60->30fps downsampling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MIN_MOTION_LENGTH = 60
MAX_MOTION_LENGTH = 300
UNIT_LENGTH = 4


def _stub_unused_clip() -> None:
    if "clip" in sys.modules:
        return
    clip_stub = types.ModuleType("clip")
    clip_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("CLIP is not used by MotionStreamer-272 BABEL fps ablation")
    )
    clip_stub.tokenize = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("CLIP is not used by MotionStreamer-272 BABEL fps ablation")
    )
    sys.modules["clip"] = clip_stub


def _eval_len(raw_len: int) -> int:
    n = min(int(raw_len), MAX_MOTION_LENGTH)
    n = (n // UNIT_LENGTH) * UNIT_LENGTH
    return n if n >= MIN_MOTION_LENGTH else 0


def _parse_stream_text(path: Path, total_frames: int) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    parts = text.split("*")
    if len(parts) != 2:
        return []
    first = parts[0].split("#")
    second = parts[1].split("#")
    if len(first) < 1 or len(second) < 5:
        return []
    try:
        boundary = int(float(second[-1]))
    except ValueError:
        return []
    boundary = max(0, min(boundary, total_frames))
    return [
        {"index": 0, "caption": first[0].strip(), "start": 0, "end": boundary},
        {"index": 1, "caption": second[0].strip(), "start": boundary, "end": total_frames},
    ]


def _caption_key(text: str, fallback: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return key or fallback


def _build_rank_candidate_indices(captions: list[str], batch_size: int, seed: int) -> list[list[int]]:
    n = len(captions)
    if n == 0:
        return []
    batch_size = max(1, min(int(batch_size), n))
    n_batches = int(np.ceil(n / batch_size))
    rng = random.Random(seed)
    keys = [_caption_key(cap, f"__empty_{i}") for i, cap in enumerate(captions)]
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        rng.shuffle(idxs)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    group_keys.sort(key=lambda key: len(groups[key]), reverse=True)
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_keys: list[set[str]] = [set() for _ in range(n_batches)]
    for key in group_keys:
        for idx in groups[key]:
            choices = [
                b for b in range(n_batches)
                if len(batches[b]) < batch_size and key not in batch_keys[b]
            ]
            if not choices:
                choices = [b for b in range(n_batches) if len(batches[b]) < batch_size]
            if not choices:
                choices = list(range(n_batches))
            min_fill = min(len(batches[b]) for b in choices)
            choices = [b for b in choices if len(batches[b]) == min_fill]
            b = choices[rng.randrange(len(choices))]
            batches[b].append(idx)
            batch_keys[b].add(key)
    by_query: list[list[int]] = [[] for _ in range(n)]
    for batch in batches:
        batch_sorted = sorted(batch)
        for idx in batch:
            by_query[idx] = batch_sorted
    return by_query


def _rank_rows(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    captions: list[str],
    candidate_indices_by_query: list[list[int]],
) -> list[dict[str, Any]]:
    from hftrainer.evaluation.evaluators.t2m_metrics import euclidean_distance_matrix

    dist = euclidean_distance_matrix(text_emb, motion_emb)
    keys = [_caption_key(cap, f"__empty_{i}") for i, cap in enumerate(captions)]
    rows = []
    for i in range(dist.shape[0]):
        cand_idx = candidate_indices_by_query[i] if candidate_indices_by_query else list(range(len(captions)))
        cand_idx = [int(j) for j in cand_idx]
        subset_order = sorted(cand_idx, key=lambda j: float(dist[i, j]))
        key_counts: dict[str, int] = {}
        for j in cand_idx:
            key_counts[keys[j]] = key_counts.get(keys[j], 0) + 1
        rank_full = int(subset_order.index(i)) + 1 if i in subset_order else 0
        seen: set[str] = set()
        rank = None
        best_same_caption_mm_dist = None
        for j in subset_order:
            key = keys[j]
            if key in seen:
                continue
            seen.add(key)
            if key == keys[i]:
                rank = len(seen)
                best_same_caption_mm_dist = float(dist[i, j])
                break
        if rank is None:
            rank = rank_full
            best_same_caption_mm_dist = float(dist[i, i])
        rows.append(
            {
                "mm_dist": float(dist[i, i]),
                "best_same_caption_mm_dist": float(best_same_caption_mm_dist),
                "rank": rank,
                "rank_full": rank_full,
                "r1": bool(rank <= 1),
                "r2": bool(rank <= 2),
                "r3": bool(rank <= 3),
                "unique_candidate_count": int(len(key_counts)),
                "full_candidate_count": int(len(cand_idx)),
                "duplicate_count_for_caption": int(key_counts[keys[i]]),
            }
        )
    return rows


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
        default=str(REPO / "outputs/evaluation/babel_stream/val/motionstreamer272_fps_ablation_20260624"),
    )
    ap.add_argument("--device", default="")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rank-batch-size", type=int, default=32)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _stub_unused_clip()

    import torch
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    from hftrainer.evaluation.evaluators.t2m_metrics import aggregate_t2m_metrics

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    motion_dir = Path(args.motion_dir)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    motion30_dir = out_dir / "motion272_30fps"
    motion30_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    captions: list[str] = []
    motions60: list[np.ndarray] = []
    motions30: list[np.ndarray] = []
    len60: list[int] = []
    len30: list[int] = []
    skipped = {"missing_text": 0, "bad_text": 0, "too_short": 0}

    motion_files = sorted(motion_dir.glob("*.npy"))
    for i, motion_path in enumerate(motion_files, 1):
        sid = motion_path.stem
        text_path = text_dir / f"{sid}.txt"
        if not text_path.exists():
            skipped["missing_text"] += 1
            continue
        motion = np.load(motion_path).astype(np.float32)
        motion30_full = motion[::2].copy()
        np.save(motion30_dir / f"{sid}.npy", motion30_full)
        segments = _parse_stream_text(text_path, len(motion))
        if not segments:
            skipped["bad_text"] += 1
            continue
        for seg in segments:
            start = int(seg["start"])
            end = int(seg["end"])
            raw60 = motion[start:end]
            raw30 = raw60[::2]
            ml60 = _eval_len(len(raw60))
            ml30 = _eval_len(len(raw30))
            if ml60 <= 0 or ml30 <= 0:
                skipped["too_short"] += 1
                continue
            captions.append(seg["caption"])
            motions60.append(raw60[:ml60])
            motions30.append(raw30[:ml30])
            len60.append(ml60)
            len30.append(ml30)
            records.append(
                {
                    "sid": sid,
                    "seg_index": int(seg["index"]),
                    "start60": start,
                    "end60": end,
                    "raw_frames60": int(len(raw60)),
                    "raw_frames30": int(len(raw30)),
                    "eval_frames60": int(ml60),
                    "eval_frames30": int(ml30),
                    "caption": seg["caption"],
                }
            )
        if i % 250 == 0:
            print(f"[load] {i}/{len(motion_files)} files, valid_segments={len(records)}", flush=True)

    print(
        f"[encode] valid_segments={len(records)} skipped={skipped} device={device}",
        flush=True,
    )
    ev = MotionStreamer272Evaluator(device=device)
    text_emb = ev.encode_text(captions, batch_size=args.batch_size)
    emb60 = ev.encode_motion(motions60, len60, batch_size=args.batch_size)
    emb30 = ev.encode_motion(motions30, len30, batch_size=args.batch_size)

    print("[metrics] aggregate", flush=True)
    metrics60 = aggregate_t2m_metrics(
        text_emb, emb60, emb60, n_repeats=args.n_repeats, chunk=32, seed=args.seed
    )
    metrics30 = aggregate_t2m_metrics(
        text_emb, emb30, emb30, n_repeats=args.n_repeats, chunk=32, seed=args.seed
    )
    metrics30_vs_60 = aggregate_t2m_metrics(
        text_emb, emb60, emb30, n_repeats=args.n_repeats, chunk=32, seed=args.seed
    )

    print("[metrics] per-segment ranks", flush=True)
    rank_batches = _build_rank_candidate_indices(captions, args.rank_batch_size, args.seed)
    rows60 = _rank_rows(text_emb, emb60, captions, rank_batches)
    rows30 = _rank_rows(text_emb, emb30, captions, rank_batches)
    emb_l2 = np.linalg.norm(emb30 - emb60, axis=1)
    csv_rows = []
    for rec, r60, r30, l2 in zip(records, rows60, rows30, emb_l2):
        csv_rows.append(
            {
                **rec,
                "mm_dist60": r60["mm_dist"],
                "best_same_caption_mm_dist60": r60["best_same_caption_mm_dist"],
                "rank60": r60["rank"],
                "rank_full60": r60["rank_full"],
                "r1_60": int(r60["r1"]),
                "r2_60": int(r60["r2"]),
                "r3_60": int(r60["r3"]),
                "mm_dist30": r30["mm_dist"],
                "best_same_caption_mm_dist30": r30["best_same_caption_mm_dist"],
                "rank30": r30["rank"],
                "rank_full30": r30["rank_full"],
                "r1_30": int(r30["r1"]),
                "r2_30": int(r30["r2"]),
                "r3_30": int(r30["r3"]),
                "unique_candidate_count": r60["unique_candidate_count"],
                "full_candidate_count": r60["full_candidate_count"],
                "duplicate_count_for_caption": r60["duplicate_count_for_caption"],
                "delta_mm_dist30_minus60": float(r30["mm_dist"] - r60["mm_dist"]),
                "emb_l2_30_vs_60": float(l2),
            }
        )

    by_bad = sorted(csv_rows, key=lambda r: (int(r["r3_60"]), -float(r["mm_dist60"]), int(r["rank60"])))
    summary = {
        "config": {
            "motion_dir": str(motion_dir),
            "text_dir": str(text_dir),
            "motion30_dir": str(motion30_dir),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "n_repeats": int(args.n_repeats),
            "seed": int(args.seed),
            "fps60_interpretation": "original MotionStreamer BABEL stream time axis",
            "fps30_processing": "temporal decimation with arr[::2]",
            "segment_rank_protocol": "deterministic 32-candidate batches with caption-deduplicated unique groups; within-batch clip rank is rank_full",
            "segment_rank_batch_size": int(args.rank_batch_size),
        },
        "counts": {
            "motion_files": int(len(motion_files)),
            "valid_segments": int(len(records)),
            "skipped": skipped,
        },
        "metrics60_self": metrics60,
        "metrics30_self": metrics30,
        "metrics30_vs_60": metrics30_vs_60,
        "rprecision_counts": {
            "r1_60": int(sum(r["r1_60"] for r in csv_rows)),
            "r2_60": int(sum(r["r2_60"] for r in csv_rows)),
            "r3_60": int(sum(r["r3_60"] for r in csv_rows)),
            "r1_30": int(sum(r["r1_30"] for r in csv_rows)),
            "r2_30": int(sum(r["r2_30"] for r in csv_rows)),
            "r3_30": int(sum(r["r3_30"] for r in csv_rows)),
        },
        "worst_r3_failures_60": by_bad[:50],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "segment_scores.csv", csv_rows)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()

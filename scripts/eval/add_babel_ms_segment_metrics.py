#!/usr/bin/env python3
"""Attach MotionStreamer evaluator diagnostics to BABEL viewer segments."""

from __future__ import annotations

import argparse
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
    """Keep unrelated InterHuman imports from requiring OpenAI CLIP."""
    if "clip" in sys.modules:
        return
    clip_stub = types.ModuleType("clip")
    clip_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("CLIP is not used by MotionStreamer segment metrics")
    )
    clip_stub.tokenize = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("CLIP is not used by MotionStreamer segment metrics")
    )
    sys.modules["clip"] = clip_stub


def _eval_len(raw_len: int) -> int:
    n = min(int(raw_len), MAX_MOTION_LENGTH)
    n = (n // UNIT_LENGTH) * UNIT_LENGTH
    return n if n >= MIN_MOTION_LENGTH else 0


def _caption_key(text: str, fallback: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return key or fallback


def _candidate_entry(
    row: dict[str, Any],
    rank: int,
    distance: float,
    self_match: bool,
    same_caption: bool,
    duplicate_count: int,
    caption_key: str,
) -> dict[str, Any]:
    seg = row["segment"]
    return {
        "rank": int(rank),
        "distance": float(distance),
        "self": bool(self_match),
        "self_in_group": bool(self_match),
        "same_caption": bool(same_caption),
        "duplicate_count": int(duplicate_count),
        "caption_key": str(caption_key),
        "sid": str(row["sid"]),
        "seg_index": int(row["seg_index"]),
        "frames": [int(seg.get("start", 0)), int(seg.get("end", 0))],
        "raw_caption": str(row["raw_text"]),
        "rewrite_caption": str(row["rewrite_text"]),
    }


def _build_rank_candidate_indices(
    candidates: list[dict[str, Any]],
    caption_field: str,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    n = len(candidates)
    if n == 0:
        return []
    batch_size = max(1, min(int(batch_size), n))
    n_batches = int(np.ceil(n / batch_size))
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(candidates):
        key = _caption_key(str(row.get(caption_field, "")), f"__empty_{i}")
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        rng.shuffle(idxs)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)

    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_keys: list[set[str]] = [set() for _ in range(n_batches)]
    for key in keys:
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
    candidates: list[dict[str, Any]],
    top_k: int,
    caption_field: str,
    candidate_indices_by_query: list[list[int]],
) -> list[dict[str, Any]]:
    from hftrainer.evaluation.evaluators.t2m_metrics import euclidean_distance_matrix

    dist = euclidean_distance_matrix(text_emb, motion_emb)
    keys = [
        _caption_key(str(row.get(caption_field, "")), f"__empty_{i}")
        for i, row in enumerate(candidates)
    ]
    rows = []
    for i in range(dist.shape[0]):
        cand_idx = candidate_indices_by_query[i] if candidate_indices_by_query else list(range(len(candidates)))
        cand_idx = [int(j) for j in cand_idx]
        subset_order = sorted(cand_idx, key=lambda j: float(dist[i, j]))
        key_counts: dict[str, int] = {}
        for j in cand_idx:
            key_counts[keys[j]] = key_counts.get(keys[j], 0) + 1
        rank_full = int(subset_order.index(i)) + 1 if i in subset_order else 0
        query_key = keys[i]
        dedup_entries: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        rank = None
        best_same_caption_mm_dist = None
        for j in subset_order:
            key = keys[j]
            entry = seen.get(key)
            if entry is None:
                entry = _candidate_entry(
                    candidates[j],
                    len(dedup_entries) + 1,
                    float(dist[i, j]),
                    j == i,
                    key == query_key,
                    key_counts[key],
                    key,
                )
                seen[key] = entry
                dedup_entries.append(entry)
                if key == query_key and rank is None:
                    rank = int(entry["rank"])
                    best_same_caption_mm_dist = float(dist[i, j])
            elif j == i:
                entry["self_in_group"] = True
            if len(dedup_entries) >= max(top_k, len(key_counts)) and rank is not None:
                break
        if rank is None:
            rank = rank_full
            best_same_caption_mm_dist = float(dist[i, i])
        top = dedup_entries[:top_k]
        rows.append(
            {
                "mm_dist": float(dist[i, i]),
                "best_same_caption_mm_dist": float(best_same_caption_mm_dist),
                "rank": rank,
                "rank_full": rank_full,
                "r1": bool(rank <= 1),
                "r2": bool(rank <= 2),
                "r3": bool(rank <= 3),
                "top_candidates": top,
                "caption_key": query_key,
                "duplicate_count_for_caption": int(key_counts[query_key]),
                "unique_candidate_count": int(len(key_counts)),
                "full_candidate_count": int(len(cand_idx)),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--viewer-data-dir",
        default=str(REPO / "motion_annot_web/babel_smpl_mesh_viewer/data"),
    )
    ap.add_argument(
        "--motion-dir",
        default=str(REPO / "data/babel_272_stream/val_stream"),
        help="Directory containing BABEL MotionStreamer-272 GT .npy files.",
    )
    ap.add_argument("--device", default="", help="cuda/cpu; default picks cuda when available")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rank-batch-size", type=int, default=32)
    ap.add_argument("--rank-seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5, help="Number of top retrieval candidates to store per segment.")
    ap.add_argument(
        "--summary-json",
        default="",
        help="Optional output path; default writes ms_segment_metrics.json next to index.json.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _stub_unused_clip()

    import torch
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    viewer_dir = Path(args.viewer_data_dir)
    motion_dir = Path(args.motion_dir)
    index_path = viewer_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    cases: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    invalid_count = 0

    for c in index.get("cases", []):
        sid = str(c["id"])
        case_path = viewer_dir / f"case_{sid}.json"
        motion_path = motion_dir / f"{sid}.npy"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        cases[sid] = case
        motion = np.load(motion_path).astype(np.float32)
        for seg in case.get("segments", []):
            start = int(seg.get("start", 0))
            end = int(seg.get("end", start))
            raw_len = max(0, min(end, len(motion)) - max(0, start))
            ml = _eval_len(raw_len)
            if ml <= 0:
                invalid_count += 1
                seg["ms_eval"] = {
                    "valid": False,
                    "reason": "too_short_for_motionstreamer_evaluator",
                    "raw_frames": int(raw_len),
                    "min_frames": MIN_MOTION_LENGTH,
                    "source": "motionstreamer_272",
                }
                continue
            raw_text = str(seg.get("raw") or seg.get("caption") or "").strip()
            rewrite_text = str(seg.get("rewrite") or seg.get("cache") or raw_text).strip()
            if not raw_text:
                raw_text = rewrite_text
            if not rewrite_text:
                rewrite_text = raw_text
            rows.append(
                {
                    "sid": sid,
                    "seg_index": int(seg.get("index", len(rows))),
                    "segment": seg,
                    "motion": motion[max(0, start) : max(0, start) + ml],
                    "length": ml,
                    "raw_frames": raw_len,
                    "raw_text": raw_text,
                    "rewrite_text": rewrite_text,
                }
            )

    if rows:
        ev = MotionStreamer272Evaluator(device=device)
        motions = [r["motion"] for r in rows]
        lengths = [int(r["length"]) for r in rows]
        raw_texts = [r["raw_text"] for r in rows]
        rewrite_texts = [r["rewrite_text"] for r in rows]
        motion_emb = ev.encode_motion(motions, lengths, batch_size=args.batch_size)
        raw_text_emb = ev.encode_text(raw_texts, batch_size=args.batch_size)
        rewrite_text_emb = ev.encode_text(rewrite_texts, batch_size=args.batch_size)
        raw_batches = _build_rank_candidate_indices(rows, "raw_text", args.rank_batch_size, args.rank_seed)
        rewrite_batches = _build_rank_candidate_indices(rows, "rewrite_text", args.rank_batch_size, args.rank_seed)
        raw_rows = _rank_rows(raw_text_emb, motion_emb, rows, max(1, args.top_k), "raw_text", raw_batches)
        rewrite_rows = _rank_rows(rewrite_text_emb, motion_emb, rows, max(1, args.top_k), "rewrite_text", rewrite_batches)
        text_l2 = np.linalg.norm(rewrite_text_emb - raw_text_emb, axis=1)

        for i, row in enumerate(rows):
            raw = raw_rows[i]
            rewrite = rewrite_rows[i]
            row["segment"]["ms_eval"] = {
                "valid": True,
                "source": "motionstreamer_272",
                "scope": "viewer_valid_segments_caption_dedup",
                "rank_protocol": "batch32_caption_dedup_unique_caption_groups",
                "rank_batch_size": int(args.rank_batch_size),
                "rank_seed": int(args.rank_seed),
                "candidate_count": int(raw["unique_candidate_count"]),
                "candidate_count_unique": int(raw["unique_candidate_count"]),
                "candidate_count_full": int(raw["full_candidate_count"]),
                "raw_unique_candidate_count": int(raw["unique_candidate_count"]),
                "rewrite_unique_candidate_count": int(rewrite["unique_candidate_count"]),
                "raw_full_candidate_count": int(raw["full_candidate_count"]),
                "rewrite_full_candidate_count": int(rewrite["full_candidate_count"]),
                "duplicates_folded_raw": int(raw["full_candidate_count"] - raw["unique_candidate_count"]),
                "duplicates_folded_rewrite": int(rewrite["full_candidate_count"] - rewrite["unique_candidate_count"]),
                "raw_frames": int(row["raw_frames"]),
                "eval_frames": int(row["length"]),
                "max_eval_frames": MAX_MOTION_LENGTH,
                "raw_caption": raw,
                "rewrite_caption": rewrite,
                "delta_mm_dist_rewrite_minus_raw": float(rewrite["mm_dist"] - raw["mm_dist"]),
                "text_emb_l2_rewrite_vs_raw": float(text_l2[i]),
            }

    for sid, case in cases.items():
        (viewer_dir / f"case_{sid}.json").write_text(
            json.dumps(case, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "source": "motionstreamer_272",
        "viewer_data_dir": str(viewer_dir),
        "motion_dir": str(motion_dir),
        "valid_segments": int(len(rows)),
        "invalid_segments": int(invalid_count),
        "device": str(device),
        "rank_batch_size": int(args.rank_batch_size),
        "rank_seed": int(args.rank_seed),
        "metric_note": (
            "Segment-level diagnostics use MotionStreamer-272 text/motion embeddings. "
            "MM-Dist is the paired text-motion distance. Retrieval rank/top matches are computed over "
            "deterministic 32-candidate batches with caption-deduplicated unique groups; "
            "the within-batch full clip rank is stored as rank_full; "
            "FID and Diversity remain distribution-level metrics and are not assigned to a single segment."
        ),
    }
    summary_path = Path(args.summary_json) if args.summary_json else viewer_dir / "ms_segment_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] wrote MotionStreamer segment metrics: valid={len(rows)} "
        f"invalid={invalid_count} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

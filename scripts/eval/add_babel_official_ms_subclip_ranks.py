#!/usr/bin/env python3
"""Precompute per-subclip MotionStreamer rank diagnostics for official BABEL.

The rows follow the same official-val protocol as
``eval_babel_seq_ms272.py``: 30 FPS, explicit transitions absorbed into
neighboring actions, per-segment canonicalization, HumanML3D evaluator stats,
and label-aware R-precision. The output JSON is consumed by
``motion_annot_web/babel_official_mesh_viewer/app.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts" / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval_motionstreamer_272 as E  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402
from eval_babel_seq_ms272 import HUMANML_MEAN_STD, norm_pad, per_seg_canon  # noqa: E402


def _balanced_batches_all(items: list[tuple[str, np.ndarray, int]], batch_size: int, seed: int) -> list[list[int]]:
    """Balanced caption batches with a final tail batch kept for visualization."""
    rng = np.random.RandomState(seed)
    order = list(rng.permutation(len(items)))
    n_batches = int(np.ceil(len(order) / float(batch_size)))
    if n_batches <= 0:
        return []
    groups: dict[str, list[int]] = {}
    for i in order:
        groups.setdefault(items[int(i)][0], []).append(int(i))
    caps = list(groups)
    rng.shuffle(caps)
    caps.sort(key=lambda cap: len(groups[cap]), reverse=True)
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_caps: list[set[str]] = [set() for _ in range(n_batches)]
    for cap in caps:
        idxs = groups[cap]
        rng.shuffle(idxs)
        for idx in idxs:
            choices = [
                b for b in range(n_batches)
                if len(batches[b]) < batch_size and cap not in batch_caps[b]
            ]
            if not choices:
                choices = [b for b in range(n_batches) if len(batches[b]) < batch_size]
            min_fill = min(len(batches[b]) for b in choices)
            choices = [b for b in choices if len(batches[b]) == min_fill]
            b = choices[int(rng.randint(len(choices)))]
            batches[b].append(int(idx))
            batch_caps[b].add(cap)
    return [b for b in batches if b]


@torch.no_grad()
def _encode_text(captions, textenc, device, batch_size):
    out = []
    for i in range(0, len(captions), batch_size):
        out.append(textenc(list(captions[i : i + batch_size])).loc.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _encode_motion(motions, lengths, motionenc, device, batch_size):
    out = []
    for i in range(0, len(motions), batch_size):
        mb = torch.from_numpy(np.stack(motions[i : i + batch_size])).float().to(device)
        lb = torch.tensor(lengths[i : i + batch_size], device=device)
        out.append(motionenc(mb, lb).loc.cpu().numpy())
    return np.concatenate(out, axis=0)


def _candidate_entry(rows, dist_row, query_label, j, rank):
    row = rows[j]
    return {
        "rank": int(rank),
        "distance": float(dist_row[j]),
        "same_caption": bool(row["eval_caption"] == query_label),
        "self": False,
        "sid": row["sid"],
        "seg_index": int(row["seg_index"]),
        "caption": row["caption"],
        "eval_caption": row["eval_caption"],
        "frames_30": [int(row["start_30"]), int(row["end_30"])],
        "frames_native": [int(row["start_native"]), int(row["end_native"])],
    }


def _rank_one(i: int, batch: list[int], rows: list[dict[str, Any]], dist: np.ndarray, top_k: int):
    order = sorted(batch, key=lambda j: float(dist[i, j]))
    query_label = rows[i]["eval_caption"]
    label_rank = None
    self_rank = None
    top = []
    for pos, j in enumerate(order, 1):
        if rows[j]["eval_caption"] == query_label and label_rank is None:
            label_rank = pos
        if j == i:
            self_rank = pos
        if len(top) < top_k:
            ent = _candidate_entry(rows, dist[i], query_label, j, pos)
            ent["self"] = bool(j == i)
            top.append(ent)
    same_count = sum(1 for j in batch if rows[j]["eval_caption"] == query_label)
    if label_rank is None:
        label_rank = self_rank or 0
    return {
        "valid": True,
        "source": "motionstreamer_272",
        "rank_protocol": "label_aware_32way_balanced_tail_included",
        "rank_batch_size": len(batch),
        "configured_rank_batch_size": 32,
        "rank": int(label_rank),
        "self_rank": int(self_rank or 0),
        "r1": bool(label_rank <= 1),
        "r2": bool(label_rank <= 2),
        "r3": bool(label_rank <= 3),
        "mm_dist": float(dist[i, i]),
        "same_caption_count": int(same_count),
        "candidate_count_full": int(len(batch)),
        "top_candidates": top,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="outputs/evaluation/sequential_t2m/babel_official_val_30fps/manifest.jsonl")
    ap.add_argument("--gt-stream-dir", default="outputs/evaluation/sequential_t2m/babel_official_val_30fps/ms272/gt_0beta")
    ap.add_argument("--out-json", default="outputs/evaluation/sequential_t2m/babel_official_val_30fps/ms272/gt_0beta/metrics/subclip_ranks_labelaware_b32_yup.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rank-batch-size", type=int, default=32)
    ap.add_argument("--rank-seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    mean = np.load(Path(HUMANML_MEAN_STD) / "Mean.npy")
    std = np.load(Path(HUMANML_MEAN_STD) / "Std.npy")
    records = [json.loads(l) for l in Path(args.manifest).open() if l.strip()]

    rows: list[dict[str, Any]] = []
    seq_cache: dict[str, np.ndarray] = {}
    invalid: list[dict[str, Any]] = []
    for rec in records:
        sid = rec["id"]
        path = Path(args.gt_stream_dir) / f"{sid}.npz"
        if not path.exists():
            continue
        seq = np.asarray(np.load(path, allow_pickle=True)["motion_272"], dtype=np.float32)
        seq_cache[sid] = seq
        source_fps = float(rec.get("fps") or 30.0)
        source_start_t = float(rec.get("source_start_t") or 0.0)
        for seg_i, seg in enumerate(rec.get("segments", [])):
            s = int(seg["start"])
            e = min(int(seg["end"]), len(seq))
            cap = str(seg["caption"]).strip()
            eval_cap = rewrite_caption(cap)
            start_t = source_start_t + s / 30.0
            end_t = source_start_t + e / 30.0
            base = {
                "sid": sid,
                "seg_index": int(seg_i),
                "caption": cap,
                "raw_label": seg.get("raw_label") or cap,
                "eval_caption": eval_cap,
                "start_30": int(s),
                "end_30": int(e),
                "start_native": int(round(start_t * source_fps)),
                "end_native": int(round(end_t * source_fps)),
                "source_start_30": int(seg.get("source_start_30", s)),
                "source_end_30": int(seg.get("source_end_30", e)),
                "length_30": int(e - s),
                "source_fps": source_fps,
            }
            if e - s < 16:
                invalid.append({**base, "valid": False, "reason": "too_short_for_motionstreamer_evaluator"})
                continue
            rows.append(base)

    print(f"[ranks] valid={len(rows)} invalid={len(invalid)} device={device}", flush=True)
    textenc, motionenc = E.load_evaluator(device)
    captions = [r["eval_caption"] for r in rows]
    text_emb = _encode_text(captions, textenc, device, args.batch_size)
    motion_chunks = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        motions = []
        lengths = []
        for row in batch_rows:
            seq = seq_cache[row["sid"]]
            raw_seg = per_seg_canon(seq[row["start_30"] : row["end_30"]].astype(np.float32))
            if raw_seg is None:
                raise RuntimeError(f"unexpected invalid segment after filtering: {row['sid']}:{row['seg_index']}")
            m, L = norm_pad(raw_seg, mean, std)
            if m is None:
                raise RuntimeError(f"unexpected invalid segment after norm_pad: {row['sid']}:{row['seg_index']}")
            motions.append(m)
            lengths.append(int(L))
            row["eval_frames"] = int(L)
        motion_chunks.append(_encode_motion(motions, lengths, motionenc, device, args.batch_size))
        done = start + len(batch_rows)
        if done % 1024 == 0 or done == len(rows):
            print(f"[ranks] encoded motion {done}/{len(rows)}", flush=True)
    motion_emb = np.concatenate(motion_chunks, axis=0)
    dist = E.euclidean_distance_matrix(text_emb, motion_emb)

    batch_items = [(r["eval_caption"], np.empty((0,), dtype=np.float32), int(r.get("eval_frames", 0))) for r in rows]
    batches = _balanced_batches_all(batch_items, args.rank_batch_size, args.rank_seed)
    for batch in batches:
        for i in batch:
            rows[i]["ms_eval"] = _rank_one(i, batch, rows, dist, max(1, args.top_k))
            rows[i]["ms_eval"]["metric_caption"] = rows[i]["eval_caption"]
            rows[i]["ms_eval"]["raw_caption"] = rows[i]["caption"]
            rows[i]["ms_eval"]["eval_frames"] = int(rows[i].get("eval_frames", 0))
            rows[i]["ms_eval"]["raw_frames_30"] = int(rows[i]["length_30"])

    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        cases.setdefault(row["sid"], {"segments": []})["segments"].append(row)
    for row in invalid:
        row["ms_eval"] = {
            "valid": False,
            "reason": row["reason"],
            "source": "motionstreamer_272",
            "raw_frames_30": int(row["length_30"]),
        }
        cases.setdefault(row["sid"], {"segments": []})["segments"].append(row)
    for case in cases.values():
        case["segments"].sort(key=lambda r: int(r["seg_index"]))

    summary = {
        "manifest": str(args.manifest),
        "gt_stream_dir": str(args.gt_stream_dir),
        "valid_segments": len(rows),
        "invalid_segments": len(invalid),
        "case_count": len(cases),
        "rank_protocol": "label_aware_32way_balanced_tail_included",
        "rank_batch_size": int(args.rank_batch_size),
        "rank_seed": int(args.rank_seed),
        "top_k": int(args.top_k),
        "note": (
            "Rank is the first retrieved motion in the 32-candidate viewer batch "
            "whose evaluator caption equals the query caption; self_rank is the "
            "paired diagonal motion rank. The final tail batch is kept for "
            "visualization, so every valid subclip has a rank."
        ),
    }
    out = {"summary": summary, "cases": cases}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ranks] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

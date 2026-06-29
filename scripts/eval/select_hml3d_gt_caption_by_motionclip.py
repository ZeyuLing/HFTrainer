#!/usr/bin/env python3
"""Select GT-aligned HumanML3D captions by scoring official candidates.

This script addresses the failure mode where the first full-clip HumanML3D
caption is not merely terse but semantically mismatched to the motion.  It keeps
the source-of-truth constrained to official ``texts/<id>.txt`` rows: no new text
is generated.  For each motion id, every full-clip candidate caption is scored
against the GT MotionCLIP-135 motion; the caption with the smallest text-motion
embedding distance is materialized into a separate caption root.

The output is evaluator-assisted, not automatically human-certified.  Review
``needs_review.tsv`` before treating the result as final ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not REPO.exists():
    REPO = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from eval_with_motionclip_evaluator import encode_dataset, load_motionclip  # noqa: E402


@dataclass
class CaptionRow:
    index: int
    raw: str
    caption: str
    tokens: str
    start: float
    end: float

    @property
    def is_full_clip(self) -> bool:
        return self.start == 0.0 and self.end == 0.0

    @property
    def word_count(self) -> int:
        return len(self.caption.split())


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return REPO / p


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _float_tag(value: str) -> float:
    if value == "nan":
        return 0.0
    out = float(value)
    return 0.0 if out != out else out


def parse_text_file(path: Path) -> list[CaptionRow]:
    rows: list[CaptionRow] = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("#")
        if len(parts) < 4 or not parts[0].strip():
            continue
        try:
            start = _float_tag(parts[2])
            end = _float_tag(parts[3])
        except ValueError:
            continue
        rows.append(CaptionRow(
            index=idx,
            raw=raw,
            caption=parts[0].strip(),
            tokens=parts[1],
            start=start,
            end=end,
        ))
    return rows


def _fallback_caption(rows: list[CaptionRow]) -> tuple[CaptionRow | None, str]:
    if not rows:
        return None, "missing_text"
    full = [row for row in rows if row.is_full_clip]
    pool = full or rows
    suffix = "" if full else "_no_full"
    return max(pool, key=lambda r: (r.word_count, len(r.caption), -r.index)), "longest_full_fallback" + suffix


def _load_motion(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 135:
        return None
    return arr


def _write_caption_artifacts(
    out_root: Path,
    source_annotation: Path,
    records: dict[str, dict[str, Any]],
    prompt_map: dict[str, str],
    split_file: Path,
    stats: dict[str, Any],
) -> None:
    out_texts = out_root / "texts"
    out_caps = out_root / "annotation_captions"
    out_split = out_root / "split"
    out_texts.mkdir(parents=True, exist_ok=True)
    out_caps.mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)
    (out_split / "test.txt").write_text(
        "".join(f"{cid}\n" for cid in records),
        encoding="utf-8",
    )

    changed_rows = []
    needs_review = []
    for cid, rec in records.items():
        selected = rec["selected"]
        (out_texts / f"{cid}.txt").write_text(selected["raw"] + "\n", encoding="utf-8")
        cap_json = out_caps / f"{cid}.json"
        cap_json.write_text(
            json.dumps({"macro": [selected["caption"]], "meso": [], "micro": []},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rec["text_path"] = _rel_or_abs(out_texts / f"{cid}.txt")
        rec["annotation_caption_path"] = _rel_or_abs(cap_json)
        if rec.get("changed_from_first_full"):
            changed_rows.append((
                cid,
                rec.get("selection_reason", ""),
                rec.get("first_full_caption", ""),
                selected["caption"],
            ))
        if rec.get("needs_review"):
            needs_review.append((
                cid,
                rec.get("selection_reason", ""),
                rec.get("best_distance", ""),
                rec.get("best_margin", ""),
                rec.get("review_reason", ""),
                rec.get("first_full_caption", ""),
                selected["caption"],
            ))

    annotation_path = None
    if source_annotation.exists():
        anno = json.loads(source_annotation.read_text(encoding="utf-8"))
        data_list = anno.get("data_list", {})
        for cid, rec in records.items():
            if cid not in data_list:
                continue
            data_list[cid]["hierarchical_caption_path"] = rec["annotation_caption_path"]
            data_list[cid]["caption_source"] = "humanml3d_official_corrected_caption"
            data_list[cid]["caption_selection_policy"] = "best_motionclip_distance_over_official_full_captions"
        anno.setdefault("meta", {})
        anno["meta"]["caption_source"] = "humanml3d_official_corrected_caption"
        anno["meta"]["caption_selection_policy"] = "best_motionclip_distance_over_official_full_captions"
        anno["meta"]["caption_root"] = _rel_or_abs(out_root)
        annotation_path = out_root / "test_hml3d_official272_gtlen_official_caption.json"
        annotation_path.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")

    stats["annotation_path"] = _rel_or_abs(annotation_path) if annotation_path else None
    stats["changed_from_first_full"] = len(changed_rows)
    stats["needs_review"] = len(needs_review)

    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_root / "caption_map.json").write_text(
        json.dumps({"meta": stats, "data": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "prompt_map.json").write_text(json.dumps(prompt_map, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_root / "changed_from_first_full.tsv").open("w", encoding="utf-8") as f:
        f.write("id\treason\tfirst_full_caption\tselected_caption\n")
        for row in changed_rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    with (out_root / "needs_review.tsv").open("w", encoding="utf-8") as f:
        f.write("id\treason\tbest_distance\tbest_margin\treview_reason\tfirst_full_caption\tselected_caption\n")
        for row in needs_review:
            f.write("\t".join(str(x) for x in row) + "\n")
    (out_root / "README.md").write_text(
        "\n".join([
            "# MotionCLIP-Selected GT Captions",
            "",
            "This is an evaluator-assisted caption selection over official HumanML3D full-clip captions.",
            "It is meant to remove complete caption-motion mismatches, not merely prefer longer descriptions.",
            "",
            f"- ids: `{stats['num_ids']}`",
            f"- records: `{stats['num_records']}`",
            f"- scored by MotionCLIP: `{stats['scored_ids']}`",
            f"- fallback without MotionCLIP motion: `{stats['fallback_ids']}`",
            f"- changed from first full caption: `{stats['changed_from_first_full']}`",
            f"- needs review: `{stats['needs_review']}`",
            "",
            "Review `needs_review.tsv` before using this as a final benchmark caption set.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    ap.add_argument("--real-motionclip-dir", default="outputs/evaluation/t2m/humanml3d_official_test/motionclip_table1_20260619/motionclip135/real")
    ap.add_argument("--source-annotation", default="data/annotation/test_hml3d_official272_gtlen.json")
    ap.add_argument("--out-root", default="outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected")
    ap.add_argument("--evaluator-ckpt", default="checkpoints/motion_clip/motionclip_base_1p_aug_hq")
    ap.add_argument("--clip-pretrained", default="checkpoints/clip-vit-base-patch32")
    ap.add_argument("--stats-file", default="data/statistic/smplx55_stats_hymotion_aug.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--forward-batch-size", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--max-ids", type=int, default=0)
    ap.add_argument("--ids", default="",
                    help="Optional comma-separated ids for targeted smoke tests.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split the selected id list into N modulo shards.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Modulo shard index to process when --num-shards > 1.")
    ap.add_argument("--margin-review-threshold", type=float, default=0.02)
    args = ap.parse_args()

    src_root = _repo_path(args.src_root)
    real_dir = _repo_path(args.real_motionclip_dir)
    out_root = _repo_path(args.out_root)
    source_annotation = _repo_path(args.source_annotation)
    split_file = src_root / "split" / "test.txt"
    text_dir = src_root / "texts"
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    source_total_ids = len(ids)
    if args.ids.strip():
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        id_set = set(ids)
        missing = [x for x in wanted if x not in id_set]
        if missing:
            raise ValueError(f"ids not in split: {missing}")
        ids = wanted
    if args.max_ids > 0:
        ids = ids[:args.max_ids]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        ids = [cid for i, cid in enumerate(ids) if i % args.num_shards == args.shard_index]

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"[load] MotionCLIP device={device}", flush=True)
    bundle = load_motionclip(
        _repo_path(args.evaluator_ckpt),
        device,
        clip_pretrained=str(_repo_path(args.clip_pretrained)),
        stats_file=str(_repo_path(args.stats_file)),
    )

    candidate_meta = []
    captions = []
    motions = []
    lengths = []
    records: dict[str, dict[str, Any]] = {}
    prompt_map: dict[str, str] = {}
    reason_counter: Counter[str] = Counter()
    full_count_counter: Counter[int] = Counter()
    fallback_ids = 0

    for cid in ids:
        rows = parse_text_file(text_dir / f"{cid}.txt")
        full_rows = [row for row in rows if row.is_full_clip]
        full_count_counter[len(full_rows)] += 1
        motion = _load_motion(real_dir / f"{cid}.npy")
        selected, reason = _fallback_caption(rows)
        if selected is None:
            records[cid] = {
                "id": cid,
                "selection_reason": reason,
                "selected": None,
                "needs_review": True,
                "review_reason": "missing_text",
                "all_rows": [],
            }
            reason_counter[reason] += 1
            continue

        first_full = min(full_rows, key=lambda r: r.index) if full_rows else rows[0]
        rec = {
            "id": cid,
            "selection_reason": reason,
            "first_full_caption": first_full.caption,
            "changed_from_first_full": bool(selected.caption != first_full.caption),
            "selected": asdict(selected) | {"is_full_clip": selected.is_full_clip},
            "all_rows": [asdict(row) | {"is_full_clip": row.is_full_clip} for row in rows],
            "candidates": [],
            "needs_review": False,
            "review_reason": "",
        }
        records[cid] = rec

        if motion is None or not full_rows:
            fallback_ids += 1
            rec["needs_review"] = True
            rec["review_reason"] = "no_motionclip_motion" if motion is None else "no_full_caption"
            reason_counter[reason] += 1
            prompt_map[cid] = selected.caption
            continue

        for row in full_rows:
            candidate_meta.append((cid, row.index))
            captions.append(row.caption)
            motions.append(motion)
            lengths.append(min(int(motion.shape[0]), int(args.max_frames)))

    print(f"[encode] candidates={len(captions)} ids={len(ids)}", flush=True)
    if captions:
        text_emb, motion_emb = encode_dataset(
            bundle,
            captions,
            motions,
            lengths,
            device,
            forward_batch_size=args.forward_batch_size,
            max_frames=args.max_frames,
            l2_normalize=True,
        )
        dists = np.linalg.norm(text_emb - motion_emb, axis=1)
        by_id: dict[str, list[tuple[int, float]]] = {}
        for (cid, row_index), dist in zip(candidate_meta, dists):
            by_id.setdefault(cid, []).append((row_index, float(dist)))

        for cid, scored in by_id.items():
            rec = records[cid]
            scored.sort(key=lambda x: x[1])
            row_by_index = {row["index"]: row for row in rec["all_rows"]}
            candidates = []
            for rank, (row_index, dist) in enumerate(scored, start=1):
                row = row_by_index[row_index]
                candidates.append({
                    "rank": rank,
                    "line_index": row_index,
                    "caption": row["caption"],
                    "distance": dist,
                })
            best = candidates[0]
            second_dist = candidates[1]["distance"] if len(candidates) > 1 else None
            margin = None if second_dist is None else second_dist - best["distance"]
            selected_row = row_by_index[best["line_index"]]
            rec["selection_reason"] = "motionclip_min_distance"
            rec["selected"] = selected_row
            rec["changed_from_first_full"] = bool(selected_row["caption"] != rec["first_full_caption"])
            rec["candidates"] = candidates
            rec["best_distance"] = float(best["distance"])
            rec["second_distance"] = None if second_dist is None else float(second_dist)
            rec["best_margin"] = None if margin is None else float(margin)
            rec["needs_review"] = bool(margin is not None and margin < args.margin_review_threshold)
            rec["review_reason"] = "low_margin" if rec["needs_review"] else ""
            reason_counter["motionclip_min_distance"] += 1
            prompt_map[cid] = selected_row["caption"]

    # Fill prompt map for records that were fallback/missing from scoring.
    for cid, rec in records.items():
        if rec.get("selected") and cid not in prompt_map:
            prompt_map[cid] = rec["selected"]["caption"]

    distances = [
        float(rec["best_distance"]) for rec in records.values()
        if rec.get("best_distance") is not None
    ]
    margins = [
        float(rec["best_margin"]) for rec in records.values()
        if rec.get("best_margin") is not None
    ]
    stats = {
        "num_ids": len(ids),
        "source_total_ids": source_total_ids,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "num_records": sum(1 for rec in records.values() if rec.get("selected")),
        "scored_ids": sum(1 for rec in records.values() if rec.get("selection_reason") == "motionclip_min_distance"),
        "fallback_ids": fallback_ids,
        "selection_reasons": dict(reason_counter),
        "full_caption_count_distribution": {str(k): int(v) for k, v in sorted(full_count_counter.items())},
        "distance_mean": float(np.mean(distances)) if distances else None,
        "distance_p95": float(np.percentile(distances, 95)) if distances else None,
        "distance_p99": float(np.percentile(distances, 99)) if distances else None,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_p05": float(np.percentile(margins, 5)) if margins else None,
        "margin_review_threshold": float(args.margin_review_threshold),
        "source_root": _rel_or_abs(src_root),
        "real_motionclip_dir": _rel_or_abs(real_dir),
        "out_root": _rel_or_abs(out_root),
    }

    _write_caption_artifacts(out_root, source_annotation, records, prompt_map, split_file, stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge sharded HumanML3D caption-selection outputs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not REPO.exists():
    REPO = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return REPO / p


def _rel_or_abs(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _load_records(shard_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = shard_root / "caption_map.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("meta", {}), payload.get("data", {})


def _write_outputs(
    out_root: Path,
    records: dict[str, dict[str, Any]],
    source_annotation: Path,
    source_split: Path,
    source_metas: list[dict[str, Any]],
) -> dict[str, Any]:
    out_texts = out_root / "texts"
    out_caps = out_root / "annotation_captions"
    out_split = out_root / "split"
    out_texts.mkdir(parents=True, exist_ok=True)
    out_caps.mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)

    ordered_ids = [x.strip() for x in source_split.read_text(encoding="utf-8").splitlines() if x.strip()]
    missing = [cid for cid in ordered_ids if cid not in records]
    extras = sorted(set(records) - set(ordered_ids))
    if missing or extras:
        raise ValueError(f"shard coverage mismatch: missing={len(missing)} extras={len(extras)}")

    records = {cid: records[cid] for cid in ordered_ids}
    prompt_map: dict[str, str] = {}
    changed_rows: list[tuple[Any, ...]] = []
    review_rows: list[tuple[Any, ...]] = []
    reason_counter: Counter[str] = Counter()
    full_count_counter: Counter[int] = Counter()
    fallback_ids = 0

    for cid, rec in records.items():
        selected = rec.get("selected")
        if not selected:
            continue
        (out_texts / f"{cid}.txt").write_text(selected["raw"] + "\n", encoding="utf-8")
        cap_json = out_caps / f"{cid}.json"
        cap_json.write_text(
            json.dumps({"macro": [selected["caption"]], "meso": [], "micro": []},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rec["text_path"] = _rel_or_abs(out_texts / f"{cid}.txt")
        rec["annotation_caption_path"] = _rel_or_abs(cap_json)
        prompt_map[cid] = selected["caption"]

        reason = rec.get("selection_reason", "")
        reason_counter[reason] += 1
        if reason != "motionclip_min_distance":
            fallback_ids += 1
        full_count_counter[sum(1 for row in rec.get("all_rows", []) if row.get("is_full_clip"))] += 1
        if rec.get("changed_from_first_full"):
            changed_rows.append((
                cid,
                reason,
                rec.get("first_full_caption", ""),
                selected["caption"],
            ))
        if rec.get("needs_review"):
            review_rows.append((
                cid,
                reason,
                rec.get("best_distance", ""),
                rec.get("best_margin", ""),
                rec.get("review_reason", ""),
                rec.get("first_full_caption", ""),
                selected["caption"],
            ))

    annotation_path: Path | None = None
    if source_annotation.exists():
        anno = json.loads(source_annotation.read_text(encoding="utf-8"))
        data_list = anno.get("data_list", {})
        for cid, rec in records.items():
            if cid not in data_list or not rec.get("selected"):
                continue
            data_list[cid]["hierarchical_caption_path"] = rec["annotation_caption_path"]
            data_list[cid]["caption_source"] = "motionclip_selected_official_humanml3d_caption"
            data_list[cid]["caption_selection_policy"] = "best_motionclip_distance_over_official_full_captions"
        anno.setdefault("meta", {})
        anno["meta"]["caption_source"] = "motionclip_selected_official_humanml3d_caption"
        anno["meta"]["caption_selection_policy"] = "best_motionclip_distance_over_official_full_captions"
        anno["meta"]["caption_root"] = _rel_or_abs(out_root)
        annotation_path = out_root / "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
        annotation_path.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")

    distances = [
        float(rec["best_distance"]) for rec in records.values()
        if rec.get("best_distance") is not None
    ]
    margins = [
        float(rec["best_margin"]) for rec in records.values()
        if rec.get("best_margin") is not None
    ]
    stats = {
        "num_ids": len(ordered_ids),
        "num_records": sum(1 for rec in records.values() if rec.get("selected")),
        "scored_ids": reason_counter.get("motionclip_min_distance", 0),
        "fallback_ids": fallback_ids,
        "selection_reasons": dict(reason_counter),
        "full_caption_count_distribution": {str(k): int(v) for k, v in sorted(full_count_counter.items())},
        "distance_mean": float(np.mean(distances)) if distances else None,
        "distance_p95": float(np.percentile(distances, 95)) if distances else None,
        "distance_p99": float(np.percentile(distances, 99)) if distances else None,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_p05": float(np.percentile(margins, 5)) if margins else None,
        "margin_review_threshold": source_metas[0].get("margin_review_threshold") if source_metas else None,
        "source_root": source_metas[0].get("source_root") if source_metas else None,
        "real_motionclip_dir": source_metas[0].get("real_motionclip_dir") if source_metas else None,
        "out_root": _rel_or_abs(out_root),
        "annotation_path": _rel_or_abs(annotation_path),
        "changed_from_first_full": len(changed_rows),
        "needs_review": len(review_rows),
        "merged_from_shards": len(source_metas),
    }

    (out_split / "test.txt").write_text("".join(f"{cid}\n" for cid in ordered_ids), encoding="utf-8")
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
        for row in review_rows:
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
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", required=True,
                    help="Directory containing shard_0, shard_1, ... outputs.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--source-annotation", default="data/annotation/test_hml3d_official272_gtlen.json")
    ap.add_argument("--source-split", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt")
    args = ap.parse_args()

    shard_root = _repo_path(args.shard_root)
    out_root = _repo_path(args.out_root)
    source_annotation = _repo_path(args.source_annotation)
    source_split = _repo_path(args.source_split)

    shard_dirs = sorted(p for p in shard_root.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* directories under {shard_root}")

    records: dict[str, dict[str, Any]] = {}
    metas: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        meta, shard_records = _load_records(shard_dir)
        metas.append(meta)
        for cid, rec in shard_records.items():
            if cid in records:
                raise ValueError(f"duplicate id {cid} in {shard_dir}")
            records[cid] = rec

    stats = _write_outputs(out_root, records, source_annotation, source_split, metas)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[merge-error] {exc}", file=sys.stderr)
        raise

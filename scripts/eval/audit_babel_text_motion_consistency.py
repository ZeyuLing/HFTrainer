#!/usr/bin/env python3
"""Audit BABEL-stream captions against the paired 272-dim motions.

This is a lightweight sanity checker for the BABEL sequential-generation split.
It does not run model inference.  The main signal is directional consistency:
for labels that explicitly say left/right and describe locomotion, estimate the
root displacement in the body's local right direction from the recovered 272
joints.  Large opposite signs are strong evidence that the text attached to the
motion segment is wrong or mirrored upstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_272_stored_positions,
)


LOCOMOTION_RE = re.compile(
    r"\b(walk|step|sidestep|side step|shuffle|move toward|move to|go|run|jump|hop|backpedal|back peddle|travel)\b"
)
OBJECT_RE = re.compile(
    r"\b(object|item|handle|phone|ball|vacuum|hose|cup|broom|suitcase|box|chair|table)\b"
)
BODY_PART_RE = re.compile(
    r"\b(hand|arm|leg|foot|knee|elbow|wrist|shoulder|head|hip)\b"
)
WORD_RE = re.compile(r"[a-z]+")
DIRECTION_CONTEXT_PATTERNS = (
    re.compile(r"\b(?:side step|sidestep)(?: to)? (?:the )?(?:left|right)\b"),
    re.compile(r"\bstep (?:over )?(?:to )?(?:the )?(?:left|right)\b"),
    re.compile(r"\bwalk (?:toward |to |[0-9]+ degrees )?(?:the )?(?:left|right)\b"),
    re.compile(r"\bmove (?:toward |to )(?:the )?(?:left|right)\b"),
    re.compile(r"\bgo (?:toward |to )(?:the )?(?:left|right)\b"),
    re.compile(r"\brun (?:toward |to )(?:the )?(?:left|right)\b"),
)

OPPOSITE_PAIRS = (
    ("left", "right"),
    ("right", "left"),
    ("forward", "backward"),
    ("forward", "backwards"),
    ("backward", "forward"),
    ("backwards", "forward"),
    ("up", "down"),
    ("down", "up"),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(REPO / "data/babel/babel_seq_val_manifest.jsonl"))
    ap.add_argument("--motion-dir", default=str(REPO / "data/babel_272_stream/val_stream"))
    ap.add_argument("--rewrite-cache", default=str(REPO / "data/babel/babel_caption_rewrites.json"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs/evaluation/babel_text_motion_audit_20260624"))
    ap.add_argument("--lat-threshold", type=float, default=0.06)
    ap.add_argument("--strong-lat-threshold", type=float, default=0.10)
    return ap.parse_args()


def words(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def single_side(text: str) -> str:
    ws = words(text)
    if "left" in ws and "right" not in ws:
        return "left"
    if "right" in ws and "left" not in ws:
        return "right"
    return ""


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rewrites(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k).strip().lower(): str(v) for k, v in payload.get("rewrites", {}).items()}


def normalized_word_text(text: str) -> str:
    return " " + re.sub(r"[^a-z]+", " ", (text or "").lower()) + " "


def rewrite_opposite_flags(rewrites: dict[str, str]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for raw, cap in sorted(rewrites.items()):
        raw_norm = normalized_word_text(raw)
        cap_norm = normalized_word_text(cap)
        for src, dst in OPPOSITE_PAIRS:
            if f" {src} " in raw_norm and f" {dst} " in cap_norm and f" {src} " not in cap_norm:
                flags.append({"raw": raw, "rewrite": cap, "source_word": src, "opposite_word": dst})
                break
    return flags


def side_is_direction(label: str) -> bool:
    text = re.sub(r"\s+", " ", (label or "").lower().replace("-", " ")).strip()
    return any(pattern.search(text) for pattern in DIRECTION_CONTEXT_PATTERNS)


def right_axis_for(pos: np.ndarray, start: int, end: int) -> np.ndarray:
    clip = pos[max(0, start): max(start + 1, end)]
    if clip.size == 0:
        clip = pos[start:start + 1]
    vectors = []
    # R - L for hips, shoulders, knees, ankles.  Averaging reduces pose noise.
    for left_idx, right_idx in ((1, 2), (16, 17), (4, 5), (7, 8)):
        vec = clip[:, right_idx, [0, 2]] - clip[:, left_idx, [0, 2]]
        norm = np.linalg.norm(vec, axis=1)
        if float(np.nanmean(norm)) > 1e-6:
            vectors.append(vec / (norm[:, None] + 1e-9))
    if not vectors:
        return np.array([1.0, 0.0], dtype=np.float32)
    axis = np.nanmean(np.concatenate(vectors, axis=0), axis=0)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (axis / axis_norm).astype(np.float32)


def segment_lateral_motion(pos: np.ndarray, start: int, end: int) -> tuple[float, float, int]:
    total = len(pos)
    s = max(0, min(int(start), total - 1))
    e = max(s + 1, min(int(end), total))
    window = max(1, min(15, (e - s) // 5 if e - s >= 5 else 1))
    start_root = pos[s:s + window, 0, :][:, [0, 2]].mean(axis=0)
    end_root = pos[e - window:e, 0, :][:, [0, 2]].mean(axis=0)
    delta = end_root - start_root
    right_axis = right_axis_for(pos, s, e)
    lateral = float(np.dot(delta, right_axis))
    distance = float(np.linalg.norm(delta))
    return lateral, distance, e - s


def audit_direction(rows: list[dict[str, Any]], motion_dir: Path, lat_threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pos_cache: dict[str, np.ndarray] = {}
    for rec in rows:
        sid = rec["id"]
        motion_path = motion_dir / f"{sid}.npy"
        if not motion_path.exists():
            continue
        for seg_idx, seg in enumerate(rec.get("segments", [])):
            label = str(seg.get("caption", ""))
            side = single_side(label)
            if not side:
                continue
            if sid not in pos_cache:
                pos_cache[sid] = recover_272_stored_positions(np.load(motion_path).astype(np.float32))
            lateral, distance, frames = segment_lateral_motion(pos_cache[sid], seg["start"], seg["end"])
            predicted = "right" if lateral > lat_threshold else "left" if lateral < -lat_threshold else "neutral"
            locomotion = bool(LOCOMOTION_RE.search(label.lower()))
            object_term = bool(OBJECT_RE.search(label.lower()))
            body_part = bool(BODY_PART_RE.search(label.lower()))
            direction_context = side_is_direction(label)
            mismatch = bool(predicted != "neutral" and predicted != side)
            high_confidence = bool(direction_context and not object_term and abs(lateral) >= 0.10)
            out.append(
                {
                    "id": sid,
                    "segment_index": seg_idx,
                    "start": int(seg["start"]),
                    "end": int(seg["end"]),
                    "frames": int(frames),
                    "caption": label,
                    "label_side": side,
                    "predicted_root_side": predicted,
                    "lateral_m": lateral,
                    "root_distance_m": distance,
                    "locomotion_label": locomotion,
                    "direction_context": direction_context,
                    "object_label": object_term,
                    "body_part_label": body_part,
                    "mismatch": mismatch,
                    "high_confidence": high_confidence,
                    "high_confidence_mismatch": bool(high_confidence and mismatch),
                }
            )
    return out


def swapped_two_segment_pairs(direction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in direction_rows:
        by_id.setdefault(str(row["id"]), []).append(row)
    out: list[dict[str, Any]] = []
    for sid, items in sorted(by_id.items()):
        if len(items) != 2:
            continue
        sides = {items[0]["label_side"], items[1]["label_side"]}
        if sides != {"left", "right"}:
            continue
        both_opposite = all(row["mismatch"] and row["predicted_root_side"] != "neutral" for row in items)
        if both_opposite:
            out.append(
                {
                    "id": sid,
                    "seg0_caption": items[0]["caption"],
                    "seg0_label": items[0]["label_side"],
                    "seg0_pred": items[0]["predicted_root_side"],
                    "seg0_lateral_m": items[0]["lateral_m"],
                    "seg1_caption": items[1]["caption"],
                    "seg1_label": items[1]["label_side"],
                    "seg1_pred": items[1]["predicted_root_side"],
                    "seg1_lateral_m": items[1]["lateral_m"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(Path(args.manifest))
    rewrites = load_rewrites(Path(args.rewrite_cache))
    rewrite_flags = rewrite_opposite_flags(rewrites)
    direction_rows = audit_direction(manifest_rows, Path(args.motion_dir), args.lat_threshold)
    high_conf_mismatch = [
        row for row in direction_rows
        if row["high_confidence_mismatch"] and abs(float(row["lateral_m"])) >= args.strong_lat_threshold
    ]
    swapped_pairs = swapped_two_segment_pairs(direction_rows)

    write_csv(out_dir / "rewrite_opposite_flags.csv", rewrite_flags)
    write_csv(out_dir / "direction_side_flags.csv", direction_rows)
    write_csv(out_dir / "high_confidence_direction_mismatches.csv", high_conf_mismatch)
    write_csv(out_dir / "swapped_two_segment_pairs.csv", swapped_pairs)

    summary = {
        "manifest": str(Path(args.manifest)),
        "motion_dir": str(Path(args.motion_dir)),
        "rewrite_cache": str(Path(args.rewrite_cache)),
        "num_manifest_rows": len(manifest_rows),
        "num_rewrite_opposite_flags": len(rewrite_flags),
        "num_side_labeled_segments": len(direction_rows),
        "num_high_confidence_direction_segments": sum(bool(row["high_confidence"]) for row in direction_rows),
        "num_high_confidence_direction_mismatches": len(high_conf_mismatch),
        "num_swapped_two_segment_pairs": len(swapped_pairs),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if high_conf_mismatch:
        print("\nTop high-confidence mismatches:")
        for row in sorted(high_conf_mismatch, key=lambda r: -abs(float(r["lateral_m"])))[:20]:
            print(
                f"{row['id']} seg{row['segment_index']} label={row['label_side']} "
                f"pred={row['predicted_root_side']} lat={float(row['lateral_m']):+.3f}m :: {row['caption']}"
            )


if __name__ == "__main__":
    main()

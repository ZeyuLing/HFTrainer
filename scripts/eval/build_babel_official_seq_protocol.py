#!/usr/bin/env python3
"""Build the official-BABEL sequential evaluation manifest at 30 FPS.

The output is compatible with ``scripts/eval/eval_babel_seq_ms272.py``. It
implements the PRISM/BABEL protocol documented in
``data/babel_official/README.md``:

* resample labels to 30 FPS;
* remove explicit ``transition`` captions from the text stream;
* split an explicit transition interval equally into the neighboring actions;
* keep the original cut for adjacent actions when there is no transition label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _is_transition(seg: dict[str, Any]) -> bool:
    label = str(seg.get("caption") or seg.get("proc_label") or seg.get("raw_label") or "")
    return bool(seg.get("is_transition")) or label.strip().lower() == "transition"


def _frame_at_time(t: float, fps: float) -> int:
    return int(round(float(t) * float(fps)))


def _seg_to_30(seg: dict[str, Any], target_fps: float, total30: int) -> dict[str, Any] | None:
    if "start_t" in seg and "end_t" in seg:
        start = _frame_at_time(seg["start_t"], target_fps)
        end = _frame_at_time(seg["end_t"], target_fps)
    else:
        src_fps = float(seg.get("fps", target_fps))
        start = int(round(float(seg["start_frame"]) * target_fps / src_fps))
        end = int(round(float(seg["end_frame"]) * target_fps / src_fps))
    start = max(0, min(start, total30))
    end = max(0, min(end, total30))
    if end <= start:
        return None
    return {
        "caption": str(seg.get("caption") or seg.get("proc_label") or seg.get("raw_label") or "").strip(),
        "raw_label": str(seg.get("raw_label") or "").strip(),
        "start": start,
        "end": end,
        "is_transition": _is_transition(seg),
        "source": seg,
    }


def _transition_between(left: dict[str, Any], right: dict[str, Any], transitions: list[dict[str, Any]]) -> tuple[int | None, int]:
    """Return midpoint of explicit transitions between two actions, if any."""
    hits = []
    for tr in transitions:
        if tr["end"] <= left["start"] or tr["start"] >= right["end"]:
            continue
        # The normal case is left.end <= transition <= right.start. Keep a small
        # tolerance because BABEL annotations are often rounded from seconds.
        if tr["start"] >= left["end"] - 1 and tr["end"] <= right["start"] + 1:
            hits.append(tr)
    if not hits:
        return None, 0
    start = min(h["start"] for h in hits)
    end = max(h["end"] for h in hits)
    return int(round((start + end) / 2.0)), len(hits)


def build_record(raw: dict[str, Any], target_fps: float, min_segments: int) -> tuple[dict[str, Any] | None, Counter]:
    stats = Counter()
    duration = raw.get("duration_sec_npz") or raw.get("duration_sec_babel")
    if duration is None:
        duration = float(raw["num_frames"]) / float(raw["fps"])
    total30_full = max(1, _frame_at_time(duration, target_fps))

    segs = []
    for seg in raw.get("segments", []):
        if seg.get("seq_level"):
            continue
        x = _seg_to_30(seg, target_fps, total30_full)
        if x is not None and x["caption"]:
            segs.append(x)
    segs.sort(key=lambda s: (s["start"], s["end"], s["caption"]))

    actions = [s for s in segs if not s["is_transition"]]
    transitions = [s for s in segs if s["is_transition"]]
    if len(actions) < min_segments:
        stats["skip_few_actions"] += 1
        return None, stats

    starts = [actions[0]["start"]]
    ends: list[int] = []
    protocol_notes = []
    for i in range(len(actions) - 1):
        left, right = actions[i], actions[i + 1]
        midpoint, n_hit = _transition_between(left, right, transitions)
        if midpoint is not None:
            cut = midpoint
            stats["explicit_transition_cuts"] += 1
            protocol_notes.append({
                "between": [left["caption"], right["caption"]],
                "kind": "transition_midpoint",
                "cut": int(cut),
                "transition_count": int(n_hit),
            })
        else:
            cut = right["start"]
            stats["native_cuts"] += 1
            if left["end"] != right["start"]:
                stats["native_overlap_or_gap_cuts"] += 1
                protocol_notes.append({
                    "between": [left["caption"], right["caption"]],
                    "kind": "native_onset_overlap_or_gap",
                    "left_end": int(left["end"]),
                    "right_start": int(right["start"]),
                    "cut": int(cut),
                })
        cut = max(starts[-1] + 1, min(int(cut), right["end"] - 1))
        ends.append(cut)
        starts.append(cut)
    ends.append(actions[-1]["end"])

    clip_start = starts[0]
    clip_end = ends[-1]
    if clip_end <= clip_start:
        stats["skip_bad_clip"] += 1
        return None, stats

    out_segments = []
    for action, start, end in zip(actions, starts, ends):
        rs = int(start - clip_start)
        re = int(end - clip_start)
        if re <= rs:
            stats["drop_empty_segment"] += 1
            continue
        out_segments.append({
            "caption": action["caption"],
            "raw_label": action["raw_label"],
            "start": rs,
            "end": re,
            "source_start_30": int(start),
            "source_end_30": int(end),
        })
    if len(out_segments) < min_segments:
        stats["skip_after_empty"] += 1
        return None, stats

    total = int(clip_end - clip_start)
    boundaries = [int(s["start"]) for s in out_segments[1:]]
    rec = {
        "id": raw["id"],
        "babel_id": raw.get("babel_id"),
        "split": raw.get("split"),
        "amass_path": raw.get("amass_path"),
        "fps": raw.get("fps"),
        "target_fps": target_fps,
        "source_start_30": int(clip_start),
        "source_end_30": int(clip_end),
        "source_start_t": float(clip_start / target_fps),
        "source_end_t": float(clip_end / target_fps),
        "total_frames": total,
        "boundaries": boundaries,
        "segments": out_segments,
        "protocol": "official_babel_transition_midpoint_30fps",
        "protocol_notes": protocol_notes,
    }
    stats["ok"] += 1
    stats[f"segments_{len(out_segments)}"] += 1
    stats["transitions_seen"] += len(transitions)
    return rec, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel_official/processed/manifests/val.jsonl")
    ap.add_argument("--out", default="outputs/evaluation/babel/official_val/msstyle_30fps_gt/manifest.jsonl")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--min-segments", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats_out = Path(args.stats_out) if args.stats_out else out.with_suffix(".stats.json")

    total_stats = Counter()
    n_in = 0
    with src.open() as fi, out.open("w") as fo:
        for line in fi:
            if not line.strip():
                continue
            n_in += 1
            raw = json.loads(line)
            rec, stats = build_record(raw, args.target_fps, args.min_segments)
            total_stats.update(stats)
            if rec is not None:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.limit and n_in >= args.limit:
                break

    summary = {"input_records": n_in, **dict(total_stats)}
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[manifest] wrote {summary.get('ok', 0)} / {n_in} -> {out}")
    print(f"[manifest] stats -> {stats_out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

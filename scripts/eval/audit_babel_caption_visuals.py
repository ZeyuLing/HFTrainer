#!/usr/bin/env python3
"""Sample BABEL sequence motions next to raw and rewritten captions.

The script is intentionally read-only with respect to BABEL data. It renders a
small static HTML audit page from the processed MS-272 validation streams so
caption rewrite mistakes can be inspected together with the actual motion.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

from babel_caption import rewrite_caption  # noqa: E402
from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_272_stored_positions,
)
from hftrainer.motion.skeleton.names import SMPL22_NAMES, SMPL22_PARENTS  # noqa: E402


HIGH_RISK_LABELS = {
    "a pose",
    "a-pose",
    "about face",
    "about face left",
    "about face right",
    "adjust",
    "animate wave",
    "place object to the right",
    "place object to the left",
    "reach to grab an object",
    "grab an object",
    "move object to the left",
    "move object to the right",
    "walk backwards",
    "step backward",
    "turn around",
    "t-pose",
}

MOTION_WORDS = {
    "left",
    "right",
    "forward",
    "backward",
    "backwards",
    "around",
    "up",
    "down",
    "object",
    "grab",
    "place",
    "throw",
    "catch",
    "turn",
    "step",
    "walk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO / "data/babel/babel_seq_val_manifest.jsonl"),
    )
    parser.add_argument(
        "--motion-dir",
        default=str(REPO / "data/babel_272_stream/val_stream"),
    )
    parser.add_argument(
        "--rewrite-cache",
        default=str(REPO / "data/babel/babel_caption_rewrites.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "outputs/evaluation/babel_caption_audit_20260623"),
    )
    parser.add_argument("--num-suspicious", type=int, default=24)
    parser.add_argument("--num-random", type=int, default=12)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rewrite_cache(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rewrites = {k.strip().lower(): v for k, v in payload.get("rewrites", {}).items()}
    meta = {k: v for k, v in payload.items() if k != "rewrites"}
    return rewrites, meta


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def segment_rewrite_info(label: str, rewrites: dict[str, str]) -> dict[str, Any]:
    raw = label.strip()
    cache = rewrites.get(raw.lower(), rewrite_caption(raw, use_cache=True))
    rule = rewrite_caption(raw, use_cache=False)
    flags: list[str] = []

    if raw.lower() in HIGH_RISK_LABELS:
        flags.append("high-risk label")
    if cache.strip().lower().rstrip(".") != rule.strip().lower().rstrip("."):
        flags.append("cache != rule")
    if raw.lower().replace("-", " ") == "a pose" and "pose" in cache.lower() and "a-pose" not in cache.lower():
        flags.append("A-pose rewritten as generic pose")
    if raw.lower().startswith("about face") and "turn" not in cache.lower() and "face" in cache.lower():
        flags.append("about-face may lose turning semantics")
    raw_motion_words = words(raw) & MOTION_WORDS
    cache_words = words(cache)
    dropped = sorted(w for w in raw_motion_words if w not in cache_words)
    if dropped:
        flags.append("dropped motion words: " + ", ".join(dropped))
    if any(phrase in cache.lower() for phrase in ("does something", "performs an action", "adjusts something")):
        flags.append("generic or hallucinated object")

    return {
        "raw": raw,
        "cache": cache,
        "rule": rule,
        "flags": flags,
        "score": len(flags),
    }


def annotate_records(records: list[dict[str, Any]], rewrites: dict[str, str]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for rec in records:
        segs = []
        score = 0
        for seg in rec.get("segments", []):
            info = segment_rewrite_info(seg.get("caption", ""), rewrites)
            segs.append({**seg, **info})
            score += info["score"]
        annotated.append({**rec, "segments": segs, "rewrite_score": score})
    return annotated


def choose_records(records: list[dict[str, Any]], motion_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    existing = [rec for rec in records if (motion_dir / f"{rec['id']}.npy").exists()]
    suspicious = sorted(
        [rec for rec in existing if rec["rewrite_score"] > 0],
        key=lambda r: (r["rewrite_score"], len(r.get("segments", [])), r["total_frames"]),
        reverse=True,
    )
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in suspicious[: args.num_suspicious]:
        picked.append(rec)
        seen.add(rec["id"])

    rng = random.Random(args.seed)
    candidates = [rec for rec in existing if rec["id"] not in seen]
    rng.shuffle(candidates)
    picked.extend(candidates[: args.num_random])
    return picked


def equalize_3d_axes(ax: Any, radius: float) -> None:
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(0.0, radius * 1.8)
    ax.set_box_aspect((1, 1, 1.1))


def draw_skeleton(ax: Any, joints: np.ndarray, color: str) -> None:
    root = joints[0].copy()
    centered = joints.copy()
    centered[:, 0] -= root[0]
    centered[:, 2] -= root[2]
    xs = centered[:, 0]
    ys = centered[:, 2]
    zs = centered[:, 1]
    for j, p in enumerate(SMPL22_PARENTS):
        if p < 0:
            continue
        ax.plot([xs[p], xs[j]], [ys[p], ys[j]], [zs[p], zs[j]], color=color, linewidth=2.0)
    ax.scatter(xs, ys, zs, s=8, color=color, alpha=0.9)
    ax.scatter([0], [0], [zs[0]], s=20, color="#111111")
    ax.view_init(elev=18, azim=-68)
    ax.set_axis_off()
    equalize_3d_axes(ax, 1.15)


def segment_frames(start: int, end: int, count: int) -> list[int]:
    lo = max(0, int(start))
    hi = max(lo + 1, int(end))
    if hi - lo <= count:
        return list(range(lo, hi))
    return sorted({int(round(x)) for x in np.linspace(lo, hi - 1, count)})


def render_record(rec: dict[str, Any], motion_path: Path, out_path: Path, frames_per_segment: int) -> dict[str, Any]:
    motion = np.load(motion_path).astype(np.float32)
    joints = recover_272_stored_positions(motion)
    segs = rec.get("segments", [])
    nseg = max(1, len(segs))
    colors = ["#2f6fed", "#c44e52", "#3f8f5b", "#8a63d2", "#b7791f", "#2b8c9c"]

    fig = plt.figure(figsize=(max(12, frames_per_segment * 2.25), 2.4 + nseg * 2.2))
    gs = fig.add_gridspec(nseg + 1, frames_per_segment, height_ratios=[1.0] + [1.5] * nseg)

    root = joints[:, 0, :]
    ax_path = fig.add_subplot(gs[0, :])
    ax_path.plot(root[:, 0], root[:, 2], color="#222222", linewidth=1.4, alpha=0.85)
    for i, seg in enumerate(segs):
        s = max(0, min(int(seg["start"]), len(root) - 1))
        e = max(s + 1, min(int(seg["end"]), len(root)))
        c = colors[i % len(colors)]
        ax_path.plot(root[s:e, 0], root[s:e, 2], color=c, linewidth=2.2)
        ax_path.scatter(root[s, 0], root[s, 2], color=c, s=28, marker="o")
        ax_path.scatter(root[e - 1, 0], root[e - 1, 2], color=c, s=30, marker="x")
    ax_path.set_title(f"{rec['id']} root trajectory (X-Z), {len(joints)} frames", fontsize=10)
    ax_path.set_xlabel("X")
    ax_path.set_ylabel("Z")
    ax_path.grid(True, linewidth=0.3, alpha=0.35)
    ax_path.set_aspect("equal", adjustable="datalim")

    rendered_frames: list[dict[str, Any]] = []
    for row, seg in enumerate(segs or [{"start": 0, "end": len(joints), "raw": "full clip"}], start=1):
        c = colors[(row - 1) % len(colors)]
        frames = segment_frames(seg.get("start", 0), min(seg.get("end", len(joints)), len(joints)), frames_per_segment)
        rendered_frames.append({"caption": seg.get("raw", ""), "frames": frames})
        for col in range(frames_per_segment):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            if col < len(frames):
                frame = frames[col]
                draw_skeleton(ax, joints[frame], c)
                ax.set_title(f"f={frame}", fontsize=8, pad=0)
            else:
                ax.set_axis_off()
        label = textwrap.shorten(seg.get("raw", "full clip"), width=44, placeholder="...")
        fig.text(0.015, 1.0 - (row + 0.18) / (nseg + 1), label, color=c, fontsize=9, va="center")

    fig.tight_layout(rect=(0.06, 0.02, 0.995, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "motion_shape": list(motion.shape),
        "rendered_frames": rendered_frames,
        "image": out_path.name,
    }


def row_html(rec: dict[str, Any], rel_img: str) -> str:
    seg_rows = []
    for seg in rec.get("segments", []):
        flags = "".join(f"<span class='flag'>{html.escape(flag)}</span>" for flag in seg.get("flags", []))
        flag_cell = flags if flags else '<span class="ok">ok</span>'
        seg_rows.append(
            "<tr>"
            f"<td>{seg.get('start', '')}-{seg.get('end', '')}</td>"
            f"<td><code>{html.escape(seg.get('raw', ''))}</code></td>"
            f"<td>{html.escape(seg.get('cache', ''))}</td>"
            f"<td>{html.escape(seg.get('rule', ''))}</td>"
            f"<td>{flag_cell}</td>"
            "</tr>"
        )
    return f"""
    <section class="case">
      <h2>{html.escape(rec['id'])} <span>{rec.get('total_frames', '?')} frames, score {rec.get('rewrite_score', 0)}</span></h2>
      <img src="{html.escape(rel_img)}" alt="{html.escape(rec['id'])} motion strip">
      <table>
        <thead><tr><th>Frames</th><th>Raw BABEL Label</th><th>Cache Rewrite</th><th>Rule Rewrite</th><th>Flags</th></tr></thead>
        <tbody>{''.join(seg_rows)}</tbody>
      </table>
    </section>
    """


def write_html(out_dir: Path, selected: list[dict[str, Any]], meta: dict[str, Any], audit: list[dict[str, Any]]) -> None:
    rows = [row_html(rec, f"images/{rec['id']}.png") for rec in selected]
    payload = html.escape(json.dumps(meta, indent=2, ensure_ascii=False))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BABEL Caption Rewrite Audit</title>
  <style>
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: #202124; background: #f7f8fa; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 14px 22px; background: #ffffff; border-bottom: 1px solid #d9dde5; }}
    h1 {{ margin: 0 0 4px; font-size: 20px; letter-spacing: 0; }}
    header p {{ margin: 2px 0; color: #5b6472; font-size: 13px; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 18px 22px 38px; }}
    .case {{ margin: 0 0 22px; padding: 16px; background: #fff; border: 1px solid #d9dde5; border-radius: 8px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; letter-spacing: 0; }}
    h2 span {{ color: #6b7280; font-weight: 500; font-size: 13px; margin-left: 8px; }}
    img {{ display: block; max-width: 100%; height: auto; border: 1px solid #e1e5ec; border-radius: 6px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-top: 1px solid #e8ebf0; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #4b5563; background: #f4f6f9; font-weight: 650; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
    .flag {{ display: inline-block; margin: 0 4px 4px 0; padding: 2px 6px; border-radius: 999px; color: #7a2e0e; background: #fff0df; border: 1px solid #ffd4a8; }}
    .ok {{ color: #256f3a; }}
    details {{ margin-top: 8px; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 10px; border-radius: 6px; overflow: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>BABEL caption rewrite audit</h1>
    <p>Selected {len(selected)} validation sequences from processed MS-272 streams. Each case shows raw BABEL label, cached LLM rewrite, rule rewrite, root trajectory, and segment-level skeleton snapshots.</p>
    <details><summary>Rewrite cache metadata</summary><pre>{payload}</pre></details>
  </header>
  <main>
    {''.join(rows)}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / "audit_records.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    motion_dir = Path(args.motion_dir)
    cache_path = Path(args.rewrite_cache)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rewrites, meta = load_rewrite_cache(cache_path)
    records = annotate_records(load_manifest(manifest), rewrites)
    selected = choose_records(records, motion_dir, args)

    audit_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(selected, 1):
        print(f"[render] {idx}/{len(selected)} {rec['id']} score={rec['rewrite_score']}", flush=True)
        image_path = img_dir / f"{rec['id']}.png"
        render_meta = render_record(
            rec,
            motion_dir / f"{rec['id']}.npy",
            image_path,
            args.frames_per_segment,
        )
        audit_rows.append({**rec, "render": render_meta})

    write_html(out_dir, selected, meta, audit_rows)
    print(f"[done] wrote {out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

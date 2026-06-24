#!/usr/bin/env python3
"""Build ViMoGen T2M evaluation metadata from MotionHub-style annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _caption_candidates(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    if all(k in data and isinstance(data[k], list) for k in ("macro", "meso", "micro")):
        out = []
        for group in ("macro", "meso", "micro"):
            for caption in data[group]:
                if isinstance(caption, str) and caption.strip():
                    out.append(caption.strip())
        return out

    if "result" in data and isinstance(data["result"], list):
        out = []
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                val = item.get(key)
                if isinstance(val, list):
                    for caption in val:
                        if isinstance(caption, str) and caption.strip():
                            out.append(caption.strip())
            for key in ("short_caption", "short caption"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
        return out
    return []


def load_caption(path: Path, style: str = "first") -> Optional[str]:
    candidates = _caption_candidates(path)
    if not candidates:
        return None
    if style == "first":
        return candidates[0]
    if style == "longest":
        return max(candidates, key=lambda x: len(x.split()))
    if style == "concat3":
        # Keep the prompt detailed enough for ViMoGen while avoiding very long
        # caption lists that can dominate the text encoder context.
        return " ".join(candidates[:3])
    raise ValueError(f"unsupported caption style: {style}")


def iter_entries(raw):
    data_list = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data_list, dict):
        yield from data_list.items()
    else:
        for idx, entry in enumerate(data_list):
            yield str(entry.get("motion_id") or entry.get("id") or idx), entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--caption-map-json", required=True)
    parser.add_argument(
        "--caption-override-json",
        default=None,
        help="Optional {sample_id: caption} JSON used to override loaded captions.",
    )
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--append-duration-to-prompt", action="store_true")
    parser.add_argument("--caption-style", choices=["first", "longest", "concat3"], default="first")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"invalid shard args: {args.shard_idx}/{args.num_shards}")

    raw = json.loads(Path(args.anno_file).read_text())
    data_dir = Path(args.data_dir)
    caption_override = None
    if args.caption_override_json:
        caption_override = json.loads(Path(args.caption_override_json).read_text())
    out = []
    caption_map = {}
    stats = {"seen": 0, "eligible": 0, "kept": 0, "no_caption": 0}

    for name, entry in iter_entries(raw):
        stats["seen"] += 1
        caption = None
        if caption_override is not None:
            caption = caption_override.get(str(name))
        caption_rel = entry.get("hierarchical_caption_path")
        if not caption:
            caption = load_caption(data_dir / caption_rel, style=args.caption_style) if caption_rel else None
        if not caption:
            stats["no_caption"] += 1
            continue

        eligible_idx = stats["eligible"]
        stats["eligible"] += 1
        if args.num_shards > 1 and eligible_idx % args.num_shards != args.shard_idx:
            continue

        duration = entry.get("duration")
        if duration is None:
            num_frames = float(entry.get("num_frames", 0))
            fps = float(entry.get("fps", 30) or 30)
            duration = num_frames / fps if fps > 0 else 5.0
        test_seq_len = int(round(float(duration) * 20.0))
        test_seq_len = max(args.min_len, min(args.max_len, test_seq_len))
        prompt = caption
        if args.append_duration_to_prompt:
            prompt = f"{caption}; motion_duration: {float(duration):.2f} seconds"

        item = {
            "sample_id": str(name),
            "global_id": str(name),
            "prompt": prompt,
            "motion_path": "data_samples/dummy_motion.pt",
            "use_ref_motion": False,
            "motion_duration": float(duration),
            "test_seq_len": test_seq_len,
            "prompt_wanvideot5_embed_path": str(Path(args.embedding_dir) / "prompt" / f"{name}.pt"),
        }
        out.append(item)
        caption_map[str(name)] = caption
        stats["kept"] += 1
        if args.max_samples and len(out) >= args.max_samples:
            break

    out_path = Path(args.out_json)
    cap_path = Path(args.caption_map_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    cap_path.write_text(json.dumps(caption_map, indent=2))
    print({
        "out_json": str(out_path),
        "caption_map_json": str(cap_path),
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        **stats,
    }, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Filter HYMotion-G1 T2M annotations by caption simplicity and retarget quality.

The G1 generator should not learn from semantically noisy text or broken robot
retargets.  This script keeps the existing annotation schema, but records reject
reasons and lightweight quality stats in ``meta_info`` so thresholds are auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


CAPTION_TO_QWEN3_DIR = {
    "human_checked_augmented_caption": "qwen3_human_checked_short",
    "human_checked_caption": "qwen3_human_checked_short",
    "improved_simple_caption": "qwen3_improved_simple_short",
    "improved_simple_augmented_caption": "qwen3_improved_simple_short",
    "augmented_caption": "qwen3_augmented",
    "editing_caption": "qwen3_editing",
    "raw_caption": "qwen3_raw_short",
}


def _caption_dir(caption_rel: str) -> Optional[str]:
    for part in str(caption_rel or "").split("/"):
        if part in CAPTION_TO_QWEN3_DIR:
            return part
    return None


def _caption_rel_to_emb_rel(caption_rel: str) -> Optional[str]:
    parts = str(caption_rel or "").split("/")
    for i, part in enumerate(parts):
        if part in CAPTION_TO_QWEN3_DIR:
            out = "/".join(parts[:i] + [CAPTION_TO_QWEN3_DIR[part]] + parts[i + 1 :])
            return out[:-5] + ".pt" if out.endswith(".json") else out
    return None


def _load_caption_from_embedding(data_dir: Path, emb_rel: str) -> str:
    import torch

    blob = torch.load(data_dir / emb_rel, map_location="cpu", weights_only=False)
    result = blob.get("result") if isinstance(blob, dict) else None
    if not result:
        return ""
    return str(result[0].get("caption", ""))


def _load_caption_from_json(data_dir: Path, caption_rel: str) -> str:
    try:
        blob = json.loads((data_dir / caption_rel).read_text())
    except Exception:
        return ""
    if isinstance(blob, str):
        return blob
    if isinstance(blob, dict):
        for key in ("caption", "text", "simple_caption", "improved_simple_caption"):
            value = blob.get(key)
            if isinstance(value, str):
                return value
        for value in blob.values():
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    if isinstance(blob, list) and blob and isinstance(blob[0], str):
        return blob[0]
    return ""


def _caption_ok(
    item: Dict[str, Any],
    *,
    data_dir: Path,
    allowed_dirs: set[str],
    min_words: int,
    max_words: int,
    max_chars: int,
    caption_source: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    cap_dir = _caption_dir(item.get("caption_rel", ""))
    if not cap_dir:
        return False, "caption:no_known_dir", {}
    if allowed_dirs and cap_dir not in allowed_dirs:
        return False, f"caption:dir:{cap_dir}", {"caption_dir": cap_dir}

    caption = ""
    # Match HyMotionG1Dataset: recompute from caption_rel and do not trust a
    # stale emb_rel stored in older annotation files.
    emb_rel = _caption_rel_to_emb_rel(item.get("caption_rel", "")) or item.get("emb_rel")
    if caption_source == "embedding" and emb_rel:
        try:
            caption = _load_caption_from_embedding(data_dir, emb_rel)
        except Exception as exc:  # noqa: BLE001
            return False, f"caption:embedding_error:{type(exc).__name__}", {"caption_dir": cap_dir}
    elif caption_source == "json":
        caption = _load_caption_from_json(data_dir, item.get("caption_rel", ""))
    if not caption:
        return False, "caption:empty", {"caption_dir": cap_dir}

    n_words = len(caption.split())
    n_chars = len(caption)
    if n_words < min_words:
        return False, "caption:too_short", {"caption_dir": cap_dir, "words": n_words, "chars": n_chars}
    if max_words > 0 and n_words > max_words:
        return False, "caption:too_many_words", {"caption_dir": cap_dir, "words": n_words, "chars": n_chars}
    if max_chars > 0 and n_chars > max_chars:
        return False, "caption:too_many_chars", {"caption_dir": cap_dir, "words": n_words, "chars": n_chars}
    return True, "ok", {"caption_dir": cap_dir, "caption": caption, "words": n_words, "chars": n_chars}


def _find_body(body_names: Iterable[Any], *needles: str) -> Optional[int]:
    names = [str(x).lower() for x in body_names]
    for needle in needles:
        needle = needle.lower()
        for i, name in enumerate(names):
            if needle in name:
                return i
    return None


def _quality_metrics(npz_path: Path, contact_h: float) -> Optional[Dict[str, float]]:
    try:
        d = np.load(npz_path, allow_pickle=True)
        body_pos = np.asarray(d["body_positions"], dtype=np.float32)
        body_names = list(d["body_names"]) if "body_names" in d else []
        dof_pos = np.asarray(d["dof_positions"], dtype=np.float32)
        fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 30.0
    except Exception:
        return None

    if body_pos.ndim != 3 or body_pos.shape[0] < 2 or body_pos.shape[-1] != 3:
        return None
    if dof_pos.ndim != 2:
        return None
    if not np.isfinite(body_pos).all() or not np.isfinite(dof_pos).all():
        return None

    left = _find_body(body_names, "left_ankle_roll", "left_foot", "left_toe")
    right = _find_body(body_names, "right_ankle_roll", "right_foot", "right_toe")
    root = body_pos[:, 0, :]
    dt = 1.0 / fps
    frame_disp = np.linalg.norm(np.diff(body_pos, axis=0), axis=-1).max(axis=1)
    root_speed = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1) / dt
    out = {
        "num_frames": float(body_pos.shape[0]),
        "fps": fps,
        "root_speed_p95": float(np.percentile(root_speed, 95)) if len(root_speed) else 0.0,
        "root_speed_mean": float(root_speed.mean()) if len(root_speed) else 0.0,
        "root_height_min": float(root[:, 2].min()),
        "root_height_max": float(root[:, 2].max()),
        "body_min_z": float(body_pos[:, :, 2].min()),
        "max_frame_disp": float(frame_disp.max()) if len(frame_disp) else 0.0,
    }

    contact_speeds = []
    for idx in (left, right):
        if idx is None:
            continue
        foot = body_pos[:, idx, :]
        h = foot[:, 2]
        spd = np.linalg.norm(np.diff(foot[:, :2], axis=0), axis=-1) / dt
        contact = h[:-1] < contact_h
        if np.any(contact):
            contact_speeds.append(spd[contact])
    if contact_speeds:
        speeds = np.concatenate(contact_speeds)
        out["foot_contact_speed_mean"] = float(speeds.mean())
        out["foot_contact_skate_ratio"] = float((speeds > 0.05).mean())
    else:
        out["foot_contact_speed_mean"] = 0.0
        out["foot_contact_skate_ratio"] = 0.0
    return out


def _quality_ok(
    item: Dict[str, Any],
    *,
    g1_dir: Path,
    contact_h: float,
    min_frames: int,
    max_frames: int,
    max_contact_speed: float,
    max_contact_skate_ratio: float,
    min_body_z: float,
    max_frame_disp: float,
    max_root_speed_p95: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    metrics = _quality_metrics(g1_dir / item["g1_path"], contact_h)
    if metrics is None:
        return False, "quality:load_or_schema", {}
    T = int(metrics["num_frames"])
    if T < min_frames:
        return False, "quality:too_short", metrics
    if max_frames > 0 and T > max_frames:
        return False, "quality:too_long", metrics
    if metrics["body_min_z"] < min_body_z:
        return False, "quality:penetration", metrics
    if metrics["max_frame_disp"] > max_frame_disp:
        return False, "quality:frame_jump", metrics
    if metrics["root_speed_p95"] > max_root_speed_p95:
        return False, "quality:root_speed", metrics
    if metrics["foot_contact_speed_mean"] > max_contact_speed:
        return False, "quality:foot_contact_speed", metrics
    if metrics["foot_contact_skate_ratio"] > max_contact_skate_ratio:
        return False, "quality:foot_skate_ratio", metrics
    return True, "ok", metrics


def _process_one(args_tuple):
    idx, item, cfg = args_tuple
    ok, reason, cmeta = _caption_ok(
        item,
        data_dir=cfg["data_dir"],
        allowed_dirs=cfg["allowed_dirs"],
        min_words=cfg["min_words"],
        max_words=cfg["max_words"],
        max_chars=cfg["max_chars"],
        caption_source=cfg["caption_source"],
    )
    if not ok:
        return idx, None, reason, cmeta
    qmeta = {}
    if not cfg["skip_quality"]:
        ok, reason, qmeta = _quality_ok(
            item,
            g1_dir=cfg["g1_dir"],
            contact_h=cfg["contact_h"],
            min_frames=cfg["min_frames"],
            max_frames=cfg["max_frames"],
            max_contact_speed=cfg["max_contact_speed"],
            max_contact_skate_ratio=cfg["max_contact_skate_ratio"],
            min_body_z=cfg["min_body_z"],
            max_frame_disp=cfg["max_frame_disp"],
            max_root_speed_p95=cfg["max_root_speed_p95"],
        )
        if not ok:
            return idx, None, reason, {**cmeta, **qmeta}
    out_item = dict(item)
    if cfg["rewrite_emb_rel"]:
        emb_rel = _caption_rel_to_emb_rel(out_item.get("caption_rel", ""))
        if emb_rel:
            out_item["emb_rel"] = emb_rel
    if cfg["store_caption"]:
        out_item["caption"] = cmeta.get("caption", "")
    return idx, out_item, "ok", {**cmeta, **qmeta}


def _summarize_numeric(rows: list[Dict[str, Any]], keys: list[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
        if vals:
            out[key] = {
                "mean": float(np.mean(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "max": float(np.max(vals)),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/annotation/train_g1_t2m_emb.json")
    ap.add_argument("--out", default="data/annotation/train_g1_t2m_clean_simple_emb.json")
    ap.add_argument("--data-dir", default="data/hymotion_data")
    ap.add_argument("--g1-dir", default="data/g1")
    ap.add_argument("--allowed-caption-dirs", default="improved_simple_augmented_caption,improved_simple_caption")
    ap.add_argument("--caption-source", choices=["embedding", "json"], default="embedding")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=120)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--contact-h", type=float, default=0.07)
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--max-contact-speed", type=float, default=0.50)
    ap.add_argument("--max-contact-skate-ratio", type=float, default=0.80)
    ap.add_argument("--min-body-z", type=float, default=-0.03)
    ap.add_argument("--max-frame-disp", type=float, default=0.35)
    ap.add_argument("--max-root-speed-p95", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--store-caption", action="store_true")
    ap.add_argument("--rewrite-emb-rel", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    anno_path = Path(args.anno)
    blob = json.loads(anno_path.read_text())
    items = list(blob["items"] if isinstance(blob, dict) else blob)
    if args.limit > 0:
        items = items[: args.limit]

    allowed_dirs = {x.strip() for x in args.allowed_caption_dirs.split(",") if x.strip()}
    cfg = {
        "data_dir": Path(args.data_dir),
        "g1_dir": Path(args.g1_dir),
        "allowed_dirs": allowed_dirs,
        "caption_source": args.caption_source,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_chars": args.max_chars,
        "skip_quality": args.skip_quality,
        "contact_h": args.contact_h,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "max_contact_speed": args.max_contact_speed,
        "max_contact_skate_ratio": args.max_contact_skate_ratio,
        "min_body_z": args.min_body_z,
        "max_frame_disp": args.max_frame_disp,
        "max_root_speed_p95": args.max_root_speed_p95,
        "store_caption": args.store_caption,
        "rewrite_emb_rel": args.rewrite_emb_rel,
    }

    kept: list[Dict[str, Any]] = []
    reject = Counter()
    kept_meta: list[Dict[str, Any]] = []
    jobs = [(i, it, cfg) for i, it in enumerate(items)]
    def consume(iterator):
        for n, (_idx, out_item, reason, meta) in enumerate(iterator, 1):
            if out_item is None:
                reject[reason] += 1
            else:
                kept.append(out_item)
                kept_meta.append(meta)
            if n % 5000 == 0:
                print(f"[filter] processed {n}/{len(items)} kept={len(kept)} reject={sum(reject.values())}", flush=True)

    if args.num_workers <= 1:
        consume(_process_one(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            consume(pool.map(_process_one, jobs, chunksize=32))

    out_blob = {
        "meta_info": {
            "source": str(anno_path),
            "num_source_items": len(items),
            "num_kept": len(kept),
            "reject_reasons": dict(reject.most_common()),
            "filters": {
                "allowed_caption_dirs": sorted(allowed_dirs),
                "caption_source": args.caption_source,
                "min_words": args.min_words,
                "max_words": args.max_words,
                "max_chars": args.max_chars,
                "skip_quality": args.skip_quality,
                "contact_h": args.contact_h,
                "min_frames": args.min_frames,
                "max_frames": args.max_frames,
                "max_contact_speed": args.max_contact_speed,
                "max_contact_skate_ratio": args.max_contact_skate_ratio,
                "min_body_z": args.min_body_z,
                "max_frame_disp": args.max_frame_disp,
                "max_root_speed_p95": args.max_root_speed_p95,
            },
            "kept_numeric_summary": _summarize_numeric(
                kept_meta,
                [
                    "words",
                    "chars",
                    "num_frames",
                    "root_speed_p95",
                    "foot_contact_speed_mean",
                    "foot_contact_skate_ratio",
                    "max_frame_disp",
                    "body_min_z",
                ],
            ),
        },
        "items": kept,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_blob, indent=1))
    print(f"[filter] wrote {len(kept)}/{len(items)} -> {out_path}")
    print("[filter] top rejects:", reject.most_common(12))


if __name__ == "__main__":
    main()

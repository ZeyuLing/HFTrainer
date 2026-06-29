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
import re
from collections import Counter, defaultdict
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


SCENE_SUPPORT_OK_RE = re.compile(
    r"\b(floor|ground|mat)\b.{0,45}\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b"
    r"|\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b.{0,45}\b(floor|ground|mat)\b"
)


SCENE_CAPTION_HARD_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "caption:stairs_ladder_platform",
        re.compile(
            r"\b(stair|stairs|staircase|stairway|upstairs|downstairs|ladder|ladders|"
            r"ramp|ramps|platform|platforms|ledge|ledges|slope|slopes|incline|inclines)\b"
        ),
    ),
    (
        "caption:climb_elevation",
        re.compile(
            r"\b(climb|climbs|climbed|climbing|ascend|ascends|ascended|ascending|"
            r"descend|descends|descended|descending)\b"
            r".{0,55}\b(up|down|onto|off|over|stair|stairs|step|steps|ladder|"
            r"platform|ledge|obstacle|wall|slope|ramp)\b"
        ),
    ),
    (
        "caption:step_on_or_over_scene",
        re.compile(
            r"\b(step|steps|stepped|stepping)\s+(up|down|onto|off)\s+"
            r"(a\s+|the\s+)?(step|steps|stair|stairs|platform|ledge|box|block|obstacle)\b"
            r"|\b(step|steps|stepped|stepping)\s+over\s+"
            r"(a\s+|the\s+)?(object|obstacle|box|block|hurdle|step|steps|stair|stairs)\b"
        ),
    ),
    (
        "caption:object_manipulation",
        re.compile(
            r"\b(pick|picks|picked|picking|lift|lifts|lifted|lifting|place|places|placed|placing|"
            r"put|puts|putting|grab|grabs|grabbing|hold|holds|holding)\b"
            r".{0,45}\b(box|boxes|object|objects|phone|suitcase|bag|bags)\b"
            r"|\b(box|boxes|object|objects|phone|suitcase|bag|bags)\b"
            r".{0,45}\b(pick|picks|picked|picking|lift|lifts|lifted|lifting|place|places|placed|placing|"
            r"put|puts|putting|grab|grabs|grabbing|hold|holds|holding)\b"
        ),
    ),
    (
        "caption:fixed_scene_support",
        re.compile(
            r"\b(support|supports|supporting|grab|grabs|grabbing|brace|braces|bracing|"
            r"hold|holds|holding|lean|leans|leaning|push|pushes|pushing|pull|pulls|pulling|"
            r"open|opens|opening|close|closes|closing)\b"
            r".{0,60}\b(platform|ledge|rail|railing|handrail|wall|door|fence|shelf|"
            r"pole|window|counter|countertop|table|desk|chair|bench|sofa|bed|obstacle|ladder)\b"
            r"|\b(platform|ledge|rail|railing|handrail|wall|door|fence|shelf|pole|window|"
            r"counter|countertop|table|desk|chair|bench|sofa|bed|obstacle|ladder)\b"
            r".{0,60}\b(support|supports|supporting|grab|grabs|grabbing|brace|braces|bracing|"
            r"hold|holds|holding|lean|leans|leaning|push|pushes|pushing|pull|pulls|pulling|"
            r"open|opens|opening|close|closes|closing)\b"
        ),
    ),
    (
        "caption:furniture_or_nonfloor_support",
        re.compile(
            r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid|rest|rests|"
            r"resting|perch|perches|perching)\b"
            r".{0,55}\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|"
            r"table|desk|bed|beds|ledge|countertop)\b"
            r"|\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|table|"
            r"desk|bed|beds|ledge|countertop)\b"
            r".{0,55}\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid|rest|"
            r"rests|resting|perch|perches|perching)\b"
        ),
    ),
    (
        "caption:vehicle_or_device",
        re.compile(
            r"\b(ride|rides|riding|drive|drives|driving)\s+(a\s+|the\s+)?"
            r"(car|cars|vehicle|vehicles|bicycle|bike|motorcycle|skateboard|paraglider)\b"
        ),
    ),
]


SCENE_CAPTION_REVIEW_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("caption:generic_obstacle", re.compile(r"\b(obstacle|obstacles|hurdle|hurdles|vault|vaults|vaulting)\b")),
    ("caption:generic_crawl", re.compile(r"\b(crawl|crawls|crawling)\b")),
    ("caption:nonfloor_sit_or_lie", re.compile(r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b")),
]


SCENE_PATH_HARD_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("path:platform", re.compile(r"\b(platform|ptfm)\b")),
    ("path:stairs_steps_ladder", re.compile(r"\b(stair|stairs|staircase|upstairs|downstairs|ladder|ldr|ascd|dscd)\b")),
    ("path:slope_ramp", re.compile(r"\b(slope|ramp|incline)\b")),
    ("path:obstacle", re.compile(r"\b(obstacle|obstacles|hurdle|hurdles)\b")),
]


SCENE_PATH_REVIEW_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("path:crawl", re.compile(r"\bcrawl\b")),
    ("path:chair_table_bed", re.compile(r"\b(chair|table|desk|bench|bed|sofa|stool)\b")),
]


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


def _caption_from_json_obj(blob: Any) -> str:
    if isinstance(blob, str):
        return blob
    if isinstance(blob, list):
        for value in blob:
            caption = _caption_from_json_obj(value)
            if caption:
                return caption
        return ""
    if not isinstance(blob, dict):
        return ""

    # Common HYMotion caption schema: result is a list of candidate caption dicts.
    result = blob.get("result")
    if isinstance(result, list):
        for entry in result:
            if not isinstance(entry, dict):
                continue
            for key in (
                "short_caption",
                "caption",
                "text",
                "simple_caption",
                "improved_simple_caption",
                "long_caption",
            ):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            rewritten = entry.get("short_caption_rewritten")
            if isinstance(rewritten, list):
                for value in rewritten:
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    for key in (
        "caption",
        "text",
        "simple_caption",
        "improved_simple_caption",
        "short_caption",
        "long_caption",
    ):
        value = blob.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_caption_from_json(data_dir: Path, caption_rel: str) -> str:
    try:
        blob = json.loads((data_dir / caption_rel).read_text())
    except Exception:
        return ""
    return _caption_from_json_obj(blob)


def _scene_normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[_/\\.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _scene_decision(
    item: Dict[str, Any],
    *,
    caption: str,
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    caption_text = _scene_normalize(caption)
    path_text = _scene_normalize(
        " ".join(
            str(item.get(k, ""))
            for k in ("g1_path", "caption_rel", "emb_rel", "motion_path")
        )
    )
    hard: list[str] = []
    review: list[str] = []
    for name, pattern in SCENE_CAPTION_HARD_RULES:
        if pattern.search(caption_text):
            hard.append(name)
    for name, pattern in SCENE_CAPTION_REVIEW_RULES:
        if pattern.search(caption_text):
            review.append(name)
    if "caption:nonfloor_sit_or_lie" in review and SCENE_SUPPORT_OK_RE.search(caption_text):
        review.remove("caption:nonfloor_sit_or_lie")

    for name, pattern in SCENE_PATH_HARD_RULES:
        if pattern.search(path_text):
            hard.append(name)
    for name, pattern in SCENE_PATH_REVIEW_RULES:
        if pattern.search(path_text):
            review.append(name)

    if quality:
        root_z = None
        if "root_height_min" in quality and "root_height_max" in quality:
            root_z = float(quality["root_height_max"]) - float(quality["root_height_min"])
        if root_z is not None and root_z > 0.75:
            review.append("motion:large_root_height_excursion")

    # Hard evidence wins, but keep review tags for audit.
    return {
        "hard_reasons": sorted(set(hard)),
        "review_reasons": sorted(set(review)),
        "caption": caption,
        "path_text": path_text[:240],
    }


def _caption_ok(
    item: Dict[str, Any],
    *,
    data_dir: Path,
    allowed_dirs: set[str],
    min_words: int,
    max_words: int,
    max_chars: int,
    caption_source: str,
    allow_empty_caption: bool,
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
    if not caption and allow_empty_caption:
        return True, "ok", {"caption_dir": cap_dir, "caption": "", "words": 0, "chars": 0}
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
        allow_empty_caption=cfg["allow_empty_caption"],
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
    if cfg["scene_filter_mode"] != "off":
        scene = _scene_decision(item, caption=cmeta.get("caption", ""), quality=qmeta)
        hard = scene["hard_reasons"]
        review = scene["review_reasons"]
        if hard or (cfg["scene_filter_mode"] == "hard_and_review" and review):
            reasons = hard or review
            return idx, None, "scene:" + "+".join(reasons[:3]), {
                **cmeta,
                **qmeta,
                "scene_hard_reasons": hard,
                "scene_review_reasons": review,
                "scene_caption": scene["caption"],
                "scene_path_text": scene["path_text"],
                "g1_path": item.get("g1_path"),
                "caption_rel": item.get("caption_rel"),
            }
        if review:
            qmeta = {
                **qmeta,
                "scene_review_reasons": review,
                "scene_caption": scene["caption"],
                "scene_path_text": scene["path_text"],
            }
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
    ap.add_argument(
        "--allow-empty-caption",
        action="store_true",
        help="Keep rows whose caption text cannot be loaded; scene filtering then falls back to path evidence.",
    )
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
    ap.add_argument(
        "--scene-filter-mode",
        choices=["off", "hard", "hard_and_review"],
        default="off",
        help=(
            "Filter scene-interaction clips. 'hard' removes only high-confidence "
            "caption/path evidence; 'hard_and_review' also removes review-only "
            "candidates such as generic crawl or large vertical motion."
        ),
    )
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
        "allow_empty_caption": args.allow_empty_caption,
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
        "scene_filter_mode": args.scene_filter_mode,
    }

    kept: list[Dict[str, Any]] = []
    reject = Counter()
    reject_examples: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    kept_meta: list[Dict[str, Any]] = []
    scene_review = Counter()
    scene_review_examples: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    jobs = [(i, it, cfg) for i, it in enumerate(items)]
    def consume(iterator):
        for n, (_idx, out_item, reason, meta) in enumerate(iterator, 1):
            if out_item is None:
                reject[reason] += 1
                if len(reject_examples[reason]) < 8:
                    reject_examples[reason].append({
                        "index": _idx,
                        "g1_path": meta.get("g1_path"),
                        "caption_rel": meta.get("caption_rel"),
                        "caption": meta.get("caption") or meta.get("scene_caption"),
                        "scene_hard_reasons": meta.get("scene_hard_reasons"),
                        "scene_review_reasons": meta.get("scene_review_reasons"),
                    })
            else:
                kept.append(out_item)
                kept_meta.append(meta)
                for scene_reason in meta.get("scene_review_reasons", []) or []:
                    scene_review[scene_reason] += 1
                    if len(scene_review_examples[scene_reason]) < 8:
                        scene_review_examples[scene_reason].append({
                            "index": _idx,
                            "g1_path": out_item.get("g1_path"),
                            "caption_rel": out_item.get("caption_rel"),
                            "caption": meta.get("scene_caption") or meta.get("caption"),
                        })
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
                "allow_empty_caption": args.allow_empty_caption,
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
                "scene_filter_mode": args.scene_filter_mode,
            },
            "reject_examples": dict(reject_examples),
            "scene_review_counts": dict(scene_review.most_common()),
            "scene_review_examples": dict(scene_review_examples),
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

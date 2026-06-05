#!/usr/bin/env python3
"""Build a PhysFlow text corpus from HYMotion annotations.

The PhysFlow KIMODO-G1 loop consumes PromptSpec JSONL files, not paired motion
supervision.  This builder expands the prompt pool beyond HumanML3D while
keeping the corpus suitable for a scene-free Unitree G1 setup:

* game data is excluded by default;
* editing-only instructions are excluded by default;
* prompts requiring body-supporting furniture, fixed scene fixtures, vehicles,
  stairs, ladders, or platforms are filtered with auditable reason codes.

The output keeps extra HYMotion metadata fields for traceability.  The runner's
PromptSpec loader ignores these extras, so the JSONL remains backward
compatible with existing PhysFlow tools.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import orjson
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("orjson is required in this repo environment") from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION = (
    ROOT
    / "data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260527.json"
)
DEFAULT_OUT_DIR = ROOT / "configs/experiments/physflow_kimodo_g1"
DEFAULT_REPORT_DIR = DEFAULT_OUT_DIR / "filter_reports"

DEFAULT_ALLOWED_SUBSETS = (
    "academic",
    "academicretarget",
    "taobao",
    "PerMo-train",
)
DEFAULT_DROPPED_SUBSETS = (
    "game",
    "PerMo-editing-train",
    "MotionFix-train",
)


SUPPORT_OK_RE = re.compile(
    r"\b(floor|ground|mat)\b.{0,40}\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b"
    r"|\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b.{0,40}\b(floor|ground|mat)\b"
)

RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "furniture_or_nonfloor_support",
        re.compile(
            r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid|rest|rests|resting|lean|leans|leaning|perch|perches|perching)\b"
            r".{0,50}\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|table|desk|bed|beds|ledge|countertop)\b"
            r"|\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|table|desk|bed|beds|ledge|countertop)\b"
            r".{0,50}\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid|rest|rests|resting|lean|leans|leaning|perch|perches|perching)\b"
            r"|\b(foot|feet|hand|hands|elbow|elbows|knee|knees)\b.{0,35}\b(on|onto|upon|against)\b"
            r".{0,35}\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|table|desk|bed|beds|ledge|countertop)\b"
        ),
    ),
    (
        "stairs_ladder_or_elevation",
        re.compile(
            r"\b(stair|stairs|staircase|ladder|ramp|platform)\b"
            r"|\b(climb|climbs|climbed|climbing|ascend|ascends|ascending|descend|descends|descending)\b"
            r".{0,40}\b(up|down|stair|stairs|ladder|step|steps|onto|off)\b"
            r"|\b(step|steps)\s+(up|down|onto|off)\b"
            r"|\b(up|down)\s+(a\s+|the\s+)?(step|steps)\b"
        ),
    ),
    (
        "fixed_scene_support",
        re.compile(
            r"\b(lean|leans|leaning|rest|rests|resting|support|supports|supporting|brace|braces|bracing|push|pushes|pushing|pull|pulls|pulling|hold|holds|holding|grab|grabs|grabbing|open|opens|opening|close|closes|closing)\b"
            r".{0,45}\b(wall|walls|door|doors|fence|fences|shelf|shelves|pole|poles|window|windows|railing|rail|rails|countertop|machine|machines)\b"
            r"|\b(wall|walls|door|doors|fence|fences|shelf|shelves|pole|poles|window|windows|railing|rail|rails|countertop|machine|machines)\b"
            r".{0,45}\b(lean|leans|leaning|rest|rests|resting|support|supports|supporting|brace|braces|bracing|push|pushes|pushing|pull|pulls|pulling|hold|holds|holding|grab|grabs|grabbing|open|opens|opening|close|closes|closing)\b"
        ),
    ),
    (
        "vehicle_or_device",
        re.compile(
            r"\b(car|cars|vehicle|vehicles|bicycle|bike|motorcycle|driver|drive|driving|paraglider|skateboard)\b"
        ),
    ),
    (
        "partner_or_multi_person",
        re.compile(
            r"\b(two people|two persons|another person|other person|partner|someone else|each other|together with)\b"
        ),
    ),
    (
        "non_floor_sit_or_lie",
        re.compile(r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b"),
    ),
]


def csv_arg(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def dedup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def drop_reasons_for_prompt(prompt: str) -> list[str]:
    text = prompt.lower()
    reasons = [name for name, pat in RULES if pat.search(text)]
    if "non_floor_sit_or_lie" in reasons and SUPPORT_OK_RE.search(text):
        reasons.remove("non_floor_sit_or_lie")
    return reasons


def categorize_prompt(prompt: str) -> str:
    text = prompt.lower()
    if any(k in text for k in ("jump", "hop", "leap", "kick", "lunge", "squat", "crouch", "crawl")):
        return "dynamic"
    if any(k in text for k in ("walk", "run", "jog", "step", "turn", "pace", "sidestep", "march")):
        return "locomotion"
    if any(k in text for k in ("arm", "hand", "wave", "clap", "gesture", "point", "reach")):
        return "upper_body"
    if any(k in text for k in ("dance", "spin", "twirl")):
        return "dance_turn"
    return "standing_misc"


def difficulty_for_prompt(prompt: str) -> int:
    text = prompt.lower()
    if any(k in text for k in ("flip", "cartwheel", "handstand", "somersault")):
        return 3
    if any(k in text for k in ("jump", "hop", "leap", "kick", "crawl", "run", "spin")):
        return 2
    if any(k in text for k in ("walk", "turn", "squat", "crouch", "dance")):
        return 1
    return 0


def resolve_candidates(raw: str | None) -> list[Path]:
    if not raw:
        return []
    path = Path(raw)
    if path.is_absolute():
        return [path]
    candidates: list[Path] = []
    raw_posix = path.as_posix()
    if raw_posix.startswith("../hymotion_data/"):
        return [ROOT / "data" / raw_posix[3:]]
    if raw_posix.startswith("data/"):
        return [ROOT / raw_posix]
    candidates.extend(
        [
            ROOT / path,
            ROOT / "data/hymotion_data" / path,
            ROOT / "data/motionhub" / path,
        ]
    )
    return candidates


def resolve_path(raw: str | None) -> Path | None:
    candidates = resolve_candidates(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def resolve_caption_path(*raw_values: str | None) -> Path | None:
    """Resolve caption path, preferring an existing fallback when available.

    Some merged HYMotion entries contain a stale ``hierarchical_caption_path``
    plus a valid ``caption_path``.  Try both instead of blindly trusting the
    first non-empty field.
    """

    fallback: Path | None = None
    for raw in raw_values:
        if not raw:
            continue
        raw_posix = Path(raw).as_posix()
        if Path(raw).is_absolute() or raw_posix.startswith("../hymotion_data/") or raw_posix.startswith("data/"):
            candidates = resolve_candidates(raw)
            return candidates[0] if candidates else None

    for raw in raw_values:
        for candidate in resolve_candidates(raw):
            if fallback is None:
                fallback = candidate
            if candidate.is_file():
                return candidate
    return fallback


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = orjson.loads(f.read())
    if "data_list" not in data or not isinstance(data["data_list"], dict):
        raise ValueError(f"{path} does not look like a HYMotion annotation with data_list")
    return data


def read_caption_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def caption_candidates(obj: dict[str, Any], rewrites_per_motion: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    result = obj.get("result")
    if not isinstance(result, list):
        return out
    for result_idx, item in enumerate(result):
        if not isinstance(item, dict):
            continue
        primary = (
            item.get("short_caption")
            or item.get("short caption")
            or item.get("simple_caption")
            or item.get("caption")
        )
        if isinstance(primary, str) and primary.strip():
            out.append((normalize_text(primary), f"r{result_idx}:short"))
        rewrites = item.get("short_caption_rewritten")
        if rewrites_per_motion > 0 and isinstance(rewrites, list):
            for rewrite_idx, rewrite in enumerate(rewrites[:rewrites_per_motion]):
                if isinstance(rewrite, str) and rewrite.strip():
                    out.append((normalize_text(rewrite), f"r{result_idx}:rewrite{rewrite_idx}"))
    return out


def iter_rows(
    data_list: dict[str, Any],
    *,
    allowed_subsets: set[str],
    dropped_subsets: set[str],
    rewrites_per_motion: int,
    min_words: int,
    max_caption_len: int,
    min_source_duration: float,
    max_source_duration: float,
    prompt_min_duration: float,
    prompt_max_duration: float,
    require_motion_exists: bool,
    limit_entries: int,
    progress_every: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    reason_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    kept_subset_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def record_drop(hymotion_id: str, subset: str, reasons: Iterable[str], prompt: str = "") -> None:
        reasons = list(reasons)
        reason_counts.update(reasons)
        row = {"hymotion_id": hymotion_id, "subset": subset, "prompt": prompt, "drop_reasons": reasons}
        dropped.append(row)
        for reason in reasons:
            if len(examples[reason]) < 10:
                examples[reason].append(row)

    caption_tasks: list[dict[str, Any]] = []

    for idx, (hymotion_id, item) in enumerate(data_list.items(), start=1):
        if limit_entries and idx > limit_entries:
            break
        if progress_every and idx % progress_every == 0:
            print(
                f"[prep] {idx}/{len(data_list)} tasks={len(caption_tasks)} dropped={len(dropped)}",
                file=sys.stderr,
            )
        if not isinstance(item, dict):
            record_drop(hymotion_id, "", ["bad_annotation_item"])
            continue

        subset = str(item.get("subset", ""))
        subset_counts[subset] += 1
        if subset in dropped_subsets:
            record_drop(hymotion_id, subset, [f"subset:{subset}"])
            continue
        if allowed_subsets and subset not in allowed_subsets:
            record_drop(hymotion_id, subset, ["subset_not_allowed"])
            continue

        duration = float(item.get("duration") or 0.0)
        if duration < min_source_duration or duration > max_source_duration:
            record_drop(hymotion_id, subset, ["duration_out_of_range"])
            continue

        raw_motion_path = item.get("smplx_path") or item.get("motion_135_path") or item.get("motion_198_path")
        if require_motion_exists:
            motion_path = resolve_path(raw_motion_path)
        else:
            motion_candidates = resolve_candidates(raw_motion_path)
            motion_path = motion_candidates[0] if motion_candidates else None
        if require_motion_exists and (motion_path is None or not motion_path.is_file()):
            record_drop(hymotion_id, subset, ["missing_motion_file"])
            continue

        caption_path = resolve_caption_path(item.get("hierarchical_caption_path"), item.get("caption_path"))
        if caption_path is None:
            record_drop(hymotion_id, subset, ["missing_caption_file"])
            continue

        caption_tasks.append(
            {
                "idx": idx,
                "hymotion_id": hymotion_id,
                "subset": subset,
                "duration": duration,
                "motion_path": motion_path,
                "caption_path": caption_path,
            }
        )

    def load_task(task: dict[str, Any]) -> dict[str, Any]:
        cap_obj = read_caption_json(task["caption_path"])
        if not cap_obj:
            return {**task, "error": "bad_caption_json", "candidates": []}
        candidates = caption_candidates(cap_obj, rewrites_per_motion)
        if not candidates:
            return {**task, "error": "no_caption", "candidates": []}
        return {**task, "error": None, "candidates": candidates}

    if num_workers > 1 and caption_tasks:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            processed_iter = pool.map(load_task, caption_tasks)
            processed_tasks = list(processed_iter)
    else:
        processed_tasks = [load_task(task) for task in caption_tasks]

    processed_tasks.sort(key=lambda row: row["idx"])

    for processed_idx, task in enumerate(processed_tasks, start=1):
        if progress_every and processed_idx % progress_every == 0:
            print(
                f"[caption] {processed_idx}/{len(processed_tasks)} kept={len(kept)} dropped={len(dropped)}",
                file=sys.stderr,
            )
        hymotion_id = task["hymotion_id"]
        subset = task["subset"]
        duration = task["duration"]
        motion_path = task["motion_path"]
        caption_path = task["caption_path"]

        if task["error"]:
            record_drop(hymotion_id, subset, [task["error"]])
            continue

        accepted_for_motion = 0
        for prompt, variant in task["candidates"]:
            words = re.findall(r"[A-Za-z0-9]+", prompt)
            reasons: list[str] = []
            if len(words) < min_words:
                reasons.append("caption_too_short")
            if len(prompt) > max_caption_len:
                reasons.append("caption_too_long")
            reasons.extend(drop_reasons_for_prompt(prompt))
            key = dedup_key(prompt)
            if key in seen_texts:
                reasons.append("duplicate_caption")
            if reasons:
                record_drop(hymotion_id, subset, reasons, prompt)
                continue

            seen_texts.add(key)
            accepted_for_motion += 1
            clipped_duration = max(prompt_min_duration, min(prompt_max_duration, duration or 4.0))
            kept_subset_counts[subset] += 1
            kept.append(
                {
                    "prompt": prompt,
                    "category": categorize_prompt(prompt),
                    "difficulty": difficulty_for_prompt(prompt),
                    "duration_sec": round(clipped_duration, 3),
                    "source": "HYMotion",
                    "hymotion_id": hymotion_id,
                    "subset": subset,
                    "caption_variant": variant,
                    "caption_path": str(caption_path.relative_to(ROOT) if caption_path.is_relative_to(ROOT) else caption_path),
                    "smplx_path": str(motion_path.relative_to(ROOT) if motion_path and motion_path.is_relative_to(ROOT) else motion_path),
                    "tags": ["hymotion", "g1_noscene", "physical_text_filter", subset],
                }
            )
        if accepted_for_motion == 0:
            # Per-candidate reasons above are already recorded; this summary
            # reason helps see motions where every textual variant failed.
            reason_counts.update(["all_caption_variants_dropped"])

    report = {
        "n_annotation_items_scanned": min(len(data_list), limit_entries or len(data_list)),
        "n_caption_tasks": len(caption_tasks),
        "n_kept_prompts": len(kept),
        "n_dropped_events": len(dropped),
        "subset_counts_scanned": dict(subset_counts.most_common()),
        "kept_subset_counts": dict(kept_subset_counts.most_common()),
        "drop_reason_counts": dict(reason_counts.most_common()),
        "examples": examples,
        "dropped": dropped[:50000],
        "dropped_truncated": len(dropped) > 50000,
    }
    return kept, report


def split_rows(rows: list[dict[str, Any]], eval_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["subset"], row["category"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    eval_rows: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(eval_rows) < eval_size and any(buckets[k] for k in keys):
        for key in keys:
            if buckets[key]:
                eval_rows.append(buckets[key].pop())
                if len(eval_rows) >= eval_size:
                    break

    train_rows: list[dict[str, Any]] = []
    for key in keys:
        train_rows.extend(buckets[key])
    rng.shuffle(train_rows)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], split: str, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            out = dict(row)
            out["id"] = f"{prefix}_{idx:06d}"
            out["split"] = split
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--name", default="physflow_text_hymotion_g1_real")
    parser.add_argument("--allowed-subsets", default=",".join(DEFAULT_ALLOWED_SUBSETS))
    parser.add_argument("--dropped-subsets", default=",".join(DEFAULT_DROPPED_SUBSETS))
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--rewrites-per-motion", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-caption-len", type=int, default=180)
    parser.add_argument("--min-source-duration", type=float, default=2.0)
    parser.add_argument("--max-source-duration", type=float, default=12.0)
    parser.add_argument("--prompt-min-duration", type=float, default=3.0)
    parser.add_argument("--prompt-max-duration", type=float, default=10.0)
    parser.add_argument("--require-motion-exists", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit-entries", type=int, default=0, help="debug only; 0 scans all annotation items")
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()

    annotation = load_annotation(args.annotation)
    allowed_subsets = set(csv_arg(args.allowed_subsets))
    dropped_subsets = set(csv_arg(args.dropped_subsets))
    rows, report = iter_rows(
        annotation["data_list"],
        allowed_subsets=allowed_subsets,
        dropped_subsets=dropped_subsets,
        rewrites_per_motion=args.rewrites_per_motion,
        min_words=args.min_words,
        max_caption_len=args.max_caption_len,
        min_source_duration=args.min_source_duration,
        max_source_duration=args.max_source_duration,
        prompt_min_duration=args.prompt_min_duration,
        prompt_max_duration=args.prompt_max_duration,
        require_motion_exists=args.require_motion_exists,
        limit_entries=args.limit_entries,
        progress_every=args.progress_every,
        num_workers=args.num_workers,
    )

    train_rows, eval_rows = split_rows(rows, eval_size=min(args.eval_size, len(rows)), seed=args.seed)
    if args.max_train and len(train_rows) > args.max_train:
        train_rows = train_rows[: args.max_train]

    train_path = args.out_dir / f"{args.name}_train.jsonl"
    eval_path = args.out_dir / f"{args.name}_eval.jsonl"
    report_path = args.report_dir / f"{args.name}.report.json"
    write_jsonl(train_path, train_rows, "train", "hytr")
    write_jsonl(eval_path, eval_rows, "test", "hyev")

    report.update(
        {
            "annotation": str(args.annotation),
            "allowed_subsets": sorted(allowed_subsets),
            "dropped_subsets": sorted(dropped_subsets),
            "rewrites_per_motion": args.rewrites_per_motion,
            "seed": args.seed,
            "eval_size": len(eval_rows),
            "train_size": len(train_rows),
            "train_path": str(train_path),
            "eval_path": str(eval_path),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"annotation items scanned: {report['n_annotation_items_scanned']}")
    print(f"kept prompts: {report['n_kept_prompts']}")
    print(f"train: {len(train_rows)} -> {train_path}")
    print(f"eval: {len(eval_rows)} -> {eval_path}")
    print(f"report: {report_path}")
    if report["drop_reason_counts"]:
        top = ", ".join(f"{k}={v}" for k, v in list(report["drop_reason_counts"].items())[:12])
        print(f"top drop reasons: {top}")


if __name__ == "__main__":
    main()

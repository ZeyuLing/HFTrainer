#!/usr/bin/env python3
"""Filter HumanML3D prompts unsuitable for scene-free G1 tracking.

PhysFlow trains a scene-free humanoid tracker/generator pair. Prompts requiring
external body support, stairs, fixed scene fixtures, vehicles, or non-floor
support inject targets that the robot cannot physically execute in the current
no-3D-scene setup.

The filter is intentionally conservative and reasoned: every dropped row is
written to a report with the matching rule names, while kept rows preserve their
original ids so existing cached KIMODO text features remain valid.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


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
            r"|\b(climb|climbs|climbed|climbing)\b.{0,35}\b(up|down|stair|stairs|ladder|step|steps)\b"
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
        re.compile(r"\b(car|cars|vehicle|vehicles|bicycle|bike|motorcycle|driver|paraglider)\b"),
    ),
    (
        "non_floor_sit_or_lie",
        re.compile(r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b"),
    ),
]


def drop_reasons(prompt: str) -> list[str]:
    text = prompt.lower()
    reasons = [name for name, pat in RULES if pat.search(text)]
    if "non_floor_sit_or_lie" in reasons and SUPPORT_OK_RE.search(text):
        reasons.remove("non_floor_sit_or_lie")
    return reasons


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_file(input_path: Path, output_path: Path, report_path: Path) -> dict:
    kept: list[dict] = []
    dropped: list[dict] = []
    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in iter_jsonl(input_path):
        reasons = drop_reasons(str(row.get("prompt", "")))
        if reasons:
            dropped_row = dict(row)
            dropped_row["drop_reasons"] = reasons
            dropped.append(dropped_row)
            reason_counts.update(reasons)
            for reason in reasons:
                if len(examples[reason]) < 8:
                    examples[reason].append(
                        {
                            "id": row.get("id"),
                            "prompt": row.get("prompt"),
                        }
                    )
        else:
            kept.append(row)

    write_jsonl(output_path, kept)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "n_input": len(kept) + len(dropped),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_rate": round(len(dropped) / max(len(kept) + len(dropped), 1), 6),
        "reason_counts": dict(reason_counts.most_common()),
        "examples": examples,
        "dropped": dropped,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--suffix", default="_g1_noscene")
    parser.add_argument("--report-dir", type=Path, default=Path("configs/experiments/physflow_kimodo_g1/filter_reports"))
    args = parser.parse_args()

    for input_path in args.inputs:
        output_path = input_path.with_name(f"{input_path.stem}{args.suffix}{input_path.suffix}")
        report_path = args.report_dir / f"{input_path.stem}{args.suffix}.report.json"
        report = filter_file(input_path, output_path, report_path)
        print(
            f"{input_path}: kept={report['n_kept']} dropped={report['n_dropped']} "
            f"({report['drop_rate'] * 100:.1f}%) -> {output_path}"
        )
        if report["reason_counts"]:
            top = ", ".join(f"{k}={v}" for k, v in list(report["reason_counts"].items())[:5])
            print(f"  reasons: {top}")


if __name__ == "__main__":
    main()

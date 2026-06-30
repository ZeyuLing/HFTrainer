#!/usr/bin/env python3
"""Split PhysFlow held-out generator cases into flat-ground and scene cases.

The main paper protocol evaluates G1 motion on an empty flat ground.  Motions
that require stairs, platforms, slopes, ladders, obstacles, or other fixed scene
support are therefore valid diagnostics but invalid main-table evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "platform_or_ledge",
        re.compile(r"\b(platform|ptfm|ledge|ledgework)\b"),
    ),
    (
        "stairs_or_steps",
        re.compile(
            r"\b(stair|stairs|staircase|stairway|upstairs|downstairs|handrail|railing|railings)\b"
            r"|\bsteps\b.{0,32}\b(up|down|onto|off|over|stone|stones|platform|stair)\b"
            r"|\b(up|down|onto|off|over)\b.{0,20}\bsteps\b"
        ),
    ),
    (
        "slope_or_ramp",
        re.compile(r"\b(slope|slopes|ramp|ramps|incline|inclines|hill|hills)\b"),
    ),
    (
        "obstacle_or_hurdle",
        re.compile(r"\b(obstacle|obstacles|hurdle|hurdles|vault|vaults|vaulting)\b"),
    ),
    (
        "ladder_or_climb",
        re.compile(
            r"\b(ladder|ladders|ldr)\b"
            r"|\b(climb|climbs|climbed|climbing|ascend|ascends|ascended|ascending|"
            r"descend|descends|descended|descending|ascd|dscd)\b"
        ),
    ),
    (
        "drop_from_elevation",
        re.compile(
            r"\b(jump|jumps|jumped|jumping|flip|flips|flipped|flipping|drop|drops|"
            r"dropped|dropping|fall|falls|fell|falling)\b"
            r".{0,60}\b(from|off|down from|down off)\b"
            r".{0,50}\b(high place|high platform|height|platform|ledge|obstacle|table)\b"
        ),
    ),
    (
        "fixed_scene_support",
        re.compile(
            r"\b(support|supports|supporting|grab|grabs|grabbing|brace|braces|bracing|"
            r"hold|holds|holding|lean|leans|leaning|push|pushes|pushing|pull|pulls|pulling)\b"
            r".{0,60}\b(platform|ledge|rail|railing|handrail|wall|door|obstacle|ladder)\b"
        ),
    ),
)


TEXT_FIELDS = (
    "prompt",
    "caption",
    "text",
    "g1_path",
    "caption_rel",
    "emb_rel",
    "motion_path",
    "source_motion_path",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def rows_from_json(blob: Any) -> list[Any]:
    if isinstance(blob, dict):
        for key in ("rows", "items", "records"):
            if isinstance(blob.get(key), list):
                return blob[key]
    if isinstance(blob, list):
        return blob
    raise TypeError("expected a list or a dict with rows/items/records")


def normalize_text(parts: Iterable[Any]) -> str:
    text = " ".join(str(part) for part in parts if part is not None)
    text = text.lower()
    text = re.sub(r"[_/\\.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_index(row: Any, fallback: int) -> int:
    if not isinstance(row, dict):
        return fallback
    if isinstance(row.get("source_index"), int):
        return int(row["source_index"])
    prompt_id = row.get("prompt_id")
    if isinstance(prompt_id, str):
        match = re.search(r"(\d+)$", prompt_id)
        if match:
            return int(match.group(1))
    if isinstance(row.get("sample_idx"), int):
        return int(row["sample_idx"])
    return fallback


def text_for_case(
    *,
    index: int,
    annotation_row: Any | None,
    manifest_rows: dict[int, dict[str, Any]],
    extra_row: Any | None = None,
) -> tuple[str, dict[str, str]]:
    parts: list[str] = []
    sources: dict[str, str] = {}
    for source_name, row in (
        ("annotation", annotation_row),
        ("manifest", manifest_rows.get(index)),
        ("extra", extra_row),
    ):
        if not isinstance(row, dict):
            continue
        source_parts: list[str] = []
        for field in TEXT_FIELDS:
            value = row.get(field)
            if value:
                source_parts.append(str(value))
        if source_parts:
            joined = " | ".join(source_parts)
            sources[source_name] = joined
            parts.append(joined)
    return normalize_text(parts), sources


def classify(text: str) -> list[str]:
    reasons = [reason for reason, pattern in PATTERNS if pattern.search(text)]
    return reasons


def load_manifest_rows(path: Path | None) -> tuple[Any | None, dict[int, dict[str, Any]]]:
    if not path:
        return None, {}
    blob = load_json(path)
    rows = rows_from_json(blob)
    indexed: dict[int, dict[str, Any]] = {}
    for fallback, row in enumerate(rows):
        if isinstance(row, dict):
            indexed[source_index(row, fallback)] = row
    return blob, indexed


def split_annotation(
    annotation_rows: list[Any],
    manifest_rows: dict[int, dict[str, Any]],
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    ground_rows: list[Any] = []
    scene_rows: list[Any] = []
    report_rows: list[dict[str, Any]] = []
    for index, row in enumerate(annotation_rows):
        text, sources = text_for_case(
            index=index,
            annotation_row=row,
            manifest_rows=manifest_rows,
        )
        reasons = classify(text)
        target = scene_rows if reasons else ground_rows
        target.append(row)
        if reasons:
            manifest_row = manifest_rows.get(index, {})
            report_rows.append(
                {
                    "source_index": index,
                    "prompt_id": manifest_row.get("prompt_id", f"gen_{index:06d}"),
                    "prompt": manifest_row.get("prompt"),
                    "reasons": reasons,
                    "sources": sources,
                }
            )
    return ground_rows, scene_rows, report_rows


def filtered_manifest(
    manifest: dict[str, Any],
    keep_indices: set[int],
    *,
    label_suffix: str,
    report_path: Path,
) -> dict[str, Any]:
    out = copy.deepcopy(manifest)
    rows = rows_from_json(out)
    new_rows = [
        row for fallback, row in enumerate(rows)
        if source_index(row, fallback) in keep_indices
    ]
    out["rows"] = new_rows
    out["group_label"] = f"{out.get('group_label', 'PhysFlow cases')} - {label_suffix}"
    out["scene_filter"] = {
        "label": label_suffix,
        "rows": len(new_rows),
        "report": str(report_path),
    }
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/annotation/_heldout_agile.json")
    ap.add_argument("--eval-manifest", default=None)
    ap.add_argument("--viz-manifest", default=None)
    ap.add_argument(
        "--out-ground",
        default="data/annotation/_heldout_agile_ground_only.json",
    )
    ap.add_argument(
        "--out-scene",
        default="data/annotation/_heldout_agile_scene_interaction.json",
    )
    ap.add_argument(
        "--report",
        default="data/annotation/_heldout_agile_scene_filter_report.json",
    )
    ap.add_argument(
        "--rewrite-viz-main",
        action="store_true",
        help="Replace --viz-manifest with flat-ground rows and keep all rows as a backup.",
    )
    args = ap.parse_args()

    annotation_path = Path(args.annotation)
    annotation_blob = load_json(annotation_path)
    annotation_rows = rows_from_json(annotation_blob)
    eval_manifest_blob, eval_rows = load_manifest_rows(
        Path(args.eval_manifest) if args.eval_manifest else None
    )
    del eval_manifest_blob

    ground_rows, scene_rows, scene_report_rows = split_annotation(
        annotation_rows,
        eval_rows,
    )

    write_json(Path(args.out_ground), ground_rows)
    write_json(Path(args.out_scene), scene_rows)

    ground_indices = set(range(len(annotation_rows))) - {
        int(row["source_index"]) for row in scene_report_rows
    }
    scene_indices = {int(row["source_index"]) for row in scene_report_rows}
    report = {
        "annotation": str(annotation_path),
        "eval_manifest": args.eval_manifest,
        "total": len(annotation_rows),
        "ground_only": len(ground_rows),
        "scene_interaction": len(scene_rows),
        "patterns": [reason for reason, _ in PATTERNS],
        "scene_cases": scene_report_rows,
    }
    report_path = Path(args.report)
    write_json(report_path, report)

    if args.viz_manifest:
        viz_path = Path(args.viz_manifest)
        viz_blob = load_json(viz_path)
        if not isinstance(viz_blob, dict):
            raise TypeError("viz manifest must be a dict")
        base_dir = viz_path.parent
        ground_manifest = filtered_manifest(
            viz_blob,
            ground_indices,
            label_suffix="flat-ground main set",
            report_path=report_path,
        )
        scene_manifest = filtered_manifest(
            viz_blob,
            scene_indices,
            label_suffix="scene-interaction diagnostics",
            report_path=report_path,
        )
        write_json(base_dir.parent / "base_vs_full_ground_viz" / "manifest.json", ground_manifest)
        write_json(
            base_dir.parent / "base_vs_full_scene_interaction_viz" / "manifest.json",
            scene_manifest,
        )
        if args.rewrite_viz_main:
            backup = base_dir / "manifest_all80_scene_unfiltered.json"
            if not backup.exists():
                write_json(backup, viz_blob)
            write_json(viz_path, ground_manifest)

    print(
        json.dumps(
            {
                "total": len(annotation_rows),
                "ground_only": len(ground_rows),
                "scene_interaction": len(scene_rows),
                "out_ground": args.out_ground,
                "out_scene": args.out_scene,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

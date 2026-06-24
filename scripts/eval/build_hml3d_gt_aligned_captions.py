#!/usr/bin/env python3
"""Build a GT-aligned HumanML3D caption root.

The official HumanML3D text files often contain multiple full-clip captions for
one motion. Some first captions are visibly wrong, so this script materializes a
separate, explicit caption set instead of overwriting the official ``texts/``.

Default policy: select the longest full-clip caption (``#0.0#0.0``) for each
test id. This is deterministic, uses only official captions, and avoids the
noisy "first full caption" convention used by some legacy evaluation scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not REPO.exists():
    REPO = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")


@dataclass
class CaptionRow:
    index: int
    raw: str
    caption: str
    tokens: str
    start: float
    end: float

    @property
    def is_full_clip(self) -> bool:
        return self.start == 0.0 and self.end == 0.0

    @property
    def word_count(self) -> int:
        return len(self.caption.split())


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return REPO / p


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _float_tag(value: str) -> float:
    if value == "nan":
        return 0.0
    out = float(value)
    return 0.0 if out != out else out


def parse_text_file(path: Path) -> list[CaptionRow]:
    rows: list[CaptionRow] = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("#")
        if len(parts) < 4 or not parts[0].strip():
            continue
        try:
            start = _float_tag(parts[2])
            end = _float_tag(parts[3])
        except ValueError:
            continue
        rows.append(CaptionRow(
            index=idx,
            raw=raw,
            caption=parts[0].strip(),
            tokens=parts[1],
            start=start,
            end=end,
        ))
    return rows


def load_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = _repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"override file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "overrides" in data:
        data = data["overrides"]
    if not isinstance(data, dict):
        raise ValueError("override JSON must be an object or contain an 'overrides' object")
    return data


def _select_override(cid: str, rows: list[CaptionRow], override: Any) -> tuple[CaptionRow, str] | None:
    if isinstance(override, int):
        for row in rows:
            if row.index == override:
                return row, "override_line_index"
        return None
    if isinstance(override, str):
        target = override.strip()
    elif isinstance(override, dict):
        if "line_index" in override:
            return _select_override(cid, rows, int(override["line_index"]))
        target = str(override.get("caption", "")).strip()
    else:
        return None
    if not target:
        return None
    for row in rows:
        if row.caption == target:
            return row, "override_caption"
    raise ValueError(f"override caption for {cid} does not match any official caption")


def select_caption(
    cid: str,
    rows: list[CaptionRow],
    policy: str,
    overrides: dict[str, Any],
) -> tuple[CaptionRow | None, str]:
    if not rows:
        return None, "missing_text"
    if cid in overrides:
        picked = _select_override(cid, rows, overrides[cid])
        if picked is None:
            raise ValueError(f"invalid override for {cid}: {overrides[cid]!r}")
        return picked

    full = [row for row in rows if row.is_full_clip]
    pool = full or rows
    reason_suffix = "" if full else "_fallback_no_full"
    if policy == "first-full":
        return min(pool, key=lambda r: r.index), "first_full" + reason_suffix
    if policy == "longest-full":
        return max(pool, key=lambda r: (r.word_count, len(r.caption), -r.index)), "longest_full" + reason_suffix
    raise ValueError(f"unknown policy: {policy}")


def write_readme(out_root: Path, stats: dict[str, Any], args: argparse.Namespace) -> None:
    (out_root / "README.md").write_text(
        "\n".join([
            "# GT-Aligned HumanML3D Captions",
            "",
            "This directory is a derived caption set for HumanML3D official test evaluation.",
            "It does not overwrite the official `texts/` files.",
            "",
            f"- source root: `{args.src_root}`",
            f"- selection policy: `{args.policy}`",
            f"- override file: `{args.overrides or ''}`",
            f"- ids: `{stats['num_ids']}`",
            f"- changed from first full caption: `{stats['changed_from_first_full']}`",
            f"- missing text files: `{stats['missing_text']}`",
            "",
            "Files:",
            "",
            "- `texts/<id>.txt`: one HumanML3D-format selected caption per id.",
            "- `prompt_map.json`: `{id: selected_caption}` for T2M inference.",
            "- `caption_map.json`: full provenance for every selected caption.",
            "- `changed_from_first_full.tsv`: ids whose selected caption differs from legacy first-full-caption evaluation.",
            "- `test_hml3d_official272_gtlen_gt_aligned_caption.json`: MotionHub-style annotation using the selected captions.",
            "",
            "Evaluation scripts should pass this root explicitly, for example:",
            "",
            "```bash",
            "python3 scripts/eval/eval_motionstreamer_272.py \\",
            "  --text-dir outputs/evaluation/t2m/humanml3d_official_test/captions/gt_aligned_longest_full_20260622/texts \\",
            "  --pred-dir outputs/evaluation/t2m/humanml3d_official_test/ms272/gt_0beta/prep/gt_0beta",
            "```",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    ap.add_argument("--source-annotation", default="data/annotation/test_hml3d_official272_gtlen.json")
    ap.add_argument(
        "--out-root",
        default="outputs/evaluation/t2m/humanml3d_official_test/captions/gt_aligned_longest_full_20260622",
    )
    ap.add_argument("--policy", choices=["first-full", "longest-full"], default="longest-full")
    ap.add_argument("--overrides", default=None)
    args = ap.parse_args()

    src_root = _repo_path(args.src_root)
    out_root = _repo_path(args.out_root)
    split_file = src_root / "split" / "test.txt"
    text_dir = src_root / "texts"
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    overrides = load_overrides(args.overrides)

    out_texts = out_root / "texts"
    out_caps = out_root / "annotation_captions"
    out_split = out_root / "split"
    out_texts.mkdir(parents=True, exist_ok=True)
    out_caps.mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(split_file, out_split / "test.txt")

    records: dict[str, Any] = {}
    prompt_map: dict[str, str] = {}
    changed_rows: list[tuple[str, str, str, str]] = []
    reason_counter: Counter[str] = Counter()
    full_count_counter: Counter[int] = Counter()
    missing_text = 0

    for cid in ids:
        rows = parse_text_file(text_dir / f"{cid}.txt")
        full_rows = [row for row in rows if row.is_full_clip]
        full_count_counter[len(full_rows)] += 1
        selected, reason = select_caption(cid, rows, args.policy, overrides)
        reason_counter[reason] += 1
        if selected is None:
            missing_text += 1
            continue
        first_full = min(full_rows, key=lambda r: r.index) if full_rows else rows[0]
        changed = selected.caption != first_full.caption

        (out_texts / f"{cid}.txt").write_text(selected.raw + "\n", encoding="utf-8")
        cap_json = out_caps / f"{cid}.json"
        cap_json.write_text(
            json.dumps({"macro": [selected.caption], "meso": [], "micro": []},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        prompt_map[cid] = selected.caption
        records[cid] = {
            "id": cid,
            "selected": asdict(selected) | {"is_full_clip": selected.is_full_clip},
            "selection_reason": reason,
            "first_full_caption": first_full.caption,
            "changed_from_first_full": bool(changed),
            "all_rows": [asdict(row) | {"is_full_clip": row.is_full_clip} for row in rows],
            "annotation_caption_path": _rel_or_abs(cap_json),
            "text_path": _rel_or_abs(out_texts / f"{cid}.txt"),
        }
        if changed:
            changed_rows.append((cid, reason, first_full.caption, selected.caption))

    annotation_path = None
    source_annotation = _repo_path(args.source_annotation)
    if source_annotation.exists():
        anno = json.loads(source_annotation.read_text(encoding="utf-8"))
        data_list = anno.get("data_list", {})
        for cid, rec in records.items():
            if cid not in data_list:
                continue
            data_list[cid]["hierarchical_caption_path"] = rec["annotation_caption_path"]
            data_list[cid]["caption_source"] = "gt_aligned_humanml3d_official"
            data_list[cid]["caption_selection_policy"] = args.policy
        anno.setdefault("meta", {})
        anno["meta"]["caption_source"] = "gt_aligned_humanml3d_official"
        anno["meta"]["caption_selection_policy"] = args.policy
        anno["meta"]["caption_root"] = _rel_or_abs(out_root)
        annotation_path = out_root / "test_hml3d_official272_gtlen_gt_aligned_caption.json"
        annotation_path.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "num_ids": len(ids),
        "num_records": len(records),
        "missing_text": missing_text,
        "selection_reasons": dict(reason_counter),
        "full_caption_count_distribution": {str(k): int(v) for k, v in sorted(full_count_counter.items())},
        "changed_from_first_full": len(changed_rows),
        "policy": args.policy,
        "source_root": _rel_or_abs(src_root),
        "out_root": _rel_or_abs(out_root),
        "annotation_path": _rel_or_abs(annotation_path) if annotation_path else None,
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_root / "caption_map.json").write_text(
        json.dumps({"meta": stats, "data": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "prompt_map.json").write_text(json.dumps(prompt_map, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_root / "changed_from_first_full.tsv").open("w", encoding="utf-8") as f:
        f.write("id\treason\tfirst_full_caption\tselected_caption\n")
        for cid, reason, first, selected in changed_rows:
            f.write(f"{cid}\t{reason}\t{first}\t{selected}\n")
    write_readme(out_root, stats, args)

    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()

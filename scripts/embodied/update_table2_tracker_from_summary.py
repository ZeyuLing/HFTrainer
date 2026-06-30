#!/usr/bin/env python3
"""Update the PhysFlow Table-2 tracker rows from unified eval summaries.

The script intentionally updates only values that are complete in the supplied
summary.  It is used by the Taiji watchdog after AMASS inference finishes, so it
must fail closed when shards are still missing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


METHOD_ROWS = {
    "any2track": "Any2Track/OpenTrack",
    "humanoid_gpt": "Humanoid-GPT",
}

SPLIT_COLUMNS = {
    "lafan1_fixed600": (0, "LAFAN1 Succ."),
    "amass_test_fixed600": (1, "AMASS-test Succ."),
    "wild_clean_fixed600": (2, "Wild-G1 Succ."),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_pct(value: Any) -> str:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite percentage source: {value!r}")
    return f"{100.0 * x:.1f}"


def _check_complete(summary: dict[str, Any], split: str) -> None:
    missing = summary.get("missing", {})
    split_missing = missing.get(split, {})
    if split_missing:
        compact = {k: len(v) for k, v in split_missing.items()}
        raise SystemExit(f"{split} is incomplete: {compact}")


def _replace_cell(table_text: str, row_label: str, col_idx: int, value: str) -> tuple[str, str]:
    pattern = re.compile(
        rf"(?P<head>\s*{re.escape(row_label)}[^\n]*&\n\s*)"
        rf"(?P<values>.*?)"
        rf"(?P<tail>\s*\\\\)",
        re.DOTALL,
    )
    match = pattern.search(table_text)
    if not match:
        raise SystemExit(f"Could not find row label in table: {row_label}")

    cells = [cell.strip() for cell in match.group("values").split("&")]
    if col_idx >= len(cells):
        raise SystemExit(f"Row {row_label} has only {len(cells)} cells")

    old = cells[col_idx]
    cells[col_idx] = value
    replacement = f"{match.group('head')}{' & '.join(cells)}{match.group('tail')}"
    return table_text[: match.start()] + replacement + table_text[match.end() :], old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--status", type=Path)
    ap.add_argument("--split", default="amass_test_fixed600")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    summary = _load(args.summary)
    _check_complete(summary, args.split)
    if args.split not in SPLIT_COLUMNS:
        raise SystemExit(f"Unsupported split for tracker table update: {args.split}")
    col_idx, field_name = SPLIT_COLUMNS[args.split]

    table_text = args.table.read_text()
    updates: dict[str, dict[str, str]] = {}
    for method, row_label in METHOD_ROWS.items():
        method_summary = summary.get("summaries", {}).get(method, {}).get(args.split)
        if not method_summary:
            raise SystemExit(f"Missing summary for {method}/{args.split}")
        if "success_rate" not in method_summary:
            raise SystemExit(f"Missing success_rate for {method}/{args.split}")
        value = _fmt_pct(method_summary["success_rate"])
        table_text, old = _replace_cell(table_text, row_label, col_idx, value)
        updates[method] = {
            "row_label": row_label,
            "split": args.split,
            "field": field_name,
            "old": old,
            "new": value,
        }

    if not args.dry_run:
        args.table.write_text(table_text)
        if args.status:
            args.status.write_text(json.dumps({"updates": updates}, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"updates": updates, "dry_run": args.dry_run}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

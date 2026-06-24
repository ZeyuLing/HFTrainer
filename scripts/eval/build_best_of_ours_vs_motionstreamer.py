#!/usr/bin/env python3
"""Build a per-case best-of ours directory from MotionStreamer Emb-L2 reports.

The baseline report is produced by ``compute_ours_vs_motionstreamer_emb_l2.py``.
Each candidate report should use the same format and point to a prep directory
whose files are named by canonical HumanML3D ids. For every case, this script
keeps the candidate with the lowest ours-vs-GT embedding L2, materializes a
single prep directory, and writes the remaining cases where best ours still
does not beat MotionStreamer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    name: str
    report: Path
    prep_dir: Path


def load_cases(report: Path) -> dict[str, dict]:
    with report.open() as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        cid = row.get("cid")
        if cid:
            out[cid] = row
    return out


def parse_candidate(text: str) -> Candidate:
    parts = text.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "candidate must be NAME:ALL_CASES_JSON:PREP_DIR"
        )
    return Candidate(parts[0], Path(parts[1]), Path(parts[2]))


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-name", default="original")
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--baseline-prep-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=[],
        help="NAME:ALL_CASES_JSON:PREP_DIR, repeatable",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    args = parser.parse_args()

    baseline = Candidate(
        args.baseline_name,
        Path(args.baseline_report),
        Path(args.baseline_prep_dir),
    )
    candidates = [baseline, *args.candidate]

    reports = {c.name: load_cases(c.report) for c in candidates}
    base_rows = reports[baseline.name]
    out_dir = Path(args.out_dir)
    prep_dir = out_dir / "prep" / "ours_best_of"
    prep_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    missing_sources = []
    for cid, base in base_rows.items():
        best_name = baseline.name
        best = base
        best_l2 = float(base["ours_emb_l2_vs_gt"])
        for cand in candidates[1:]:
            row = reports[cand.name].get(cid)
            if row is None:
                continue
            l2 = float(row["ours_emb_l2_vs_gt"])
            if l2 < best_l2:
                best_name = cand.name
                best = row
                best_l2 = l2

        source = next(c for c in candidates if c.name == best_name)
        src_file = source.prep_dir / f"{cid}.npz"
        dst_file = prep_dir / f"{cid}.npz"
        ms_l2 = float(base["motionstreamer_emb_l2_vs_gt"])
        out_row = dict(base)
        out_row.update(
            {
                "selected_candidate": best_name,
                "selected_ours_emb_l2_vs_gt": best_l2,
                "selected_delta_ours_minus_ms": best_l2 - ms_l2,
                "selected_ours_better": best_l2 <= ms_l2,
                "selected_source_file": str(src_file),
            }
        )
        if src_file.exists():
            materialize(src_file, dst_file, args.link_mode)
        else:
            out_row["selected_source_missing"] = True
            missing_sources.append(cid)
        selected_rows.append(out_row)

    selected_rows.sort(key=lambda r: r["selected_delta_ours_minus_ms"], reverse=True)
    bad_rows = [r for r in selected_rows if not r["selected_ours_better"]]
    usage = {}
    for row in selected_rows:
        usage[row["selected_candidate"]] = usage.get(row["selected_candidate"], 0) + 1

    summary = {
        "criterion": "bad iff selected_ours_emb_l2_vs_gt > motionstreamer_emb_l2_vs_gt",
        "baseline_report": str(baseline.report),
        "baseline_prep_dir": str(baseline.prep_dir),
        "candidates": [
            {"name": c.name, "report": str(c.report), "prep_dir": str(c.prep_dir)}
            for c in candidates
        ],
        "out_prep_dir": str(prep_dir),
        "link_mode": args.link_mode,
        "n_total": len(selected_rows),
        "n_selected_better_or_equal": len(selected_rows) - len(bad_rows),
        "n_selected_worse": len(bad_rows),
        "selected_better_rate": (
            (len(selected_rows) - len(bad_rows)) / max(1, len(selected_rows))
        ),
        "candidate_usage": usage,
        "missing_sources": missing_sources,
        "remaining_bad_ids": [r["cid"] for r in bad_rows],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "all_cases.json").write_text(json.dumps(selected_rows, indent=2))
    (out_dir / "remaining_bad_cases.json").write_text(json.dumps(bad_rows, indent=2))
    (out_dir / "remaining_bad_ids.txt").write_text(
        "\n".join(summary["remaining_bad_ids"]) + ("\n" if bad_rows else "")
    )
    with (out_dir / "remaining_bad_cases.csv").open("w", newline="") as f:
        fields = [
            "cid",
            "selected_delta_ours_minus_ms",
            "selected_ours_emb_l2_vs_gt",
            "motionstreamer_emb_l2_vs_gt",
            "selected_candidate",
            "gt_len",
            "ours_len",
            "caption",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in bad_rows:
            writer.writerow({k: row.get(k) for k in fields})

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

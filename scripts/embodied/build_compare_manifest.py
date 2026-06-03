#!/usr/bin/env python3
"""Merge a base (un-optimized) and an optimized PhysFlow viz manifest into a
single 3-column comparison manifest for the embodied_viz triplet dashboard.

The triplet page renders three fixed column slots (raw_reference,
optimized_reference, tracked_rollout) but reads each column's title from the
manifest, so we repurpose the slots to show, per prompt:

    col1  Base KIMODO-G1 (un-optimized T2M)        <- base.raw_reference
    col2  PhysFlow-optimized T2M                   <- opt.raw_reference
    col3  Optimized -> frozen MuJoCo tracker        <- opt.tracked_rollout

This is the view that directly supports the paper claim "does our optimization
make the generated robot motion more physically realistic / executable": the
reader sees the un-optimized vs optimized generation side by side, then whether
the optimized motion is actually trackable on G1.

Usage:
    python3 scripts/embodied/build_compare_manifest.py \
        --base work_dirs/.../hml3dtest_base_manifest/manifest.json \
        --opt  work_dirs/.../hml3dtest_v3_manifest/manifest.json \
        --out  work_dirs/.../hml3dtest_compare_manifest/manifest.json \
        --opt-label "PhysFlow-v3 (iter_1050)"
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def _rows_by_id(manifest: dict) -> dict:
    return {r["prompt_id"]: r for r in manifest.get("rows", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="un-optimized arm manifest.json")
    ap.add_argument("--opt", required=True, help="optimized arm manifest.json")
    ap.add_argument("--out", required=True, help="output combined manifest.json")
    ap.add_argument("--base-label", default="Base KIMODO-G1 (un-optimized)")
    ap.add_argument("--opt-label", default="PhysFlow-optimized")
    args = ap.parse_args()

    base = json.load(open(args.base))
    opt = json.load(open(args.opt))
    base_rows = _rows_by_id(base)
    opt_rows = _rows_by_id(opt)

    rows = []
    for idx, (pid, br) in enumerate(sorted(base_rows.items())):
        orow = opt_rows.get(pid)
        if orow is None:
            continue
        b_raw = br["columns"]["raw_reference"]
        o_raw = orow["columns"]["raw_reference"]
        o_trk = orow["columns"]["tracked_rollout"]
        # This is a per-prompt base-vs-optimized comparison, NOT a training-iteration
        # timeline, so the per-card heading is the CASE index + prompt id (the old
        # iteration_label like "iter_000900" was a leftover from the viz run and
        # confused which prompt each card belonged to).
        rows.append({
            "iteration": idx,
            "iteration_label": f"Case {idx:02d}  \u00b7  {pid}",
            "prompt_id": pid,
            "prompt": br.get("prompt", ""),
            "category": br.get("category", ""),
            "difficulty": br.get("difficulty"),
            "seed": br.get("seed"),
            "sample_idx": br.get("sample_idx", 0),
            "columns": {
                "raw_reference": {
                    "title": args.base_label,
                    "path": b_raw.get("path", ""),
                    "status": b_raw.get("status", "pending"),
                    "metrics": b_raw.get("metrics", {}),
                },
                "optimized_reference": {
                    "title": args.opt_label,
                    "path": o_raw.get("path", ""),
                    "status": o_raw.get("status", "pending"),
                    "metrics": o_raw.get("metrics", {}),
                },
                "tracked_rollout": {
                    "title": f"{args.opt_label} -> MuJoCo tracker",
                    "path": o_trk.get("path", ""),
                    "status": o_trk.get("status", "pending"),
                    "metrics": o_trk.get("metrics", {}),
                },
            },
        })

    out = {
        "schema_version": base.get("schema_version", 1),
        "project": "PhysFlow KIMODO-G1 | base vs optimized (HumanML3D test)",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {"base": args.base, "optimized": args.opt},
        # tells the triplet dashboard to label the filter dropdown by "case"
        # instead of the default "iter" (this manifest is not an iteration timeline)
        "group_label": "case",
        "rows": rows,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_p, "w"), ensure_ascii=False, indent=2)
    print(f"[compare-manifest] wrote {out_p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

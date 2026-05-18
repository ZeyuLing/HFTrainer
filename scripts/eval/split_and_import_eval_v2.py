#!/usr/bin/env python3
"""Split eval_v2_*.json files into per-(model, task, setting) JSONs and import
into the 8082 eval dashboard DB.

Usage:
    python scripts/eval/split_and_import_eval_v2.py <run_root_dir> [--db <db_path>]

Looks for ``eval_v2_*.json`` recursively under ``run_root_dir`` (each job
directory produced by run_e3_e8d_e14_e15_latest_v2.sh contains exactly one),
splits each by (model, task, setting), writes the slim per-run files into
``<run_root>/import_jsons/`` and calls ``data_importer.import_result_json`` on
each one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "motion_annot_web" / "eval_dashboard"))

SKIP_TASK_IDS = {"E6"}
SKIP_SETTINGS = {
    ("E4", "A_rhand_sparse"),
    ("E4", "B_ankles_sparse"),
    ("E4", "C_rhand_lfoot"),
    ("E4", "D_both_hands"),
    ("E4", "E_all4_sparse"),
    ("E4", "F_rhand_dense"),
    ("E14", "M_c5_t60"),
    ("E14", "M_c5_t120"),
    ("E14", "M_c5_t180"),
    ("E14", "M_c15_t60"),
    ("E14", "M_c15_t120"),
    ("E14", "M_c15_t180"),
    ("E14", "M_c30_t60"),
    ("E14", "M_c30_t120"),
    ("E14", "M_c30_t180"),
    ("E14", "M_c45_t120"),
    ("E14", "M_c60_t120"),
    ("E14", "M_c75_t120"),
    ("E14", "M_c90_t120"),
    ("E14", "M_c45_t150"),
    ("E14", "M_c45_t180"),
    ("E14", "M_c45_t240"),
}


def _slim_run(model_name: str, task_data: Dict[str, Any], parent: Dict[str, Any]) -> Dict[str, Any]:
    """Build the JSON shape that data_importer.import_result_json expects."""
    return {
        "model": model_name,
        "checkpoint": parent.get("checkpoint", ""),
        "rotation_space": parent.get("rotation_space", "local"),
        "has_caption": parent.get("has_caption", False),
        "motion_dim": parent.get("motion_dim", 198),
        "num_steps": parent.get("num_steps", 50),
        "replacement_guidance": parent.get("replacement_guidance", "skip_last"),
        "timestamp": parent.get("timestamp", ""),
        "task_id": task_data.get("task_id"),
        "setting": task_data.get("setting"),
        "num_prompts": task_data.get("num_samples", 0),
        "aggregated": task_data.get("aggregated", {}),
        "per_sample": task_data.get("per_sample", []),
    }


def _checkpoint_from(parent: Dict[str, Any]) -> str:
    return parent.get("checkpoint", "")


def _has_caption_from(model_name: str, parent: Dict[str, Any]) -> bool:
    if "caption_" in model_name:
        return True
    return bool(parent.get("has_caption", False))


def split_one(eval_v2_path: Path, out_dir: Path) -> List[Path]:
    """Split a single eval_v2_*.json into per-(model, task, setting) JSONs."""
    with open(eval_v2_path, "r") as f:
        all_results = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for model_name, model_block in all_results.items():
        if not isinstance(model_block, dict):
            continue
        if "tasks" not in model_block:
            continue
        # Add some defaults pulled from the per-model block
        parent = {
            "checkpoint": _checkpoint_from(model_block),
            "rotation_space": model_block.get("rotation_space", "local"),
            "has_caption": _has_caption_from(model_name, model_block),
            "motion_dim": model_block.get("motion_dim", 198),
            "num_steps": model_block.get("num_steps", 50),
            "replacement_guidance": model_block.get("replacement_guidance", "skip_last"),
            "timestamp": model_block.get("timestamp", ""),
        }
        for task_key, task_data in (model_block.get("tasks") or {}).items():
            if not isinstance(task_data, dict):
                continue
            slim = _slim_run(model_name, task_data, parent)
            if not slim["task_id"] or not slim["setting"]:
                continue
            if slim["task_id"] in SKIP_TASK_IDS:
                continue
            if (slim["task_id"], slim["setting"]) in SKIP_SETTINGS:
                continue
            fname = f"{model_name}__{slim['task_id']}_{slim['setting']}.json"
            fpath = out_dir / fname
            with open(fpath, "w") as f:
                json.dump(slim, f)
            written.append(fpath)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path,
                        help="Directory that contains job sub-directories with eval_v2_*.json")
    parser.add_argument("--db", default=None, help="Path to eval_dashboard.db (default: project default)")
    parser.add_argument("--no-import", action="store_true",
                        help="Only split JSONs (skip DB import).")
    parser.add_argument("--notes", default="", help="Notes attached to imported runs.")
    args = parser.parse_args()

    if not args.run_root.exists():
        raise FileNotFoundError(args.run_root)

    out_dir = args.run_root / "import_jsons"
    written: List[Path] = []
    eval_files = sorted(args.run_root.rglob("eval_v2_*.json"))
    print(f"[split] found {len(eval_files)} eval_v2_*.json under {args.run_root}")
    for ef in eval_files:
        new_files = split_one(ef, out_dir)
        print(f"  {ef.relative_to(args.run_root)} -> {len(new_files)} slim files")
        written.extend(new_files)

    print(f"[split] wrote {len(written)} slim files to {out_dir}")

    if args.no_import:
        return

    from db_manager import EvalDashboardDB
    from data_importer import import_result_json

    if args.db is None:
        db_path = str(PROJECT_ROOT / "motion_annot_web" / "eval_dashboard" / "eval_dashboard.db")
    else:
        db_path = args.db
    db = EvalDashboardDB(db_path)
    notes = args.notes or f"Imported from {args.run_root.name}"
    ok = 0
    err = 0
    for path in written:
        result = import_result_json(db, str(path), notes=notes)
        status = result.get("status", "error")
        if status == "ok":
            ok += 1
        else:
            err += 1
            print(f"  [ERROR] {path.name}: {result}")
    print(f"[import] ok={ok} err={err}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Watch eval_v2_*.json files and import each changed file once."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scripts.eval.split_and_import_eval_v2 import split_one


def _run_key_and_samples(path: Path) -> Tuple[str, str, str, int]:
    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    model = str(data.get("model", "unknown"))
    task_id = str(data.get("task_id", ""))
    setting = str(data.get("setting", "default"))
    num_samples = int(data.get("num_prompts", 0) or len(data.get("per_sample", [])))
    return model, task_id, setting, num_samples


def _existing_num_samples(db: Any, model: str, task_id: str, setting: str) -> Optional[int]:
    conn = db._get_conn()
    row = conn.execute(
        """
        SELECT er.num_samples
        FROM eval_runs er
        JOIN models m ON er.model_id = m.id
        WHERE m.name = ? AND er.task_id = ? AND er.setting = ?
        """,
        (model, task_id, setting),
    ).fetchone()
    if row is None:
        return None
    return int(row["num_samples"] or 0)


def _load_state(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    try:
        data = json.load(open(path))
    except Exception:
        return {}
    return {str(k): float(v) for k, v in data.items()}


def _save_state(path: Path, state: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--db", default="motion_annot_web/eval_dashboard/eval_dashboard.db")
    parser.add_argument("--state", default="outputs/eval_caption_qwen3_import_state.json")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--notes", default="qwen3 caption auto import")
    parser.add_argument(
        "--skip-smaller-existing",
        action="store_true",
        help="Do not replace an existing run when the new split file has fewer samples.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Also import settings that split_and_import_eval_v2 normally suppresses.",
    )
    args = parser.parse_args()

    from motion_annot_web.eval_dashboard.db_manager import EvalDashboardDB
    from motion_annot_web.eval_dashboard.data_importer import import_result_json

    db = EvalDashboardDB(args.db)
    state_path = Path(args.state)
    state = _load_state(state_path)
    print(f"[watch] db={args.db}", flush=True)
    print(f"[watch] state={state_path}", flush=True)
    for root in args.roots:
        print(f"[watch] root={root}", flush=True)

    while True:
        changed = False
        for root_raw in args.roots:
            root = Path(root_raw)
            if not root.exists():
                continue
            for eval_path in sorted(root.rglob("eval_v2_*.json")):
                key = str(eval_path.resolve())
                mtime = eval_path.stat().st_mtime
                if state.get(key) == mtime:
                    continue
                try:
                    out_dir = eval_path.parent / "import_jsons"
                    written = split_one(
                        eval_path,
                        out_dir,
                        include_skipped=args.include_skipped,
                    )
                    ok = 0
                    skipped = 0
                    for slim in written:
                        if args.skip_smaller_existing:
                            model, task_id, setting, num_samples = _run_key_and_samples(slim)
                            existing = _existing_num_samples(db, model, task_id, setting)
                            if existing is not None and num_samples < existing:
                                skipped += 1
                                print(
                                    f"[watch] skipped smaller {slim}: "
                                    f"new={num_samples} existing={existing}",
                                    flush=True,
                                )
                                continue
                        result = import_result_json(db, str(slim), notes=args.notes)
                        if result.get("status") == "ok":
                            ok += 1
                        else:
                            raise RuntimeError(f"{slim}: {result}")
                    state[key] = mtime
                    _save_state(state_path, state)
                    changed = True
                    print(
                        f"[watch] imported {eval_path}: ok={ok} skipped={skipped} total={len(written)}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[watch] import failed for {eval_path}: {exc}", flush=True)
        if not changed:
            print("[watch] idle", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

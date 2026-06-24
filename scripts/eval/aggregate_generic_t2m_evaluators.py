#!/usr/bin/env python3
"""Aggregate per-method generic T2M evaluator JSON files into one table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

from scripts.eval.run_generic_t2m_evaluators import (  # noqa: E402
    DEFAULT_MANIFEST,
    _count_files,
    _json_safe,
    _slug,
    flatten_for_csv,
    load_methods,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_safe))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--evaluators", default="tmr")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    evaluators = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    methods = load_methods(Path(args.manifest))

    all_results = {}
    rows = []
    for method in methods:
        label = method["label"]
        paths = {
            "motion135_dir": str(method["motion135_dir"]),
            "hml263_dir": str(method["hml263_dir"]) if method["hml263_dir"] else None,
            "motion135_count": _count_files(method["motion135_dir"], ".npz"),
            "hml263_count": _count_files(method["hml263_dir"], ".npy"),
        }
        all_results[label] = {"paths": paths, "evaluators": {}}
        for evaluator in evaluators:
            result_path = out_dir / evaluator / f"{_slug(label)}.json"
            if result_path.exists():
                res = json.loads(result_path.read_text())
            else:
                res = {
                    "method": label,
                    "status": "missing_result",
                    "reason": f"no result JSON at {result_path}",
                    "paths": paths,
                }
            res.setdefault("method", label)
            res.setdefault("paths", paths)
            all_results[label]["evaluators"][evaluator] = res
            rows.append(flatten_for_csv(label, evaluator, res))

    write_json(out_dir / "summary.json", all_results)
    fieldnames = sorted({k for row in rows for k in row})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aggregate] wrote {out_dir / 'summary.json'}")
    print(f"[aggregate] wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

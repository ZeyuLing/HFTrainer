#!/usr/bin/env python3
"""Build a viewer manifest for HYMotion/AIST/FineDance translation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


METHODS = [
    ("hymotion", "HYMotion raw", "reference"),
    ("aist", "AIST++ fixed official", "ours"),
    ("finedance", "FineDance fixed official", "baseline"),
]


def read_rows(path: Path, subset: str | None = None) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("rows", []) if row.get("ok")]
    if subset is not None:
        rows = [row for row in rows if row.get("subset") == subset]
    if not rows:
        suffix = "" if subset is None else f" for subset={subset}"
        raise ValueError(f"{path} has no ok rows{suffix}")
    return rows


def debug_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mesh_min_y": row.get("model_trans_mesh_min_y"),
        "mesh_frame_min_y_mean": row.get("model_trans_mesh_frame_min_y_mean"),
        "joint_min_y": row.get("model_trans_joint_min_y"),
        "classification": row.get("classification"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hymotion-report", required=True)
    parser.add_argument("--aist-report", required=True)
    parser.add_argument("--finedance-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-cases", type=int, default=8)
    args = parser.parse_args()

    grouped = {
        "hymotion": read_rows(Path(args.hymotion_report)),
        "aist": read_rows(Path(args.aist_report), "aist"),
        "finedance": read_rows(Path(args.finedance_report), "finedance"),
    }
    n = min(args.num_cases, *(len(rows) for rows in grouped.values()))

    cases = []
    for idx in range(n):
        motions = []
        caption_lines = []
        for method_id, label, kind in METHODS:
            row = grouped[method_id][idx]
            motions.append({
                "id": method_id,
                "label": label,
                "kind": kind,
                "smpl_path": row["path"],
                "debug": debug_from_row(row),
            })
            caption_lines.append(
                f"{label}: {row['classification']} | mesh_min_y={row['model_trans_mesh_min_y']:.4f} | {row['path']}"
            )
        cases.append({
            "key": f"translation_compare_{idx + 1:02d}",
            "dataset": "HYMotion vs MotionHub",
            "genre": "translation-alignment",
            "split": "comparison",
            "caption": "\n".join(caption_lines),
            "motions": motions,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "description": (
                    "HYMotion raw / AIST++ fixed official / FineDance fixed official "
                    "comparison. Viewer must use Th = transl + shaped_rest_root_joint "
                    "and must not floor-align or canonicalize in frontend."
                ),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "num_cases": n}, indent=2))


if __name__ == "__main__":
    main()

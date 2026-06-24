#!/usr/bin/env python3
"""Build a HumanML3D-263 id list after excluding implausible recovered GT clips."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.hml263_to_smpl_ik import N_JOINTS, recover_from_ric, resample_linear


T2M22_EDGES = [
    [0, 2], [2, 5], [5, 8], [8, 11],
    [0, 1], [1, 4], [4, 7], [7, 10],
    [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
    [9, 14], [14, 17], [17, 19], [19, 21],
    [9, 13], [13, 16], [16, 18], [18, 20],
]


def load_ids(gt_dir: Path, ids_path: Path | None) -> list[str]:
    if ids_path is None:
        return sorted(path.stem for path in gt_dir.glob("*.npy"))
    return [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quality_issue(
    path: Path,
    *,
    source_fps: float,
    target_fps: float,
    floor_tol: float,
    max_span_y: float,
    max_root_y: float,
    max_bone: float,
) -> str | None:
    if not path.exists():
        return "missing"
    feat = np.load(path).astype(np.float32)
    if feat.ndim != 2 or feat.shape[-1] != 263:
        return f"bad_shape={feat.shape}"
    joints = resample_linear(recover_from_ric(feat, N_JOINTS), source_fps, target_fps)
    if not np.isfinite(joints).all():
        return "non_finite"
    flat = joints.reshape(-1, 3)
    y_min = float(flat[:, 1].min())
    y_span = float(flat[:, 1].max() - y_min)
    root_y_max = float(joints[:, 0, 1].max())
    bones = [np.linalg.norm(joints[:, a] - joints[:, b], axis=-1) for a, b in T2M22_EDGES]
    bone_max = float(np.stack(bones, axis=-1).max()) if bones else 0.0
    if abs(y_min) > floor_tol:
        return f"floor_y={y_min:.3f}"
    if y_span > max_span_y:
        return f"span_y={y_span:.3f}"
    if root_y_max > max_root_y:
        return f"root_y_max={root_y_max:.3f}"
    if bone_max > max_bone:
        return f"bone_max={bone_max:.3f}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out-ids", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--floor-tol", type=float, default=0.08)
    parser.add_argument("--max-span-y", type=float, default=2.50)
    parser.add_argument("--max-root-y", type=float, default=2.40)
    parser.add_argument("--max-bone", type=float, default=0.80)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    ids = load_ids(gt_dir, Path(args.ids) if args.ids else None)
    kept: list[str] = []
    excluded: list[dict[str, str]] = []
    for sid in ids:
        issue = quality_issue(
            gt_dir / f"{sid}.npy",
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            floor_tol=args.floor_tol,
            max_span_y=args.max_span_y,
            max_root_y=args.max_root_y,
            max_bone=args.max_bone,
        )
        if issue is None:
            kept.append(sid)
        else:
            excluded.append({"key": sid, "reason": issue})

    out_ids = Path(args.out_ids)
    out_ids.parent.mkdir(parents=True, exist_ok=True)
    out_ids.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    report = {
        "gt_dir": str(gt_dir),
        "ids": args.ids,
        "out_ids": str(out_ids),
        "selected": len(ids),
        "kept": len(kept),
        "excluded": len(excluded),
        "excluded_items": excluded,
        "thresholds": {
            "floor_tol": args.floor_tol,
            "max_span_y": args.max_span_y,
            "max_root_y": args.max_root_y,
            "max_bone": args.max_bone,
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected", "kept", "excluded")}, indent=2))
    print(f"[hml263-quality-filter] wrote {out_ids}")
    print(f"[hml263-quality-filter] wrote {out_json}")


if __name__ == "__main__":
    main()

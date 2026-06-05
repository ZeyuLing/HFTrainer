#!/usr/bin/env python3
"""Convert row/interleaved SMPL-135 NPZ predictions to MotionCLIP-evaluator input.

The M2M viewer and SMPL FK files use row/interleaved 6D rotations:
``[R00, R01, R10, R11, R20, R21]``.  ``eval_with_motionclip_evaluator.py``
loads GT motions with ``matrix_to_rotation_6d(..., convention="column")`` and
therefore expects predictions in column-major layout:
``[R00, R10, R20, R01, R11, R21]``.

This helper also remaps canonical HumanML3D ids (e.g. ``000000``) to annotation
keys (e.g. ``humanml3d_...``), matching the evaluator's alignment protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROW_TO_COL = [0, 2, 4, 1, 3, 5]


def _annotation_map(anno_file: Path, include_mirrors: bool) -> list[tuple[str, str]]:
    raw = json.loads(anno_file.read_text())
    data = raw["data_list"] if isinstance(raw, dict) and "data_list" in raw else raw
    if not isinstance(data, dict):
        raise ValueError(f"expected dict data_list in {anno_file}")
    pairs: list[tuple[str, str]] = []
    for anno_name, entry in data.items():
        cid = Path(str(entry.get("smplx_path") or "")).stem
        if not cid:
            continue
        if cid.startswith("M") and not include_mirrors:
            continue
        pairs.append((cid, str(anno_name)))
    return pairs


def _load_motion135(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        if "motion_135" not in z.files:
            raise KeyError(f"{path} has no motion_135")
        motion = np.asarray(z["motion_135"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[-1] != 135:
        raise ValueError(f"{path}: expected (T,135), got {motion.shape}")
    return motion


def _row_to_column_motion(motion: np.ndarray) -> np.ndarray:
    out = np.asarray(motion, dtype=np.float32).copy()
    r6 = out[:, 3:].reshape(len(out), 22, 6)
    out[:, 3:] = r6[:, :, ROW_TO_COL].reshape(len(out), 132)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--src-dir", required=True, help="Canonical-id NPZ dir with row/interleaved motion_135.")
    ap.add_argument("--out-dir", required=True, help="Annotation-key .npy dir for MotionCLIP evaluator.")
    ap.add_argument("--include-mirrors", action="store_true", default=True)
    ap.add_argument("--no-include-mirrors", dest="include_mirrors", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = missing = skipped = failed = 0
    for cid, anno_name in _annotation_map(Path(args.anno_file), args.include_mirrors):
        src = src_dir / f"{cid}.npz"
        dst = out_dir / f"{anno_name}.npy"
        if not src.exists():
            missing += 1
            continue
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            np.save(dst, _row_to_column_motion(_load_motion135(src)))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 10:
                print(f"[fail] {cid}->{anno_name}: {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "ok": ok,
        "missing": missing,
        "skipped": skipped,
        "failed": failed,
        "include_mirrors": bool(args.include_mirrors),
        "format": "motionclip_column_major_135",
    }
    (out_dir / "_convert_row135_to_motionclip_col_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[done] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Map HumanML3D canonical-id predictions to annotation-key predictions.

HumanML3D-263 baselines are keyed by official motion ids such as ``010541``.
``eval_with_motionclip_evaluator.py`` aligns predictions by annotation keys such
as ``humanml3d_194``.  This helper creates a lightweight symlink/copy directory
with annotation-key file names so the MotionCLIP evaluator can score the same
predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", default=".npz", choices=[".npz", ".npy"])
    ap.add_argument("--include-mirrors", action="store_true")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    anno = json.loads(Path(args.anno_file).read_text())["data_list"]
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = skipped = missing = 0
    for name, entry in anno.items():
        smplx_path = str(entry.get("smplx_path") or "")
        cid = Path(smplx_path).stem
        if not cid:
            missing += 1
            continue
        if cid.startswith("M") and not args.include_mirrors:
            skipped += 1
            continue
        src = src_dir / f"{cid}{args.suffix}"
        if not src.exists():
            missing += 1
            continue
        dst = out_dir / f"{name}{args.suffix}"
        if dst.exists() or dst.is_symlink():
            if not args.overwrite:
                skipped += 1
                continue
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.relpath(src, dst.parent), dst)
        linked += 1

    print(
        f"[done] src={src_dir} out={out_dir} linked={linked} "
        f"skipped={skipped} missing={missing} include_mirrors={args.include_mirrors}",
        flush=True,
    )


if __name__ == "__main__":
    main()

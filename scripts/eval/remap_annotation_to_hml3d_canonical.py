#!/usr/bin/env python3
"""Map annotation-key predictions to HumanML3D canonical-id predictions."""
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
    ap.add_argument("--suffix", default=".npy", choices=[".npy", ".npz"])
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    anno = json.loads(Path(args.anno_file).read_text())["data_list"]
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = skipped = missing = duplicate = 0
    used: set[str] = set()
    for name, entry in anno.items():
        cid = Path(str(entry.get("smplx_path") or "")).stem
        if not cid:
            missing += 1
            continue
        if cid in used:
            duplicate += 1
            continue
        src = src_dir / f"{name}{args.suffix}"
        if not src.exists():
            missing += 1
            continue
        dst = out_dir / f"{cid}{args.suffix}"
        if dst.exists() or dst.is_symlink():
            if not args.overwrite:
                skipped += 1
                continue
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        used.add(cid)
        linked += 1

    print(
        f"[done] src={src_dir} out={out_dir} linked={linked} "
        f"skipped={skipped} missing={missing} duplicate={duplicate}",
        flush=True,
    )


if __name__ == "__main__":
    main()

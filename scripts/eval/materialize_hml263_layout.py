#!/usr/bin/env python3
"""Materialize flat HML263 ``<id>.npy`` files as a HumanML-style split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _iter_annotation(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
        return
    for i, entry in enumerate(data):
        key = entry.get("motion_id") or entry.get("id") or str(i)
        yield str(key), entry


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, dst.parent), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--anno-file", default="")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    flat_dir = Path(args.flat_dir)
    out_dir = Path(args.out_dir)
    out_motion = out_dir / "new_joint_vecs"
    out_motion.mkdir(parents=True, exist_ok=True)

    if args.anno_file:
        ordered = [key for key, _ in _iter_annotation(Path(args.anno_file))]
    else:
        ordered = sorted(path.stem for path in flat_dir.glob("*.npy"))

    ids = []
    missing = 0
    for key in ordered:
        src = flat_dir / f"{key}.npy"
        if not src.exists():
            missing += 1
            continue
        _link_or_copy(src.resolve(), out_motion / f"{key}.npy", args.copy)
        ids.append(key)

    (out_dir / "test.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    summary = {
        "flat_dir": str(flat_dir),
        "out_dir": str(out_dir),
        "anno_file": args.anno_file,
        "num_ids": len(ids),
        "missing_from_order": missing,
        "copy": args.copy,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

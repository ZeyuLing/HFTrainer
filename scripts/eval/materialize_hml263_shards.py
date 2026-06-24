#!/usr/bin/env python3
"""Combine sharded HumanML-style outputs into one HumanML-style split."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True, help="Directory containing shard_*/new_joint_vecs.")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    out_dir = Path(args.out_dir)
    out_motion = out_dir / "new_joint_vecs"
    out_motion.mkdir(parents=True, exist_ok=True)
    written = []
    missing = []
    for idx, sid in enumerate(_ids(Path(args.ids))):
        shard = idx % args.num_shards
        src = shard_root / f"shard_{shard}" / "new_joint_vecs" / f"{sid}.npy"
        if not src.exists():
            # Some callers may materialize shards with different partitioning;
            # fall back to a tiny linear search to keep the tool robust.
            src = next(shard_root.glob(f"shard_*/new_joint_vecs/{sid}.npy"), None)
        if src is None or not src.exists():
            missing.append(sid)
            continue
        dst = out_motion / f"{sid}.npy"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
        written.append(sid)
    (out_dir / "test.txt").write_text("\n".join(written) + ("\n" if written else ""), encoding="utf-8")
    print({
        "shard_root": str(shard_root),
        "out_dir": str(out_dir),
        "written": len(written),
        "missing": len(missing),
        "missing_first": missing[:10],
    })


if __name__ == "__main__":
    main()

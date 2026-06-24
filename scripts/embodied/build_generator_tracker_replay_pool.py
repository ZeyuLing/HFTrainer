#!/usr/bin/env python3
"""Pack PhysFlow generator eval outputs into tracker replay pools.

Inputs are PhysFlow frozen-eval directories that contain:

    generated/csv/*.csv      # qpos [T, 36], 30 fps
    generated/proto/*.motion # ProtoMotions replay files converted from CSV

The script writes one shared replay bundle:

    <out>/csv/*.csv
    <out>/qpos_npz/*.npz      # qpos + frequency, for Any2Track/HumanoidGPT
    <out>/proto/*.motion      # symlinks/copies, for ProtoMotions
    <out>/manifest.json

Names are prefixed by the eval directory so repeated stems such as e0000 do not
collide. This keeps generator-to-tracker experiments reproducible and makes the
three tracker families consume the same generated motion set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "item"


def _link_or_copy(src: Path, dst: Path, mode: str, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if force:
            dst.unlink()
        else:
            return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"bad mode: {mode}")


def _digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_eval_dirs(roots: Iterable[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "generated" / "csv").is_dir():
            dirs.append(root)
            continue
        for csv_dir in sorted(root.glob("*/generated/csv")):
            dirs.append(csv_dir.parents[1])
    return sorted(set(d.resolve() for d in dirs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        required=True,
        help="Eval root or directory containing *_frozen_eval/generated/csv.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Skip duplicate CSV payloads. Off by default because repeated noise seeds across checkpoints are useful.",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    csv_out = out / "csv"
    npz_out = out / "qpos_npz"
    proto_out = out / "proto"
    for d in (csv_out, npz_out, proto_out):
        d.mkdir(parents=True, exist_ok=True)

    eval_dirs = _iter_eval_dirs(args.source_root)
    if not eval_dirs:
        raise SystemExit("No generated/csv directories found.")

    seen: set[str] = set()
    items: list[dict[str, object]] = []
    for eval_dir in eval_dirs:
        rel = eval_dir
        try:
            rel = eval_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        prefix = _safe_name(str(rel).replace("/", "__"))
        csv_dir = eval_dir / "generated" / "csv"
        proto_dir = eval_dir / "generated" / "proto"
        for csv_path in sorted(csv_dir.glob("*.csv")):
            digest = _digest(csv_path) if args.dedupe else ""
            if args.dedupe and digest in seen:
                continue
            if args.dedupe:
                seen.add(digest)
            name = f"{prefix}__{csv_path.stem}"
            qpos = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
            if qpos.ndim == 1:
                qpos = qpos[None]
            csv_dst = csv_out / f"{name}.csv"
            npz_dst = npz_out / f"{name}.npz"
            _link_or_copy(csv_path, csv_dst, args.mode, args.force)
            np.savez(npz_dst, qpos=qpos.astype(np.float32), frequency=np.float32(args.fps))

            proto_src = proto_dir / f"{csv_path.stem}.motion"
            proto_dst = None
            if proto_src.is_file():
                proto_dst = proto_out / f"{name}.motion"
                _link_or_copy(proto_src, proto_dst, args.mode, args.force)

            items.append(
                {
                    "name": name,
                    "source_eval": str(eval_dir),
                    "source_csv": str(csv_path),
                    "csv": str(csv_dst),
                    "qpos_npz": str(npz_dst),
                    "proto_motion": str(proto_dst) if proto_dst else None,
                    "frames": int(qpos.shape[0]),
                    "qpos_dim": int(qpos.shape[1]),
                    "frequency": float(args.fps),
                    "sha1": digest or None,
                }
            )

    if not items:
        raise SystemExit("No CSV files packed.")
    manifest = {
        "created_from": [str(p.resolve()) for p in args.source_root],
        "count": len(items),
        "proto_count": sum(1 for x in items if x["proto_motion"]),
        "fps": float(args.fps),
        "items": items,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "qpos_npz_manifest.txt").write_text("\n".join(Path(x["qpos_npz"]).stem for x in items) + "\n")
    print(f"[generator-replay] eval_dirs={len(eval_dirs)} packed={len(items)} proto={manifest['proto_count']}")
    print(f"[generator-replay] out={out}")


if __name__ == "__main__":
    main()

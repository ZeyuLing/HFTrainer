#!/usr/bin/env python3
"""Utility ops for canonical TP2M HumanML3D leaderboard artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANNO = ROOT / (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/"
    "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)


def _ids() -> list[str]:
    raw = json.loads(ANNO.read_text())
    data = raw.get("data_list", raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"bad annotation format: {ANNO}")
    return sorted(str(k) for k in data)


def _parse_conds(value: str) -> list[int]:
    return [int(x) for x in str(value).replace(",", " ").split() if x.strip()]


def _parse_methods(value: str) -> list[str]:
    methods = [x.strip() for x in str(value).replace(",", " ").split() if x.strip()]
    if methods == ["all"]:
        return ["motionstreamer", "flowmdm", "motionlab", "kimodo"]
    return methods


def _dataset(cond: int) -> str:
    return f"humanml3d_official_test_c{cond}"


def _rep_dir(cond: int, rep: str, method: str) -> Path:
    return ROOT / "outputs/evaluation/tp2m" / _dataset(cond) / rep / method


def _count(path: Path, suffixes: tuple[str, ...] = (".npy", ".npz")) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix in suffixes and not p.name.startswith("_"))


def _stems(path: Path, suffixes: tuple[str, ...] = (".npy", ".npz")) -> set[str]:
    if not path.exists():
        return set()
    return {p.stem for p in path.iterdir() if p.is_file() and p.suffix in suffixes and not p.name.startswith("_")}


def _write_run_metadata(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (path / "command.txt").write_text(
        "Produced by canonical TP2M leaderboard completion scripts. "
        "See run_config.json for protocol and source details.\n"
    )


def cmd_status(args: argparse.Namespace) -> int:
    official = set(_ids())
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in _parse_conds(args.conds):
        for method in _parse_methods(args.methods):
            reps = ["smplx"] if method == "kimodo" else ["motion135", "ms272"]
            if method in {"flowmdm", "motionlab"}:
                reps = ["hml263", "motion135", "ms272"]
            for rep in reps:
                path = _rep_dir(cond, rep, method)
                stems = _stems(path)
                missing = sorted(official - stems)
                extra = sorted(stems - official)
                missing_file = out_root / f"{method}_c{cond}_{rep}_missing.txt"
                missing_file.write_text("\n".join(missing) + ("\n" if missing else ""))
                row = {
                    "cond": cond,
                    "method": method,
                    "rep": rep,
                    "count": len(stems & official),
                    "expected": len(official),
                    "missing": len(missing),
                    "extra": len(extra),
                    "path": str(path.relative_to(ROOT)),
                    "missing_file": str(missing_file.relative_to(ROOT)),
                }
                rows.append(row)
                print(
                    f"{method:14s} c{cond:<2d} {rep:9s} "
                    f"{row['count']:4d}/{row['expected']} missing={row['missing']:4d} extra={row['extra']:3d} "
                    f"{row['path']}",
                    flush=True,
                )
    (out_root / "status.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return 0


def _iter_ids_file(path: Path | None, fallback: Iterable[str]) -> list[str]:
    if path is None:
        return list(fallback)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def cmd_convert_ms272(args: argparse.Namespace) -> int:
    from hftrainer.motion.representation.motion272 import motion135_to_272

    official = set(_ids())
    total_written = total_skipped = total_missing = total_failed = 0
    for cond in _parse_conds(args.conds):
        for method in _parse_methods(args.methods):
            if method == "kimodo" and not _rep_dir(cond, "motion135", method).exists():
                continue
            src_dir = _rep_dir(cond, "motion135", method)
            dst_dir = _rep_dir(cond, "ms272", method)
            dst_dir.mkdir(parents=True, exist_ok=True)
            ids = _iter_ids_file(Path(args.ids) if args.ids else None, sorted(official))
            written = skipped = missing = failed = 0
            for sid in ids:
                if sid not in official:
                    continue
                src = src_dir / f"{sid}.npz"
                dst = dst_dir / f"{sid}.npz"
                if args.skip_existing and dst.exists():
                    skipped += 1
                    continue
                if not src.exists():
                    missing += 1
                    continue
                try:
                    with np.load(src, allow_pickle=True) as data:
                        m135 = np.asarray(data["motion_135"], dtype=np.float32)
                    m272 = np.asarray(motion135_to_272(m135), dtype=np.float32)
                    np.savez_compressed(dst, motion_272=m272)
                    written += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    if failed <= 10:
                        print(f"[fail] c{cond} {method} {sid}: {type(exc).__name__}: {exc}", flush=True)
            _write_run_metadata(
                dst_dir,
                {
                    "task": "tp2m",
                    "test_dataset": _dataset(cond),
                    "representation": "ms272",
                    "method": method,
                    "condition_frames": cond,
                    "expected_count": len(official),
                    "source_motion135": str(src_dir.relative_to(ROOT)),
                    "motion135_to_ms272": "hftrainer.motion.representation.motion272::motion135_to_272",
                    "annotation": str(ANNO.relative_to(ROOT)),
                },
            )
            total_written += written
            total_skipped += skipped
            total_missing += missing
            total_failed += failed
            print(
                f"[convert-ms272] c{cond} {method} written={written} skipped={skipped} "
                f"missing_src={missing} failed={failed} dst_count={_count(dst_dir)}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "written": total_written,
                "skipped": total_skipped,
                "missing_src": total_missing,
                "failed": total_failed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if total_failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.add_argument("--methods", default="all")
    p.add_argument("--conds", default="1 5 9")
    p.add_argument("--out-dir", default="outputs/evaluation/tp2m/_runs/leaderboard_missing_20260629/missing")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("convert-ms272")
    p.add_argument("--methods", default="motionstreamer flowmdm motionlab kimodo")
    p.add_argument("--conds", default="1 5 9")
    p.add_argument("--ids", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.set_defaults(func=cmd_convert_ms272)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert reconstruction motion135 directories to MotionStreamer ms272 NPZs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

from motionstreamer_272_encoder import motion135_to_272  # noqa: E402


DEFAULT_BASE = ROOT / "outputs/evaluation/reconstruction/humanml3d_official_test"
DEFAULT_SPLIT = DEFAULT_BASE / "_meta/test_ids.txt"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def alt_ids(sid: str) -> list[str]:
    out = [sid]
    if sid.startswith("M") and sid[1:].isdigit():
        out.append(sid[1:])
    elif sid and sid[0].isdigit():
        out.append("M" + sid)
    return out


def find_motion135(src_dir: Path, sid: str) -> Path | None:
    for aid in alt_ids(sid):
        path = src_dir / f"{aid}.npz"
        if path.exists():
            return path
    return None


def load_motion135(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        arr = np.asarray(data["motion_135"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 135:
        raise ValueError(f"expected motion_135 shape (T,>=135), got {arr.shape}")
    return arr[:, :135]


def convert_method(base: Path, method: str, ids: list[str], force: bool) -> dict[str, object]:
    src_dir = base / "motion135" / method
    dst_dir = base / "ms272" / method
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    for sid in ids:
        dst = dst_dir / f"{sid}.npz"
        if dst.exists() and not force:
            skipped_existing += 1
            continue
        src = find_motion135(src_dir, sid)
        if src is None:
            missing.append(sid)
            continue
        try:
            m272 = np.asarray(motion135_to_272(load_motion135(src)), dtype=np.float32)
            np.savez(dst, motion_272=m272)
            written += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": sid, "source": str(src), "error": f"{type(exc).__name__}: {exc}"})
            if len(errors) <= 10:
                print(f"[fail] {method}/{sid}: {type(exc).__name__}: {exc}", flush=True)
    return {
        "method": method,
        "source": str(src_dir),
        "target": str(dst_dir),
        "count": len(list(dst_dir.glob("*.npz"))),
        "written": written,
        "skipped_existing": skipped_existing,
        "missing": len(missing),
        "errors": len(errors),
        "missing_examples": missing[:10],
        "error_examples": errors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--methods", default="t2mgpt,momask,mld,mogents,motiongpt3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    split = Path(args.split).expanduser().resolve()
    ids = read_ids(split)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = [convert_method(base, method, ids, args.force) for method in methods]
    payload = {"base": str(base), "split": str(split), "ids": len(ids), "methods": rows}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.summary_json:
        out = Path(args.summary_json).expanduser().resolve()
    else:
        out = base / "ms272/materialize_from_motion135_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, flush=True)
    return 1 if any(row["missing"] or row["errors"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

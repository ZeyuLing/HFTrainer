#!/usr/bin/env python3
"""Restore exact root translation in motion_135 files from HML263 sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fit_length_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == target_len:
        return x
    if len(x) < 2:
        return np.repeat(x[:1], target_len, axis=0).astype(np.float32)
    grid = np.linspace(0.0, len(x) - 1, target_len)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (grid - lo).astype(np.float32)
    return (x[lo] * (1.0 - w[:, None]) + x[hi] * w[:, None]).astype(np.float32)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez(tmp, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def restore_one(src: Path, dst: Path, meta_path: Path) -> dict:
    data = np.load(str(src), allow_pickle=True)
    meta = np.load(str(meta_path), allow_pickle=True)
    if "motion_135" not in data.files:
        raise KeyError(f"{src} missing motion_135")
    if "source_motion135_transl" not in meta.files:
        raise KeyError(f"{meta_path} missing source_motion135_transl")
    arrays = {key: np.asarray(data[key]) for key in data.files}
    motion = np.asarray(arrays["motion_135"], dtype=np.float32).copy()
    old_transl = motion[:, :3].copy()
    restored = _fit_length_linear(np.asarray(meta["source_motion135_transl"], dtype=np.float32), len(motion))
    motion[:, :3] = restored
    arrays["motion_135"] = motion
    arrays["transl"] = restored
    arrays.setdefault("canonical_transl", old_transl)
    arrays["root_translation_restore_mode"] = np.array("source_transl")
    arrays["root_translation_restored"] = np.array(True)
    arrays["root_translation_source_frames"] = np.array(len(meta["source_motion135_transl"]), dtype=np.int32)
    _save_npz(dst, arrays)
    source = np.asarray(meta["source_motion135_transl"], dtype=np.float32)
    if len(source) == len(restored):
        err = np.linalg.norm(restored - source, axis=-1)
        mean_err = float(err.mean() * 1000.0)
        max_err = float(err.max() * 1000.0)
    else:
        mean_err = None
        max_err = None
    return {
        "id": src.stem,
        "frames": int(len(motion)),
        "mean_internal_restore_error_mm": mean_err,
        "max_internal_restore_error_mm": max_err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    meta_dir = Path(args.meta_dir)
    files = sorted(p for p in in_dir.glob("*.npz") if not p.name.startswith("_"))
    if args.limit:
        files = files[: args.limit]
    rows = []
    skipped = missing = failed = 0
    failures = []
    for i, src in enumerate(files, 1):
        dst = out_dir / src.name
        meta_path = meta_dir / src.name
        if args.skip_existing and dst.exists():
            skipped += 1
            continue
        if not meta_path.exists():
            missing += 1
            continue
        try:
            rows.append(restore_one(src, dst, meta_path))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"id": src.stem, "error": f"{type(exc).__name__}: {exc}"})
            if failed <= 10:
                print(f"[fail] {src.stem}: {type(exc).__name__}: {exc}", flush=True)
        if i % 500 == 0 or i == len(files):
            print(f"[progress] {i}/{len(files)} ok={len(rows)} skipped={skipped} missing={missing} failed={failed}", flush=True)

    summary = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "meta_dir": str(meta_dir),
        "inputs": len(files),
        "ok": len(rows),
        "skipped": skipped,
        "missing_metadata": missing,
        "failed": failed,
        "mean_internal_restore_error_mm": (
            float(np.mean([r["mean_internal_restore_error_mm"] for r in rows if r["mean_internal_restore_error_mm"] is not None]))
            if any(r["mean_internal_restore_error_mm"] is not None for r in rows) else None
        ),
        "max_internal_restore_error_mm": (
            float(max([r["max_internal_restore_error_mm"] for r in rows if r["max_internal_restore_error_mm"] is not None]))
            if any(r["max_internal_restore_error_mm"] is not None for r in rows) else None
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_root_restore_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_root_restore_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if failures:
        (out_dir / "_root_restore_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary), flush=True)
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

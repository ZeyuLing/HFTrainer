#!/usr/bin/env python3
"""Retarget KIMODO SOMA debug NPZs to SMPL ``motion_135``.

This is the preferred KIMODO mesh path. New debug NPZs should contain
``global_rot_mats``; those are transferred to SMPL rotations directly. The
position-only IK fallback is deliberately opt-in because it cannot reliably
recover upper-body twist for mesh visualization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.motion.retarget import KIMODOSOMAToSMPLRetargeter  # noqa: E402


def _iter_files(in_dir: Path, ids: Path | None, limit: int | None) -> Iterable[Path]:
    files = sorted(in_dir.glob("*.npz")) or sorted(in_dir.glob("*.npy"))
    if ids is not None:
        wanted = [line.strip() for line in ids.read_text().splitlines() if line.strip()]
        suffix = files[0].suffix if files else ".npz"
        files = [in_dir / f"{sid}{suffix}" for sid in wanted]
    files = [p for p in files if p.exists()]
    return files[:limit] if limit else files


def _read_caption(data) -> str:
    if "caption" not in data.files:
        return ""
    try:
        return str(np.asarray(data["caption"]).item())
    except Exception:
        return str(data["caption"])


def _metadata(path: Path) -> dict[str, object]:
    if path.suffix != ".npz":
        return {"caption": "", "has_global_rot_mats": False}
    with np.load(path, allow_pickle=True) as data:
        return {
            "caption": _read_caption(data),
            "has_global_rot_mats": "global_rot_mats" in data.files,
            "has_local_rot_mats": "local_rot_mats" in data.files,
            "has_root_positions": "root_positions" in data.files,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    parser.add_argument("--allow-position-fallback", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-floor-align", action="store_true")
    parser.add_argument("--no-foot-height-align", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ids = Path(args.ids) if args.ids else None
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(_iter_files(in_dir, ids, args.limit))
    retargeter = KIMODOSOMAToSMPLRetargeter(
        model_dir=args.model_dir,
        device=args.device,
        floor_align=not args.no_floor_align,
        foot_height_align=not args.no_foot_height_align,
    )

    rows = []
    ok = skipped = failed = degraded = 0
    for idx, path in enumerate(files, 1):
        dst = out_dir / f"{path.stem}.npz"
        if args.skip_existing and dst.exists():
            skipped += 1
            continue
        try:
            meta = _metadata(path)
            if not meta["has_global_rot_mats"] and not args.allow_position_fallback:
                raise KeyError(
                    f"{path} has no global_rot_mats; rerun KIMODO generation with the "
                    "updated debug exporter or pass --allow-position-fallback explicitly"
                )
            result = retargeter.retarget_file(path)
            method = str(np.asarray(result.get("retarget_method", "")).item())
            if method == "soma_position_ik_fallback":
                degraded += 1
            np.savez_compressed(
                dst,
                **result,
                caption=np.array(meta["caption"], dtype=object),
                source_id=np.array(path.stem, dtype=object),
                source_skeleton_path=np.array(str(path), dtype=object),
                source_fps=np.array(30.0, dtype=np.float32),
                target_fps=np.array(30.0, dtype=np.float32),
                has_global_rot_mats=np.array(bool(meta["has_global_rot_mats"]), dtype=np.bool_),
                has_local_rot_mats=np.array(bool(meta["has_local_rot_mats"]), dtype=np.bool_),
                has_root_positions=np.array(bool(meta["has_root_positions"]), dtype=np.bool_),
            )
            rows.append({
                "sid": path.stem,
                "frames": int(result["motion_135"].shape[0]),
                "method": method,
                "has_global_rot_mats": bool(meta["has_global_rot_mats"]),
                "mpjpe_mm_mean": (
                    float(np.asarray(result["fit_mpjpe_mm"]).mean())
                    if "fit_mpjpe_mm" in result
                    else None
                ),
            })
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 10:
                print(f"  [fail] {path.name}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 100 == 0 or idx == len(files):
            print(
                f"  {idx}/{len(files)} ok={ok} skipped={skipped} "
                f"failed={failed} degraded={degraded}",
                flush=True,
            )

    summary = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "files": len(files),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "position_only_fallback": degraded,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else "")
    )
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

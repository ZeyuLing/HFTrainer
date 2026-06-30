#!/usr/bin/env python3
"""Convert MotionHub SMPL translation convention in-place or into a new root.

MotionHub SMPL/SMPL-X files should store the body-model translation parameter
when they are meant to align with HYMotion raw data. Some older repair passes
stored the pelvis/root joint world position instead. The two conventions differ
by the rest-pose root joint offset:

    root_world = model_trans + rest_root

This tool applies the reversible offset and records a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "motion_annot_web" / "m2m_database" / "static" / "assets"


def root_offset(model: str) -> np.ndarray:
    if model == "smplx":
        path = ASSET_ROOT / "dump_smplx" / "j_template.bin"
        joints = np.fromfile(path, dtype=np.float32).reshape(55, 3)
    elif model == "smplh":
        path = ASSET_ROOT / "dump_smplh" / "j_template.bin"
        joints = np.fromfile(path, dtype=np.float32).reshape(52, 3)
    else:
        raise ValueError(f"unsupported model: {model}")
    return joints[0].astype(np.float32)


def conversion_delta(src: str, dst: str, rest_root: np.ndarray) -> np.ndarray:
    if src == dst:
        return np.zeros(3, dtype=np.float32)
    if src == "root_world" and dst == "model_trans":
        return -rest_root
    if src == "model_trans" and dst == "root_world":
        return rest_root
    raise ValueError(f"unsupported conversion: {src} -> {dst}")


def iter_motion_files(subset_root: Path, motion_dir: str) -> Iterable[Path]:
    yield from sorted((subset_root / motion_dir).glob("*.npz"))


def copy_static_subset_files(src_subset: Path, dst_subset: Path, motion_dir: str) -> None:
    for path in src_subset.iterdir():
        if path.name == motion_dir:
            continue
        dst = dst_subset / path.name
        if path.is_dir():
            if dst.exists():
                continue
            shutil.copytree(path, dst, symlinks=True)
        elif path.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def convert_file(src: Path, dst: Path, delta: np.ndarray, write: bool) -> Dict[str, Any]:
    data = np.load(src, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    keys = [key for key in ("transl", "trans") if key in arrays]
    if not keys and "motion_135" not in arrays:
        raise KeyError(f"{src} has no transl/trans/motion_135")

    before = None
    after = None
    out = dict(arrays)
    for key in keys:
        arr = np.asarray(out[key], dtype=np.float32).copy()
        if before is None:
            before = arr[:, :3].mean(axis=0)
        arr[:, :3] += delta[None, :]
        if after is None:
            after = arr[:, :3].mean(axis=0)
        out[key] = arr

    if "motion_135" in out:
        motion = np.asarray(out["motion_135"], dtype=np.float32).copy()
        if before is None:
            before = motion[:, :3].mean(axis=0)
        motion[:, :3] += delta[None, :]
        if after is None:
            after = motion[:, :3].mean(axis=0)
        out["motion_135"] = motion

    if write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **out)

    return {
        "source": str(src),
        "target": str(dst),
        "frames": int(np.asarray(next(iter(out.values()))).shape[0]) if out else 0,
        "mean_before": before.tolist() if before is not None else None,
        "mean_after": after.tolist() if after is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--motion-dir", default="smplx_55")
    parser.add_argument("--output-root", default="", help="Output subset root. Omit for in-place conversion.")
    parser.add_argument("--model", choices=("smplx", "smplh"), default="smplx")
    parser.add_argument("--from-convention", choices=("model_trans", "root_world"), required=True)
    parser.add_argument("--to-convention", choices=("model_trans", "root_world"), required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    subset_root = Path(args.subset_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else subset_root
    rest_root = root_offset(args.model)
    delta = conversion_delta(args.from_convention, args.to_convention, rest_root)
    files = list(iter_motion_files(subset_root, args.motion_dir))

    if args.write and output_root != subset_root:
        copy_static_subset_files(subset_root, output_root, args.motion_dir)

    rows: List[Dict[str, Any]] = []
    for idx, src in enumerate(files, start=1):
        dst = output_root / args.motion_dir / src.name
        rows.append(convert_file(src, dst, delta, args.write))
        if idx % 500 == 0:
            print(f"[convert] {idx}/{len(files)}", flush=True)

    means_before = np.array([r["mean_before"] for r in rows if r["mean_before"] is not None], dtype=np.float64)
    means_after = np.array([r["mean_after"] for r in rows if r["mean_after"] is not None], dtype=np.float64)
    summary: Dict[str, Any] = {
        "subset_root": str(subset_root),
        "output_root": str(output_root),
        "motion_dir": args.motion_dir,
        "model": args.model,
        "from_convention": args.from_convention,
        "to_convention": args.to_convention,
        "rest_root": rest_root.tolist(),
        "delta": delta.tolist(),
        "write": bool(args.write),
        "num_files": len(rows),
        "mean_before": means_before.mean(axis=0).tolist() if means_before.size else None,
        "mean_after": means_after.mean(axis=0).tolist() if means_after.size else None,
    }
    payload = {"summary": summary, "files": rows}
    print(json.dumps(summary, indent=2), flush=True)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

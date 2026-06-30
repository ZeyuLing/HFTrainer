#!/usr/bin/env python3
"""Floor-align MotionHub SMPL-X files without changing heading or XZ motion."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_J_TEMPLATE = (
    PROJECT_ROOT
    / "motion_annot_web"
    / "m2m_database"
    / "static"
    / "assets"
    / "dump_smplx"
    / "j_template.bin"
)

SMPLX_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_rows(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    raw = read_json(path)
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            yield str(row.get("id", idx) if isinstance(row, dict) else idx), row
    else:
        raise TypeError(f"unsupported annotation type in {path}: {type(data).__name__}")


def collect_motion_paths(subset_root: Path, splits: List[str], exclude_invalid: bool) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    data_root = subset_root.parent
    for split in splits:
        split_path = subset_root / f"{split}.json"
        for _, row in iter_rows(split_path):
            if exclude_invalid and row.get("invalid"):
                continue
            rel = row.get("smplx_path") or row.get("motion_path")
            values = rel if isinstance(rel, list) else [rel]
            for value in values:
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = data_root / path
                path = path.resolve()
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    return paths


def aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = aa / np.where(small, 1.0, theta)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(theta)[..., 0]
    s = np.sin(theta)[..., 0]
    C = 1.0 - c
    R = np.empty(aa.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    R[small[..., 0]] = np.eye(3, dtype=np.float32)
    return R


def fk_min_y(poses: np.ndarray, transl: np.ndarray, j_template: np.ndarray) -> float:
    T = poses.shape[0]
    offsets = np.zeros((55, 3), dtype=np.float32)
    for i in range(1, 55):
        offsets[i] = j_template[i] - j_template[SMPLX_PARENTS[i]]
    local = aa_to_matrix(poses.reshape(T, 55, 3))
    grot = np.zeros((T, 55, 3, 3), dtype=np.float32)
    gpos = np.zeros((T, 55, 3), dtype=np.float32)
    grot[:, 0] = local[:, 0]
    gpos[:, 0] = transl
    for i in range(1, 55):
        parent = SMPLX_PARENTS[i]
        grot[:, i] = grot[:, parent] @ local[:, i]
        gpos[:, i] = gpos[:, parent] + np.einsum("tij,j->ti", grot[:, parent], offsets[i])
    return float(gpos[:, :, 1].min())


def load_j_template(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(55, 3)


def canonicalize_file(src: Path, dst: Path, j_template: np.ndarray, write: bool) -> Dict[str, Any]:
    data = np.load(src, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    poses = np.asarray(arrays["poses"], dtype=np.float32)
    transl_key = "transl" if "transl" in arrays else "trans"
    transl = np.asarray(arrays[transl_key], dtype=np.float32)
    min_y = fk_min_y(poses, transl, j_template)

    if write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out = dict(arrays)
        for key in ("transl", "trans"):
            if key in out:
                arr = np.asarray(out[key], dtype=np.float32).copy()
                arr[:, 1] -= min_y
                out[key] = arr
        if "motion_135" in out:
            motion = np.asarray(out["motion_135"], dtype=np.float32).copy()
            motion[:, 1] -= min_y
            out["motion_135"] = motion
        np.savez_compressed(dst, **out)

    return {
        "source": str(src),
        "target": str(dst),
        "frames": int(poses.shape[0]),
        "shift_y": min_y,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", required=True, help="Subset root, e.g. data/motionhub/finedance")
    parser.add_argument("--output-root", required=True, help="Output MotionHub data root or subset root")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--exclude-invalid", action="store_true")
    parser.add_argument("--j-template", default=str(DEFAULT_J_TEMPLATE))
    parser.add_argument("--report", default="")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    subset_root = Path(args.subset_root).resolve()
    output_root = Path(args.output_root).resolve()
    data_root = subset_root.parent
    subset_name = subset_root.name
    if output_root.name == subset_name:
        output_subset_root = output_root
        output_data_root = output_root.parent
    else:
        output_data_root = output_root
        output_subset_root = output_data_root / subset_name

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    j_template = load_j_template(Path(args.j_template))
    motion_paths = collect_motion_paths(subset_root, splits, args.exclude_invalid)

    rows = []
    for idx, src in enumerate(motion_paths, start=1):
        rel = src.relative_to(data_root.resolve())
        dst = output_data_root / rel
        rows.append(canonicalize_file(src, dst, j_template, args.write))
        if idx % 500 == 0:
            print(f"[canonicalize] {idx}/{len(motion_paths)}", flush=True)

    if args.write:
        output_subset_root.mkdir(parents=True, exist_ok=True)
        for split in splits:
            shutil.copy2(subset_root / f"{split}.json", output_subset_root / f"{split}.json")

    shifts = np.array([r["shift_y"] for r in rows], dtype=np.float64)
    summary = {
        "subset_root": str(subset_root),
        "output_root": str(output_root),
        "splits": splits,
        "exclude_invalid": args.exclude_invalid,
        "num_files": len(rows),
        "shift_y_mean": float(shifts.mean()) if shifts.size else None,
        "shift_y_std": float(shifts.std()) if shifts.size else None,
        "shift_y_min": float(shifts.min()) if shifts.size else None,
        "shift_y_max": float(shifts.max()) if shifts.size else None,
    }
    payload = {"summary": summary, "files": rows}
    print(json.dumps(summary, indent=2), flush=True)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

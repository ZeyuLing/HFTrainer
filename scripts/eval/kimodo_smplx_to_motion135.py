#!/usr/bin/env python3
"""Convert KIMODO-SMPLX native debug NPZs to SMPL ``motion_135``.

KIMODO-SMPLX-RP emits a 22-joint SMPL-X body skeleton with local/global
rotation matrices and root positions. For the hftrainer evaluators we convert
those local rotations to the row-major ``motion_135`` convention:

    transl(3) + 22 local rotations as row-major 6D (132)

The output NPZ also carries SMPL-X-style axis-angle fields so downstream tools
that expect ``transl/global_orient/body_pose`` can consume the same directory.
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

# Some KIMODO debug NPZs were written with NumPy 2.x object arrays, whose pickle
# payloads reference ``numpy._core``. NumPy 1.x environments expose the same
# modules under ``numpy.core`` instead, so install aliases before ``np.load``.
try:  # pragma: no cover - exercised only on NumPy 1.x reading NumPy 2.x pickles
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    import numpy.core.numeric as _np_numeric
    import numpy.core.umath as _np_umath

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
    sys.modules.setdefault("numpy._core.umath", _np_umath)
except Exception:
    pass

SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
_COL_TO_ROW = [0, 3, 1, 4, 2, 5]


def _rotmat_to_rot6d_row_major(rotmat: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat, dtype=np.float32)
    col6d = np.concatenate([rotmat[..., 0:3, 0], rotmat[..., 0:3, 1]], axis=-1)
    return col6d[..., _COL_TO_ROW]


def _matrix_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat, dtype=np.float32)
    flat = rotmat.reshape(-1, 3, 3)
    trace = np.trace(flat, axis1=1, axis2=2)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos)
    vec = np.stack(
        [
            flat[:, 2, 1] - flat[:, 1, 2],
            flat[:, 0, 2] - flat[:, 2, 0],
            flat[:, 1, 0] - flat[:, 0, 1],
        ],
        axis=1,
    )
    sin = np.sin(angle)
    scale = np.zeros_like(angle, dtype=np.float32)
    mask = np.abs(sin) > 1e-6
    scale[mask] = (angle[mask] / (2.0 * sin[mask])).astype(np.float32)
    out = vec.astype(np.float32) * scale[:, None]
    near_zero = angle < 1e-6
    out[near_zero] = 0.0
    return out.reshape(rotmat.shape[:-2] + (3,))


def _iter_files(in_dir: Path, ids: Path | None, limit: int | None) -> Iterable[Path]:
    files = sorted(p for p in in_dir.glob("*.npz") if not p.name.startswith("_"))
    if ids is not None:
        wanted = [line.strip() for line in ids.read_text().splitlines() if line.strip()]
        files = [in_dir / f"{sid}.npz" for sid in wanted]
    files = [p for p in files if p.exists()]
    return files[:limit] if limit else files


def _read_object(data: np.lib.npyio.NpzFile, key: str, default=""):
    if key not in data.files:
        return default
    try:
        return np.asarray(data[key]).item()
    except Exception:
        return data[key]


def _local_from_global(global_rot: np.ndarray) -> np.ndarray:
    global_rot = np.asarray(global_rot, dtype=np.float32)
    local = np.empty_like(global_rot)
    for j, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            local[:, j] = global_rot[:, j]
        else:
            local[:, j] = np.matmul(
                np.swapaxes(global_rot[:, parent], -1, -2),
                global_rot[:, j],
            )
    return local


def _load_local_rots(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "local_rot_mats" in data.files:
        local = np.asarray(data["local_rot_mats"], dtype=np.float32)
    elif "global_rot_mats" in data.files:
        local = _local_from_global(np.asarray(data["global_rot_mats"], dtype=np.float32))
    else:
        raise KeyError(f"{path} has neither local_rot_mats nor global_rot_mats")
    if local.ndim != 4 or local.shape[1:] != (22, 3, 3):
        raise ValueError(f"{path} expected SMPLX22 local rotations, got {local.shape}")
    return local


def _load_root(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "root_positions" in data.files:
        root = np.asarray(data["root_positions"], dtype=np.float32)
    elif "posed_joints" in data.files:
        posed = np.asarray(data["posed_joints"], dtype=np.float32)
        if posed.ndim != 3 or posed.shape[1] < 1:
            raise ValueError(f"{path} has bad posed_joints shape {posed.shape}")
        root = posed[:, 0]
    else:
        raise KeyError(f"{path} has neither root_positions nor posed_joints")
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"{path} expected root_positions (T,3), got {root.shape}")
    return root


def convert_one(path: Path) -> dict[str, np.ndarray | object]:
    with np.load(path, allow_pickle=True) as data:
        local = _load_local_rots(data, path)
        root = _load_root(data, path)
        t = min(len(root), len(local))
        if t <= 0:
            raise ValueError(f"{path} is empty")
        root = root[:t]
        local = local[:t]
        if not np.isfinite(root).all() or not np.isfinite(local).all():
            raise ValueError(f"{path} contains non-finite values")

        rot6d = _rotmat_to_rot6d_row_major(local).reshape(t, 132)
        motion_135 = np.concatenate([root.astype(np.float32), rot6d.astype(np.float32)], axis=1)
        aa_np = _matrix_to_axis_angle(local.reshape(-1, 3, 3)).reshape(t, 22, 3).astype(np.float32)

        poses = np.zeros((t, 55, 3), dtype=np.float32)
        poses[:, :22] = aa_np
        return {
            "motion_135": motion_135.astype(np.float32),
            "transl": root.astype(np.float32),
            "trans": root.astype(np.float32),
            "global_orient": aa_np[:, 0].astype(np.float32),
            "body_pose": aa_np[:, 1:22].reshape(t, 63).astype(np.float32),
            "poses": poses.reshape(t, 165).astype(np.float32),
            "mocap_frame_rate": np.array(30.0, dtype=np.float32),
            "model": np.array("smplx2020", dtype=object),
            "gender": np.array("neutral", dtype=object),
            "source_id": np.array(path.stem, dtype=object),
            "source_skeleton_path": np.array(str(path), dtype=object),
            "retarget_method": np.array("kimodo_smplx22_local_rot_transfer", dtype=object),
            "caption": np.array(_read_object(data, "caption", ""), dtype=object),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ids = Path(args.ids) if args.ids else None
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(_iter_files(in_dir, ids, args.limit))

    rows = []
    ok = skipped = failed = 0
    for idx, path in enumerate(files, 1):
        dst = out_dir / f"{path.stem}.npz"
        if args.skip_existing and dst.exists():
            skipped += 1
            continue
        try:
            payload = convert_one(path)
            np.savez_compressed(dst, **payload)
            rows.append({
                "sid": path.stem,
                "frames": int(payload["motion_135"].shape[0]),
                "path": str(dst),
            })
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 10:
                print(f"  [fail] {path.name}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 100 == 0 or idx == len(files):
            print(f"  {idx}/{len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)

    summary = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "files": len(files),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Floor-align MotionHub SMPL-X files with exact full-mesh LBS.

The processed files keep the HYMotion-compatible body-model translation
convention:

    vertices_world = smpl_vertices_without_translation + transl

Only the y component of ``transl``/``trans`` and compatible packed translation
fields is shifted.  No yaw, x/z centering, or front-end canonicalization is
performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSATILE_ROOT = PROJECT_ROOT.parent / "versatilemotion"
if str(VERSATILE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSATILE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def as_repeated(arr: Any, frames: int, width: int) -> np.ndarray:
    if arr is None:
        return np.zeros((frames, width), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    if out.ndim == 1:
        out = out[None, :]
    if out.shape[0] == 1 and frames > 1:
        out = np.repeat(out, frames, axis=0)
    if out.shape[0] != frames:
        if out.shape[0] > frames:
            out = out[:frames]
        elif out.shape[0] > 0:
            out = np.concatenate([out, np.repeat(out[-1:], frames - out.shape[0], axis=0)], axis=0)
    if out.shape[1] < width:
        out = np.concatenate(
            [out, np.zeros((out.shape[0], width - out.shape[1]), dtype=np.float32)],
            axis=1,
        )
    return out[:, :width].astype(np.float32, copy=False)


def motion_files(subset_root: Path, motion_dir: str) -> List[Path]:
    files = sorted((subset_root / motion_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz files found under {subset_root / motion_dir}")
    return files


def load_arrays(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def smplx_components(arrays: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if "transl" in arrays:
        transl = np.asarray(arrays["transl"], dtype=np.float32)
    elif "trans" in arrays:
        transl = np.asarray(arrays["trans"], dtype=np.float32)
    else:
        raise KeyError("missing transl/trans")
    if transl.ndim != 2 or transl.shape[1] < 3:
        raise ValueError(f"bad transl shape: {transl.shape}")
    transl = transl[:, :3]
    frames = transl.shape[0]

    if "poses" in arrays:
        poses = np.asarray(arrays["poses"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 165:
            raise ValueError(f"expected SMPL-X poses shape (*,165), got {poses.shape}")
        poses = poses[:frames, :165]
        global_orient = poses[:, 0:3]
        body_pose = poses[:, 3:66]
    else:
        global_orient = as_repeated(arrays.get("global_orient"), frames, 3)
        body_pose = as_repeated(arrays.get("body_pose"), frames, 63)

    return {
        "frames": np.asarray([frames], dtype=np.int64),
        "transl": transl,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "betas": as_repeated(arrays.get("betas"), frames, 10),
    }


def mesh_min_y(
    model: torch.nn.Module,
    comp: Dict[str, np.ndarray],
    device: torch.device,
    chunk: int,
) -> float:
    mins: List[float] = []
    frames = int(comp["frames"][0])
    for start in range(0, frames, chunk):
        sl = slice(start, min(start + chunk, frames))
        with torch.no_grad():
            verts = model(
                body_pose=torch.from_numpy(comp["body_pose"][sl]).to(device=device, dtype=torch.float32),
                betas=torch.from_numpy(comp["betas"][sl]).to(device=device, dtype=torch.float32),
                global_orient=torch.from_numpy(comp["global_orient"][sl]).to(device=device, dtype=torch.float32),
                transl=torch.from_numpy(comp["transl"][sl]).to(device=device, dtype=torch.float32),
            )
        mins.append(float(verts[..., 1].min().item()))
    return float(min(mins))


def shift_arrays(
    arrays: Dict[str, Any],
    shift_y: float,
    version: str,
    mesh_min_before: float,
    mesh_min_after: float,
) -> Dict[str, Any]:
    out = dict(arrays)
    for key in ("transl", "trans"):
        if key in out:
            arr = np.asarray(out[key], dtype=np.float32).copy()
            arr[:, 1] += np.float32(shift_y)
            out[key] = arr
    if "motion_135" in out:
        motion = np.asarray(out["motion_135"], dtype=np.float32).copy()
        motion[:, 1] += np.float32(shift_y)
        out["motion_135"] = motion
    out["canonicalize_version"] = np.asarray(version)
    out["canonicalize_mode"] = np.asarray("mesh_floor_model_trans")
    out["canonicalize_mesh_min_y_before"] = np.asarray(mesh_min_before, dtype=np.float32)
    out["canonicalize_mesh_min_y_after"] = np.asarray(mesh_min_after, dtype=np.float32)
    out["canonicalize_mesh_shift_y"] = np.asarray(shift_y, dtype=np.float32)
    return out


def process_file(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    chunk: int,
    write: bool,
    version: str,
    tolerance: float,
) -> Dict[str, Any]:
    arrays = load_arrays(path)
    comp = smplx_components(arrays)
    before = mesh_min_y(model, comp, device, chunk)
    shift_y = -before
    after = before + shift_y
    changed = abs(shift_y) > tolerance
    if write:
        out = shift_arrays(arrays, shift_y, version, before, after)
        np.savez_compressed(path, **out)
    return {
        "path": str(path),
        "frames": int(comp["frames"][0]),
        "mesh_min_y_before": before,
        "shift_y": shift_y,
        "mesh_min_y_after_expected": after,
        "changed_over_tolerance": bool(changed),
    }


def summarize(rows: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    before = np.asarray([row["mesh_min_y_before"] for row in rows], dtype=np.float64)
    shifts = np.asarray([row["shift_y"] for row in rows], dtype=np.float64)
    frames = np.asarray([row["frames"] for row in rows], dtype=np.int64)
    return {
        "num_files": len(rows),
        "total_frames": int(frames.sum()),
        "tolerance": tolerance,
        "changed_over_tolerance": int(np.sum(np.abs(shifts) > tolerance)),
        "mesh_min_y_before_min": float(before.min()),
        "mesh_min_y_before_max": float(before.max()),
        "mesh_min_y_before_mean": float(before.mean()),
        "shift_y_min": float(shifts.min()),
        "shift_y_max": float(shifts.max()),
        "shift_y_mean": float(shifts.mean()),
        "shift_y_abs_max": float(np.max(np.abs(shifts))),
    }


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--motion-dir", default="smplx_55")
    parser.add_argument("--report", required=True)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--max-files", type=int, default=0, help="Debug limit; 0 means all files.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    from mmotion.models.body_models.smplx_lite import SmplxLite

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = SmplxLite(
        model_path=str(VERSATILE_ROOT / "checkpoints/smpl_models/smplx"),
        gender="neutral",
        num_betas=10,
    ).to(device=device, dtype=torch.float32).eval()

    subset_root = Path(args.subset_root)
    files = motion_files(subset_root, args.motion_dir)
    if args.max_files > 0:
        files = files[: args.max_files]
    version = "motionhub_mesh_floor_20260629"
    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(files, start=1):
        rows.append(
            process_file(
                path,
                model,
                device,
                max(1, args.chunk),
                args.write,
                version,
                args.tolerance,
            )
        )
        if idx % 100 == 0 or idx == len(files):
            print(f"[mesh-floor] {idx}/{len(files)} {subset_root.name}", flush=True)

    payload = {
        "meta": {
            "subset_root": str(subset_root),
            "motion_dir": args.motion_dir,
            "write": bool(args.write),
            "device": str(device),
            "chunk": max(1, args.chunk),
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "translation_convention": "body_model_transl",
        },
        "summary": summarize(rows, args.tolerance),
        "files": rows,
    }
    write_json(Path(args.report), payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"[report] {args.report}", flush=True)


if __name__ == "__main__":
    main()

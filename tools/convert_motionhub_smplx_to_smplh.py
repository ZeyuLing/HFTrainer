#!/usr/bin/env python3
"""Convert MotionHub SMPL-X npz files to SMPL-H npz files.

This is intended for representation-level audits where MotionHub subsets need
to be compared against HYMotion raw SMPL-H data.  The conversion keeps the
shared body and hand axis-angle pose channels:

    SMPL-X: [global, body, jaw, leye, reye, left_hand, right_hand]
    SMPL-H: [global, body, left_hand, right_hand]

Translation is optionally floor-aligned in the target SMPL-H body model using
the same lightweight J24 regressor used by HYMotion quality tools.  Existing
SMPL-X files are never modified; outputs are written to a separate motion dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSATILE_ROOT = PROJECT_ROOT.parent / "versatilemotion"
if str(VERSATILE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSATILE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VERSION = "motionhub_smplx_to_smplh_j24_floor_20260629"


def motion_files(
    subset_root: Path,
    input_motion_dir: str,
    recursive: bool,
    exclude_dir_names: set[str],
) -> List[Path]:
    if recursive:
        files = sorted(
            path
            for path in subset_root.rglob("*.npz")
            if input_motion_dir in path.relative_to(subset_root).parts
            and not (set(path.relative_to(subset_root).parts) & exclude_dir_names)
        )
    else:
        files = sorted((subset_root / input_motion_dir).glob("*.npz"))
    if not files:
        suffix = f"**/{input_motion_dir}" if recursive else input_motion_dir
        raise FileNotFoundError(f"no npz files found under {subset_root / suffix}")
    return files


def output_path_for(
    subset_root: Path,
    source_path: Path,
    input_motion_dir: str,
    output_motion_dir: str,
    recursive: bool,
) -> Path:
    if not recursive:
        return subset_root / output_motion_dir / source_path.name
    rel = source_path.relative_to(subset_root)
    parts = list(rel.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == input_motion_dir:
            parts[idx] = output_motion_dir
            return subset_root.joinpath(*parts)
    raise ValueError(f"{source_path} is not under a {input_motion_dir} directory")


def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def as_repeated(arr: Any, frames: int, width: int) -> np.ndarray:
    if arr is None:
        return np.zeros((frames, width), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
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


def components_from_smplx(arrays: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if "poses" in arrays:
        poses = np.asarray(arrays["poses"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] < 165:
            raise ValueError(f"expected SMPL-X poses shape (T,165), got {poses.shape}")
        frames = poses.shape[0]
        global_orient = poses[:, 0:3]
        body_pose = poses[:, 3:66]
        left_hand_pose = poses[:, 75:120]
        right_hand_pose = poses[:, 120:165]
    else:
        transl_key = "transl" if "transl" in arrays else "trans"
        frames = int(np.asarray(arrays[transl_key]).shape[0])
        global_orient = as_repeated(arrays.get("global_orient"), frames, 3)
        body_pose = as_repeated(arrays.get("body_pose"), frames, 63)
        left_hand_pose = as_repeated(arrays.get("left_hand_pose"), frames, 45)
        right_hand_pose = as_repeated(arrays.get("right_hand_pose"), frames, 45)

    if "transl" in arrays:
        transl = np.asarray(arrays["transl"], dtype=np.float32)[:frames, :3]
    elif "trans" in arrays:
        transl = np.asarray(arrays["trans"], dtype=np.float32)[:frames, :3]
    else:
        raise KeyError("missing transl/trans")

    betas = as_repeated(arrays.get("betas"), frames, 16)
    return {
        "frames": np.asarray([frames], dtype=np.int64),
        "transl": transl,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "betas": betas,
    }


def target_floor_min_y(
    model: torch.nn.Module,
    comp: Dict[str, np.ndarray],
    device: torch.device,
    chunk: int,
    mode: str,
) -> float:
    if mode == "none":
        return 0.0
    mins: List[float] = []
    frames = int(comp["frames"][0])
    for start in range(0, frames, chunk):
        sl = slice(start, min(start + chunk, frames))
        with torch.no_grad():
            values = model(
                body_pose=torch.from_numpy(comp["body_pose"][sl]).to(device=device, dtype=torch.float32),
                left_hand_pose=torch.from_numpy(comp["left_hand_pose"][sl]).to(device=device, dtype=torch.float32),
                right_hand_pose=torch.from_numpy(comp["right_hand_pose"][sl]).to(device=device, dtype=torch.float32),
                betas=torch.from_numpy(comp["betas"][sl]).to(device=device, dtype=torch.float32),
                global_orient=torch.from_numpy(comp["global_orient"][sl]).to(device=device, dtype=torch.float32),
                transl=torch.from_numpy(comp["transl"][sl]).to(device=device, dtype=torch.float32),
                rotation_mode="aa",
            )
        mins.append(float(values[..., 1].min().item()))
    return float(min(mins))


def convert_file(
    path: Path,
    out_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    chunk: int,
    floor_mode: str,
    write: bool,
) -> Dict[str, Any]:
    arrays = load_npz(path)
    comp = components_from_smplx(arrays)
    floor_before = target_floor_min_y(model, comp, device, chunk, floor_mode)
    shift_y = -floor_before if floor_mode != "none" else 0.0

    transl = comp["transl"].copy()
    transl[:, 1] += np.float32(shift_y)
    poses_h = np.concatenate(
        [
            comp["global_orient"],
            comp["body_pose"],
            comp["left_hand_pose"],
            comp["right_hand_pose"],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    betas = comp["betas"][0].astype(np.float32, copy=False)

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pack: Dict[str, Any] = {
            "poses": poses_h,
            "global_orient": comp["global_orient"].astype(np.float32, copy=False),
            "body_pose": comp["body_pose"].astype(np.float32, copy=False),
            "left_hand_pose": comp["left_hand_pose"].astype(np.float32, copy=False),
            "right_hand_pose": comp["right_hand_pose"].astype(np.float32, copy=False),
            "trans": transl,
            "transl": transl,
            "betas": betas,
            "gender": arrays.get("gender", np.asarray("neutral")),
            "mocap_framerate": arrays.get("mocap_framerate", np.asarray(30)),
            "num_frames": np.asarray(int(comp["frames"][0]), dtype=np.int64),
            "source_smpl_type": np.asarray("smplx"),
            "smpl_type": np.asarray("smplh"),
            "conversion_version": np.asarray(VERSION),
            "conversion_floor_mode": np.asarray(floor_mode),
            "conversion_floor_min_y_before": np.asarray(floor_before, dtype=np.float32),
            "conversion_y_shift": np.asarray(shift_y, dtype=np.float32),
            "source_path": np.asarray(str(path)),
        }
        np.savez_compressed(out_path, **pack)

    return {
        "source": str(path),
        "target": str(out_path),
        "frames": int(comp["frames"][0]),
        "transl_y_mean_before": float(comp["transl"][:, 1].mean()),
        "floor_min_y_before": float(floor_before),
        "shift_y": float(shift_y),
        "transl_y_mean_after": float(transl[:, 1].mean()),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = np.asarray([row["frames"] for row in rows], dtype=np.int64)
    out: Dict[str, Any] = {"num_files": len(rows), "total_frames": int(frames.sum())}
    for key in ("transl_y_mean_before", "floor_min_y_before", "shift_y", "transl_y_mean_after"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
        out[f"{key}_min"] = float(values.min())
        out[f"{key}_max"] = float(values.max())
    return out


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def has_current_conversion(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return str(data.get("conversion_version", "")) == VERSION
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--input-motion-dir", default="smplx_55")
    parser.add_argument("--output-motion-dir", default="smplh_52")
    parser.add_argument("--report", required=True)
    parser.add_argument("--floor-mode", choices=("j24", "none"), default="j24")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--recursive", action="store_true", help="Find **/{input_motion_dir}/*.npz recursively.")
    parser.add_argument(
        "--exclude-dir-name",
        action="append",
        default=["__MACOSX"],
        help="Directory name to ignore during recursive scans. May be passed multiple times.",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-current-existing",
        action="store_true",
        help="When writing, skip outputs that already have the current conversion_version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from mmotion.models.body_models.smplx_lite import SmplxLiteJ24

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = SmplxLiteJ24(
        model_path=str(VERSATILE_ROOT / "checkpoints/smpl_models/smplh"),
        gender="neutral",
        num_betas=16,
    ).to(device=device, dtype=torch.float32).eval()

    subset_root = Path(args.subset_root)
    files = motion_files(
        subset_root,
        args.input_motion_dir,
        recursive=args.recursive,
        exclude_dir_names=set(args.exclude_dir_name or []),
    )
    if args.max_files > 0:
        files = files[: args.max_files]

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    skipped_current = 0
    for idx, path in enumerate(files, start=1):
        out_path = output_path_for(
            subset_root,
            path,
            args.input_motion_dir,
            args.output_motion_dir,
            recursive=args.recursive,
        )
        if args.write and args.skip_current_existing and has_current_conversion(out_path):
            skipped_current += 1
            if idx % 250 == 0 or idx == len(files):
                print(
                    f"[smplx->smplh] {idx}/{len(files)} {subset_root.name} "
                    f"(skipped_current={skipped_current})",
                    flush=True,
                )
            continue
        if args.write and out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace")
        try:
            rows.append(
                convert_file(
                    path=path,
                    out_path=out_path,
                    model=model,
                    device=device,
                    chunk=max(1, args.chunk),
                    floor_mode=args.floor_mode,
                    write=args.write,
                )
            )
        except Exception as exc:
            errors.append({
                "source": str(path),
                "target": str(out_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
        if idx % 250 == 0 or idx == len(files):
            print(f"[smplx->smplh] {idx}/{len(files)} {subset_root.name}", flush=True)

    summary = summarize(rows)
    summary["num_errors"] = len(errors)
    summary["num_skipped_current"] = int(skipped_current)
    payload = {
        "meta": {
            "subset_root": str(subset_root),
            "input_motion_dir": args.input_motion_dir,
            "output_motion_dir": args.output_motion_dir,
            "floor_mode": args.floor_mode,
            "write": bool(args.write),
            "device": str(device),
            "chunk": max(1, args.chunk),
            "recursive": bool(args.recursive),
            "exclude_dir_names": sorted(set(args.exclude_dir_name or [])),
            "version": VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "errors": errors,
        "files": rows,
    }
    write_json(Path(args.report), payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

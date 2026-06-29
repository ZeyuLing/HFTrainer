#!/usr/bin/env python3
"""Convert ViMoGen/DART 276D outputs to SMPL-style motion135 files.

The default output is repository-canonical ``motion_135``:
``translation + 22 * row-major local rot6d``. Use
``--rotation-convention column`` only when creating legacy MotionCLIP evaluator
inputs directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
sys.path.insert(0, str(REPO))

from hftrainer.motion.representation.dart276 import dart276_to_motion135  # noqa: E402


def resample_motion(
    motion: np.ndarray,
    src_fps: float,
    dst_fps: float,
    *,
    target_frames: int = 0,
) -> np.ndarray:
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"fps must be positive, got src={src_fps}, dst={dst_fps}")
    if abs(src_fps - dst_fps) < 1e-6 or len(motion) < 2:
        return motion.astype(np.float32, copy=False)
    out_len = (
        int(target_frames)
        if target_frames
        else max(1, int(round(len(motion) * dst_fps / src_fps)))
    )
    src_t = np.linspace(0.0, 1.0, len(motion), dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    out = np.empty((out_len, motion.shape[1]), dtype=np.float32)
    for dim in range(motion.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, motion[:, dim])
    return out


def to_motionclip135(
    motion276: torch.Tensor,
    *,
    recover_from_velocity: bool = True,
    equal_length: bool = True,
    coord_conversion: str = "mbench",
    translation_source: str = "floor_aligned_smpl_transl",
    rotation_convention: str = "row",
) -> np.ndarray:
    return dart276_to_motion135(
        motion276.float(),
        recover_from_velocity=recover_from_velocity,
        equal_length=equal_length,
        coord_conversion=coord_conversion,
        translation_source=translation_source,
        rotation_convention=rotation_convention,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-recover-from-velocity", action="store_true")
    parser.add_argument("--no-equal-length", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--src-fps", type=float, default=20.0,
                        help="Frame rate of ViMoGen motion latents.")
    parser.add_argument("--dst-fps", type=float, default=30.0,
                        help="Frame rate to write for framework/evaluator motion135 input.")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Clamp resampled motion to this many frames; use 0 to disable.")
    parser.add_argument("--coord-conversion", choices=["mbench", "none"], default="mbench")
    parser.add_argument(
        "--translation-source",
        choices=[
            "floor_aligned_smpl_transl",
            "floor_aligned_joints_pelvis",
            "joints_pelvis",
            "smpl_transl",
        ],
        default="floor_aligned_smpl_transl",
        help="Root translation convention passed to dart276_to_motion135.",
    )
    parser.add_argument("--rotation-convention", choices=["row", "column"], default="row")
    parser.add_argument("--out-format", choices=["npz", "npy"], default="npz")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.glob(args.pattern))
    if args.max_files:
        files = files[:args.max_files]
    ok = failed = skipped = 0
    for idx, path in enumerate(files, 1):
        sample_id = path.parent.name if path.name == "motion_gen_condition_on_text.pt" else path.stem
        out_path = out_dir / f"{sample_id}.{args.out_format}"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            if path.suffix == ".npy":
                motion = torch.from_numpy(np.load(path).astype(np.float32))
            else:
                motion = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(motion, dict):
                motion = motion["motion"]
            if motion.ndim == 3:
                motion = motion[0]
            motion135 = to_motionclip135(
                motion,
                recover_from_velocity=not args.no_recover_from_velocity,
                equal_length=not args.no_equal_length,
                coord_conversion=args.coord_conversion,
                translation_source=args.translation_source,
                rotation_convention=args.rotation_convention,
            )
            max_frames = int(args.max_frames)
            target_frames = 0
            if abs(args.src_fps - args.dst_fps) >= 1e-6 and max_frames > 0:
                target_frames = min(
                    max_frames,
                    max(1, int(round(float(len(motion)) * args.dst_fps / args.src_fps))),
                )
            motion135 = resample_motion(
                motion135,
                args.src_fps,
                args.dst_fps,
                target_frames=target_frames,
            )
            if max_frames > 0 and len(motion135) > max_frames:
                motion135 = motion135[:max_frames]
            if args.out_format == "npz":
                np.savez(out_path, motion_135=motion135)
            else:
                np.save(out_path, motion135)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 20:
                print(f"[fail] {path}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 200 == 0:
            print(f"[progress] {idx}/{len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)
    manifest = {
        "input_root": str(input_root),
        "out_dir": str(out_dir),
        "files": len(files),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "recover_from_velocity": not args.no_recover_from_velocity,
        "equal_length": not args.no_equal_length,
        "src_fps": args.src_fps,
        "dst_fps": args.dst_fps,
        "max_frames": args.max_frames,
        "coord_conversion": args.coord_conversion,
        "translation_source": args.translation_source,
        "rotation_convention": args.rotation_convention,
        "out_format": args.out_format,
    }
    (out_dir / "_manifest.json").write_text(__import__("json").dumps(manifest, indent=2) + "\n")
    print(f"[done] input={input_root} out={out_dir} files={len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

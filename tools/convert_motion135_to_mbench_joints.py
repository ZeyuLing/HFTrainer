#!/usr/bin/env python3
"""Convert SMPL-22 135D motions to MBench raw joint inputs.

The input convention defaults to the row-major 6D layout used by the local
viewer/export files:

    [transl(3), 22 * rot6d(row-major)]

MBench expects one ``{id}.npy`` per sample with shape ``(T, 22, 3)`` in its
z-up coordinate system.  This tool uses the same SMPLPoseProcessor/body model
configuration as VerMo export, avoiding the skeleton mismatch seen with the
lightweight FK helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from mmengine.config import Config

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Import registers the body model class used by the processor config.
import hftrainer.models.motion.components.body_models.smplx_lite  # noqa: F401,E402
import hftrainer.models.motion.components.motion_processor.smpl_processor  # noqa: F401,E402
from hftrainer.registry import MODELS  # noqa: E402


SMPL_YUP_TO_MBENCH_ZUP = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def motion135_row_to_column(motion135: np.ndarray) -> np.ndarray:
    motion135 = np.asarray(motion135, dtype=np.float32).copy()
    rot = motion135[..., 3:135].reshape(*motion135.shape[:-1], 22, 6)
    motion135[..., 3:135] = rot[..., [0, 2, 4, 1, 3, 5]].reshape(*motion135.shape[:-1], 132)
    return motion135


def resample_linear(values: np.ndarray, new_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    old_len = values.shape[0]
    new_len = int(new_len)
    if old_len == new_len:
        return values.astype(np.float32, copy=False)
    if old_len <= 1 or new_len <= 1:
        return np.repeat(values[:1], max(new_len, 1), axis=0).astype(np.float32, copy=False)
    xs = np.linspace(0.0, old_len - 1, new_len, dtype=np.float32)
    lo = np.floor(xs).astype(np.int64)
    hi = np.minimum(lo + 1, old_len - 1)
    w = (xs - lo).reshape(-1, *([1] * (values.ndim - 1))).astype(np.float32)
    return (values[lo] * (1.0 - w) + values[hi] * w).astype(np.float32)


def resampled_length(num_frames: int, source_fps: float, target_fps: float) -> int:
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return int(num_frames)
    return max(1, int(round(int(num_frames) * float(target_fps) / float(source_fps))))


def load_eval_frame_map(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    frame_map: Dict[str, int] = {}
    for entry in raw:
        key = str(entry["id"])
        frames = int(entry["motion_duration"])
        old = frame_map.get(key)
        if old is not None and old != frames:
            raise ValueError(f"conflicting frame count for id={key}: {old} vs {frames}")
        frame_map[key] = frames
    return frame_map


def load_motion(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if key not in data.files:
            raise KeyError(f"{path} missing key {key!r}; keys={data.files}")
        motion = data[key]
    else:
        motion = np.load(path)
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[-1] < 135:
        raise ValueError(f"expected (T, >=135) motion in {path}, got {motion.shape}")
    return motion[:, :135].astype(np.float32, copy=False)


def iter_motion_files(in_dir: Path, ext: str) -> Iterable[Path]:
    yield from sorted(in_dir.glob(f"*{ext}"), key=lambda p: p.stem)


def build_smpl_processor(config_path: Path, device: torch.device):
    cfg = Config.fromfile(str(config_path))
    processor_cfg = cfg.model["processor"]["smpl_pose_processor"]
    processor = MODELS.build(processor_cfg).to(device)
    processor.eval()
    return processor


@torch.no_grad()
def motion_to_mbench_joints(
    processor,
    motion135: np.ndarray,
    *,
    input_convention: str,
    device: torch.device,
    chunk_size: int,
    source_fps: float,
    target_fps: float,
    target_frames: Optional[int] = None,
) -> np.ndarray:
    if input_convention == "row":
        motion135 = motion135_row_to_column(motion135)
    elif input_convention != "column":
        raise ValueError(f"unsupported input convention: {input_convention}")

    chunks: List[np.ndarray] = []
    for start in range(0, motion135.shape[0], chunk_size):
        chunk = motion135[start : start + chunk_size]
        motion_t = torch.as_tensor(chunk, dtype=torch.float32, device=device)
        transl = motion_t[:, :3].unsqueeze(0)
        rot6d = motion_t[:, 3:135].unsqueeze(0)
        joints = processor.fk(transl, rot6d, rot_type="rotation_6d").squeeze(0)
        joints_np = joints.detach().cpu().numpy().astype(np.float32)
        chunks.append(joints_np)
    joints = np.concatenate(chunks, axis=0)
    if target_frames is not None:
        joints = resample_linear(joints, int(target_frames))
    elif abs(float(source_fps) - float(target_fps)) > 1e-6:
        joints = resample_linear(joints, resampled_length(joints.shape[0], source_fps, target_fps))
    joints = np.einsum("ij,tvj->tvi", SMPL_YUP_TO_MBENCH_ZUP, joints).astype(np.float32)
    return joints


def joint_stats(joints: np.ndarray) -> Dict[str, Any]:
    feet = joints[:, [10, 11], :]
    return {
        "shape": list(joints.shape),
        "nan_count": int(np.isnan(joints).sum()),
        "foot_min_z": float(feet[..., 2].min()),
        "root_start_xyz": [float(x) for x in joints[0, 0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ext", default=".npy", choices=[".npy", ".npz"])
    parser.add_argument("--npz-key", default="motion_135")
    parser.add_argument("--input-convention", choices=["row", "column"], default="row")
    parser.add_argument(
        "--processor-config",
        default="configs/vermo/_base_vermo_pretrain_wavtokenizer.py",
    )
    parser.add_argument(
        "--eval-info-json",
        default="ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument(
        "--match-eval-frames",
        action="store_true",
        help="Resample FK joints to the exact frame count in --eval-info-json.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    eval_input = out_root / "mbench_eval_input"
    eval_input.mkdir(parents=True, exist_ok=True)
    frame_map = load_eval_frame_map(Path(args.eval_info_json))
    device = torch.device(args.device)
    processor = build_smpl_processor(Path(args.processor_config), device)

    records = []
    for path in iter_motion_files(in_dir, args.ext):
        out_path = eval_input / f"{path.stem}.npy"
        if out_path.exists() and not args.force:
            records.append({"id": path.stem, "status": "skipped_existing"})
            continue
        try:
            motion = load_motion(path, args.npz_key)
            joints = motion_to_mbench_joints(
                processor,
                motion,
                input_convention=args.input_convention,
                device=device,
                chunk_size=args.chunk_size,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                target_frames=frame_map.get(path.stem) if args.match_eval_frames else None,
            )
            np.save(out_path, joints)
            expected = frame_map.get(path.stem)
            records.append(
                {
                    "id": path.stem,
                    "status": "ok",
                    "input_path": str(path),
                    "output_path": str(out_path),
                    "pred_frames": int(joints.shape[0]),
                    "expected_frames": expected,
                    "frame_abs_error": None if expected is None else abs(int(joints.shape[0]) - expected),
                    "joint_stats": joint_stats(joints),
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append({"id": path.stem, "status": "error", "error": repr(exc), "input_path": str(path)})
            print(f"[error] {path}: {type(exc).__name__}: {exc}", flush=True)

    statuses = Counter(record["status"] for record in records)
    frame_errors = [
        record["frame_abs_error"]
        for record in records
        if record.get("status") == "ok" and record.get("frame_abs_error") is not None
    ]
    summary = {
        "num_records": len(records),
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "match_eval_frames": bool(args.match_eval_frames),
        "statuses": dict(statuses),
        "ok": int(statuses.get("ok", 0)),
        "complete": int(statuses.get("ok", 0)) == 450,
        "frame_abs_error_mean": float(np.mean(frame_errors)) if frame_errors else None,
        "frame_abs_error_max": int(np.max(frame_errors)) if frame_errors else None,
    }
    payload = {"summary": summary, "records": records}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[convert-motion135-mbench] wrote {out_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

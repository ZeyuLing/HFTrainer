#!/usr/bin/env python3
"""Build a GT/reference vs before/after tracker visualization manifest.

The input files are ProtoMotions packaged MotionLib ``.pt`` files. The reference
file is the target motion shard, while the before/after files are predicted
MotionLibs exported by ``MimicEvaluator._save_predicted_motion_lib``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    DEFAULT_BODIES,
    MESHES_BY_BODY,
    _parse_g1_body_meshes,
)


def _load_motion_lib(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    required = {"gts", "grs", "length_starts", "motion_num_frames"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return data


def _motion_count(data: dict[str, Any]) -> int:
    return int(torch.as_tensor(data["motion_num_frames"]).numel())


def _slice_motion(data: dict[str, Any], idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.as_tensor(data["length_starts"]).long()
    lengths = torch.as_tensor(data["motion_num_frames"]).long()
    start = int(starts[idx].item())
    length = int(lengths[idx].item())
    end = start + length
    body_pos = torch.as_tensor(data["gts"][start:end]).float().cpu()
    body_quat_xyzw = torch.as_tensor(data["grs"][start:end]).float().cpu()
    if body_pos.ndim != 3 or body_pos.shape[1] != len(DEFAULT_BODIES):
        raise ValueError(
            f"motion {idx} has body_pos shape {tuple(body_pos.shape)}, "
            f"expected [T, {len(DEFAULT_BODIES)}, 3]"
        )
    if body_quat_xyzw.ndim != 3 or body_quat_xyzw.shape[1:] != (len(DEFAULT_BODIES), 4):
        raise ValueError(
            f"motion {idx} has body_quat shape {tuple(body_quat_xyzw.shape)}, "
            f"expected [T, {len(DEFAULT_BODIES)}, 4]"
        )
    body_quat_wxyz = body_quat_xyzw[..., [3, 0, 1, 2]]
    return body_pos, body_quat_wxyz


def _body_records() -> list[dict[str, Any]]:
    try:
        bodies = _parse_g1_body_meshes()
    except Exception:
        bodies = []
    if [body.get("name") for body in bodies] == DEFAULT_BODIES:
        return bodies
    return [
        {
            "name": name,
            "meshes": [
                {
                    "file": mesh,
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                }
                for mesh in MESHES_BY_BODY.get(name, [])
            ],
        }
        for name in DEFAULT_BODIES
    ]


def _jsonify_tensor(value: torch.Tensor) -> Any:
    return value.detach().cpu().numpy().tolist()


def _write_robot_frames(
    path: Path,
    body_pos: torch.Tensor,
    body_quat_wxyz: torch.Tensor,
    fps: int,
    bodies: list[dict[str, Any]],
) -> None:
    frames = [
        {
            "body_pos": _jsonify_tensor(pos),
            "body_quat": _jsonify_tensor(quat),
        }
        for pos, quat in zip(body_pos, body_quat_wxyz)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "robot_frames",
                "robot": "g1",
                "fps": fps,
                "num_frames": len(frames),
                "num_bodies": len(bodies),
                "bodies": bodies,
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )


def _motion_label(data: dict[str, Any], idx: int) -> tuple[str, str]:
    motion_files = data.get("motion_files") or ()
    if idx >= len(motion_files):
        return f"motion_{idx:04d}", ""
    raw = str(motion_files[idx])
    parts = Path(raw).parts
    for marker in ("CMU", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset"):
        if marker in parts:
            rel = Path(*parts[parts.index(marker) :])
            return rel.with_suffix("").as_posix(), raw
    return Path(raw).with_suffix("").name, raw


def _align_rollout_xy(ref_pos: torch.Tensor, pred_pos: torch.Tensor) -> torch.Tensor:
    """Remove IsaacGym environment XY origin offset from exported rollouts."""
    if ref_pos.shape[0] == 0 or pred_pos.shape[0] == 0:
        return pred_pos
    aligned = pred_pos.clone()
    offset_xy = aligned[0, 0, :2] - ref_pos[0, 0, :2]
    aligned[..., :2] -= offset_xy
    return aligned


def _resample_positions(
    body_pos: torch.Tensor,
    src_fps: int,
    dst_times: torch.Tensor,
) -> torch.Tensor:
    if body_pos.shape[0] == 0:
        return body_pos
    frame = dst_times * float(src_fps)
    lo = torch.floor(frame).long().clamp(0, body_pos.shape[0] - 1)
    hi = (lo + 1).clamp(0, body_pos.shape[0] - 1)
    alpha = (frame - lo.float()).view(-1, 1, 1)
    return body_pos[lo] * (1.0 - alpha) + body_pos[hi] * alpha


def _mean_errors_mm(
    ref_pos: torch.Tensor,
    ref_fps: int,
    pred_pos: torch.Tensor,
    pred_fps: int,
) -> tuple[float, float]:
    duration = min(ref_pos.shape[0] / float(ref_fps), pred_pos.shape[0] / float(pred_fps))
    if duration <= 0:
        return float("nan"), float("nan")
    frame_count = max(1, int(duration * ref_fps))
    times = torch.arange(frame_count, dtype=torch.float32) / float(ref_fps)
    ref_sampled = _resample_positions(ref_pos, ref_fps, times)
    pred_sampled = _resample_positions(pred_pos, pred_fps, times)
    delta = torch.linalg.norm(ref_sampled - pred_sampled, dim=-1)
    mean_body = float(delta.mean().item() * 1000.0)
    mean_max_body = float(delta.max(dim=-1).values.mean().item() * 1000.0)
    return mean_body, mean_max_body


def _fps(data: dict[str, Any], default: int = 30) -> int:
    if "fps" in data:
        return int(data["fps"])
    if "motion_dt" in data:
        dt = torch.as_tensor(data["motion_dt"]).flatten()
        if dt.numel() > 0 and float(dt[0]) > 0:
            return round(1.0 / float(dt[0]))
    return default


def build_manifest(
    ref_motion_lib: Path,
    before_pred_lib: Path,
    after_pred_lib: Path,
    out_dir: Path,
    max_cases: int | None,
) -> Path:
    ref_data = _load_motion_lib(ref_motion_lib)
    before_data = _load_motion_lib(before_pred_lib)
    after_data = _load_motion_lib(after_pred_lib)

    count = min(_motion_count(ref_data), _motion_count(before_data), _motion_count(after_data))
    if max_cases is not None:
        count = min(count, max_cases)
    if count <= 0:
        raise ValueError("No overlapping motions to visualize")

    bodies = _body_records()
    ref_fps = _fps(ref_data)
    before_fps = _fps(before_data)
    after_fps = _fps(after_data)
    frames_dir = out_dir / "robot_frames"
    rows: list[dict[str, Any]] = []

    for idx in range(count):
        prompt_id, source_path = _motion_label(ref_data, idx)
        ref_pos, ref_quat = _slice_motion(ref_data, idx)
        before_pos, before_quat = _slice_motion(before_data, idx)
        after_pos, after_quat = _slice_motion(after_data, idx)
        before_pos = _align_rollout_xy(ref_pos, before_pos)
        after_pos = _align_rollout_xy(ref_pos, after_pos)

        ref_json = frames_dir / f"case_{idx:02d}.reference.json"
        before_json = frames_dir / f"case_{idx:02d}.tracker_before.json"
        after_json = frames_dir / f"case_{idx:02d}.tracker_after.json"
        _write_robot_frames(ref_json, ref_pos, ref_quat, ref_fps, bodies)
        _write_robot_frames(before_json, before_pos, before_quat, before_fps, bodies)
        _write_robot_frames(after_json, after_pos, after_quat, after_fps, bodies)

        before_mean, before_max = _mean_errors_mm(ref_pos, ref_fps, before_pos, before_fps)
        after_mean, after_max = _mean_errors_mm(ref_pos, ref_fps, after_pos, after_fps)
        rows.append(
            {
                "case": idx,
                "prompt_id": prompt_id,
                "prompt": source_path,
                "columns": [
                    {
                        "title": "GT Reference",
                        "group": "target",
                        "path": str(ref_json.resolve()),
                    },
                    {
                        "title": "Tracker Before",
                        "group": "pretrained",
                        "path": str(before_json.resolve()),
                    },
                    {
                        "title": "Tracker After",
                        "group": "GT replay",
                        "path": str(after_json.resolve()),
                    },
                ],
                "metrics": {
                    "ref_frames": int(ref_pos.shape[0]),
                    "before_frames": int(before_pos.shape[0]),
                    "after_frames": int(after_pos.shape[0]),
                    "ref_fps": ref_fps,
                    "before_fps": before_fps,
                    "after_fps": after_fps,
                    "before_gt_error_mm": before_mean,
                    "after_gt_error_mm": after_mean,
                    "delta_gt_error_mm": after_mean - before_mean,
                    "before_joint_error": before_max,
                    "after_joint_error": after_max,
                    "delta_joint_error": after_max - before_max,
                },
            }
        )

    manifest = {
        "title": "GT Replay Tracker Before/After",
        "reference_motion_lib": str(ref_motion_lib.resolve()),
        "before_predicted_motion_lib": str(before_pred_lib.resolve()),
        "after_predicted_motion_lib": str(after_pred_lib.resolve()),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-motion-lib", required=True, type=Path)
    parser.add_argument("--before-pred-lib", required=True, type=Path)
    parser.add_argument("--after-pred-lib", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    manifest_path = build_manifest(
        ref_motion_lib=args.ref_motion_lib,
        before_pred_lib=args.before_pred_lib,
        after_pred_lib=args.after_pred_lib,
        out_dir=args.out_dir,
        max_cases=args.max_cases,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

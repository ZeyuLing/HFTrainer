#!/usr/bin/env python3
"""Materialize ProtoMotions predicted MotionLib shards into canonical NPZ files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.physflow_canonical_rollouts import save_body, write_run_config
from scripts.embodied.physflow_triplet_manifest import DEFAULT_BODIES


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _starts_lens(lib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    starts = _as_np(lib["length_starts"]).astype(np.int64)
    if "motion_num_frames" in lib:
        lens = _as_np(lib["motion_num_frames"]).astype(np.int64)
    else:
        total = int(lib["gts"].shape[0])
        lens = np.asarray(
            [int(starts[i + 1] - starts[i]) if i + 1 < len(starts) else int(total - starts[i]) for i in range(len(starts))],
            dtype=np.int64,
        )
    return starts, lens


def _motion_files(lib: dict[str, Any], n: int) -> list[str]:
    files = list(lib.get("motion_files", []))
    if len(files) < n:
        files.extend([f"predicted_motion_{i}" for i in range(len(files), n)])
    return [str(x) for x in files[:n]]


def _slice(lib: dict[str, Any], motion_id: int) -> tuple[np.ndarray, np.ndarray, float, str]:
    starts, lens = _starts_lens(lib)
    start = int(starts[motion_id])
    frames = int(lens[motion_id])
    pos = _as_np(lib["gts"][start : start + frames]).astype(np.float32).copy()
    quat_xyzw = _as_np(lib["grs"][start : start + frames]).astype(np.float32).copy()
    dt = float(_as_np(lib["motion_dt"]).reshape(-1)[min(motion_id, len(_as_np(lib["motion_dt"]).reshape(-1)) - 1)])
    files = _motion_files(lib, len(lens))
    return pos, quat_xyzw, dt, files[motion_id]


def _latest_predicted(root: Path) -> Path | None:
    candidates = sorted((root / "results").glob("predicted_motion_lib_epoch_*.pt"))
    return candidates[-1] if candidates else None


def _shard_id(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("predicted_shard_"):
            try:
                return int(part.replace("predicted_shard_", ""))
            except ValueError:
                return None
    return None


def _case_id(motion_file: str, fallback: str) -> str:
    stem = Path(motion_file).stem
    return stem or fallback


def _align_xy(pred_pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    out = pred_pos.copy()
    if len(out) and len(ref_pos):
        out[..., :2] -= out[0, 0, :2] - ref_pos[0, 0, :2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-root", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--method-name", default="protomotions_g1_bones")
    ap.add_argument("--canonical-root", type=Path, default=Path("outputs/evaluation/physflow"))
    ap.add_argument("--canonical-method", default="protomotion")
    ap.add_argument("--output-fps", type=float, default=30.0)
    args = ap.parse_args()

    eval_dir = args.protocol_root / "runs" / "protomotions" / args.split / f"eval_{args.method_name}"
    motion_base = args.protocol_root / "proto_motions" / args.split
    if not eval_dir.is_dir():
        raise FileNotFoundError(eval_dir)

    rows = []
    for pred_root in sorted(eval_dir.glob("predicted_shard_*")):
        pred_path = _latest_predicted(pred_root)
        shard = _shard_id(pred_root)
        if pred_path is None or shard is None:
            continue
        ref_path = motion_base / f"{args.split}_g1_shard_{shard}.pt"
        if not ref_path.is_file():
            continue
        pred = _load(pred_path)
        ref = _load(ref_path)
        _, pred_lens = _starts_lens(pred)
        _, ref_lens = _starts_lens(ref)
        n = min(len(pred_lens), len(ref_lens))
        for motion_id in range(n):
            pred_pos, pred_quat_xyzw, pred_dt, motion_file = _slice(pred, motion_id)
            ref_pos, ref_quat_xyzw, ref_dt, _ = _slice(ref, motion_id)
            case_id = _case_id(motion_file, f"shard{shard}_motion{motion_id}")
            pred_pos = _align_xy(pred_pos, ref_pos)
            ref_quat_wxyz = ref_quat_xyzw[..., [3, 0, 1, 2]]
            pred_quat_wxyz = pred_quat_xyzw[..., [3, 0, 1, 2]]
            save_body(
                args.canonical_root,
                args.split,
                "reference",
                case_id,
                ref_pos,
                ref_quat_wxyz,
                DEFAULT_BODIES,
                source_fps=1.0 / max(ref_dt, 1e-9),
                target_fps=args.output_fps,
                metadata={
                    "runner": "scripts/embodied/materialize_protomotions_canonical_rollouts.py",
                    "reference_motion_lib": str(ref_path),
                    "motion_file": motion_file,
                },
            )
            bpath = save_body(
                args.canonical_root,
                args.split,
                args.canonical_method,
                case_id,
                pred_pos,
                pred_quat_wxyz,
                DEFAULT_BODIES,
                source_fps=1.0 / max(pred_dt, 1e-9),
                target_fps=args.output_fps,
                metadata={
                    "runner": "scripts/embodied/materialize_protomotions_canonical_rollouts.py",
                    "predicted_motion_lib": str(pred_path),
                    "reference_motion_lib": str(ref_path),
                    "motion_file": motion_file,
                    "shard": shard,
                    "motion_id": motion_id,
                },
            )
            rows.append({"case_id": case_id, "body": str(bpath), "frames": int(pred_pos.shape[0])})

    write_run_config(
        args.canonical_root,
        args.split,
        "g1_body30",
        args.canonical_method,
        {
            "method": args.canonical_method,
            "source_method": args.method_name,
            "runner": "scripts/embodied/materialize_protomotions_canonical_rollouts.py",
            "protocol_root": str(args.protocol_root),
            "split": args.split,
            "output_fps": args.output_fps,
            "num_cases": len(rows),
            "note": "ProtoMotions predicted MotionLib provides rigid body frames; execution qpos is not exported.",
        },
    )
    print(json.dumps({"num_cases": len(rows), "rows": rows[:5]}, indent=2))


if __name__ == "__main__":
    main()

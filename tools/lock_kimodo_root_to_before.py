#!/usr/bin/env python3
"""Lock KIMODO Base Pose root translation to the source motion.

Base Pose Edit changes pose, not locomotion. KIMODO can still introduce root
trajectory wobble between dense constraints, so this tool applies a rigid
per-frame XZ shift to the whole SOMA/SMPL output so the root follows the
source retargeted SOMA root exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np  # noqa: E402


def _case_metrics(pred_pos: np.ndarray, before: np.ndarray, after: np.ndarray,
                  keypose_indices: list[int], bone_offsets: np.ndarray) -> dict:
    gt_pos = motion135_to_positions_np(after, bone_offsets)
    before_pos = motion135_to_positions_np(before, bone_offsets)
    n = min(pred_pos.shape[0], gt_pos.shape[0], before_pos.shape[0])
    pred = pred_pos[:n]
    gt = gt_pos[:n]
    src = before_pos[:n]
    kp = [int(k) for k in keypose_indices if 0 <= int(k) < n]
    acc = np.diff(pred, n=2, axis=0)
    feet = pred[:, [7, 8, 10, 11]]
    vel = np.diff(feet[..., [0, 2]], axis=0)
    return {
        "kf_mpjpe": float(np.linalg.norm(pred[kp] - gt[kp], axis=-1).mean()) if kp else 0.0,
        "global_mpjpe": float(np.linalg.norm(pred - gt, axis=-1).mean()),
        "src_mpjpe": float(np.linalg.norm(pred - src, axis=-1).mean()),
        "overall_smoothness": float(np.linalg.norm(acc, axis=-1).mean()) if len(acc) else 0.0,
        "foot_skating": float(np.linalg.norm(vel, axis=-1).mean()) if len(vel) else 0.0,
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = ["kf_mpjpe", "global_mpjpe", "src_mpjpe", "overall_smoothness", "foot_skating"]
    return {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}


def _shift_array(arr: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = arr.copy()
    n = min(out.shape[0], delta.shape[0])
    out[:n] += delta[:n, None, :]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lock-y", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--smooth-passes", type=int, default=1)
    args = parser.parse_args()

    in_dir = PROJECT_ROOT / args.in_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bone_offsets = torch.load(
        PROJECT_ROOT / "data/hymotion_m2m_data/bone_offsets_22.pt",
        map_location="cpu",
    ).numpy()

    rows = []
    for src_path in sorted(in_dir.glob("case*.npz")):
        data = np.load(src_path, allow_pickle=True)
        before = data["before_motion"].astype(np.float32)
        after = data["after_motion"].astype(np.float32)
        target_root = before[:, :3].astype(np.float32)
        if args.smooth_window > 1:
            win = int(args.smooth_window)
            if win % 2 == 0:
                win += 1
            pad = win // 2
            kernel = np.ones(win, dtype=np.float32) / float(win)
            root_xz = target_root[:, [0, 2]].copy()
            for _ in range(max(1, int(args.smooth_passes))):
                padded = np.pad(root_xz, ((pad, pad), (0, 0)), mode="edge")
                root_xz = np.stack([
                    np.convolve(padded[:, i], kernel, mode="valid")
                    for i in range(root_xz.shape[1])
                ], axis=1).astype(np.float32)
            target_root = target_root.copy()
            target_root[:, [0, 2]] = root_xz

        posed = data["posed_joints"].astype(np.float32)
        current_root = posed[:, 0, :].astype(np.float32)
        n = min(len(posed), len(target_root))
        delta = np.zeros((len(posed), 3), dtype=np.float32)
        delta[:n, [0, 2]] = target_root[:n, [0, 2]] - current_root[:n, [0, 2]]
        if args.lock_y:
            delta[:n, 1] = target_root[:n, 1] - current_root[:n, 1]

        out_items = {k: data[k] for k in data.files}
        out_items["posed_joints"] = _shift_array(posed, delta).astype(np.float32)
        if "output_positions" in data.files:
            out_items["output_positions"] = _shift_array(
                data["output_positions"].astype(np.float32), delta
            ).astype(np.float32)
        out_items["root_lock_delta"] = delta.astype(np.float32)
        out_items["root_lock_source"] = np.array("before_root_xz", dtype="<U20")
        out_items["root_lock_smooth_window"] = np.array(args.smooth_window, dtype=np.int64)
        out_items["root_lock_smooth_passes"] = np.array(args.smooth_passes, dtype=np.int64)

        kp = data["keypose_indices"].astype(int).tolist() if "keypose_indices" in data.files else []
        metrics = _case_metrics(out_items["output_positions"], before, after, kp, bone_offsets)
        row = {
            "case_key": src_path.stem,
            "filename": str(data["before_motion"].shape),
            "root_lock": "xz",
            **metrics,
        }
        rows.append(row)
        np.savez_compressed(out_dir / src_path.name, **out_items)

    result = {"aggregate": _aggregate(rows) if rows else {}, "cases": rows}
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"out_dir": str(out_dir), "num_cases": len(rows), **result}, indent=2))


if __name__ == "__main__":
    main()

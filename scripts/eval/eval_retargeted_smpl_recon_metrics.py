#!/usr/bin/env python3
"""Compute reconstruction metrics between two retargeted SMPL npz directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.evaluation.motion.m2m_eval_metrics import compute_pa_mpjpe
from hftrainer.models.motion.components.utils.geometry.rotation_convert import axis_angle_to_matrix


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_samples": int(arr.size)}


def _rotmats(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    go = torch.from_numpy(np.asarray(z["global_orient"], dtype=np.float32)).reshape(-1, 1, 3)
    bp = torch.from_numpy(np.asarray(z["body_pose"], dtype=np.float32)).reshape(-1, 21, 3)
    aa = torch.cat([go, bp], dim=1)
    return axis_angle_to_matrix(aa).numpy().astype(np.float32)


def _geodesic_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    rel = np.matmul(np.swapaxes(pred, -1, -2), gt)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _load_ids(gt_dir: Path, pred_dir: Path, ids_file: str | None) -> list[str]:
    if ids_file:
        raw = [line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = sorted(path.stem for path in gt_dir.glob("*.npz"))
    return [sid for sid in raw if (gt_dir / f"{sid}.npz").exists() and (pred_dir / f"{sid}.npz").exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    ids = _load_ids(gt_dir, pred_dir, args.ids or None)
    if args.limit:
        ids = ids[: args.limit]

    values: dict[str, list[float]] = {
        "mpjpe_mm": [],
        "pa_mpjpe_mm": [],
        "mpjre_deg": [],
    }
    per_case: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for sid in ids:
        try:
            gt_path = gt_dir / f"{sid}.npz"
            pred_path = pred_dir / f"{sid}.npz"
            gt = np.load(gt_path, allow_pickle=True)
            pred = np.load(pred_path, allow_pickle=True)
            gt_pos = np.asarray(gt["fitted_joints"], dtype=np.float32)
            pred_pos = np.asarray(pred["fitted_joints"], dtype=np.float32)
            gt_rot = _rotmats(gt_path)
            pred_rot = _rotmats(pred_path)
            t = min(len(gt_pos), len(pred_pos), len(gt_rot), len(pred_rot))
            if t <= 0:
                raise ValueError("empty sequence")
            gt_pos = gt_pos[:t]
            pred_pos = pred_pos[:t]
            gt_rot = gt_rot[:t]
            pred_rot = pred_rot[:t]
            mpjpe = float(np.linalg.norm(pred_pos - gt_pos, axis=-1).mean() * 1000.0)
            pa = float(compute_pa_mpjpe(pred_pos, gt_pos)["pa_mpjpe_mean"] * 1000.0)
            mpjre = _geodesic_deg(pred_rot, gt_rot)
            values["mpjpe_mm"].append(mpjpe)
            values["pa_mpjpe_mm"].append(pa)
            values["mpjre_deg"].append(mpjre)
            per_case.append(
                {
                    "key": sid,
                    "frames": int(t),
                    "mpjpe_mm": mpjpe,
                    "pa_mpjpe_mm": pa,
                    "mpjre_deg": mpjre,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": sid, "error": repr(exc)})

    payload = {
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "ids": args.ids,
        "selected": len(ids),
        "summary": {
            "mpjpe_mm": _summary(values["mpjpe_mm"]),
            "pa_mpjpe_mm": _summary(values["pa_mpjpe_mm"]),
            "mpjre_deg": _summary(values["mpjre_deg"]),
            "num_failures": len(failures),
        },
        "failures": failures,
        "per_case": per_case,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"[retargeted-smpl-recon-metrics] wrote {out}")


if __name__ == "__main__":
    main()

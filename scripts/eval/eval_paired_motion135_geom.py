#!/usr/bin/env python3
"""Paired geometry metrics directly on row-major SMPL motion_135 clips."""
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

from hftrainer.evaluation.motion.m2m_eval_metrics import compute_pa_mpjpe  # noqa: E402
from hftrainer.motion.skeleton.fk import motion135_to_fk  # noqa: E402


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_samples": int(arr.size)}


def _geodesic_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    rel = np.matmul(np.swapaxes(pred, -1, -2), gt)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _root_aligned_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_ra = pred - pred[..., :1, :]
    gt_ra = gt - gt[..., :1, :]
    return float(np.linalg.norm(pred_ra - gt_ra, axis=-1).mean() * 1000.0)


def _load_m135(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    arr = np.asarray(data["motion_135"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 135:
        raise ValueError(f"expected motion_135 shape (T,135), got {arr.shape}")
    return arr


def _fk(m135: np.ndarray, bone_offsets: torch.Tensor):
    x = torch.from_numpy(m135).float()
    with torch.no_grad():
        pos, world_rot, _trans, local_rot = motion135_to_fk(x, bone_offsets, rotation_space="local")
    return (
        pos.detach().cpu().numpy().astype(np.float32),
        local_rot.detach().cpu().numpy().astype(np.float32),
        world_rot.detach().cpu().numpy().astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bone-offsets", default=str(REPO / "data/hymotion_m2m_data/bone_offsets_22.pt"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--skip-pa-mpjpe",
        action="store_true",
        help="Skip PA-MPJPE; useful for full-split conversion diagnostics where the SVD alignment is the bottleneck.",
    )
    args = parser.parse_args()

    ids = [line.strip() for line in Path(args.split).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]
    ref_dir = Path(args.ref_dir)
    pred_dir = Path(args.pred_dir)
    bone_offsets = torch.load(args.bone_offsets, map_location="cpu").float()

    values: dict[str, list[float]] = {
        "mpjpe_mm": [],
        "root_aligned_mpjpe_mm": [],
        "mpjre_deg": [],
        "local_mpjre_deg": [],
        "motion135_l1": [],
    }
    if not args.skip_pa_mpjpe:
        values["pa_mpjpe_mm"] = []
    failures: list[dict[str, str]] = []
    skipped = {"missing": 0, "empty": 0, "error": 0}
    per_case: list[dict[str, Any]] = []

    for idx, sid in enumerate(ids, 1):
        rpath = ref_dir / f"{sid}.npz"
        ppath = pred_dir / f"{sid}.npz"
        if not rpath.exists() or not ppath.exists():
            skipped["missing"] += 1
            continue
        try:
            ref = _load_m135(rpath)
            pred = _load_m135(ppath)
            t = min(len(ref), len(pred))
            if t <= 0:
                skipped["empty"] += 1
                continue
            ref = ref[:t]
            pred = pred[:t]
            ref_pos, ref_local, ref_world = _fk(ref, bone_offsets)
            pred_pos, pred_local, pred_world = _fk(pred, bone_offsets)
            mpjpe = float(np.linalg.norm(pred_pos - ref_pos, axis=-1).mean() * 1000.0)
            root_mpjpe = _root_aligned_mpjpe_mm(pred_pos, ref_pos)
            pa = None
            if not args.skip_pa_mpjpe:
                pa = float(compute_pa_mpjpe(pred_pos, ref_pos)["pa_mpjpe_mean"] * 1000.0)
            mpjre = _geodesic_deg(pred_world, ref_world)
            local_mpjre = _geodesic_deg(pred_local, ref_local)
            l1 = float(np.abs(pred - ref).mean())
            values["mpjpe_mm"].append(mpjpe)
            values["root_aligned_mpjpe_mm"].append(root_mpjpe)
            if pa is not None:
                values["pa_mpjpe_mm"].append(pa)
            values["mpjre_deg"].append(mpjre)
            values["local_mpjre_deg"].append(local_mpjre)
            values["motion135_l1"].append(l1)
            per_case.append(
                {
                    "id": sid,
                    "frames": int(t),
                    "mpjpe_mm": mpjpe,
                    "root_aligned_mpjpe_mm": root_mpjpe,
                    "pa_mpjpe_mm": pa,
                    "mpjre_deg": mpjre,
                    "local_mpjre_deg": local_mpjre,
                    "motion135_l1": l1,
                }
            )
        except Exception as exc:  # noqa: BLE001
            skipped["error"] += 1
            failures.append({"id": sid, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 10:
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 500 == 0:
            print(f"[progress] {idx}/{len(ids)} used={len(per_case)} skipped={skipped}", flush=True)

    payload = {
        "ref_dir": str(ref_dir),
        "pred_dir": str(pred_dir),
        "split": args.split,
        "bone_offsets": args.bone_offsets,
        "skip_pa_mpjpe": bool(args.skip_pa_mpjpe),
        "selected": len(ids),
        "used": len(per_case),
        "skipped": skipped,
        "summary": {key: _summary(val) for key, val in values.items()},
        "failures": failures,
        "per_case": per_case,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload["summary"], "used": payload["used"], "skipped": skipped}, indent=2), flush=True)


if __name__ == "__main__":
    main()

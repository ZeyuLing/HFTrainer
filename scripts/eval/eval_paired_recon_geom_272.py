#!/usr/bin/env python3
"""Paired reconstruction geometry metrics in MotionStreamer-272 space."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MS = REPO / "ref_repo/MotionStreamer/MotionStreamer"
SPLIT_TEST = MS / "humanml3d_272/split/test.txt"
GT_MOTION_DIR = MS / "humanml3d_272/motion_data"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_272_stored_positions,
    recover_local_rotations_and_root,
)
from hftrainer.evaluation.motion.m2m_eval_metrics import compute_pa_mpjpe  # noqa: E402
from motionstreamer_272_encoder import motion135_to_272  # noqa: E402


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_samples": int(arr.size)}


def _path_for(root: Path, sid: str, kind: str) -> Path:
    return root / f"{sid}.npy" if kind == "ms272-npy" else root / f"{sid}.npz"


def _load_motion(path: Path, kind: str) -> np.ndarray:
    if kind == "ms272-npy":
        return np.load(path).astype(np.float32)
    data = np.load(path, allow_pickle=True)
    if kind == "npz272":
        return np.asarray(data["motion_272"], dtype=np.float32)
    if kind == "npz135":
        return np.asarray(motion135_to_272(data["motion_135"]), dtype=np.float32)
    raise ValueError(f"unsupported kind: {kind}")


def _geodesic_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    rel = np.matmul(np.swapaxes(pred, -1, -2), gt)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _root_aligned_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_ra = pred - pred[..., :1, :]
    gt_ra = gt - gt[..., :1, :]
    return float(np.linalg.norm(pred_ra - gt_ra, axis=-1).mean() * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", default=str(GT_MOTION_DIR))
    parser.add_argument("--ref-kind", choices=["ms272-npy", "npz272", "npz135"], default="ms272-npy")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--pred-kind", choices=["ms272-npy", "npz272", "npz135"], required=True)
    parser.add_argument("--split", default=str(SPLIT_TEST))
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    pred_dir = Path(args.pred_dir)
    ids = [line.strip() for line in Path(args.split).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]

    values: dict[str, list[float]] = {
        "mpjpe_mm": [],
        "root_aligned_mpjpe_mm": [],
        "pa_mpjpe_mm": [],
        "mpjre_deg": [],
    }
    failures: list[dict[str, str]] = []
    skipped = {"missing": 0, "empty": 0, "error": 0}
    per_case: list[dict[str, Any]] = []

    for idx, sid in enumerate(ids, 1):
        rpath = _path_for(ref_dir, sid, args.ref_kind)
        ppath = _path_for(pred_dir, sid, args.pred_kind)
        if not rpath.exists() or not ppath.exists():
            skipped["missing"] += 1
            continue
        try:
            ref = _load_motion(rpath, args.ref_kind)
            pred = _load_motion(ppath, args.pred_kind)
            t = min(len(ref), len(pred))
            if t <= 0:
                skipped["empty"] += 1
                continue
            ref = ref[:t]
            pred = pred[:t]
            ref_pos = recover_272_stored_positions(ref)
            pred_pos = recover_272_stored_positions(pred)
            ref_rot, _ = recover_local_rotations_and_root(ref)
            pred_rot, _ = recover_local_rotations_and_root(pred)
            mpjpe = float(np.linalg.norm(pred_pos - ref_pos, axis=-1).mean() * 1000.0)
            root_mpjpe = _root_aligned_mpjpe_mm(pred_pos, ref_pos)
            pa = float(compute_pa_mpjpe(pred_pos, ref_pos)["pa_mpjpe_mean"] * 1000.0)
            mpjre = _geodesic_deg(pred_rot, ref_rot)
            values["mpjpe_mm"].append(mpjpe)
            values["root_aligned_mpjpe_mm"].append(root_mpjpe)
            values["pa_mpjpe_mm"].append(pa)
            values["mpjre_deg"].append(mpjre)
            per_case.append(
                {
                    "id": sid,
                    "frames": int(t),
                    "mpjpe_mm": mpjpe,
                    "root_aligned_mpjpe_mm": root_mpjpe,
                    "pa_mpjpe_mm": pa,
                    "mpjre_deg": mpjre,
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
        "ref_kind": args.ref_kind,
        "pred_dir": str(pred_dir),
        "pred_kind": args.pred_kind,
        "split": args.split,
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

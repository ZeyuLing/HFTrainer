#!/usr/bin/env python3
"""Measure root and FK foot-height drift for motion_135 prediction folders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.motion.skeleton.fk import motion135_to_fk  # noqa: E402


FOOT_JOINTS = (7, 8, 10, 11)


def _load_anno_lengths(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    items = data.get("data_list", data)
    if not isinstance(items, dict):
        raise TypeError(f"Expected dict-style annotation, got {type(items).__name__}")
    lengths: dict[str, int] = {}
    for sid, item in items.items():
        can = Path(item.get("smplx_path", sid)).stem
        length = item.get("num_frames") or item.get("length")
        if length is None:
            raise KeyError(f"Missing num_frames/length for {sid}")
        lengths[can] = int(length)
    return lengths


def _load_motion135(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "motion_135" in data:
            arr = data["motion_135"]
        elif "motion" in data:
            arr = data["motion"]
        else:
            raise KeyError(f"{path} does not contain motion_135")
    else:
        arr = data
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 135:
        raise ValueError(f"Bad motion_135 shape for {path}: {arr.shape}")
    return arr[:, :135]


def _edge_mean(values: np.ndarray, side: str, window: int) -> float:
    n = min(window, len(values))
    if n <= 0:
        return float("nan")
    if side == "start":
        return float(np.mean(values[:n]))
    return float(np.mean(values[-n:]))


def _summ(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return {
            "count": 0,
            "mean_cm": float("nan"),
            "median_cm": float("nan"),
            "p90_cm": float("nan"),
            "max_cm": float("nan"),
        }
    arr = np.asarray(vals, dtype=np.float64) * 100.0
    return {
        "count": int(len(arr)),
        "mean_cm": float(np.mean(arr)),
        "median_cm": float(np.median(arr)),
        "p90_cm": float(np.percentile(arr, 90)),
        "max_cm": float(np.max(arr)),
    }


def _bucket(length: int) -> str:
    if length < 120:
        return "<120"
    if length < 180:
        return "120-179"
    if length < 240:
        return "180-239"
    return ">=240"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m135-dir", required=True, help="Directory of <id>.npz motion_135 files.")
    ap.add_argument("--anno-file", required=True, help="Official annotation JSON used for coverage and lengths.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    ap.add_argument("--edge-window", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    lengths = _load_anno_lengths(Path(args.anno_file))
    m135_dir = Path(args.m135_dir)
    bone_offsets = torch.load(args.bone_offsets, map_location=args.device)
    if not isinstance(bone_offsets, torch.Tensor):
        bone_offsets = torch.as_tensor(bone_offsets)
    bone_offsets = bone_offsets.to(args.device).float()

    rows = []
    missing = []
    failed = []
    bucket_vals: dict[str, dict[str, list[float]]] = {}
    for cid, official_len in sorted(lengths.items()):
        path = m135_dir / f"{cid}.npz"
        if not path.exists():
            missing.append(cid)
            continue
        try:
            motion = _load_motion135(path)
            final_len = int(motion.shape[0])
            valid = motion[:official_len]
            root_y = valid[:, 1]
            root_start = _edge_mean(root_y, "start", args.edge_window)
            root_end = _edge_mean(root_y, "end", args.edge_window)
            with torch.no_grad():
                mt = torch.from_numpy(valid).to(args.device).float()
                pos, _, _, _ = motion135_to_fk(mt, bone_offsets, rotation_space="local")
                foot_y = pos[:, FOOT_JOINTS, 1].amin(dim=1).detach().cpu().numpy()
            foot_start = _edge_mean(foot_y, "start", args.edge_window)
            foot_end = _edge_mean(foot_y, "end", args.edge_window)
            row = {
                "id": cid,
                "official_len": official_len,
                "final_len": final_len,
                "length_match": final_len == official_len,
                "root_y_start_m": root_start,
                "root_y_end_m": root_end,
                "root_y_drift_m": root_end - root_start,
                "root_y_abs_drift_m": abs(root_end - root_start),
                "foot_min_y_start_m": foot_start,
                "foot_min_y_end_m": foot_end,
                "foot_min_y_drift_m": foot_end - foot_start,
                "foot_min_y_abs_drift_m": abs(foot_end - foot_start),
                "mean_root_y_m": float(np.mean(root_y)),
                "mean_foot_min_y_m": float(np.mean(foot_y)),
            }
            rows.append(row)
            b = _bucket(official_len)
            bucket_vals.setdefault(b, {
                "root_y_abs_drift_m": [],
                "foot_min_y_abs_drift_m": [],
                "root_y_drift_m": [],
                "foot_min_y_drift_m": [],
            })
            for key in bucket_vals[b]:
                bucket_vals[b][key].append(float(row[key]))
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": cid, "error": str(exc)})

    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    root_signed = vals("root_y_drift_m")
    foot_signed = vals("foot_min_y_drift_m")
    summary = {
        "m135_dir": str(m135_dir),
        "anno_file": str(args.anno_file),
        "coverage": {
            "expected": int(len(lengths)),
            "found": int(len(rows)),
            "missing": int(len(missing)),
            "failed": int(len(failed)),
            "length_match": int(sum(r["length_match"] for r in rows)),
        },
        "root_y_abs_drift": _summ(vals("root_y_abs_drift_m")),
        "foot_min_y_abs_drift": _summ(vals("foot_min_y_abs_drift_m")),
        "root_y_signed_drift_mean_cm": float(mean(root_signed) * 100.0) if root_signed else float("nan"),
        "root_y_signed_drift_median_cm": float(median(root_signed) * 100.0) if root_signed else float("nan"),
        "foot_min_y_signed_drift_mean_cm": float(mean(foot_signed) * 100.0) if foot_signed else float("nan"),
        "foot_min_y_signed_drift_median_cm": float(median(foot_signed) * 100.0) if foot_signed else float("nan"),
        "buckets": {
            b: {k: _summ(v) for k, v in metrics.items()}
            for b, metrics in sorted(bucket_vals.items())
        },
        "top_root_abs_drift": sorted(rows, key=lambda r: r["root_y_abs_drift_m"], reverse=True)[:args.top_k],
        "top_foot_abs_drift": sorted(rows, key=lambda r: r["foot_min_y_abs_drift_m"], reverse=True)[:args.top_k],
        "missing_ids": missing[:200],
        "failed": failed[:200],
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["coverage"], indent=2))
    print("root_y_abs_drift", summary["root_y_abs_drift"])
    print("foot_min_y_abs_drift", summary["foot_min_y_abs_drift"])


if __name__ == "__main__":
    main()

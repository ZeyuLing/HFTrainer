#!/usr/bin/env python3
"""Diagnose whether TP2M baselines actually consume pose prefixes.

The script supports two output formats used in the Table-2 reruns:

* HumanML3D-263 ``.npy`` predictions from FlowMDM / MotionLab.
* MotionStreamer ``.npz`` predictions that store ``motion_272`` and
  ``gt272_path`` metadata.

It reports prefix feature error, recovered-joint error, and root-drift style
statistics so broken conditioning can be detected before full evaluator runs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
MOTIONLAB_ROOT = REPO / "ref_repo" / "MotionLab"
if str(MOTIONLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONLAB_ROOT))


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_entries(raw):
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
        return
    for i, entry in enumerate(data):
        yield str(entry.get("motion_id") or entry.get("id") or i), entry


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def _build_gt_map(anno_file: Path | None, gt_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if anno_file is not None and anno_file.exists():
        for name, entry in _iter_entries(_load_json(anno_file)):
            stem = Path(str(entry.get("smplx_path") or "")).stem
            candidates = [gt_dir / f"{name}.npy"]
            if stem:
                candidates.append(gt_dir / f"{stem}.npy")
            found = next((p for p in candidates if p.exists()), None)
            if found is not None:
                out[str(name)] = found
                out[_safe_name(str(name))] = found
                if stem:
                    out[stem] = found
                    out[_safe_name(stem)] = found
    for path in gt_dir.glob("*.npy"):
        out.setdefault(path.stem, path)
        out.setdefault(_safe_name(path.stem), path)
    return out


def _find_pred_files(pred_dir: Path, suffix: str) -> list[Path]:
    return sorted(p for p in pred_dir.glob(f"*{suffix}") if p.is_file())


def _mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _percentile_or_nan(values: Iterable[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(vals, q)) if vals else float("nan")


def _recover_joints_hml263(arr: np.ndarray) -> np.ndarray:
    from rfmotion.data.humanml.scripts.motion_process import recover_from_ric  # noqa: WPS433

    x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        joints = recover_from_ric(x, 22)
    return joints.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _stats_from_hml263(pred_dir: Path, gt_dir: Path, anno_file: Path | None, cond: int, limit: int):
    gt_map = _build_gt_map(anno_file, gt_dir)
    rows = []
    for pred_path in _find_pred_files(pred_dir, ".npy"):
        sid = pred_path.stem
        gt_path = gt_map.get(sid)
        if gt_path is None:
            continue
        pred = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        t = min(pred.shape[0], gt.shape[0])
        n = min(cond, t)
        if n <= 0:
            continue
        pred = pred[:t]
        gt = gt[:t]
        pj = _recover_joints_hml263(pred[: max(t, n)])
        gj = _recover_joints_hml263(gt[: max(t, n)])
        root = slice(0, 3)
        rows.append({
            "sid": sid,
            "prefix_feat_mae": float(np.mean(np.abs(pred[:n] - gt[:n]))),
            "prefix_feat_rmse": float(np.sqrt(np.mean((pred[:n] - gt[:n]) ** 2))),
            "prefix_root_mae": float(np.mean(np.abs(pred[:n, root] - gt[:n, root]))),
            "prefix_joint_mpjpe": float(np.mean(np.linalg.norm(pj[:n] - gj[:n], axis=-1))),
            "after_joint_mpjpe": float(np.mean(np.linalg.norm(pj[n:] - gj[n:], axis=-1))) if t > n else float("nan"),
            "pred_root_span": float(np.linalg.norm(pred[:t, root].max(axis=0) - pred[:t, root].min(axis=0))),
            "gt_root_span": float(np.linalg.norm(gt[:t, root].max(axis=0) - gt[:t, root].min(axis=0))),
            "length": int(t),
        })
        if limit and len(rows) >= limit:
            break
    return rows


def _stats_from_ms272(pred_dir: Path, gt_dir: Path | None, cond: int, limit: int):
    rows = []
    for pred_path in _find_pred_files(pred_dir, ".npz"):
        data = np.load(pred_path, allow_pickle=True)
        if "motion_272" not in data:
            continue
        gt_path = None
        if "gt272_path" in data:
            raw = str(data["gt272_path"])
            if raw:
                gt_path = Path(raw)
        if (gt_path is None or not gt_path.exists()) and gt_dir is not None:
            cand = gt_dir / f"{pred_path.stem}.npy"
            if cand.exists():
                gt_path = cand
        if gt_path is None or not gt_path.exists():
            continue
        pred = np.asarray(data["motion_272"], dtype=np.float32)
        gt = np.load(gt_path).astype(np.float32)
        t = min(pred.shape[0], gt.shape[0])
        n = min(cond, t)
        if n <= 0:
            continue
        pred = pred[:t]
        gt = gt[:t]
        root = slice(0, 3)
        rows.append({
            "sid": pred_path.stem,
            "prefix_feat_mae": float(np.mean(np.abs(pred[:n] - gt[:n]))),
            "prefix_feat_rmse": float(np.sqrt(np.mean((pred[:n] - gt[:n]) ** 2))),
            "prefix_root_mae": float(np.mean(np.abs(pred[:n, root] - gt[:n, root]))),
            "prefix_joint_mpjpe": float("nan"),
            "after_joint_mpjpe": float("nan"),
            "pred_root_span": float(np.linalg.norm(pred[:t, root].max(axis=0) - pred[:t, root].min(axis=0))),
            "gt_root_span": float(np.linalg.norm(gt[:t, root].max(axis=0) - gt[:t, root].min(axis=0))),
            "length": int(t),
        })
        if limit and len(rows) >= limit:
            break
    return rows


def _summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    keys = [
        "prefix_feat_mae",
        "prefix_feat_rmse",
        "prefix_root_mae",
        "prefix_joint_mpjpe",
        "after_joint_mpjpe",
        "pred_root_span",
        "gt_root_span",
    ]
    out = {
        "name": args.name,
        "format": args.format,
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir) if args.gt_dir else None,
        "anno_file": str(args.anno_file) if args.anno_file else None,
        "condition_num_frames": args.condition_num_frames,
        "samples": len(rows),
    }
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        out[f"{key}_mean"] = _mean_or_nan(vals)
        out[f"{key}_p50"] = _percentile_or_nan(vals, 50)
        out[f"{key}_p95"] = _percentile_or_nan(vals, 95)
    out["examples"] = rows[: min(5, len(rows))]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--format", choices=["hml263", "ms272_npz"], required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--anno-file", type=Path, default=None)
    parser.add_argument("--condition-num-frames", type=int, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.format == "hml263":
        if args.gt_dir is None:
            raise ValueError("--gt-dir is required for hml263 diagnostics")
        rows = _stats_from_hml263(args.pred_dir, args.gt_dir, args.anno_file, args.condition_num_frames, args.limit)
    else:
        rows = _stats_from_ms272(args.pred_dir, args.gt_dir, args.condition_num_frames, args.limit)
    summary = _summarize(rows, args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")


if __name__ == "__main__":
    main()

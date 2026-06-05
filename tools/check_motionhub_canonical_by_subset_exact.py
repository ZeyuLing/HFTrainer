#!/usr/bin/env python3
"""Check MotionHub floor canonicalization by subset with exact body-model FK."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSATILE_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT), "versatilemotion"
)
if VERSATILE_ROOT not in sys.path:
    sys.path.insert(0, VERSATILE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def iter_annotation(anno_path: str) -> Iterable[Dict[str, Any]]:
    with open(anno_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data_list", obj) if isinstance(obj, dict) else obj
    if isinstance(data, dict):
        yield from data.values()
    else:
        yield from data


def collect_paths(anno_path: str) -> Dict[str, List[str]]:
    by_subset: Dict[str, List[str]] = defaultdict(list)
    for row in iter_annotation(anno_path):
        value = row.get("smplx_path") or row.get("motion_path")
        paths = value if isinstance(value, list) else [value]
        for path in paths:
            if not path or not isinstance(path, str):
                continue
            if path.startswith("../") or path.startswith("/"):
                continue
            subset = path.split("/", 1)[0]
            by_subset[subset].append(path)
    return dict(by_subset)


def as_repeated(arr: Any, frames: int, width: int, default: float = 0.0) -> np.ndarray:
    if arr is None:
        return np.full((frames, width), default, dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out[None, :]
    if out.shape[0] == 1 and frames > 1:
        out = np.repeat(out, frames, axis=0)
    if out.shape[0] != frames:
        out = out[:frames]
        if out.shape[0] < frames:
            pad = np.repeat(out[-1:], frames - out.shape[0], axis=0)
            out = np.concatenate([out, pad], axis=0)
    if out.shape[1] < width:
        pad = np.zeros((frames, width - out.shape[1]), dtype=np.float32)
        out = np.concatenate([out, pad], axis=1)
    return out[:, :width].astype(np.float32, copy=False)


def smplx_components(data: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    if "global_orient" in data:
        transl = np.asarray(data.get("transl", data.get("trans")), dtype=np.float32)
        frames = transl.shape[0]
        return (
            transl,
            as_repeated(data.get("global_orient"), frames, 3),
            as_repeated(data.get("body_pose"), frames, 63),
            as_repeated(data.get("jaw_pose"), frames, 3),
            as_repeated(data.get("leye_pose"), frames, 3),
            as_repeated(data.get("reye_pose"), frames, 3),
            as_repeated(data.get("left_hand_pose"), frames, 45),
            as_repeated(data.get("right_hand_pose"), frames, 45),
            as_repeated(data.get("betas"), frames, 10),
        )
    poses = np.asarray(data["poses"], dtype=np.float32)
    if poses.shape[1] != 165:
        raise ValueError(f"expected SMPL-X poses shape (T,165), got {poses.shape}")
    transl = np.asarray(data.get("transl", data.get("trans")), dtype=np.float32)
    frames = transl.shape[0]
    return (
        transl,
        poses[:, 0:3],
        poses[:, 3:66],
        poses[:, 66:69],
        poses[:, 69:72],
        poses[:, 72:75],
        poses[:, 75:120],
        poses[:, 120:165],
        as_repeated(data.get("betas"), frames, 10),
    )


def smplh_components(data: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    poses = np.asarray(data["poses"], dtype=np.float32)
    if poses.shape[1] != 156:
        raise ValueError(f"expected SMPL-H poses shape (T,156), got {poses.shape}")
    transl = np.asarray(data.get("trans", data.get("transl")), dtype=np.float32)
    frames = transl.shape[0]
    return (
        transl,
        poses[:, 0:3],
        poses[:, 3:66],
        poses[:, 66:111],
        poses[:, 111:156],
        as_repeated(data.get("betas"), frames, 16),
    )


def check_smplx(data: Dict[str, Any], model, device, chunk: int) -> Dict[str, Any]:
    transl, go, bp, jaw, leye, reye, lh, rh, betas = smplx_components(data)
    all_mins: List[float] = []
    foot_mins: List[float] = []
    foot_maxs: List[float] = []
    for start in range(0, transl.shape[0], chunk):
        sl = slice(start, start + chunk)
        kwargs = {
            "transl": torch.from_numpy(transl[sl]).to(device)[None],
            "global_orient": torch.from_numpy(go[sl]).to(device)[None],
            "body_pose": torch.from_numpy(bp[sl]).to(device)[None],
            "jaw_pose": torch.from_numpy(jaw[sl]).to(device)[None],
            "leye_pose": torch.from_numpy(leye[sl]).to(device)[None],
            "reye_pose": torch.from_numpy(reye[sl]).to(device)[None],
            "left_hand_pose": torch.from_numpy(lh[sl]).to(device)[None],
            "right_hand_pose": torch.from_numpy(rh[sl]).to(device)[None],
            "betas": torch.from_numpy(betas[sl]).to(device)[None],
        }
        with torch.no_grad():
            joints, _, _ = model.fk(**kwargs)
        j = joints[0].detach().cpu().numpy()
        feet = j[:, [7, 8, 10, 11], :]
        all_mins.append(float(j[:, :, 1].min()))
        foot_mins.append(float(feet[:, :, 1].min()))
        foot_maxs.append(float(feet[:, :, 1].max()))
    return {
        "frames": int(transl.shape[0]),
        "model": "smplx55",
        "all_min_y": float(min(all_mins)),
        "foot_min_y": float(min(foot_mins)),
        "foot_max_y": float(max(foot_maxs)),
    }


def check_smplh(data: Dict[str, Any], model, device, chunk: int) -> Dict[str, Any]:
    transl, go, bp, lh, rh, betas = smplh_components(data)
    mins: List[float] = []
    for start in range(0, transl.shape[0], chunk):
        sl = slice(start, start + chunk)
        with torch.no_grad():
            joints = model(
                body_pose=torch.from_numpy(bp[sl]).to(device),
                left_hand_pose=torch.from_numpy(lh[sl]).to(device),
                right_hand_pose=torch.from_numpy(rh[sl]).to(device),
                betas=torch.from_numpy(betas[sl]).to(device),
                global_orient=torch.from_numpy(go[sl]).to(device),
                transl=torch.from_numpy(transl[sl]).to(device),
                rotation_mode="aa",
            )
        mins.append(float(joints.detach().cpu().numpy()[:, :, 1].min()))
    min_y = float(min(mins))
    return {
        "frames": int(transl.shape[0]),
        "model": "smplh24",
        "all_min_y": min_y,
        "foot_min_y": min_y,
        "foot_max_y": float("nan"),
    }


def summarize(rows: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    by_subset: Dict[str, Dict[str, Any]] = {}
    for subset in sorted({r["subset"] for r in rows}):
        group = [r for r in rows if r["subset"] == subset]
        ok = [r for r in group if r.get("ok")]
        errors = len(group) - len(ok)
        vals = np.array([r["all_min_y"] for r in ok], dtype=np.float64)
        foot = np.array([r["foot_min_y"] for r in ok], dtype=np.float64)
        if len(ok):
            by_subset[subset] = {
                "checked": len(group),
                "ok": len(ok),
                "errors": errors,
                "all_near": int(np.sum(np.abs(vals) <= tolerance)),
                "all_floating": int(np.sum(vals > tolerance)),
                "all_penetrating": int(np.sum(vals < -tolerance)),
                "all_min": float(vals.min()),
                "all_max": float(vals.max()),
                "all_mean": float(vals.mean()),
                "foot_near": int(np.sum(np.abs(foot) <= tolerance)),
                "foot_floating": int(np.sum(foot > tolerance)),
                "foot_penetrating": int(np.sum(foot < -tolerance)),
                "foot_min": float(foot.min()),
                "foot_max": float(foot.max()),
                "foot_mean": float(foot.mean()),
            }
        else:
            by_subset[subset] = {
                "checked": len(group),
                "ok": 0,
                "errors": errors,
            }
    return by_subset


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "ok", "subset", "path", "model", "frames", "all_min_y",
        "foot_min_y", "foot_max_y", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", default="data/annotation/train_hq_motionhub_hymotion.json")
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--per-subset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    from mmotion.models.body_models.smplx_lite import SmplxLiteJ24, SmplxLiteV437Coco17

    rng = random.Random(args.seed)
    by_subset = collect_paths(args.anno)
    selected: List[Tuple[str, str]] = []
    counts = {k: len(v) for k, v in by_subset.items()}
    for subset, paths in sorted(by_subset.items()):
        picks = paths if args.per_subset <= 0 else rng.sample(paths, min(args.per_subset, len(paths)))
        selected.extend((subset, p) for p in picks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = SmplxLiteV437Coco17().to(device=device, dtype=torch.float32).eval()
    smplh = SmplxLiteJ24(
        model_path=os.path.join(VERSATILE_ROOT, "checkpoints/smpl_models/smplh"),
        gender="neutral",
    ).to(device=device, dtype=torch.float32).eval()

    rows: List[Dict[str, Any]] = []
    for idx, (subset, rel_path) in enumerate(selected, start=1):
        full_path = os.path.join(args.data_dir, rel_path)
        row: Dict[str, Any] = {
            "ok": False,
            "subset": subset,
            "path": rel_path,
            "error": "",
        }
        try:
            data = dict(np.load(full_path, allow_pickle=True))
            poses = np.asarray(data["poses"], dtype=np.float32)
            if poses.shape[1] == 156:
                row.update(check_smplh(data, smplh, device, args.chunk))
            else:
                row.update(check_smplx(data, smplx, device, args.chunk))
            row["ok"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if idx % 50 == 0:
            print(f"[canonical] checked {idx}/{len(selected)}", flush=True)

    write_csv(args.csv, rows)
    print(json.dumps({
        "anno": args.anno,
        "data_dir": args.data_dir,
        "per_subset": args.per_subset,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "subset_counts": counts,
        "summary": summarize(rows, args.tolerance),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

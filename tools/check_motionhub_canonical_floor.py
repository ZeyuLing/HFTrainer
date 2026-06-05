#!/usr/bin/env python3
"""Check MotionHub canonical floor via SMPL-22 FK minimum joint y."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk


def read_scalar(value: Any, default: str = "") -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        return str(arr.tolist())
    except Exception:
        return default


def motion135_column_to_row(motion135: np.ndarray) -> np.ndarray:
    motion135 = np.asarray(motion135, dtype=np.float32).copy()
    rot = motion135[..., 3:135].reshape(*motion135.shape[:-1], 22, 6)
    motion135[..., 3:135] = rot[..., [0, 3, 1, 4, 2, 5]].reshape(*motion135.shape[:-1], 132)
    return motion135


def resolve_path(data_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(data_dir, path))


def load_motion135_from_npz(path: str) -> Tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    if "motion_135" in data.files:
        motion = np.asarray(data["motion_135"], dtype=np.float32)
        convention = read_scalar(data["rot6d_convention"], "") if "rot6d_convention" in data.files else ""
        if convention == "column":
            motion = motion135_column_to_row(motion)
        return motion[..., :135], "motion_135"
    if "trans" not in data.files or "poses" not in data.files:
        raise KeyError(f"{path} has neither motion_135 nor trans+poses")
    trans = np.asarray(data["trans"], dtype=np.float32)
    poses = np.asarray(data["poses"], dtype=np.float32)
    pose22 = process_smplx_pose(
        poses,
        rot_type="rotation_6d",
        out_type="smpl_22",
        rot6d_convention="row",
    )
    return np.concatenate([trans[:, :3], pose22], axis=-1).astype(np.float32), "trans_poses"


def floor_stats(motion135: np.ndarray, bone_offsets: torch.Tensor, chunk_frames: int = 4096) -> Dict[str, Any]:
    mins: List[float] = []
    arg_frame = 0
    arg_joint = 0
    best = math.inf
    for start in range(0, motion135.shape[0], chunk_frames):
        chunk = torch.from_numpy(np.asarray(motion135[start:start + chunk_frames], dtype=np.float32))
        with torch.no_grad():
            pos, _, _, _ = motion135_to_fk(chunk, bone_offsets, rotation_space="local")
        y = pos[..., 1]
        min_val = float(y.min().item())
        mins.append(min_val)
        if min_val < best:
            flat = int(y.argmin().item())
            frame = flat // y.shape[1]
            joint = flat % y.shape[1]
            best = min_val
            arg_frame = start + frame
            arg_joint = joint
    return {
        "min_y": best,
        "argmin_frame": int(arg_frame),
        "argmin_joint": int(arg_joint),
    }


def iter_annotation(anno_path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with open(anno_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data_list", obj) if isinstance(obj, dict) else obj
    if isinstance(data, dict):
        yield from data.items()
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            yield str(row.get("id", idx) if isinstance(row, dict) else idx), row
    else:
        raise TypeError(f"Unsupported annotation data_list type: {type(data)}")


def iter_viewer_manifest(manifest_path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    root = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for case in manifest.get("cases", []):
        for group in ("inputs", "targets", "predictions"):
            for item in case.get(group, []):
                if item.get("kind") != "motion" or not item.get("path"):
                    continue
                yield f"{case.get('case_id')}::{group}::{item.get('label') or item.get('modal')}", {
                    "subset": case.get("task", ""),
                    "smplx_path": os.path.join(root, item["path"]),
                    "viewer_group": group,
                    "viewer_case": case.get("case_id"),
                    "dataset_idx": case.get("dataset_idx"),
                }


def motion_paths(row: Dict[str, Any]) -> List[str]:
    for key in ("smplx_path", "motion_path"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]
    return []


def summarize(records: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    valid = [r for r in records if r.get("ok")]
    floaters = [r for r in valid if r["min_y"] > tolerance]
    penetrations = [r for r in valid if r["min_y"] < -tolerance]
    near = [r for r in valid if abs(r["min_y"]) <= tolerance]
    by_subset = defaultdict(lambda: Counter(total=0, float=0, penetrate=0, near=0))
    for r in valid:
        subset = r.get("subset") or "unknown"
        by_subset[subset]["total"] += 1
        if r["min_y"] > tolerance:
            by_subset[subset]["float"] += 1
        elif r["min_y"] < -tolerance:
            by_subset[subset]["penetrate"] += 1
        else:
            by_subset[subset]["near"] += 1
    min_values = [r["min_y"] for r in valid]
    return {
        "checked": len(records),
        "valid": len(valid),
        "errors": len(records) - len(valid),
        "near_floor": len(near),
        "floating": len(floaters),
        "penetrating": len(penetrations),
        "min_y_mean": float(np.mean(min_values)) if min_values else None,
        "min_y_min": float(np.min(min_values)) if min_values else None,
        "min_y_max": float(np.max(min_values)) if min_values else None,
        "by_subset": {k: dict(v) for k, v in sorted(by_subset.items())},
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    fields = [
        "ok", "key", "person", "subset", "path", "source", "frames",
        "min_y", "argmin_frame", "argmin_joint", "error",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--anno", help="MotionHub-style annotation JSON")
    source.add_argument("--viewer-manifest", help="VerMo viewer manifest JSON")
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    bone_offsets = torch.load(args.bone_offsets, map_location="cpu").float()
    iterator: Iterable[Tuple[str, Dict[str, Any]]]
    if args.anno:
        iterator = iter_annotation(args.anno)
        source_name = args.anno
    else:
        iterator = iter_viewer_manifest(args.viewer_manifest)
        source_name = args.viewer_manifest

    rows: List[Dict[str, Any]] = []
    seen = 0
    selected = 0
    for key, row in iterator:
        seen += 1
        if args.stride > 1 and (seen - 1) % args.stride != 0:
            continue
        paths = motion_paths(row)
        if not paths:
            continue
        for person_idx, rel_path in enumerate(paths):
            if args.limit and selected >= args.limit:
                break
            selected += 1
            path = resolve_path(args.data_dir, rel_path)
            rec = {
                "ok": False,
                "key": key,
                "person": person_idx,
                "subset": row.get("subset", ""),
                "path": path,
                "error": "",
            }
            try:
                motion, src = load_motion135_from_npz(path)
                stats = floor_stats(motion, bone_offsets)
                rec.update({
                    "ok": True,
                    "source": src,
                    "frames": int(motion.shape[0]),
                    **stats,
                })
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)
        if args.limit and selected >= args.limit:
            break
        if selected and selected % 500 == 0:
            print(f"[floor] selected={selected} seen={seen}", flush=True)

    summary = summarize(rows, args.tolerance)
    write_csv(args.csv, rows)
    print(json.dumps({
        "source": source_name,
        "data_dir": args.data_dir,
        "tolerance": args.tolerance,
        "summary": summary,
    }, ensure_ascii=False, indent=2), flush=True)

    valid = [r for r in rows if r.get("ok")]
    print("\nTop floating min_y:", flush=True)
    for r in sorted(valid, key=lambda x: x["min_y"], reverse=True)[:args.topk]:
        print(f"{r['min_y']:+.4f} {r['subset']} {r['key']} person={r['person']} frames={r['frames']} {r['path']}", flush=True)
    print("\nTop penetrating min_y:", flush=True)
    for r in sorted(valid, key=lambda x: x["min_y"])[:args.topk]:
        print(f"{r['min_y']:+.4f} {r['subset']} {r['key']} person={r['person']} frames={r['frames']} {r['path']}", flush=True)

    errors = [r for r in rows if not r.get("ok")]
    if errors:
        print("\nErrors:", flush=True)
        for r in errors[:args.topk]:
            print(f"{r['key']} {r['path']} {r['error']}", flush=True)


if __name__ == "__main__":
    main()

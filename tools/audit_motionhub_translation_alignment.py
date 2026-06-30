#!/usr/bin/env python3
"""Audit MotionHub/HYMotion SMPL translation convention and floor alignment.

The training convention for this repository is HYMotion-compatible
body-model translation:

    vertices_world = smpl_vertices_without_translation + transl

The Three.js viewer must pass ``Th = transl + shaped_rest_root_joint`` because
its root bone is bound at the shaped rest root.  Dataset files themselves must
still store the raw body-model ``trans``/``transl`` value.

This script is intentionally read-only.  It classifies each file by checking
mesh min-y under two hypotheses:

* ``model_trans``: the stored translation is already body-model translation.
* ``root_world``: the stored translation is a root/pelvis world position and
  should be converted to body-model translation by subtracting the shaped root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSATILE_ROOT = PROJECT_ROOT.parent / "versatilemotion"
if str(VERSATILE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSATILE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_annotation_rows(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    raw = read_json(path)
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            yield str(row.get("id", idx) if isinstance(row, dict) else idx), row
    else:
        raise TypeError(f"unsupported annotation type in {path}: {type(data).__name__}")


def resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Avoid Path.resolve here: annotation inventories can contain hundreds of
    # thousands of CEPH paths, and resolving every symlink before sampling is
    # prohibitively slow. Normalize lexically; selected files are validated
    # later when they are actually loaded.
    return Path(os.path.abspath(os.path.normpath(str(data_root / path))))


def collect_paths_from_annotation(
    anno_path: Path,
    data_root: Path,
    include_external: bool,
    subsets: Optional[set[str]],
) -> Dict[str, List[Tuple[str, Path]]]:
    by_subset: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for key, row in iter_annotation_rows(anno_path):
        value = row.get("smplx_path") or row.get("motion_path")
        if value is None:
            continue
        paths = value if isinstance(value, list) else [value]
        for person_idx, rel in enumerate(paths):
            if not rel or not isinstance(rel, str):
                continue
            if rel.startswith("../") or rel.startswith("/"):
                if not include_external:
                    continue
                subset = row.get("subset") or "external"
            else:
                subset = rel.split("/", 1)[0]
            if subsets and subset not in subsets:
                continue
            by_subset[str(subset)].append((f"{key}#{person_idx}", resolve_path(data_root, rel)))
    return dict(by_subset)


def collect_paths_from_subset_roots(subset_roots: List[Path], motion_dir: str) -> Dict[str, List[Tuple[str, Path]]]:
    by_subset: Dict[str, List[Tuple[str, Path]]] = {}
    for root in subset_roots:
        paths = sorted((root / motion_dir).glob("*.npz"))
        by_subset[root.name] = [(path.stem, path.resolve()) for path in paths]
    return by_subset


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
        if out.shape[0] < frames and out.shape[0] > 0:
            out = np.concatenate([out, np.repeat(out[-1:], frames - out.shape[0], axis=0)], axis=0)
    if out.shape[1] < width:
        out = np.concatenate([out, np.zeros((frames, width - out.shape[1]), dtype=np.float32)], axis=1)
    return out[:, :width].astype(np.float32, copy=False)


def load_components(path: Path) -> Dict[str, Any]:
    data = dict(np.load(path, allow_pickle=True))
    if "poses" not in data:
        raise KeyError("missing poses")
    poses = np.asarray(data["poses"], dtype=np.float32)
    transl = np.asarray(data.get("transl", data.get("trans")), dtype=np.float32)
    if transl.ndim != 2 or transl.shape[1] < 3:
        raise ValueError(f"bad transl shape {transl.shape}")
    transl = transl[:, :3]
    frames = transl.shape[0]
    if poses.shape[0] != frames:
        poses = poses[:frames]

    if poses.shape[1] == 165:
        return {
            "smpl_type": "smplx",
            "frames": frames,
            "transl": transl,
            "global_orient": poses[:, :3],
            "body_pose": poses[:, 3:66],
            "jaw_pose": poses[:, 66:69],
            "leye_pose": poses[:, 69:72],
            "reye_pose": poses[:, 72:75],
            "left_hand_pose": poses[:, 75:120],
            "right_hand_pose": poses[:, 120:165],
            "betas": as_repeated(data.get("betas"), frames, 10),
        }
    if poses.shape[1] == 156:
        return {
            "smpl_type": "smplh",
            "frames": frames,
            "transl": transl,
            "global_orient": poses[:, :3],
            "body_pose": poses[:, 3:66],
            "left_hand_pose": poses[:, 66:111],
            "right_hand_pose": poses[:, 111:156],
            "betas": as_repeated(data.get("betas"), frames, 16),
        }
    raise ValueError(f"unsupported poses shape {poses.shape}")


def shaped_root_from_model(model: torch.nn.Module, comp: Dict[str, Any], device: torch.device) -> np.ndarray:
    betas = torch.from_numpy(comp["betas"]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if comp["smpl_type"] == "smplx":
            root = model.get_skeleton(betas)[:, 0, :]
        else:
            root = model.get_skeleton(betas)[:, 0, :]
    return root.detach().cpu().numpy().astype(np.float32)


def mesh_stats_for_hypothesis(
    model: torch.nn.Module,
    comp: Dict[str, Any],
    transl: np.ndarray,
    device: torch.device,
    chunk: int,
    frame_stride: int,
    max_frames: int,
) -> Dict[str, float]:
    min_values: List[float] = []
    frame_min_values: List[float] = []
    joint_min_values: List[float] = []
    frames = comp["frames"]
    frame_idx = np.arange(frames, dtype=np.int64)
    if frame_stride > 1:
        frame_idx = frame_idx[::frame_stride]
    if max_frames > 0 and frame_idx.shape[0] > max_frames:
        frame_idx = np.linspace(0, frames - 1, max_frames, dtype=np.int64)
    if frame_idx.shape[0] == 0:
        frame_idx = np.array([0], dtype=np.int64)

    for start in range(0, frame_idx.shape[0], chunk):
        idx = frame_idx[start : start + chunk]
        t = torch.from_numpy(transl[idx]).to(device=device, dtype=torch.float32)
        betas = torch.from_numpy(comp["betas"][idx]).to(device=device, dtype=torch.float32)
        go = torch.from_numpy(comp["global_orient"][idx]).to(device=device, dtype=torch.float32)
        bp = torch.from_numpy(comp["body_pose"][idx]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            if comp["smpl_type"] == "smplx":
                verts = model(
                    body_pose=bp,
                    betas=betas,
                    global_orient=go,
                    transl=t,
                )
                joints = model.get_skeleton(betas) + t[:, None, :]
            else:
                body_pose = torch.cat(
                    [
                        bp,
                        torch.from_numpy(comp["left_hand_pose"][idx]).to(device=device, dtype=torch.float32),
                        torch.from_numpy(comp["right_hand_pose"][idx]).to(device=device, dtype=torch.float32),
                    ],
                    dim=-1,
                )
                verts = model(
                    body_pose=body_pose,
                    betas=betas,
                    global_orient=go,
                    transl=t,
                    rotation_mode="aa",
                )
                joints = model.get_skeleton(betas) + t[:, None, :]
        mesh_y = verts[..., 1]
        joint_y = joints[..., 1]
        min_values.append(float(mesh_y.min().item()))
        frame_min_values.extend(mesh_y.min(dim=-1).values.detach().cpu().numpy().astype(np.float64).tolist())
        joint_min_values.append(float(joint_y.min().item()))
    return {
        "mesh_min_y": float(min(min_values)),
        "mesh_frame_min_y_mean": float(np.mean(frame_min_values)),
        "joint_min_y": float(min(joint_min_values)),
    }


def classify(model_min: float, root_min: float, tolerance: float) -> str:
    model_ok = abs(model_min) <= tolerance
    root_ok = abs(root_min) <= tolerance
    if model_ok and not root_ok:
        return "already_model_trans"
    if root_ok and not model_ok:
        return "root_world_encoded"
    if model_ok and root_ok:
        return "ambiguous_both_near_floor"
    if abs(model_min) < abs(root_min):
        return "floor_residual_abnormal_model_trans"
    if abs(root_min) < abs(model_min):
        return "floor_residual_abnormal_root_world"
    return "error_or_ambiguous"


def audit_one(
    key: str,
    subset: str,
    path: Path,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    chunk: int,
    tolerance: float,
    frame_stride: int,
    max_frames: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ok": False,
        "key": key,
        "subset": subset,
        "path": str(path),
        "error": "",
    }
    try:
        comp = load_components(path)
        model = models[comp["smpl_type"]]
        root = shaped_root_from_model(model, comp, device)
        stored = comp["transl"]
        model_trans_stats = mesh_stats_for_hypothesis(
            model, comp, stored, device, chunk, frame_stride, max_frames
        )
        root_world_stats = mesh_stats_for_hypothesis(
            model, comp, stored - root, device, chunk, frame_stride, max_frames
        )
        cls = classify(
            model_trans_stats["mesh_min_y"],
            root_world_stats["mesh_min_y"],
            tolerance,
        )
        row.update({
            "ok": True,
            "smpl_type": comp["smpl_type"],
            "frames": int(comp["frames"]),
            "classification": cls,
            "stored_transl_y_mean": float(stored[:, 1].mean()),
            "stored_transl_y_min": float(stored[:, 1].min()),
            "stored_transl_y_max": float(stored[:, 1].max()),
            "shaped_root_y_mean": float(root[:, 1].mean()),
            "model_trans_mesh_min_y": model_trans_stats["mesh_min_y"],
            "model_trans_mesh_frame_min_y_mean": model_trans_stats["mesh_frame_min_y_mean"],
            "model_trans_joint_min_y": model_trans_stats["joint_min_y"],
            "root_world_mesh_min_y": root_world_stats["mesh_min_y"],
            "root_world_mesh_frame_min_y_mean": root_world_stats["mesh_frame_min_y_mean"],
            "root_world_joint_min_y": root_world_stats["joint_min_y"],
            "suggested_model_trans_y_shift": float(-model_trans_stats["mesh_min_y"]),
        })
    except Exception as exc:
        row["classification"] = "error_or_ambiguous"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "ok", "classification", "subset", "key", "path", "smpl_type", "frames",
        "stored_transl_y_mean", "stored_transl_y_min", "stored_transl_y_max",
        "shaped_root_y_mean", "model_trans_mesh_min_y",
        "model_trans_mesh_frame_min_y_mean", "model_trans_joint_min_y",
        "root_world_mesh_min_y", "root_world_mesh_frame_min_y_mean",
        "root_world_joint_min_y", "suggested_model_trans_y_shift", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_subset: Dict[str, Any] = {}
    for subset in sorted({row["subset"] for row in rows}):
        group = [row for row in rows if row["subset"] == subset]
        ok = [row for row in group if row.get("ok")]
        cls = Counter(row.get("classification", "unknown") for row in group)
        mesh_vals = np.array([row["model_trans_mesh_min_y"] for row in ok], dtype=np.float64)
        by_subset[subset] = {
            "checked": len(group),
            "ok": len(ok),
            "errors": len(group) - len(ok),
            "classification_counts": dict(cls),
            "model_trans_mesh_min_y_min": float(mesh_vals.min()) if mesh_vals.size else None,
            "model_trans_mesh_min_y_max": float(mesh_vals.max()) if mesh_vals.size else None,
            "model_trans_mesh_min_y_mean": float(mesh_vals.mean()) if mesh_vals.size else None,
        }
    return by_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--anno", help="MotionHub-style annotation JSON")
    src.add_argument("--subset-root", action="append", default=[], help="Subset root; may repeat")
    parser.add_argument("--data-root", default="data/motionhub")
    parser.add_argument("--motion-dir", default="smplx_55")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--subsets", default="", help="Comma-separated subset filter for --anno")
    parser.add_argument("--per-subset", type=int, default=0, help="0 means all selected paths")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--report", required=True)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    from mmotion.models.body_models.smplx_lite import SmplLite, SmplxLite

    data_root = Path(args.data_root).resolve()
    subset_filter = {x.strip() for x in args.subsets.split(",") if x.strip()} or None
    if args.anno:
        grouped = collect_paths_from_annotation(
            Path(args.anno), data_root, args.include_external, subset_filter
        )
    else:
        grouped = collect_paths_from_subset_roots([Path(p) for p in args.subset_root], args.motion_dir)

    rng = random.Random(args.seed)
    selected: List[Tuple[str, str, Path]] = []
    for subset, rows in sorted(grouped.items()):
        picks = list(rows)
        if args.per_subset > 0 and len(picks) > args.per_subset:
            picks = rng.sample(picks, args.per_subset)
        selected.extend((key, subset, path) for key, path in picks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "smplx": SmplxLite(
            model_path=str(VERSATILE_ROOT / "checkpoints/smpl_models/smplx"),
            gender="neutral",
            num_betas=10,
        ).to(device=device, dtype=torch.float32).eval(),
        "smplh": SmplLite(
            model_path=str(VERSATILE_ROOT / "checkpoints/smpl_models/smplh"),
            gender="neutral",
            num_betas=16,
        ).to(device=device, dtype=torch.float32).eval(),
    }

    rows: List[Dict[str, Any]] = []
    for idx, (key, subset, path) in enumerate(selected, start=1):
        rows.append(
            audit_one(
                key,
                subset,
                path,
                models,
                device,
                args.chunk,
                args.tolerance,
                max(1, args.frame_stride),
                max(0, args.max_frames),
            )
        )
        if idx % 25 == 0:
            print(f"[audit] {idx}/{len(selected)}", flush=True)

    payload = {
        "meta": {
            "data_root": str(data_root),
            "source": args.anno or args.subset_root,
            "include_external": args.include_external,
            "subsets": sorted(grouped),
            "per_subset": args.per_subset,
            "seed": args.seed,
            "tolerance": args.tolerance,
            "chunk": args.chunk,
            "frame_stride": max(1, args.frame_stride),
            "max_frames": max(0, args.max_frames),
            "device": str(device),
            "num_rows": len(rows),
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.csv:
        write_csv(Path(args.csv), rows)
    print(json.dumps({"meta": payload["meta"], "summary": payload["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

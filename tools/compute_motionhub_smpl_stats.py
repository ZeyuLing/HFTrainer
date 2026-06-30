#!/usr/bin/env python3
"""Compute reusable MotionHub SMPL/SMPL-X normalization statistics.

The output schema is consumed by ``SMPLPoseProcessor``:

    transl, transl_vel, global_orient, body_pose, left/right_hand_pose,
    jaw_pose, leye_pose, reye_pose, betas, expression

Rotation blocks include canonical keys (``rotation_6d`` and ``matrix``) plus
legacy aliases (``rot6d`` and ``rotmat``) so old stats readers keep working.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


POSE_BLOCKS: Tuple[Tuple[str, int], ...] = (
    ("global_orient", 1),
    ("body_pose", 21),
    ("jaw_pose", 1),
    ("leye_pose", 1),
    ("reye_pose", 1),
    ("left_hand_pose", 15),
    ("right_hand_pose", 15),
)

POSE_SLICES = {
    "global_orient": (0, 3),
    "body_pose": (3, 66),
    "jaw_pose": (66, 69),
    "leye_pose": (69, 72),
    "reye_pose": (72, 75),
    "left_hand_pose": (75, 120),
    "right_hand_pose": (120, 165),
}

DIRECT_BLOCKS: Tuple[Tuple[str, int], ...] = (
    ("transl", 3),
    ("transl_vel", 3),
    ("betas", 10),
    ("expression", 10),
)

ROTATION_KEYS = ("axis_angle", "quaternion", "rotation_6d", "matrix")
LEGACY_ROTATION_ALIASES = {
    "rotation_6d": "rot6d",
    "matrix": "rotmat",
}


@dataclass
class RunningStats:
    width: int
    count: int = 0
    total: Optional[np.ndarray] = None
    total_sq: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.total = np.zeros(self.width, dtype=np.float64)
        self.total_sq = np.zeros(self.width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError(f"expected (*, {self.width}), got {values.shape}")
        if values.shape[0] == 0:
            return
        self.count += int(values.shape[0])
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)

    def as_dict(self) -> Dict[str, Any]:
        if self.count <= 0:
            mean = np.zeros(self.width, dtype=np.float64)
            std = np.zeros(self.width, dtype=np.float64)
        else:
            mean = self.total / float(self.count)
            var = self.total_sq / float(self.count) - np.square(mean)
            std = np.sqrt(np.maximum(var, 0.0))
        return {
            "count_frames": int(self.count),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


def _as_repeated(arr: Any, frames: int, width: int) -> np.ndarray:
    if arr is None:
        return np.zeros((frames, width), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out[None, :]
    if out.shape[0] == 1 and frames > 1:
        out = np.repeat(out, frames, axis=0)
    if out.shape[0] != frames:
        if out.shape[0] > frames:
            out = out[:frames]
        else:
            pad = np.repeat(out[-1:], frames - out.shape[0], axis=0)
            out = np.concatenate([out, pad], axis=0)
    if out.shape[1] < width:
        pad = np.zeros((frames, width - out.shape[1]), dtype=np.float32)
        out = np.concatenate([out, pad], axis=1)
    return out[:, :width].astype(np.float32, copy=False)


def _load_transl(data: Dict[str, Any]) -> np.ndarray:
    if "transl" in data:
        transl = data["transl"]
    elif "trans" in data:
        transl = data["trans"]
    else:
        raise KeyError("missing transl/trans")
    transl = np.asarray(transl, dtype=np.float32)
    if transl.ndim != 2 or transl.shape[1] < 3:
        raise ValueError(f"expected transl shape (T,3), got {transl.shape}")
    return transl[:, :3]


def _load_pose_block(data: Dict[str, Any], key: str, frames: int, joints: int) -> np.ndarray:
    width = joints * 3
    if key in data:
        return _as_repeated(data[key], frames, width)
    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        if poses.ndim == 2 and poses.shape[1] >= 165:
            start, end = POSE_SLICES[key]
            return _as_repeated(poses[:, start:end], frames, width)
    return np.zeros((frames, width), dtype=np.float32)


def _rotation_representations(axis_angle_flat: np.ndarray, joints: int) -> Dict[str, np.ndarray]:
    axis_angle_flat = np.asarray(axis_angle_flat, dtype=np.float64)
    frames = axis_angle_flat.shape[0]
    aa = axis_angle_flat.reshape(frames, joints, 3)
    rot = R.from_rotvec(aa.reshape(-1, 3))
    mat = rot.as_matrix().reshape(frames, joints, 3, 3)
    quat_xyzw = rot.as_quat().reshape(frames, joints, 4)
    quat_wxyz = np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)
    rot6d = np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)
    out = {
        "axis_angle": aa.reshape(frames, joints * 3),
        "quaternion": quat_wxyz.reshape(frames, joints * 4),
        "rotation_6d": rot6d.reshape(frames, joints * 6),
        "matrix": mat.reshape(frames, joints * 9),
    }
    return out


def _stats_to_plain(stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, RunningStats):
            out[key] = value.as_dict()
        elif isinstance(value, dict):
            out[key] = _stats_to_plain(value)
        else:
            out[key] = value
    return out


def _iter_annotation_rows(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    data = obj.get("data_list", obj) if isinstance(obj, dict) else obj
    if isinstance(data, dict):
        for key, row in data.items():
            if isinstance(row, dict):
                yield str(key), row
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                yield str(row.get("id", idx)), row


def _annotation_motion_relpath(subset_root: Path, rel: str, motion_dir: str) -> str:
    parts = list(Path(str(rel)).parts)
    if parts and parts[0] == subset_root.name:
        parts = parts[1:]
    for idx, part in enumerate(parts):
        if part in {"smplx_55", "smplh_52"}:
            parts[idx] = motion_dir
            break
    return Path(*parts).as_posix()


def _included_motion_refs(
    subset_root: Path,
    splits: Optional[List[str]],
    exclude_invalid: bool,
    motion_dir: str,
) -> Tuple[Optional[set[str]], Dict[str, Any]]:
    if not splits:
        return None, {}
    refs: set[str] = set()
    missing_refs = 0
    skipped_invalid = 0
    rows = 0
    for split in splits:
        split_path = subset_root / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"missing annotation split: {split_path}")
        for _, row in _iter_annotation_rows(split_path):
            if exclude_invalid and row.get("invalid") is True:
                skipped_invalid += 1
                continue
            rows += 1
            rel = row.get("smplx_path")
            if not rel:
                missing_refs += 1
                continue
            refs.add(_annotation_motion_relpath(subset_root, str(rel), motion_dir))
    return refs, {
        "annotation_splits": splits,
        "annotation_rows": rows,
        "skipped_invalid_rows": skipped_invalid,
        "missing_smplx_refs": missing_refs,
        "included_motion_refs": len(refs),
    }


def _motion_files(
    subset_root: Path,
    motion_dir: str,
    include_refs: Optional[set[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    if recursive:
        files = sorted(
            path
            for path in subset_root.rglob("*.npz")
            if motion_dir in path.relative_to(subset_root).parts
        )
    else:
        files = sorted((subset_root / motion_dir).glob("*.npz"))
    if include_refs is not None:
        files = [path for path in files if path.relative_to(subset_root).as_posix() in include_refs]
    if not files:
        suffix = f"**/{motion_dir}" if recursive else motion_dir
        raise FileNotFoundError(f"no .npz files found under {subset_root / suffix}")
    return files


def compute_stats(
    subset_root: Path,
    motion_dir: str,
    annotation_splits: Optional[List[str]] = None,
    exclude_invalid: bool = False,
    recursive: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    include_refs, annotation_meta = _included_motion_refs(
        subset_root,
        annotation_splits,
        exclude_invalid,
        motion_dir,
    )
    files = _motion_files(subset_root, motion_dir, include_refs=include_refs, recursive=recursive)
    direct = {key: RunningStats(width) for key, width in DIRECT_BLOCKS}
    pose_stats = {
        key: {rot_key: RunningStats(joints * _rot_width(rot_key)) for rot_key in ROTATION_KEYS}
        for key, joints in POSE_BLOCKS
    }
    errors: List[Dict[str, str]] = []
    total_frames = 0

    for idx, path in enumerate(files, start=1):
        try:
            with np.load(path, allow_pickle=True) as data:
                transl = _load_transl(data)
                frames = int(transl.shape[0])
                total_frames += frames
                direct["transl"].update(transl)
                if frames > 1:
                    direct["transl_vel"].update(transl[1:] - transl[:-1])
                direct["betas"].update(_as_repeated(data["betas"] if "betas" in data else None, frames, 10))
                direct["expression"].update(
                    _as_repeated(data["expression"] if "expression" in data else None, frames, 10)
                )
                for key, joints in POSE_BLOCKS:
                    aa_flat = _load_pose_block(data, key, frames, joints)
                    reps = _rotation_representations(aa_flat, joints)
                    for rot_key, values in reps.items():
                        pose_stats[key][rot_key].update(values)
        except Exception as exc:  # keep the script useful for auditing partial subsets
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if idx % 250 == 0:
            print(f"[stats] processed {idx}/{len(files)} files", flush=True)

    raw_stats = {**direct, **pose_stats}
    stats = _stats_to_plain(raw_stats)
    for key, block in list(stats.items()):
        if key in dict(POSE_BLOCKS):
            block["count_frames"] = block["axis_angle"]["count_frames"]
            for canonical, legacy in LEGACY_ROTATION_ALIASES.items():
                block[legacy] = json.loads(json.dumps(block[canonical]))
    meta = {
        "subset_root": str(subset_root),
        "motion_dir": motion_dir,
        "recursive": bool(recursive),
        "num_files": len(files),
        "total_frames": total_frames,
        "errors": errors,
        **annotation_meta,
    }
    return stats, meta, raw_stats


def _rot_width(key: str) -> int:
    if key == "axis_angle":
        return 3
    if key == "quaternion":
        return 4
    if key == "rotation_6d":
        return 6
    if key == "matrix":
        return 9
    raise KeyError(key)


def _leaf_blocks(
    stats: Dict[str, Any],
    prefix: str = "",
    parent_count: Optional[int] = None,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, value in stats.items():
        if key == "count_frames":
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "mean" in value and "std" in value:
            if "count_frames" not in value and parent_count is not None:
                value = {**value, "count_frames": parent_count}
            yield name, value
        elif isinstance(value, dict):
            next_count = value.get("count_frames", parent_count)
            yield from _leaf_blocks(value, prefix=name, parent_count=next_count)


def compare_stats(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return {}
    old_blocks = dict(_leaf_blocks(old))
    new_blocks = dict(_leaf_blocks(new))
    comparisons: Dict[str, Any] = {}
    for name, new_block in sorted(new_blocks.items()):
        old_name = name
        if old_name not in old_blocks and name.endswith(".rotation_6d"):
            old_name = name.rsplit(".", 1)[0] + ".rot6d"
        if old_name not in old_blocks and name.endswith(".matrix"):
            old_name = name.rsplit(".", 1)[0] + ".rotmat"
        if old_name not in old_blocks:
            comparisons[name] = {"status": "new"}
            continue
        old_block = old_blocks[old_name]
        old_mean = np.asarray(old_block["mean"], dtype=np.float64)
        old_std = np.asarray(old_block["std"], dtype=np.float64)
        new_mean = np.asarray(new_block["mean"], dtype=np.float64)
        new_std = np.asarray(new_block["std"], dtype=np.float64)
        if old_mean.shape != new_mean.shape or old_std.shape != new_std.shape:
            comparisons[name] = {
                "status": "shape_mismatch",
                "old_mean_shape": list(old_mean.shape),
                "new_mean_shape": list(new_mean.shape),
                "old_std_shape": list(old_std.shape),
                "new_std_shape": list(new_std.shape),
            }
            continue
        comparisons[name] = {
            "status": "ok",
            "old_count_frames": int(old_block.get("count_frames", 0)),
            "new_count_frames": int(new_block.get("count_frames", 0)),
            "mean_max_abs_diff": float(np.max(np.abs(new_mean - old_mean))) if new_mean.size else 0.0,
            "std_max_abs_diff": float(np.max(np.abs(new_std - old_std))) if new_std.size else 0.0,
        }
    return comparisons


def _plain_block(plain: Dict[str, Any], name: str) -> Dict[str, Any]:
    value: Dict[str, Any] = plain
    for part in name.split("."):
        value = value[part]
    return value


def _running_leaf_blocks(
    raw_stats: Dict[str, Any],
    prefix: str = "",
) -> Iterable[Tuple[str, RunningStats]]:
    for key, value in raw_stats.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, RunningStats):
            yield name, value
        elif isinstance(value, dict):
            yield from _running_leaf_blocks(value, prefix=name)


def validation_from_running(
    raw_stats: Dict[str, Any],
    stats: Dict[str, Any],
    *,
    eps: float,
) -> Dict[str, Any]:
    constant_channels: Dict[str, List[int]] = {}
    max_abs_norm_mean = 0.0
    max_abs_norm_std_minus_one = 0.0
    checked_blocks = 0
    for name, running in _running_leaf_blocks(raw_stats):
        if running.count <= 0:
            continue
        block = _plain_block(stats, name)
        mean = np.asarray(block["mean"], dtype=np.float64)
        std = np.asarray(block["std"], dtype=np.float64)
        data_mean = running.total / float(running.count)
        data_var = running.total_sq / float(running.count) - np.square(data_mean)
        data_std = np.sqrt(np.maximum(data_var, 0.0))
        constant = np.where(std <= eps)[0]
        variable = np.where(std > eps)[0]
        if constant.size:
            constant_channels[name] = constant.astype(int).tolist()
        if variable.size:
            norm_mean = (data_mean[variable] - mean[variable]) / std[variable]
            norm_std = data_std[variable] / std[variable]
            max_abs_norm_mean = max(max_abs_norm_mean, float(np.max(np.abs(norm_mean))))
            max_abs_norm_std_minus_one = max(
                max_abs_norm_std_minus_one,
                float(np.max(np.abs(norm_std - 1.0))),
            )
        checked_blocks += 1
    return {
        "eps": eps,
        "checked_blocks": checked_blocks,
        "max_abs_norm_mean_nonconstant": max_abs_norm_mean,
        "max_abs_norm_std_minus_one_nonconstant": max_abs_norm_std_minus_one,
        "constant_channels": constant_channels,
    }


def old_translation_validation(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return {}
    out = {}
    for key in ("transl", "transl_vel"):
        old_mean = np.asarray(old[key]["mean"], dtype=np.float64)
        old_std = np.asarray(old[key]["std"], dtype=np.float64)
        new_mean = np.asarray(new[key]["mean"], dtype=np.float64)
        new_std = np.asarray(new[key]["std"], dtype=np.float64)
        denom = np.maximum(old_std, 1e-12)
        out[key] = {
            "mean_after_old_normalize": ((new_mean - old_mean) / denom).tolist(),
            "std_after_old_normalize": (new_std / denom).tolist(),
        }
    return out


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_backup(path: Path, backup: Optional[Path]) -> None:
    if backup is None or not path.exists():
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        return
    shutil.copy2(path, backup)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--motion-dir", default="smplx_55")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-old")
    parser.add_argument("--report", required=True)
    parser.add_argument("--backup")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--annotation-splits",
        help="Comma-separated split names used to select motion files, e.g. train,test.",
    )
    parser.add_argument("--exclude-invalid", action="store_true", help="Skip annotation rows with invalid=true.")
    parser.add_argument("--recursive", action="store_true", help="Find **/{motion_dir}/*.npz recursively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_root = Path(args.subset_root)
    output = Path(args.output)
    compare_old = Path(args.compare_old) if args.compare_old else None
    report_path = Path(args.report)
    backup = Path(args.backup) if args.backup else None

    old_stats = load_json(compare_old)
    annotation_splits = (
        [item.strip() for item in args.annotation_splits.split(",") if item.strip()]
        if args.annotation_splits
        else None
    )
    stats, meta, raw_stats = compute_stats(
        subset_root,
        args.motion_dir,
        annotation_splits=annotation_splits,
        exclude_invalid=args.exclude_invalid,
        recursive=args.recursive,
    )
    report = {
        "meta": meta,
        "output": str(output),
        "compare_old": str(compare_old) if compare_old else None,
        "comparison": compare_stats(old_stats, stats),
        "validation": validation_from_running(raw_stats, stats, eps=args.eps),
        "old_stats_translation_validation": old_translation_validation(old_stats, stats),
    }
    if old_stats is not None:
        report["translation_highlight"] = {
            "old_transl_y_mean": old_stats["transl"]["mean"][1],
            "new_transl_y_mean": stats["transl"]["mean"][1],
            "oldstats_norm_y_mean": report["old_stats_translation_validation"]["transl"][
                "mean_after_old_normalize"
            ][1],
        }

    write_json(report_path, report)
    if args.write:
        maybe_backup(output, backup)
        write_json(output, stats)
        print(f"[write] stats -> {output}", flush=True)
    else:
        print("[dry-run] pass --write to update stats", flush=True)
    print(f"[report] {report_path}", flush=True)
    print(
        "[summary] files={num_files} frames={total_frames} errors={errors}".format(
            num_files=meta["num_files"],
            total_frames=meta["total_frames"],
            errors=len(meta["errors"]),
        ),
        flush=True,
    )
    if "translation_highlight" in report:
        h = report["translation_highlight"]
        print(
            "[translation] old_y_mean={:.6f} new_y_mean={:.6f} old_norm_y_mean={:.6f}".format(
                h["old_transl_y_mean"],
                h["new_transl_y_mean"],
                h["oldstats_norm_y_mean"],
            ),
            flush=True,
        )
    if meta["errors"]:
        raise SystemExit(f"failed to read {len(meta['errors'])} motion files")


if __name__ == "__main__":
    main()

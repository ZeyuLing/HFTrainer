#!/usr/bin/env python3
"""Build fixed-window G1 qpos inputs for unified Table-2 tracker evaluation.

The prior diagnostic runs mixed full-source clips, selected visual segments,
and baseline-specific conversion paths.  This script writes one explicit input
protocol shared by released tracker baselines:

* G1 qpos: root xyz + root quaternion WXYZ + 29 G1 DOFs.
* Stored frequency: 30 FPS.
* Window: first ``max_frames`` frames after resampling, preserving shorter clips.
* Manifest: JSON list of flat stems consumed by Any2Track and Humanoid-GPT.
* AMASS-G1 root rotations are read with their explicit source quaternion order.
* The main paper protocol uses an explicit AMASS-test manifest.  The full-AMASS
  sweep is an opt-in stress test, not the default Table-2 input.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


TARGET_G1_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _as_list(value: np.ndarray) -> list[str]:
    return [str(x) for x in value.tolist()]


def _scalar(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in data.files:
            arr = np.asarray(data[key]).reshape(-1)
            if arr.size:
                return float(arr[0])
    return float(default)


def _safe_stem(name: str) -> str:
    return "__".join(Path(name).with_suffix("").parts).replace(" ", "_")


def _qpos_slices(joint_names: list[str], jnt_type: np.ndarray | None) -> dict[str, slice]:
    if jnt_type is None:
        jnt_type = np.array([0] + [3] * (len(joint_names) - 1), dtype=np.int32)
    qpos_slices: dict[str, slice] = {}
    qpos_i = 0
    for name, typ in zip(joint_names, jnt_type):
        typ = int(typ)
        if typ == 0:
            qpos_slices[name] = slice(qpos_i, qpos_i + 7)
            qpos_i += 7
        elif typ in (2, 3):
            qpos_slices[name] = slice(qpos_i, qpos_i + 1)
            qpos_i += 1
        else:
            raise ValueError(f"Unsupported joint type {typ} for {name}")
    return qpos_slices


def _expand_named_qpos_to_g1(
    qpos: np.ndarray,
    joint_names: list[str],
    jnt_type: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    if qpos.shape[1] == 36:
        return qpos.astype(np.float32, copy=False), []
    source_slices = _qpos_slices(joint_names, jnt_type)
    out = np.zeros((qpos.shape[0], 36), dtype=np.float32)
    if "root" in source_slices:
        out[:, :7] = qpos[:, source_slices["root"]]
    elif qpos.shape[1] >= 7:
        out[:, :7] = qpos[:, :7]
    else:
        raise ValueError(f"qpos shape {qpos.shape} cannot provide root")
    missing: list[str] = []
    for i, name in enumerate(TARGET_G1_DOF_NAMES):
        sl = source_slices.get(name)
        if sl is None:
            missing.append(name)
            continue
        out[:, 7 + i] = qpos[:, sl].reshape(qpos.shape[0])
    return out, missing


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)


def _apply_root_frame_correction_wxyz(root_quat: np.ndarray, correction: str) -> np.ndarray:
    if correction == "none":
        return _normalize_quat(root_quat).astype(np.float32)
    if correction != "rx_neg90":
        raise ValueError(f"Unsupported root frame correction: {correction}")
    src_xyzw = _normalize_quat(root_quat)[:, [1, 2, 3, 0]]
    corrected_xyzw = (Rotation.from_euler("x", -90, degrees=True) * Rotation.from_quat(src_xyzw)).as_quat()
    return _normalize_quat(corrected_xyzw[:, [3, 0, 1, 2]]).astype(np.float32)


def _resample_qpos_wxyz(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-6 or len(qpos) < 2:
        return qpos.astype(np.float32, copy=False)
    duration = (len(qpos) - 1) / source_fps
    src_t = np.arange(len(qpos), dtype=np.float64) / source_fps
    out_n = int(round(duration * target_fps)) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / target_fps
    dst_t[-1] = min(dst_t[-1], src_t[-1])
    out = np.empty((out_n, qpos.shape[1]), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    src_xyzw = _normalize_quat(qpos[:, 3:7])[:, [1, 2, 3, 0]]
    out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
    out[:, 3:7] = out_xyzw[:, [3, 0, 1, 2]]
    for i in range(7, qpos.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    return out.astype(np.float32)


def _amp_source_to_qpos(
    data: np.lib.npyio.NpzFile,
    body_quat_order: str,
    root_frame_correction: str,
) -> tuple[np.ndarray, float, list[str]]:
    body_names = _as_list(data["body_names"])
    dof_names = _as_list(data["dof_names"])
    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    root_pos = np.asarray(data["body_positions"][:, pelvis_idx, :], dtype=np.float32)
    root_quat = np.asarray(data["body_rotations"][:, pelvis_idx, :], dtype=np.float32)
    if body_quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    elif body_quat_order != "wxyz":
        raise ValueError(f"Unsupported body_quat_order={body_quat_order}")
    root_quat = _apply_root_frame_correction_wxyz(root_quat, root_frame_correction)
    dof_src = np.asarray(data["dof_positions"], dtype=np.float32)
    dof = np.zeros((dof_src.shape[0], len(TARGET_G1_DOF_NAMES)), dtype=np.float32)
    missing: list[str] = []
    for i, name in enumerate(TARGET_G1_DOF_NAMES):
        if name not in dof_names:
            missing.append(name)
            continue
        dof[:, i] = dof_src[:, dof_names.index(name)]
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    return qpos, _scalar(data, ("frequency", "fps"), 30.0), missing


def _load_g1_qpos(
    path: Path,
    body_quat_order: str,
    root_frame_correction: str,
) -> tuple[np.ndarray, float, list[str]]:
    data = np.load(path, allow_pickle=True)
    if "qpos" in data.files:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = _scalar(data, ("frequency", "fps"), 30.0)
        if qpos.shape[1] == 36:
            return qpos, fps, []
        if "joint_names" not in data.files:
            raise ValueError(f"{path}: qpos dim {qpos.shape[1]} and no joint_names")
        joint_names = _as_list(data["joint_names"])
        jnt_type = np.asarray(data["jnt_type"]) if "jnt_type" in data.files else None
        qpos36, missing = _expand_named_qpos_to_g1(qpos, joint_names, jnt_type)
        return qpos36, fps, missing
    required = {"body_positions", "body_rotations", "dof_positions", "dof_names", "body_names"}
    if required.issubset(data.files):
        return _amp_source_to_qpos(data, body_quat_order, root_frame_correction)
    raise ValueError(f"{path}: unsupported npz fields {data.files}")


def _write_motion(
    src: Path,
    out_dir: Path,
    stem: str,
    target_fps: float,
    max_frames: int,
    body_quat_order: str,
    root_frame_correction: str,
) -> dict[str, Any]:
    qpos, src_fps, missing = _load_g1_qpos(src, body_quat_order, root_frame_correction)
    qpos = _resample_qpos_wxyz(qpos, src_fps, target_fps)
    if max_frames > 0:
        qpos = qpos[:max_frames]
    out = out_dir / f"{stem}.npz"
    np.savez(out, qpos=qpos.astype(np.float32), frequency=np.float32(target_fps), source=str(src))
    return {
        "stem": stem,
        "source": str(src),
        "source_fps": src_fps,
        "target_fps": target_fps,
        "frames": int(qpos.shape[0]),
        "missing_joints_filled_zero": missing,
    }


def _build_lafan(args: argparse.Namespace, split_dir: Path) -> dict[str, Any]:
    names = json.loads(args.lafan_manifest.read_text())
    npz_dir = split_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for name in names:
        stem = _safe_stem(name)
        rows.append(_write_motion(args.lafan_root / f"{name}.npz", npz_dir, stem, args.target_fps, args.max_frames, "wxyz", "none"))
        manifest.append(stem)
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"count": len(manifest), "motions": rows}


def _build_wild(args: argparse.Namespace, split_dir: Path) -> dict[str, Any]:
    names = json.loads(args.wild_manifest.read_text())
    npz_dir = split_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for name in names:
        stem = _safe_stem(name)
        rows.append(_write_motion(args.wild_root / f"{name}.npz", npz_dir, stem, args.target_fps, args.max_frames, "wxyz", "none"))
        manifest.append(stem)
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"count": len(manifest), "motions": rows}


def _resolve_manifest_paths(root: Path, manifest_path: Path) -> list[Path]:
    names = json.loads(manifest_path.read_text())
    out: list[Path] = []
    for name in names:
        rel = Path(name)
        if rel.suffix != ".npz":
            rel = rel.with_suffix(".npz")
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Manifest entry {name!r} does not exist under {root}")
        out.append(path)
    return out


def _read_protomotions_amass_yaml(yaml_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in yaml_path.read_text().splitlines():
        match = re.match(r"\s*-\s*file:\s*(.+?)\s*$", line)
        if match:
            current = {"file": match.group(1), "fps": None}
            rows.append(current)
            continue
        match = re.match(r"\s*fps:\s*([0-9.]+)\s*$", line)
        if match and current is not None and current["fps"] is None:
            current["fps"] = float(match.group(1))
    if not rows:
        raise ValueError(f"{yaml_path}: no ProtoMotions AMASS entries found")
    missing_fps = [row["file"] for row in rows if row["fps"] is None]
    if missing_fps:
        raise ValueError(f"{yaml_path}: missing fps for {missing_fps[:5]}")
    return rows


def _resolve_protomotions_amass_test_paths(
    root: Path,
    yaml_path: Path,
) -> tuple[list[Path], dict[str, Any]]:
    rows = _read_protomotions_amass_yaml(yaml_path)
    out: list[Path] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        rel_motion = Path(row["file"]).with_suffix("")
        fps = int(round(float(row["fps"])))
        candidates = [
            rel_motion.parent / f"{rel_motion.name}_{fps}_jpos.npz",
            rel_motion.parent / f"{rel_motion.name}_jpos.npz",
            rel_motion.with_suffix(".npz"),
        ]
        hit = next((root / rel for rel in candidates if (root / rel).is_file()), None)
        if hit is None:
            missing.append({
                "file": row["file"],
                "fps": row["fps"],
                "candidates": [str(rel) for rel in candidates],
            })
            continue
        out.append(hit)
    meta = {
        "source": str(yaml_path),
        "source_format": "ProtoMotions data/yaml_files/amass_smpl_test.yaml",
        "source_entries": len(rows),
        "matched_entries": len(out),
        "missing_entries": missing,
    }
    return out, meta


def _build_amass(args: argparse.Namespace, split_dir: Path, src_files: list[Path] | None = None) -> dict[str, Any]:
    if src_files is None:
        src_files = sorted(args.amass_root.glob("**/*.npz"))
        if args.amass_limit:
            src_files = src_files[: args.amass_limit]
    npz_dir = split_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for idx, src in enumerate(src_files, start=1):
        rel = src.relative_to(args.amass_root).with_suffix("")
        stem = _safe_stem(str(rel))
        rows.append(_write_motion(
            src,
            npz_dir,
            stem,
            args.target_fps,
            args.max_frames,
            args.amass_body_quat_order,
            args.amass_root_frame_correction,
        ))
        manifest.append(stem)
        if idx % 1000 == 0:
            print(f"[build-table2] amass {idx}/{len(src_files)}", flush=True)
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"count": len(manifest), "motions": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1"))
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--lafan-root", type=Path, default=Path("data/LAFAN1_Retargeted_for_G1/UnitreeG1"))
    ap.add_argument("--lafan-manifest", type=Path, default=Path("output/opentrack_lafan1_g1/local_py311_full_localmetric_20260604_233656/manifests/all.json"))
    ap.add_argument("--wild-root", type=Path, default=Path("output/heldout_frozen_score"))
    ap.add_argument("--wild-manifest", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker/wild_g1_clean/manifest.json"))
    ap.add_argument("--amass-root", type=Path, default=Path("data/AMASS_Retarged_for_G1/g1"))
    ap.add_argument("--amass-test-manifest", type=Path, default=None, help="Optional relative AMASS-test npz paths; overrides --amass-test-yaml.")
    ap.add_argument("--amass-test-yaml", type=Path, default=Path("ref_repo/ProtoMotions/data/yaml_files/amass_smpl_test.yaml"), help="ProtoMotions AMASS test yaml used for the main AMASS-test-G1 split.")
    ap.add_argument("--skip-amass-test", action="store_true", help="Do not build the main AMASS-test-G1 split.")
    ap.add_argument("--strict-amass-test", action="store_true", help="Fail if any ProtoMotions AMASS-test entry is missing from the local G1 retarget root.")
    ap.add_argument("--include-full-amass-stress", action="store_true", help="Also build the optional full-AMASS stress split.")
    ap.add_argument("--skip-wild", action="store_true", help="Do not build the main Wild-G1 split.")
    ap.add_argument("--include-wild-stress", action="store_true", help="Deprecated compatibility flag; Wild-G1 is a main split by default.")
    ap.add_argument("--amass-limit", type=int, default=0, help="0 means all AMASS files")
    ap.add_argument("--amass-body-quat-order", choices=["xyzw", "wxyz"], default="wxyz")
    ap.add_argument(
        "--amass-root-frame-correction",
        choices=["none", "rx_neg90"],
        default="none",
        help="Optional diagnostic correction from AMASS-G1 pelvis rotations into the G1 free-joint frame.",
    )
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = {
        "lafan1_fixed600": _build_lafan(args, args.out_root / "inputs" / "lafan1_fixed600"),
    }
    amass_test_source: dict[str, Any] | None = None
    if not args.skip_amass_test:
        if args.amass_test_manifest is not None:
            amass_test_files = _resolve_manifest_paths(args.amass_root, args.amass_test_manifest)
            amass_test_source = {
                "source": str(args.amass_test_manifest),
                "source_format": "relative npz manifest",
                "source_entries": len(amass_test_files),
                "matched_entries": len(amass_test_files),
                "missing_entries": [],
            }
        else:
            amass_test_files, amass_test_source = _resolve_protomotions_amass_test_paths(
                args.amass_root,
                args.amass_test_yaml,
            )
        if amass_test_source.get("missing_entries") and args.strict_amass_test:
            raise FileNotFoundError(f"Missing AMASS-test-G1 files: {amass_test_source['missing_entries']}")
        if not amass_test_files:
            raise RuntimeError("AMASS-test-G1 split resolved to zero local files")
        splits["amass_test_fixed600"] = _build_amass(
            args,
            args.out_root / "inputs" / "amass_test_fixed600",
            amass_test_files,
        )
    if not args.skip_wild or args.include_wild_stress:
        splits["wild_clean_fixed600"] = _build_wild(args, args.out_root / "inputs" / "wild_clean_fixed600")
    if args.include_full_amass_stress:
        splits["amass_fixed600"] = _build_amass(args, args.out_root / "inputs" / "amass_fixed600")

    summary = {
        "schema": "table2_unified_protocol_v1",
        "target_fps": args.target_fps,
        "max_frames": args.max_frames,
        "amass_body_quat_order": args.amass_body_quat_order,
        "amass_root_frame_correction": args.amass_root_frame_correction,
        "amass_test_source": amass_test_source,
        "splits": splits,
    }
    (args.out_root / "protocol_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    compact = {
        "target_fps": args.target_fps,
        "max_frames": args.max_frames,
        "counts": {k: v["count"] for k, v in summary["splits"].items()},
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

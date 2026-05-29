#!/usr/bin/env python3
"""Build visualization manifests for PhysFlow online triplet monitoring.

The dashboard consumes a compact JSON manifest with rows keyed by
iteration/prompt. Each row may contain three robot-frame sources:

  raw_reference -> optimized_reference -> tracked_rollout

For G1/KIMODO runs, raw and optimized references are stored as ProtoMotions
.motion files, while tracked rollouts are exported as robot_frames JSON. This
script converts .motion references to the same robot_frames JSON format so the
viewer can render all three columns with one code path.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from math import cos, sin
from pathlib import Path
import sys
from typing import Any, Optional

import torch


DEFAULT_BODIES = [
    "pelvis",
    "head",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]


MESHES_BY_BODY = {
    "pelvis": ["pelvis.stl", "pelvis_contour_link.stl"],
    "head": [],
    "left_hip_pitch_link": ["left_hip_pitch_link.stl"],
    "left_hip_roll_link": ["left_hip_roll_link.stl"],
    "left_hip_yaw_link": ["left_hip_yaw_link.stl"],
    "left_knee_link": ["left_knee_link.stl"],
    "left_ankle_pitch_link": ["left_ankle_pitch_link.stl"],
    "left_ankle_roll_link": ["left_ankle_roll_link.stl"],
    "right_hip_pitch_link": ["right_hip_pitch_link.stl"],
    "right_hip_roll_link": ["right_hip_roll_link.stl"],
    "right_hip_yaw_link": ["right_hip_yaw_link.stl"],
    "right_knee_link": ["right_knee_link.stl"],
    "right_ankle_pitch_link": ["right_ankle_pitch_link.stl"],
    "right_ankle_roll_link": ["right_ankle_roll_link.stl"],
    "waist_yaw_link": ["waist_yaw_link_rev_1_0.stl"],
    "waist_roll_link": ["waist_roll_link_rev_1_0.stl"],
    "torso_link": ["torso_link_rev_1_0.stl", "logo_link.stl", "head_link.stl"],
    "left_shoulder_pitch_link": ["left_shoulder_pitch_link.stl"],
    "left_shoulder_roll_link": ["left_shoulder_roll_link.stl"],
    "left_shoulder_yaw_link": ["left_shoulder_yaw_link.stl"],
    "left_elbow_link": ["left_elbow_link.stl"],
    "left_wrist_roll_link": ["left_wrist_roll_link.stl"],
    "left_wrist_pitch_link": ["left_wrist_pitch_link.stl"],
    "left_wrist_yaw_link": ["left_wrist_yaw_link.stl", "left_rubber_hand.stl"],
    "left_rubber_hand": [],
    "right_shoulder_pitch_link": ["right_shoulder_pitch_link.stl"],
    "right_shoulder_roll_link": ["right_shoulder_roll_link.stl"],
    "right_shoulder_yaw_link": ["right_shoulder_yaw_link.stl"],
    "right_elbow_link": ["right_elbow_link.stl"],
    "right_wrist_roll_link": ["right_wrist_roll_link.stl"],
    "right_wrist_pitch_link": ["right_wrist_pitch_link.stl"],
    "right_wrist_yaw_link": ["right_wrist_yaw_link.stl", "right_rubber_hand.stl"],
    "right_rubber_hand": [],
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G1_MJCF = (
    PROJECT_ROOT
    / "ref_repo"
    / "ProtoMotions"
    / "protomotions"
    / "data"
    / "assets"
    / "mjcf"
    / "g1_holo_compat.xml"
)

PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROTOMOTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOMOTIONS_ROOT))

def _float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    return [float(x) for x in value.split()]


def _quat_from_axis_angle(axis_angle: str | None) -> list[float] | None:
    if axis_angle is None:
        return None
    values = [float(x) for x in axis_angle.split()]
    if len(values) != 4:
        return None
    axis = torch.tensor(values[:3], dtype=torch.float64)
    norm = torch.linalg.norm(axis)
    if float(norm) < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    axis = axis / norm
    half = 0.5 * values[3]
    xyz = axis * sin(half)
    return [float(cos(half)), float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _quat_from_euler(euler: str | None) -> list[float] | None:
    if euler is None:
        return None
    values = [float(x) for x in euler.split()]
    if len(values) != 3:
        return None
    cx, cy, cz = (cos(v * 0.5) for v in values)
    sx, sy, sz = (sin(v * 0.5) for v in values)
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


def _parse_g1_body_meshes(mjcf_path: Path = DEFAULT_G1_MJCF) -> list[dict]:
    """Return raw STL visual mesh transforms from the MJCF XML."""
    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()
    mesh_name_to_file = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh_elem in asset.findall("mesh"):
            name = mesh_elem.get("name", "")
            filename = mesh_elem.get("file", "")
            if name and filename:
                mesh_name_to_file[name] = filename

    bodies = []

    def geom_record(geom: ET.Element) -> dict | None:
        if geom.get("type") not in (None, "mesh") and not geom.get("mesh"):
            return None
        mesh_name = geom.get("mesh", "")
        if mesh_name not in mesh_name_to_file:
            return None
        quat = (
            _float_list(geom.get("quat"), [1.0, 0.0, 0.0, 0.0])
            if geom.get("quat") is not None
            else _quat_from_axis_angle(geom.get("axisangle"))
            or _quat_from_euler(geom.get("euler"))
            or [1.0, 0.0, 0.0, 0.0]
        )
        return {
            "file": mesh_name_to_file[mesh_name],
            "pos": _float_list(geom.get("pos"), [0.0, 0.0, 0.0]),
            "quat": quat,
        }

    def walk_body(elem: ET.Element) -> None:
        body_name = elem.get("name", "unnamed")
        meshes = []
        seen_files = set()
        for geom in elem.findall("geom"):
            record = geom_record(geom)
            if record is None:
                continue
            if record["file"] in seen_files:
                continue
            seen_files.add(record["file"])
            meshes.append(record)
        bodies.append({"name": body_name, "meshes": meshes})
        for child in elem.findall("body"):
            walk_body(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            walk_body(top_body)
    return bodies


def _jsonify_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    return value


def motion_to_robot_frames(motion_path: Path, out_path: Path) -> Path:
    motion = torch.load(motion_path, map_location="cpu")
    body_pos = motion["rigid_body_pos"]
    # ProtoMotions RobotState uses XYZW quaternions, while the Three.js G1
    # viewer path shares the MuJoCo export format and expects WXYZ.
    body_quat_xyzw = motion["rigid_body_rot"]
    body_quat_wxyz = body_quat_xyzw[..., [3, 0, 1, 2]]
    fps = int(motion.get("fps", 30))

    if body_pos.shape[1] != len(DEFAULT_BODIES):
        raise ValueError(
            f"{motion_path} has {body_pos.shape[1]} bodies, expected {len(DEFAULT_BODIES)}"
        )

    bodies = _parse_g1_body_meshes()
    if [body["name"] for body in bodies] != DEFAULT_BODIES:
        bodies = [
            {
                "name": name,
                "meshes": [{"file": mesh, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]} for mesh in MESHES_BY_BODY.get(name, [])],
            }
            for name in DEFAULT_BODIES
        ]
    frames = []
    for pos, quat in zip(body_pos, body_quat_wxyz):
        frames.append(
            {
                "body_pos": _jsonify_tensor(pos),
                "body_quat": _jsonify_tensor(quat),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "type": "robot_frames",
                "robot": "g1",
                "fps": fps,
                "num_frames": len(frames),
                "num_bodies": len(bodies),
                "bodies": bodies,
                "frames": frames,
            },
            indent=2,
        )
    )
    return out_path


def _resolve_existing_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def _load_summary_records(run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    return list(summary.get("records", []))


def _record_pair_key(record: dict) -> tuple[str, int]:
    return (str(record.get("prompt_id", record.get("output_stem", ""))), int(record.get("sample_idx", 0)))


def _reference_column(
    record: dict,
    out_dir: Path,
    label: str,
    title: str,
    status: str = "ready",
) -> dict:
    stem = record["output_stem"]
    motion_path = _resolve_existing_path(record["motion_path"], Path.cwd())
    ref_path = motion_to_robot_frames(
        motion_path,
        out_dir / "robot_frames_reference" / f"{stem}.{label}.json",
    )
    return {
        "status": status,
        "title": title,
        "path": str(ref_path),
        "metrics": {
            "source_score": record.get("adversarial_score"),
            "source_error": record.get("max_joint_error_rad"),
            "root_displacement_ref_m": record.get("root_displacement_ref_m"),
        },
    }


def build_from_runs(
    raw_run_dir: Path,
    out_dir: Path,
    iteration: int,
    optimized_run_dir: Optional[Path] = None,
) -> Path:
    raw_records = _load_summary_records(raw_run_dir)
    optimized_records = []
    if optimized_run_dir is not None and (optimized_run_dir / "summary.json").is_file():
        optimized_records = _load_summary_records(optimized_run_dir)

    optimized_by_key = {
        _record_pair_key(record): record
        for record in optimized_records
        if record.get("status") == "scored" and record.get("motion_path")
    }

    rows = []

    for record in raw_records:
        if record.get("status") != "scored" or not record.get("motion_path"):
            continue
        stem = record.get("output_stem") or record.get("prompt_id") or "motion"
        tracked_path = _resolve_existing_path(record["robot_json_path"], Path.cwd())

        raw_column = _reference_column(
            record,
            out_dir,
            label="raw_reference",
            title="Raw T2M Reference",
        )

        optimized_record = optimized_by_key.get(_record_pair_key(record))
        if optimized_record:
            optimized_column = _reference_column(
                optimized_record,
                out_dir,
                label="optimized_reference",
                title="Optimized / Selected Reference",
            )
        else:
            optimized_column = {
                "status": "pending",
                "title": "Optimized / Selected Reference",
                "path": "",
                "metrics": {},
            }

        rows.append(
            {
                "iteration": iteration,
                "iteration_label": f"iter_{iteration:06d}",
                "prompt_id": record.get("prompt_id", stem),
                "prompt": record.get("prompt", ""),
                "category": record.get("category", ""),
                "difficulty": record.get("difficulty"),
                "seed": record.get("seed"),
                "sample_idx": record.get("sample_idx"),
                "columns": {
                    "raw_reference": raw_column,
                    "optimized_reference": optimized_column,
                    "tracked_rollout": {
                        "status": "ready",
                        "title": "Motion Tracking Rollout",
                        "path": str(tracked_path),
                        "metrics": {
                            "completion": record.get("completion_ratio"),
                            "max_joint_error_rad": record.get("max_joint_error_rad"),
                            "fall": record.get("fall_detected"),
                            "root_height_final": record.get("root_height_final"),
                            "root_displacement_ref_m": record.get("root_displacement_ref_m"),
                            "root_displacement_track_m": record.get("root_displacement_track_m"),
                            "root_displacement_error_m": record.get("root_displacement_error_m"),
                            "root_trajectory_error_mean_m": record.get("root_trajectory_error_mean_m"),
                            "root_trajectory_error_final_m": record.get("root_trajectory_error_final_m"),
                        },
                    },
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "project": "PhysFlow KIMODO-G1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {
            "raw": str(raw_run_dir),
            "optimized": str(optimized_run_dir) if optimized_run_dir else None,
        },
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-run-dir", type=Path, default=None, help="Backward-compatible alias for --raw-run-dir.")
    parser.add_argument("--raw-run-dir", type=Path, default=None)
    parser.add_argument("--optimized-run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=0)
    args = parser.parse_args()

    raw_run_dir = args.raw_run_dir or args.smoke_run_dir
    if raw_run_dir is None:
        raise SystemExit("Provide --raw-run-dir or --smoke-run-dir.")

    manifest_path = build_from_runs(
        raw_run_dir=raw_run_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        iteration=args.iteration,
        optimized_run_dir=args.optimized_run_dir.resolve() if args.optimized_run_dir else None,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""G1 Robot RL Tracker Export: motion_135 NPZ → G1 robot body poses JSON.

End-to-end pipeline:
  1. Retarget motion_135 NPZ → ProtoMotions .motion file (via pipeline_motion_to_robot.py)
  2. Run G1 ONNX policy in MuJoCo simulation
  3. Export per-frame body positions (xpos) and quaternions (xquat) as JSON
     for Three.js visualization

Output JSON format:
    {
        "type": "robot_frames",
        "robot": "g1",
        "fps": 50,
        "num_frames": N,
        "bodies": [
            {"name": "pelvis", "meshes": ["pelvis.stl", "pelvis_contour_link.stl"]},
            {"name": "left_hip_pitch_link", "meshes": ["left_hip_pitch_link.stl"]},
            ...
        ],
        "frames": [
            {
                "body_pos": [[x,y,z], ...],   # per-body world position (num_bodies x 3)
                "body_quat": [[w,x,y,z], ...] # per-body world quaternion wxyz (num_bodies x 4)
            },
            ...
        ]
    }

Usage:
    # Single file
    python3 scripts/embodied/run_g1_rl_tracker_export.py \
        --input output/physflow/eval_demo/data/npz/original_000_a_person_stands_still.npz \
        --output-dir output/physflow/eval_demo/data/robot_mesh_rl/

    # Batch (all NPZ in a directory)
    python3 scripts/embodied/run_g1_rl_tracker_export.py \
        --input-dir output/physflow/eval_demo/data/npz/ \
        --output-dir output/physflow/eval_demo/data/robot_mesh_rl/
"""
import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"

# Add ProtoMotions to path for deployment utilities
_PROTO_ROOT = str(PROTOMOTIONS_ROOT)
if _PROTO_ROOT not in sys.path:
    sys.path.insert(0, _PROTO_ROOT)

from deployment.state_utils import (
    mujoco_wxyz_to_xyzw,
    compute_anchor_rot_np,
    compute_yaw_offset_np,
    apply_heading_offset_np,
)
from deployment.motion_utils import MotionPlayer

# Default paths
DEFAULT_ONNX = PROTOMOTIONS_ROOT / "data" / "pretrained_models" / "motion_tracker" / "g1-bones-deploy" / "compiled_models" / "unified_pipeline.onnx"
DEFAULT_MJCF = PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"
DEFAULT_SMPL_MODEL = PROJECT_ROOT / "checkpoints" / "smpl_models"
DEFAULT_URDF = PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / "urdf" / "for_retargeting" / "g1.urdf"

# Retargeting scripts
PIPELINE_SCRIPT = SCRIPT_DIR / "pipeline_motion_to_robot.py"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# MJCF parsing: body→mesh mapping
# ---------------------------------------------------------------------------

def parse_body_mesh_mapping(mjcf_path: pathlib.Path) -> list:
    """Parse MJCF XML to extract body name → mesh file mapping.

    Returns a list of dicts: [{"name": body_name, "meshes": [file1.stl, ...]}, ...]
    in the order bodies appear in MuJoCo (excluding world body).
    """
    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()

    # Build mesh_name → file mapping from <asset>
    mesh_name_to_file = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh_elem in asset.findall("mesh"):
            name = mesh_elem.get("name", "")
            filename = mesh_elem.get("file", "")
            if name and filename:
                mesh_name_to_file[name] = filename

    # Walk body hierarchy in DFS order (same as MuJoCo body indexing)
    bodies = []

    def walk_body(elem):
        body_name = elem.get("name", "unnamed")
        meshes = []
        # Find geoms with type="mesh" in this body
        for geom in elem.findall("geom"):
            if geom.get("type") == "mesh":
                mesh_name = geom.get("mesh", "")
                if mesh_name in mesh_name_to_file:
                    stl_file = mesh_name_to_file[mesh_name]
                    if stl_file not in meshes:
                        meshes.append(stl_file)
        bodies.append({"name": body_name, "meshes": meshes})
        # Recurse into child bodies
        for child in elem.findall("body"):
            walk_body(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for top_body in worldbody.findall("body"):
            walk_body(top_body)

    return bodies


# ---------------------------------------------------------------------------
# MuJoCo model loading (from test_tracker_mujoco.py)
# ---------------------------------------------------------------------------

def _patch_mjcf_xml(xml_path: pathlib.Path) -> str:
    """Patch MJCF for standalone MuJoCo use (strip sensors, add ground)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or "ground" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("rgba", "0.7 0.7 0.7 1")

    return ET.tostring(root, encoding="unicode")


def load_mujoco_model(mjcf_path: str, stiffness: list, damping: list, physics_dt: float):
    """Load MuJoCo model configured for G1 policy deployment."""
    mjcf_file = pathlib.Path(mjcf_path)
    if not mjcf_file.is_absolute():
        candidates = [
            PROTOMOTIONS_ROOT / mjcf_path,
            PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / mjcf_path,
        ]
        for c in candidates:
            if c.exists():
                mjcf_file = c
                break

    if not mjcf_file.exists():
        raise FileNotFoundError(f"Cannot find MJCF: {mjcf_path}")

    log.info(f"Loading MuJoCo model: {mjcf_file}")
    patched_xml = _patch_mjcf_xml(mjcf_file)

    asset_dir = str(mjcf_file.parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)

    # Set physics timestep
    model.opt.timestep = physics_dt
    log.info(f"  Physics timestep: {physics_dt}s ({1.0/physics_dt:.0f}Hz)")

    # Zero passive forces
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Configure implicit PD actuators
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator mismatch: nu={num_actuators}, stiff={len(stiffness)}, damp={len(damping)}"
    )
    for i in range(num_actuators):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0

    log.info(f"  {num_actuators} actuators, {model.nbody} bodies, {model.nq} qpos, {model.nv} qvel")
    return model, data


# ---------------------------------------------------------------------------
# Robot state reading (from test_tracker_mujoco.py)
# ---------------------------------------------------------------------------

def read_robot_state(data, anchor_body_index: int, root_body_index: int = 0):
    """Read robot state from MuJoCo buffers."""
    body_rot_wxyz = data.xquat[1:].copy()
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)

    # For root body, use canonical free-joint quaternion
    root_rot_wxyz = data.qpos[3:7].copy()
    body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz)

    root_local_ang_vel = data.qvel[3:6].copy().astype(np.float32)

    return {
        "dof_pos":            data.qpos[7:].copy().astype(np.float32),
        "dof_vel":            data.qvel[6:].copy().astype(np.float32),
        "body_rot":           body_rot.astype(np.float32),
        "root_local_ang_vel": root_local_ang_vel,
    }


def set_initial_pose(model, data, motion_player):
    """Initialize robot at first frame of motion."""
    frame0 = motion_player.get_state_at_frame(0)
    root_pos = frame0["body_pos"][0]
    root_quat = frame0["body_rot"][0]  # xyzw

    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat[[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:] = frame0["dof_pos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def build_onnx_inputs(robot_state, future_refs, onnx_name_to_key, anchor_body_index, num_dofs, prev_actions=None):
    """Assemble ONNX input dict from robot state + motion futures."""
    dof_pos = robot_state["dof_pos"]
    dof_vel = robot_state["dof_vel"]
    body_rot = robot_state["body_rot"]
    root_local_ang_vel = robot_state["root_local_ang_vel"]

    anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)

    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]

    key_to_array = {
        "current.dof_pos":             dof_pos[None],
        "current.dof_vel":             dof_vel[None],
        "current.anchor_rot":          anchor_rot[None],
        "current.root_local_ang_vel":  root_local_ang_vel[None],
        "historical.processed_actions": prev_actions[None, None],
        "mimic.future_anchor_rot":     future_anchor_rot[None],
        "mimic.future_rot":            future_refs["body_rot"][None],
        "mimic.future_dof_pos":        future_refs["dof_pos"][None],
        "mimic.future_dof_vel":        future_refs["dof_vel"][None],
    }

    onnx_inputs = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

    return onnx_inputs


# ---------------------------------------------------------------------------
# Simulation + Export
# ---------------------------------------------------------------------------

def simulate_and_export(
    onnx_path: str,
    motion_file: str,
    output_json_path: str,
    mjcf_path: str,
    body_mesh_mapping: list,
    subsample_factor: int = 1,
) -> dict:
    """Run G1 ONNX policy simulation and export body poses as JSON.

    Parameters
    ----------
    onnx_path: Path to unified_pipeline.onnx
    motion_file: Path to .motion file (ProtoMotions format)
    output_json_path: Where to save the output JSON
    mjcf_path: Path to G1 MJCF XML
    body_mesh_mapping: List of {name, meshes} from parse_body_mesh_mapping
    subsample_factor: Export every Nth frame (1=all frames at 50fps)

    Returns
    -------
    stats dict with simulation metrics
    """
    yaml_path = onnx_path.replace(".onnx", ".yaml")

    # Load YAML metadata
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    robot_meta = meta["robot"]
    timing = meta["timing"]
    motion_meta = meta["motion"]
    control = meta["control"]
    runtime = meta["_runtime"]

    anchor_body_index = robot_meta["anchor_body_index"]
    root_body_index = robot_meta["root_body_index"]
    num_bodies = robot_meta["num_bodies"]
    num_dofs = robot_meta["num_dofs"]
    control_dt = timing["control_dt"]
    decimation = timing["decimation"]
    physics_dt = timing["physics_dt"]
    future_step_indices = motion_meta["future_step_indices"]
    stiffness = control["stiffness"]
    damping = control["damping"]
    pd_target_max_accel = control.get("pd_target_max_accel")
    action_ema_alpha = control.get("action_ema_alpha", 1.0)
    onnx_name_to_key = runtime["onnx_name_to_in_key"]

    log.info(f"Loading ONNX: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_out_names = [out.name for out in session.get_outputs()]

    # Load motion
    log.info(f"Loading motion: {motion_file}")
    player = MotionPlayer(motion_file, control_dt=control_dt)
    log.info(f"  Motion: {player.total_frames} frames @ {1.0/control_dt:.0f}Hz "
             f"(duration={player.total_frames * control_dt:.2f}s)")

    # Load MuJoCo model
    model, data = load_mujoco_model(mjcf_path, stiffness, damping, physics_dt)

    # Verify body count matches
    mj_num_bodies = model.nbody - 1  # exclude world body
    if mj_num_bodies != num_bodies:
        log.warning(f"Body count mismatch: MuJoCo={mj_num_bodies}, YAML={num_bodies}")

    # Initialize
    set_initial_pose(model, data, player)

    # EMA state
    use_ema = action_ema_alpha < 1.0
    ema_prev_targets = None
    prev_pd = None
    prev_prev_pd = None
    prev_actions = None
    heading_offset = None

    # Collect frames
    frames_data = []
    total_steps = 0
    max_pd_diff = 0.0
    fall_detected = False

    t_start = time.perf_counter()

    for frame_idx in range(player.total_frames):
        # Read robot state
        robot_state = read_robot_state(data, anchor_body_index, root_body_index)

        # Heading offset on first step
        if heading_offset is None:
            robot_anchor_rot = robot_state["body_rot"][anchor_body_index]
            motion_anchor_rot = player.get_state_at_frame(0)["body_rot"][anchor_body_index]
            heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)

        # Get future motion references
        future_refs = player.get_future_references(frame_idx, future_step_indices)
        future_refs["body_rot"] = apply_heading_offset_np(heading_offset, future_refs["body_rot"])

        # Build ONNX inputs
        onnx_inputs = build_onnx_inputs(
            robot_state=robot_state,
            future_refs=future_refs,
            onnx_name_to_key=onnx_name_to_key,
            anchor_body_index=anchor_body_index,
            num_dofs=num_dofs,
            prev_actions=prev_actions,
        )

        # ONNX inference
        ort_out = session.run(actual_out_names, onnx_inputs)
        pd_targets = ort_out[1].squeeze().copy()  # joint_pos_targets

        # PD target acceleration clamp
        if pd_target_max_accel is not None and prev_pd is not None and prev_prev_pd is not None:
            delta = pd_targets - prev_pd
            prev_delta = prev_pd - prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
            pd_targets = prev_pd + prev_delta + clamped_accel

        prev_prev_pd = prev_pd
        prev_pd = pd_targets.copy()

        # EMA filter
        if use_ema:
            if ema_prev_targets is None:
                ema_prev_targets = pd_targets.copy()
            pd_targets = action_ema_alpha * pd_targets + (1.0 - action_ema_alpha) * ema_prev_targets
            ema_prev_targets = pd_targets.copy()

        prev_actions = pd_targets.copy()

        # Write PD targets and step physics
        data.ctrl[:] = pd_targets
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # Check for fall (root height too low)
        root_height = float(data.qpos[2])
        if root_height < 0.3:
            log.warning(f"  Fall detected at frame {frame_idx}, root_h={root_height:.3f}")
            fall_detected = True
            break

        # Export frame (subsample)
        if frame_idx % subsample_factor == 0:
            # body_pos: data.xpos[1:] (skip world body), shape [num_bodies, 3]
            body_pos = data.xpos[1:num_bodies+1].copy()
            # body_quat: data.xquat[1:] in wxyz format, shape [num_bodies, 4]
            body_quat = data.xquat[1:num_bodies+1].copy()

            frames_data.append({
                "body_pos": body_pos.tolist(),
                "body_quat": body_quat.tolist(),  # wxyz for Three.js
            })

        # Track ref error
        ref_dof_pos = player.get_state_at_frame(frame_idx)["dof_pos"]
        diff = float(np.abs(data.qpos[7:] - ref_dof_pos).max())
        if diff > max_pd_diff:
            max_pd_diff = diff

        total_steps += 1

        if frame_idx % 200 == 0:
            log.info(f"  frame={frame_idx:4d}/{player.total_frames}  "
                     f"root_h={root_height:.3f}  max_err={max_pd_diff:.4f}")

    elapsed = time.perf_counter() - t_start
    log.info(f"  Simulation done: {total_steps} steps in {elapsed:.1f}s "
             f"({total_steps/max(elapsed,1e-6):.0f} steps/s)")

    # Determine output fps
    output_fps = 50.0 / subsample_factor  # control at 50Hz

    # Build output JSON
    output_data = {
        "type": "robot_frames",
        "robot": "g1",
        "fps": output_fps,
        "num_frames": len(frames_data),
        "num_bodies": num_bodies,
        "bodies": body_mesh_mapping[:num_bodies],
        "frames": frames_data,
    }

    # Save
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

    file_size_mb = os.path.getsize(output_json_path) / 1e6
    log.info(f"  Saved: {output_json_path} ({file_size_mb:.1f} MB, {len(frames_data)} frames)")

    stats = {
        "total_steps": total_steps,
        "total_frames_exported": len(frames_data),
        "fall_detected": fall_detected,
        "max_joint_error_rad": max_pd_diff,
        "root_height_final": float(data.qpos[2]),
        "elapsed_seconds": elapsed,
        "output_fps": output_fps,
    }
    return stats


# ---------------------------------------------------------------------------
# Retargeting step
# ---------------------------------------------------------------------------

def retarget_npz_to_motion(
    npz_path: pathlib.Path,
    output_dir: pathlib.Path,
    smpl_model_path: str = None,
    urdf_path: str = None,
    fps: int = 30,
) -> pathlib.Path:
    """Run retargeting pipeline: motion_135 NPZ → .motion file.

    Uses pipeline_motion_to_robot.py as subprocess.
    Returns path to generated .motion file.
    """
    if smpl_model_path is None:
        smpl_model_path = str(DEFAULT_SMPL_MODEL)
    if urdf_path is None:
        urdf_path = str(DEFAULT_URDF)

    motion_output_dir = output_dir / "motion_files"
    motion_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if .motion file already exists
    stem = npz_path.stem
    existing_motion = motion_output_dir / f"{stem}.motion"
    if existing_motion.exists():
        log.info(f"  Retargeted .motion already exists: {existing_motion}")
        return existing_motion

    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--input", str(npz_path),
        "--output", str(motion_output_dir),
        "--smpl-model-path", smpl_model_path,
        "--urdf", urdf_path,
        "--fps", str(fps),
        "--keep-intermediates",
    ]

    log.info(f"  Running retargeting: {npz_path.name}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error(f"  Retargeting failed for {npz_path.name}:")
        log.error(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Retargeting failed: {npz_path.name}")

    # Find the generated .motion file
    motion_files = list(motion_output_dir.glob(f"{stem}*.motion"))
    if not motion_files:
        # Also check in subdirectories
        motion_files = list(motion_output_dir.glob(f"**/{stem}*.motion"))

    if not motion_files:
        raise FileNotFoundError(f"No .motion file found after retargeting {npz_path.name}")

    return motion_files[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G1 Robot RL Tracker Export: NPZ → robot body poses JSON"
    )
    parser.add_argument("--input", type=str, help="Single input motion_135 NPZ")
    parser.add_argument("--input-dir", type=str, help="Directory of NPZ files to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for robot JSON files")
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX),
                        help="Path to G1 ONNX model")
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF),
                        help="Path to G1 MJCF XML")
    parser.add_argument("--smpl-model-path", type=str, default=str(DEFAULT_SMPL_MODEL),
                        help="Path to SMPL model directory")
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF),
                        help="Path to G1 URDF for retargeting")
    parser.add_argument("--fps", type=int, default=30,
                        help="Input motion FPS")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Export every Nth control frame (default 2 → 25fps output)")
    parser.add_argument("--skip-retarget", action="store_true",
                        help="Skip retargeting, assume .motion files exist")
    parser.add_argument("--motion-dir", type=str,
                        help="Directory containing pre-generated .motion files")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect NPZ files
    npz_files = []
    if args.input:
        npz_files = [pathlib.Path(args.input)]
    elif args.input_dir:
        npz_files = sorted(pathlib.Path(args.input_dir).glob("*.npz"))
    else:
        parser.error("Must specify --input or --input-dir")

    if not npz_files:
        log.error("No NPZ files found!")
        sys.exit(1)

    log.info(f"Processing {len(npz_files)} NPZ files")
    log.info(f"Output: {output_dir}")

    # Parse body-mesh mapping from MJCF
    mjcf_path = pathlib.Path(args.mjcf)
    body_mesh_mapping = parse_body_mesh_mapping(mjcf_path)
    log.info(f"Parsed {len(body_mesh_mapping)} bodies from MJCF")

    # Verify ONNX exists
    onnx_path = args.onnx
    if not pathlib.Path(onnx_path).exists():
        log.error(f"ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Process each NPZ
    all_stats = []
    for i, npz_path in enumerate(npz_files):
        log.info(f"\n{'='*60}")
        log.info(f"  [{i+1}/{len(npz_files)}] {npz_path.name}")
        log.info(f"{'='*60}")

        stem = npz_path.stem
        output_json = output_dir / f"{stem}.json"

        # Skip if already exists
        if output_json.exists():
            log.info(f"  Output already exists, skipping: {output_json}")
            all_stats.append({"name": stem, "skipped": True})
            continue

        try:
            # Step 1: Retarget NPZ → .motion
            if args.skip_retarget and args.motion_dir:
                motion_dir = pathlib.Path(args.motion_dir)
                motion_candidates = list(motion_dir.glob(f"{stem}*.motion"))
                if not motion_candidates:
                    log.error(f"  No .motion file found for {stem} in {motion_dir}")
                    all_stats.append({"name": stem, "error": "no_motion_file"})
                    continue
                motion_path = motion_candidates[0]
            else:
                motion_path = retarget_npz_to_motion(
                    npz_path, output_dir,
                    smpl_model_path=args.smpl_model_path,
                    urdf_path=args.urdf,
                    fps=args.fps,
                )

            log.info(f"  Motion file: {motion_path}")

            # Step 2: Simulate + Export
            stats = simulate_and_export(
                onnx_path=onnx_path,
                motion_file=str(motion_path),
                output_json_path=str(output_json),
                mjcf_path=args.mjcf,
                body_mesh_mapping=body_mesh_mapping,
                subsample_factor=args.subsample,
            )
            stats["name"] = stem
            all_stats.append(stats)

        except Exception as e:
            log.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({"name": stem, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"  G1 RL Tracker Export Complete")
    print(f"{'='*60}")
    successes = [s for s in all_stats if "error" not in s and not s.get("skipped")]
    failures = [s for s in all_stats if "error" in s]
    skipped = [s for s in all_stats if s.get("skipped")]
    print(f"  Total: {len(npz_files)}")
    print(f"  Success: {len(successes)}")
    print(f"  Skipped (already exist): {len(skipped)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print(f"\n  Failures:")
        for f in failures:
            print(f"    - {f['name']}: {f['error']}")

    if successes:
        avg_steps = np.mean([s["total_steps"] for s in successes])
        avg_error = np.mean([s["max_joint_error_rad"] for s in successes])
        falls = sum(1 for s in successes if s.get("fall_detected"))
        print(f"\n  Stats (successful):")
        print(f"    Avg steps: {avg_steps:.0f}")
        print(f"    Avg max joint error: {avg_error:.4f} rad")
        print(f"    Falls: {falls}/{len(successes)}")

    # Save summary
    summary_path = output_dir / "_export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

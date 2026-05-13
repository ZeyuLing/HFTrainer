#!/usr/bin/env python3
"""Run ONNX tracker in MuJoCo physics sim and export tracked motion as cache .pt.

Takes a reference motion cache .pt (kinematic FK data from gmr_to_protomotions)
and runs the ONNX tracker policy in closed-loop MuJoCo simulation. Exports the
resulting physics-simulated body states as a new cache .pt file in the same
format, suitable for convert_cache_to_json.py → Three.js visualization.

The KEY difference from the reference cache:
  - Reference: body positions/rotations from mj_forward() (pure FK, no physics)
  - Tracked:   body positions/rotations from mj_step() (full physics simulation
               with PD control, contact forces, gravity, etc.)

This means tracked motion will show:
  - Feet that don't slide through ground (contact constraints)
  - Robot falling over if motion is physically implausible
  - Realistic dynamics (inertia, momentum, ground reaction forces)

Usage
-----
::

    # Single motion
    python scripts/embodied/run_tracker_export.py \
        --motion output/embodied_comparison/data/caches/pipeline_00000.pt \
        --output output/embodied_comparison/data/tracked_caches/tracked_00000.pt

    # Batch mode (all .pt files in a directory)
    python scripts/embodied/run_tracker_export.py \
        --motion-dir output/embodied_comparison/data/caches/ \
        --output-dir output/embodied_comparison/data/tracked_caches/ \
        --pattern 'pipeline_*.pt'
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Locate repo root (hf_trainer/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # scripts/embodied/../../ = hf_trainer/

_DEFAULT_ONNX = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker"
    / "g1-bones-deploy/compiled_models/unified_pipeline.onnx"
)
_DEFAULT_MJCF = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
)


# ---------------------------------------------------------------------------
# Import ProtoMotions deployment utilities
# ---------------------------------------------------------------------------


def _setup_protomotions_imports():
    """Add ProtoMotions to sys.path so deployment.* imports work."""
    proto_root = _REPO_ROOT / "ref_repo" / "ProtoMotions"
    if str(proto_root) not in sys.path:
        sys.path.insert(0, str(proto_root))


# ---------------------------------------------------------------------------
# MuJoCo model loading (with PD actuators — needed for simulation)
# ---------------------------------------------------------------------------


def _patch_mjcf_xml(xml_path: Path) -> str:
    """Patch MJCF for standalone MuJoCo (strip sensors, add ground/light)."""
    import xml.etree.ElementTree as ET

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

        if not worldbody.findall("light"):
            light = ET.SubElement(worldbody, "light")
            light.set("pos", "2 0 5.0")
            light.set("dir", "0 0 -1")
            light.set("diffuse", "0.4 0.4 0.4")
            light.set("specular", "0.1 0.1 0.1")
            light.set("directional", "true")

    return ET.tostring(root, encoding="unicode")


def load_mujoco_model_for_sim(
    mjcf_path: str, stiffness: list, damping: list, physics_dt: float
):
    """Load MuJoCo model configured for physics simulation (with PD actuators).

    Replicates the setup from test_tracker_mujoco.py::load_mujoco_model().
    """
    import tempfile

    mjcf_file = Path(mjcf_path)
    if not mjcf_file.is_absolute():
        mjcf_file = _REPO_ROOT / mjcf_path
    if not mjcf_file.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf_file}")

    log.info(f"Loading MuJoCo model: {mjcf_file}")
    patched_xml = _patch_mjcf_xml(mjcf_file)

    asset_dir = str(mjcf_file.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as tmp:
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

    # Zero passive forces (match training)
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Configure implicit PD actuators
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator count mismatch: nu={num_actuators}, "
        f"stiffness={len(stiffness)}, damping={len(damping)}"
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

    log.info(
        f"  {num_actuators} actuators, {model.nbody} bodies, "
        f"nq={model.nq}, nv={model.nv}"
    )
    return model, data


# ---------------------------------------------------------------------------
# Core: run tracker and export states
# ---------------------------------------------------------------------------


def run_tracker_and_export(
    motion_cache_path: str,
    output_path: str,
    onnx_path: str = _DEFAULT_ONNX,
    mjcf_path: str = _DEFAULT_MJCF,
) -> dict:
    """Run ONNX tracker on a reference motion and export tracked body states.

    Parameters
    ----------
    motion_cache_path:
        Path to reference motion cache .pt (kinematic FK data).
    output_path:
        Path to write the tracked motion cache .pt.
    onnx_path:
        Path to ONNX tracker model.
    mjcf_path:
        Path to MJCF XML model file.

    Returns
    -------
    dict with summary info: status, num_frames, fall_frame, root_height_min, etc.
    """
    import onnxruntime as ort
    import torch
    import yaml

    _setup_protomotions_imports()
    from deployment.state_utils import (
        mujoco_wxyz_to_xyzw,
        compute_anchor_rot_np,
        compute_yaw_offset_np,
        apply_heading_offset_np,
    )
    from deployment.motion_utils import MotionPlayer

    # ------------------------------------------------------------------
    # Load YAML metadata
    # ------------------------------------------------------------------
    yaml_path = onnx_path.replace(".onnx", ".yaml")
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
    damping_ctrl = control["damping"]
    pd_target_max_accel = control.get("pd_target_max_accel")
    action_ema_alpha = control.get("action_ema_alpha", 1.0)
    onnx_name_to_key = runtime["onnx_name_to_in_key"]

    # ------------------------------------------------------------------
    # Load ONNX session
    # ------------------------------------------------------------------
    log.info(f"Loading ONNX: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    # ------------------------------------------------------------------
    # Load motion via MotionPlayer
    # ------------------------------------------------------------------
    player = MotionPlayer(motion_cache_path, control_dt=control_dt)
    num_frames = player.total_frames
    log.info(
        f"Motion: {num_frames} frames @ {1.0/control_dt:.0f} Hz "
        f"(duration={num_frames * control_dt:.2f}s)"
    )

    # ------------------------------------------------------------------
    # Load MuJoCo model (with PD actuators for simulation)
    # ------------------------------------------------------------------
    model, data = load_mujoco_model_for_sim(
        mjcf_path, stiffness, damping_ctrl, physics_dt
    )

    # ------------------------------------------------------------------
    # Set initial pose
    # ------------------------------------------------------------------
    frame0 = player.get_state_at_frame(0)
    root_pos = frame0["body_pos"][0]
    root_quat_xyzw = frame0["body_rot"][0]
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:] = frame0["dof_pos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    out_body_pos = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    out_body_rot = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    out_body_vel = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    out_body_ang_vel = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    out_dof_pos = np.zeros((num_frames, num_dofs), dtype=np.float32)
    out_dof_vel = np.zeros((num_frames, num_dofs), dtype=np.float32)

    # ------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------
    use_ema = action_ema_alpha < 1.0
    ema_prev_targets = None
    prev_pd = None
    prev_prev_pd = None
    prev_actions = None
    heading_offset = None

    # Fall detection
    FALL_HEIGHT_THRESHOLD = 0.3  # root height below this = fallen
    fall_frame = None
    root_height_min = float("inf")

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    for frame_idx in range(num_frames):
        # ---- Record current state BEFORE stepping ----
        # Body positions from xpos (skip world body at index 0)
        out_body_pos[frame_idx] = data.xpos[1 : num_bodies + 1].copy().astype(np.float32)
        # Body rotations from xquat (wxyz -> xyzw)
        body_rot_wxyz = data.xquat[1 : num_bodies + 1].copy()
        out_body_rot[frame_idx] = mujoco_wxyz_to_xyzw(body_rot_wxyz).astype(np.float32)
        # Use canonical free-joint quat for root body
        root_rot_wxyz = data.qpos[3:7].copy()
        out_body_rot[frame_idx, root_body_index] = mujoco_wxyz_to_xyzw(
            root_rot_wxyz
        ).astype(np.float32)
        # DOF state
        out_dof_pos[frame_idx] = data.qpos[7:].copy().astype(np.float32)
        out_dof_vel[frame_idx] = data.qvel[6:].copy().astype(np.float32)
        # Body velocities from cvel: [ang_vel(3), lin_vel(3)]
        cvel = data.cvel[1 : num_bodies + 1].copy()
        out_body_ang_vel[frame_idx] = cvel[:, 0:3].astype(np.float32)
        out_body_vel[frame_idx] = cvel[:, 3:6].astype(np.float32)

        # ---- Fall detection ----
        root_h = float(data.qpos[2])
        root_height_min = min(root_height_min, root_h)
        if root_h < FALL_HEIGHT_THRESHOLD and fall_frame is None:
            fall_frame = frame_idx
            log.warning(f"  FALL detected at frame {frame_idx} (root_h={root_h:.3f})")

        # ---- Read robot state for policy ----
        robot_state = {
            "dof_pos": data.qpos[7:].copy().astype(np.float32),
            "dof_vel": data.qvel[6:].copy().astype(np.float32),
            "body_rot": out_body_rot[frame_idx].copy(),
            "root_local_ang_vel": data.qvel[3:6].copy().astype(np.float32),
        }

        # ---- Heading offset (first step) ----
        if heading_offset is None:
            robot_anchor_rot = robot_state["body_rot"][anchor_body_index]
            motion_anchor_rot = player.get_state_at_frame(0)["body_rot"][
                anchor_body_index
            ]
            heading_offset = compute_yaw_offset_np(
                robot_anchor_rot, motion_anchor_rot
            )

        # ---- Future motion references ----
        future_refs = player.get_future_references(frame_idx, future_step_indices)
        future_refs["body_rot"] = apply_heading_offset_np(
            heading_offset, future_refs["body_rot"]
        )

        # ---- Build ONNX inputs ----
        anchor_rot = compute_anchor_rot_np(
            robot_state["body_rot"], anchor_body_index
        )

        if prev_actions is None:
            prev_actions_input = np.zeros(num_dofs, dtype=np.float32)
        else:
            prev_actions_input = prev_actions

        future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]

        key_to_array = {
            "current.dof_pos": robot_state["dof_pos"][None],
            "current.dof_vel": robot_state["dof_vel"][None],
            "current.anchor_rot": anchor_rot[None],
            "current.root_local_ang_vel": robot_state["root_local_ang_vel"][None],
            "historical.processed_actions": prev_actions_input[None, None],
            "mimic.future_anchor_rot": future_anchor_rot[None],
            "mimic.future_rot": future_refs["body_rot"][None],
            "mimic.future_dof_pos": future_refs["dof_pos"][None],
            "mimic.future_dof_vel": future_refs["dof_vel"][None],
        }

        onnx_inputs = {}
        for onnx_name, sem_key in onnx_name_to_key.items():
            if sem_key in key_to_array:
                onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

        # ---- ONNX inference ----
        ort_out = session.run(actual_out_names, onnx_inputs)
        pd_targets = ort_out[1].squeeze().copy()  # joint_pos_targets

        # ---- PD target acceleration clamp ----
        if (
            pd_target_max_accel is not None
            and prev_pd is not None
            and prev_prev_pd is not None
        ):
            delta = pd_targets - prev_pd
            prev_delta = prev_pd - prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(
                accel, -pd_target_max_accel, pd_target_max_accel
            )
            pd_targets = prev_pd + prev_delta + clamped_accel

        prev_prev_pd = prev_pd
        prev_pd = pd_targets.copy()

        # ---- EMA action filter ----
        if use_ema:
            if ema_prev_targets is None:
                ema_prev_targets = pd_targets.copy()
            pd_targets = (
                action_ema_alpha * pd_targets
                + (1.0 - action_ema_alpha) * ema_prev_targets
            )
            ema_prev_targets = pd_targets.copy()

        prev_actions = pd_targets.copy()

        # ---- Apply control and step physics ----
        data.ctrl[:] = pd_targets
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # ---- Progress logging ----
        if frame_idx % 100 == 0:
            elapsed = time.perf_counter() - t_start
            speed = (frame_idx + 1) * control_dt / max(elapsed, 1e-6)
            log.info(
                f"  frame={frame_idx:4d}/{num_frames}  "
                f"root_h={root_h:.3f}  speed={speed:.1f}x"
            )

    elapsed = time.perf_counter() - t_start
    log.info(
        f"Simulation complete: {num_frames} frames in {elapsed:.1f}s "
        f"({num_frames * control_dt / max(elapsed, 1e-6):.1f}x realtime)"
    )

    # ------------------------------------------------------------------
    # Export tracked cache .pt
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tracked_cache = {
        "dof_pos": out_dof_pos,
        "dof_vel": out_dof_vel,
        "body_rot": out_body_rot,
        "body_pos": out_body_pos,
        "body_vel": out_body_vel,
        "body_ang_vel": out_body_ang_vel,
        "control_dt": control_dt,
        "num_frames": num_frames,
    }
    torch.save(tracked_cache, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"Tracked cache saved: {output_path} ({file_size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Determine status
    # ------------------------------------------------------------------
    if fall_frame is not None:
        status = "fell"
    elif root_height_min < 0.4:
        status = "unstable"
    else:
        status = "success"

    summary = {
        "status": status,
        "num_frames": num_frames,
        "fall_frame": fall_frame,
        "root_height_min": float(root_height_min),
        "duration_s": num_frames * control_dt,
        "sim_time_s": elapsed,
        "output_path": output_path,
    }
    log.info(f"Status: {status} | root_height_min={root_height_min:.3f}")
    return summary


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def batch_run(
    motion_dir: str,
    output_dir: str,
    onnx_path: str,
    mjcf_path: str,
    pattern: str = "pipeline_*.pt",
    prefix: str = "tracked_",
    max_motions: int | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run tracker on all matching motion caches in a directory.

    For each pipeline_XXXXX.pt, produces tracked_XXXXX.pt.

    Returns list of summary dicts.
    """
    files = sorted(glob.glob(os.path.join(motion_dir, pattern)))
    if not files:
        log.error(f"No files matching {pattern} in {motion_dir}")
        return []

    if max_motions is not None:
        files = files[:max_motions]

    os.makedirs(output_dir, exist_ok=True)
    results = []

    log.info(f"\n{'='*60}")
    log.info(f"Batch tracker export: {len(files)} motions")
    log.info(f"  Input:  {motion_dir}")
    log.info(f"  Output: {output_dir}")
    log.info(f"{'='*60}\n")

    for i, cache_path in enumerate(files):
        name = Path(cache_path).stem  # e.g. pipeline_00000
        # Replace pipeline_ prefix with tracked_ prefix
        tracked_name = name.replace("pipeline_", prefix)
        output_path = os.path.join(output_dir, f"{tracked_name}.pt")

        log.info(f"\n--- [{i+1}/{len(files)}] {name} ---")

        if skip_existing and os.path.exists(output_path):
            log.info(f"  Skipping (exists): {output_path}")
            results.append({"id": name, "status": "skipped", "output_path": output_path})
            continue

        try:
            summary = run_tracker_and_export(
                motion_cache_path=cache_path,
                output_path=output_path,
                onnx_path=onnx_path,
                mjcf_path=mjcf_path,
            )
            summary["id"] = name
            results.append(summary)
        except Exception as e:
            log.error(f"  FAILED: {e}")
            results.append({"id": name, "status": "error", "error": str(e)})

    # Write summary JSON
    summary_path = os.path.join(output_dir, "tracker_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSummary written to {summary_path}")

    # Print summary table
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")
    for r in results:
        status = r.get("status", "?")
        name = r.get("id", "?")
        fall = r.get("fall_frame")
        rh = r.get("root_height_min")
        info = ""
        if fall is not None:
            info = f" (fell at frame {fall})"
        elif rh is not None:
            info = f" (root_h_min={rh:.3f})"
        log.info(f"  {name:20s} {status:10s}{info}")

    n_success = sum(1 for r in results if r.get("status") == "success")
    n_fell = sum(1 for r in results if r.get("status") == "fell")
    n_err = sum(1 for r in results if r.get("status") == "error")
    log.info(f"\nTotal: {len(results)} | Success: {n_success} | Fell: {n_fell} | Error: {n_err}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX tracker in MuJoCo and export tracked motion cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Single motion mode
    parser.add_argument(
        "--motion",
        help="Path to single reference motion cache .pt",
    )
    parser.add_argument(
        "--output",
        help="Output path for tracked cache .pt (single mode)",
    )

    # Batch mode
    parser.add_argument(
        "--motion-dir",
        help="Directory with reference cache .pt files (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for tracked caches (batch mode)",
    )
    parser.add_argument(
        "--pattern",
        default="pipeline_*.pt",
        help="Glob pattern for batch mode",
    )
    parser.add_argument(
        "--prefix",
        default="tracked_",
        help="Prefix for output filenames (replaces 'pipeline_')",
    )
    parser.add_argument(
        "--max-motions",
        type=int,
        default=None,
        help="Max motions to process in batch mode",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip motions that already have tracked output",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even if tracked output exists",
    )

    # Model paths
    parser.add_argument(
        "--onnx",
        default=_DEFAULT_ONNX,
        help="Path to ONNX tracker model",
    )
    parser.add_argument(
        "--mjcf",
        default=_DEFAULT_MJCF,
        help="Path to MJCF XML model file",
    )

    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip_existing

    # Resolve ONNX path
    onnx_path = args.onnx
    if not Path(onnx_path).is_absolute():
        onnx_path = str(_REPO_ROOT / onnx_path)

    # Batch mode
    if args.motion_dir:
        if not args.output_dir:
            args.output_dir = os.path.join(
                os.path.dirname(args.motion_dir.rstrip("/")),
                "tracked_caches",
            )
        batch_run(
            motion_dir=args.motion_dir,
            output_dir=args.output_dir,
            onnx_path=onnx_path,
            mjcf_path=args.mjcf,
            pattern=args.pattern,
            prefix=args.prefix,
            max_motions=args.max_motions,
            skip_existing=skip_existing,
        )
    # Single motion mode
    elif args.motion:
        if not args.output:
            # Auto-generate output path
            name = Path(args.motion).stem.replace("pipeline_", "tracked_")
            args.output = str(
                Path(args.motion).parent.parent / "tracked_caches" / f"{name}.pt"
            )
        run_tracker_and_export(
            motion_cache_path=args.motion,
            output_path=args.output,
            onnx_path=onnx_path,
            mjcf_path=args.mjcf,
        )
    else:
        parser.error("Specify either --motion (single) or --motion-dir (batch)")


if __name__ == "__main__":
    main()

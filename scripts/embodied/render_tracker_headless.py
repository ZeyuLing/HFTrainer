#!/usr/bin/env python3
"""Headless rendering of G1 robot motion from a motion cache.

Renders a G1 robot in MuJoCo without a display, saving frames as PNG images
and optionally compositing them into a video via ffmpeg.

Two modes:
  --mode reference  : Directly set qpos from the motion cache (no simulation).
                      Shows what the generated/reference motion looks like.
  --mode tracked    : Run the ONNX tracker policy in closed-loop simulation,
                      then render the resulting motion.

Examples
--------
::

    # Render reference motion (fast, no ONNX needed)
    python scripts/embodied/render_tracker_headless.py \
        --motion path/to/motion_cache.pt \
        --output-dir /tmp/render_ref \
        --mode reference

    # Render tracked motion (requires ONNX model)
    python scripts/embodied/render_tracker_headless.py \
        --motion path/to/motion_cache.pt \
        --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
        --output-dir /tmp/render_tracked \
        --mode tracked

    # Create video from rendered frames (requires ffmpeg)
    python scripts/embodied/render_tracker_headless.py \
        --motion path/to/motion_cache.pt \
        --output-dir /tmp/render_ref \
        --mode reference --video
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Default paths
_DEFAULT_MJCF = (
    "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
)
_DEFAULT_ONNX = (
    "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/"
    "g1-bones-deploy/compiled_models/unified_pipeline.onnx"
)

# Locate repo root (hf_trainer/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # scripts/embodied/../../ = hf_trainer/


# ---------------------------------------------------------------------------
# Motion cache loader
# ---------------------------------------------------------------------------


def load_motion_cache(motion_path: str) -> dict:
    """Load a motion cache .pt file.

    Expected keys:
        dof_pos:      (T, 29)      - joint angles
        dof_vel:      (T, 29)      - joint velocities
        body_rot:     (T, 33, 4)   - body rotations (xyzw quaternion)
        body_pos:     (T, 33, 3)   - body positions
        body_vel:     (T, 33, 3)   - body linear velocities
        body_ang_vel: (T, 33, 3)   - body angular velocities
        control_dt:   float        - time between frames
        num_frames:   int          - total number of frames
    """
    import torch

    data = torch.load(motion_path, map_location="cpu", weights_only=False)

    cache = {}
    for key in ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel"):
        if key in data:
            arr = data[key]
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            cache[key] = np.asarray(arr, dtype=np.float32)
        else:
            raise KeyError(f"Motion cache missing key: '{key}'")

    cache["control_dt"] = float(data["control_dt"])
    cache["num_frames"] = int(data["num_frames"])

    log.info(
        f"Loaded motion cache: {cache['num_frames']} frames, "
        f"dt={cache['control_dt']:.4f}s ({1.0/cache['control_dt']:.0f} Hz), "
        f"duration={cache['num_frames'] * cache['control_dt']:.2f}s"
    )
    log.info(
        f"  dof_pos: {cache['dof_pos'].shape}, body_rot: {cache['body_rot'].shape}, "
        f"body_pos: {cache['body_pos'].shape}"
    )
    return cache


# ---------------------------------------------------------------------------
# MuJoCo model loading (simplified for rendering — no PD actuators needed)
# ---------------------------------------------------------------------------


def load_mujoco_model_for_rendering(
    mjcf_path: str,
    render_width: int = 1280,
    render_height: int = 720,
) -> tuple:
    """Load a MuJoCo model for rendering purposes (no actuator setup).

    Patches the MJCF to add ground plane, lighting, and offscreen framebuffer.

    Returns (model, data).
    """
    import tempfile
    import xml.etree.ElementTree as ET

    mjcf_file = Path(mjcf_path)
    if not mjcf_file.is_absolute():
        mjcf_file = _REPO_ROOT / mjcf_path
    if not mjcf_file.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf_file}")

    log.info(f"Loading MuJoCo model: {mjcf_file}")

    # Parse and patch XML
    tree = ET.parse(str(mjcf_file))
    root = tree.getroot()

    # Strip sensors (may reference missing sites)
    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    # Ensure offscreen framebuffer is large enough for rendering
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(render_width))
    glob.set("offheight", str(render_height))

    # Ensure worldbody has ground plane and light
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
            ground.set("size", "10 10 0.05")
            ground.set("rgba", "0.8 0.8 0.8 1")
            ground.set("conaffinity", "1")
            ground.set("condim", "3")

        # Add better lighting for rendering
        if not worldbody.findall("light"):
            # Key light
            light1 = ET.SubElement(worldbody, "light")
            light1.set("name", "key_light")
            light1.set("pos", "3 -2 5")
            light1.set("dir", "-0.3 0.2 -1")
            light1.set("diffuse", "0.6 0.6 0.6")
            light1.set("specular", "0.2 0.2 0.2")
            light1.set("directional", "true")
            # Fill light
            light2 = ET.SubElement(worldbody, "light")
            light2.set("name", "fill_light")
            light2.set("pos", "-2 3 4")
            light2.set("dir", "0.2 -0.3 -1")
            light2.set("diffuse", "0.3 0.3 0.3")
            light2.set("specular", "0.05 0.05 0.05")
            light2.set("directional", "true")

    patched_xml = ET.tostring(root, encoding="unicode")

    # Write to temp file in same directory (for mesh asset resolution)
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
    log.info(f"  Model: {model.nbody} bodies, nq={model.nq}, nv={model.nv}, nu={model.nu}")
    return model, data


# ---------------------------------------------------------------------------
# Reference mode: render motion cache directly
# ---------------------------------------------------------------------------


def render_reference_mode(
    cache: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_dir: Path,
    width: int = 1280,
    height: int = 720,
    skip_frames: int = 2,
    max_frames: int | None = None,
    camera_distance: float = 3.0,
    camera_elevation: float = -20.0,
    camera_azimuth: float = 135.0,
) -> int:
    """Render reference motion by setting qpos directly from the cache.

    For each frame:
      qpos = [root_pos(3), root_rot_wxyz(4), dof_pos(29)]

    The cache body_rot is in xyzw format; MuJoCo qpos expects wxyz.

    Returns the number of frames rendered.
    """
    num_frames = cache["num_frames"]
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    body_pos = cache["body_pos"]    # (T, 33, 3)
    body_rot = cache["body_rot"]    # (T, 33, 4) xyzw
    dof_pos = cache["dof_pos"]      # (T, 29)

    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Setup camera scene options
    scene_option = mujoco.MjvOption()

    # Track pelvis body (body index 1 in MuJoCo = first non-world body)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1  # pelvis (first non-world body)
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = camera_azimuth

    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_count = 0
    t_start = time.perf_counter()

    for frame_idx in range(0, num_frames, skip_frames):
        # Extract root pose from body 0 (pelvis)
        root_pos = body_pos[frame_idx, 0, :]           # (3,)
        root_rot_xyzw = body_rot[frame_idx, 0, :]      # (4,) xyzw
        # Convert xyzw -> wxyz for MuJoCo
        root_rot_wxyz = np.array([
            root_rot_xyzw[3],  # w
            root_rot_xyzw[0],  # x
            root_rot_xyzw[1],  # y
            root_rot_xyzw[2],  # z
        ], dtype=np.float32)

        dof = dof_pos[frame_idx, :]                    # (29,)

        # Set qpos: [root_pos(3), root_rot_wxyz(4), dof(29)] = 36
        # But check model.nq to be safe
        qpos = np.concatenate([root_pos, root_rot_wxyz, dof])
        assert len(qpos) <= model.nq, (
            f"qpos length mismatch: constructed {len(qpos)}, model expects {model.nq}"
        )
        data.qpos[: len(qpos)] = qpos

        # Zero velocities (just rendering, no simulation)
        data.qvel[:] = 0.0

        # Forward kinematics (computes body positions/orientations for rendering)
        mujoco.mj_forward(model, data)

        # Render
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        pixels = renderer.render()

        # Save frame
        frame_path = output_dir / f"frame_{rendered_count:06d}.png"
        _save_png(pixels, frame_path, frame_idx=frame_idx)

        rendered_count += 1

        if rendered_count % 50 == 0:
            elapsed = time.perf_counter() - t_start
            fps_render = rendered_count / max(elapsed, 1e-6)
            log.info(
                f"  Rendered {rendered_count} frames "
                f"(source frame {frame_idx}/{num_frames}) "
                f"[{fps_render:.1f} render fps]"
            )

    elapsed = time.perf_counter() - t_start
    log.info(
        f"Reference rendering complete: {rendered_count} frames in {elapsed:.1f}s "
        f"({rendered_count / max(elapsed, 1e-6):.1f} fps)"
    )

    renderer.close()
    return rendered_count


# ---------------------------------------------------------------------------
# Tracked mode: run ONNX policy and render
# ---------------------------------------------------------------------------


def render_tracked_mode(
    cache: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    onnx_path: str,
    output_dir: Path,
    width: int = 1280,
    height: int = 720,
    skip_frames: int = 2,
    max_frames: int | None = None,
    camera_distance: float = 3.0,
    camera_elevation: float = -20.0,
    camera_azimuth: float = 135.0,
) -> int:
    """Run the ONNX tracker policy in simulation and render each frame.

    This replicates the core loop from test_tracker_mujoco.py but renders
    headlessly instead of using a viewer.

    Returns the number of frames rendered.
    """
    import onnxruntime as ort
    import yaml

    # Load YAML metadata alongside the ONNX model
    yaml_path = onnx_path.replace(".onnx", ".yaml")
    if not Path(yaml_path).exists():
        raise FileNotFoundError(
            f"ONNX metadata YAML not found: {yaml_path}\n"
            "The .yaml file must be alongside the .onnx file."
        )

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

    log.info(f"Tracker config: {num_dofs} DOFs, {num_bodies} bodies")
    log.info(f"  control_dt={control_dt}s, decimation={decimation}, physics_dt={physics_dt}s")
    log.info(f"  anchor_body={anchor_body_index}, root_body={root_body_index}")
    log.info(f"  future_steps={future_step_indices}")

    # Configure model physics (match training)
    model.opt.timestep = physics_dt
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Configure implicit PD actuators
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator count mismatch: model.nu={num_actuators}, "
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

    # Load ONNX session
    log.info(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]
    log.info(f"  ONNX inputs:  {actual_in_names}")
    log.info(f"  ONNX outputs: {actual_out_names}")

    # Setup ProtoMotions deployment utilities
    # ONNX path: .../ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/xxx.onnx
    # Need to go up 5 levels to reach ProtoMotions root
    proto_root = Path(onnx_path).resolve().parents[5]
    if not (proto_root / "deployment").exists():
        # Try _REPO_ROOT based path
        proto_root = _REPO_ROOT / "ref_repo" / "ProtoMotions"
    log.info(f"  ProtoMotions root: {proto_root}")
    if str(proto_root) not in sys.path:
        sys.path.insert(0, str(proto_root))

    from deployment.state_utils import (
        mujoco_wxyz_to_xyzw,
        compute_anchor_rot_np,
        compute_yaw_offset_np,
        apply_heading_offset_np,
    )
    from deployment.motion_utils import MotionPlayer

    # Create MotionPlayer from cache
    import torch
    # Save cache to temp file so MotionPlayer can load it
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(cache, tmp.name)
        tmp_motion_path = tmp.name

    try:
        player = MotionPlayer(tmp_motion_path, control_dt=control_dt)
    finally:
        os.unlink(tmp_motion_path)

    num_frames = player.total_frames
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    # Set initial pose
    frame0 = player.get_state_at_frame(0)
    root_pos = frame0["body_pos"][0]
    root_quat_xyzw = frame0["body_rot"][0]
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:] = frame0["dof_pos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = camera_azimuth

    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulation state
    use_ema = action_ema_alpha < 1.0
    ema_prev_targets = None
    prev_pd = None
    prev_prev_pd = None
    prev_actions = None
    heading_offset = None

    rendered_count = 0
    t_start = time.perf_counter()

    for frame_idx in range(num_frames):
        # ---- Read robot state ----
        body_rot_wxyz = data.xquat[1:].copy()
        body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz).astype(np.float32)
        # Use canonical free-joint quat for root
        root_rot_wxyz_mj = data.qpos[3:7].copy()
        body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz_mj).astype(np.float32)

        robot_state = {
            "dof_pos": data.qpos[7:].copy().astype(np.float32),
            "dof_vel": data.qvel[6:].copy().astype(np.float32),
            "body_rot": body_rot,
            "root_local_ang_vel": data.qvel[3:6].copy().astype(np.float32),
        }

        # ---- Heading offset (first step) ----
        if heading_offset is None:
            robot_anchor_rot = robot_state["body_rot"][anchor_body_index]
            motion_anchor_rot = player.get_state_at_frame(0)["body_rot"][anchor_body_index]
            heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)

        # ---- Future motion references ----
        future_refs = player.get_future_references(frame_idx, future_step_indices)
        future_refs["body_rot"] = apply_heading_offset_np(
            heading_offset, future_refs["body_rot"]
        )

        # ---- Build ONNX inputs ----
        anchor_rot = compute_anchor_rot_np(robot_state["body_rot"], anchor_body_index)

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
        if pd_target_max_accel is not None and prev_pd is not None and prev_prev_pd is not None:
            delta = pd_targets - prev_pd
            prev_delta = prev_pd - prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
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

        # ---- Render (every skip_frames) ----
        if frame_idx % skip_frames == 0:
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            pixels = renderer.render()

            frame_path = output_dir / f"frame_{rendered_count:06d}.png"
            _save_png(pixels, frame_path, frame_idx=frame_idx)
            rendered_count += 1

            if rendered_count % 50 == 0:
                elapsed = time.perf_counter() - t_start
                fps_render = rendered_count / max(elapsed, 1e-6)
                root_h = float(data.qpos[2])
                log.info(
                    f"  Rendered {rendered_count} frames "
                    f"(sim frame {frame_idx}/{num_frames}, root_h={root_h:.3f}) "
                    f"[{fps_render:.1f} render fps]"
                )

    elapsed = time.perf_counter() - t_start
    log.info(
        f"Tracked rendering complete: {rendered_count} frames in {elapsed:.1f}s "
        f"({rendered_count / max(elapsed, 1e-6):.1f} fps)"
    )

    renderer.close()
    return rendered_count


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def _save_png(pixels: np.ndarray, path: Path, frame_idx: int | None = None) -> None:
    """Save an RGB pixel array as PNG.

    Tries PIL first for frame-number overlay, falls back to mujoco or raw PNG.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(pixels)
        if frame_idx is not None:
            draw = ImageDraw.Draw(img)
            text = f"Frame {frame_idx}"
            # Use default font (no external font file needed)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except (OSError, IOError):
                font = ImageFont.load_default()
            # Draw text with shadow for visibility
            draw.text((12, 12), text, fill=(0, 0, 0), font=font)
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        img.save(str(path))
    except ImportError:
        # Fallback: save raw PNG without overlay using basic PNG writing
        _save_png_raw(pixels, path)


def _save_png_raw(pixels: np.ndarray, path: Path) -> None:
    """Minimal PNG save without PIL (uses zlib for DEFLATE compression)."""
    import struct
    import zlib

    h, w, c = pixels.shape
    assert c == 3, f"Expected RGB, got {c} channels"

    def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = _make_chunk(b"IHDR", ihdr_data)

    # IDAT: filter byte (0=None) prepended to each row
    raw_data = b""
    for row in range(h):
        raw_data += b"\x00" + pixels[row].tobytes()
    compressed = zlib.compress(raw_data, 6)
    idat = _make_chunk(b"IDAT", compressed)

    # IEND
    iend = _make_chunk(b"IEND", b"")

    with open(str(path), "wb") as f:
        f.write(signature + ihdr + idat + iend)


# ---------------------------------------------------------------------------
# Video compositing
# ---------------------------------------------------------------------------


def create_video(
    output_dir: Path,
    fps: float = 25.0,
    video_name: str = "output.mp4",
) -> Path | None:
    """Create a video from rendered PNG frames using ffmpeg.

    Returns the path to the video file, or None if ffmpeg is not available.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("ffmpeg not found in PATH — skipping video creation")
        return None

    video_path = output_dir / video_name
    frame_pattern = str(output_dir / "frame_%06d.png")

    # Try encoders in order: libx264 (best quality), h264_nvenc (NVIDIA HW), mpeg4 (fallback)
    encoders = [
        ["libx264", "-preset", "medium", "-crf", "18"],
        ["h264_nvenc", "-preset", "medium"],
        ["mpeg4", "-q:v", "5"],
    ]

    log.info(f"Creating video: {video_path}")

    for enc_args in encoders:
        encoder_name = enc_args[0]
        cmd = [
            ffmpeg,
            "-y",  # overwrite
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", *enc_args,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(video_path),
        ]
        log.info(f"  Trying encoder: {encoder_name}")
        log.info(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            break
        log.warning(f"  Encoder {encoder_name} failed, trying next...")

    if result.returncode != 0:
        log.error(f"All video encoders failed. Last error:\n{result.stderr}")
        return None

    log.info(f"Video saved: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return video_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Headless rendering of G1 robot motion from a motion cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--motion", required=True,
        help="Path to motion cache .pt file",
    )
    p.add_argument(
        "--mjcf", default=_DEFAULT_MJCF,
        help="Path to MJCF XML model file",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to save rendered PNG frames",
    )
    p.add_argument(
        "--mode", choices=["reference", "tracked"], default="reference",
        help="Rendering mode: 'reference' sets qpos directly, 'tracked' runs ONNX policy",
    )
    p.add_argument(
        "--onnx", default=_DEFAULT_ONNX,
        help="Path to ONNX tracker model (only for --mode tracked)",
    )
    p.add_argument(
        "--skip-frames", type=int, default=2,
        help="Render every N-th frame from the motion",
    )
    p.add_argument(
        "--width", type=int, default=1280,
        help="Render width in pixels",
    )
    p.add_argument(
        "--height", type=int, default=720,
        help="Render height in pixels",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of source frames to process (default: all)",
    )
    p.add_argument(
        "--camera-distance", type=float, default=3.0,
        help="Camera distance from tracked body",
    )
    p.add_argument(
        "--camera-elevation", type=float, default=-20.0,
        help="Camera elevation angle (degrees)",
    )
    p.add_argument(
        "--camera-azimuth", type=float, default=135.0,
        help="Camera azimuth angle (degrees)",
    )
    p.add_argument(
        "--video", action="store_true", default=False,
        help="Also create an MP4 video from the rendered frames",
    )
    p.add_argument(
        "--video-fps", type=float, default=None,
        help="Video frame rate (default: derived from motion dt and skip-frames)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    # Load motion cache
    cache = load_motion_cache(args.motion)

    # Load MuJoCo model
    model, data = load_mujoco_model_for_rendering(
        args.mjcf, render_width=args.width, render_height=args.height
    )

    # Run rendering
    if args.mode == "reference":
        log.info("=== Reference mode: rendering motion cache directly ===")
        num_rendered = render_reference_mode(
            cache=cache,
            model=model,
            data=data,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            camera_distance=args.camera_distance,
            camera_elevation=args.camera_elevation,
            camera_azimuth=args.camera_azimuth,
        )

    elif args.mode == "tracked":
        log.info("=== Tracked mode: running ONNX policy simulation ===")
        onnx_path = args.onnx
        if not Path(onnx_path).is_absolute():
            onnx_path = str(_REPO_ROOT / onnx_path)
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        num_rendered = render_tracked_mode(
            cache=cache,
            model=model,
            data=data,
            onnx_path=onnx_path,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            camera_distance=args.camera_distance,
            camera_elevation=args.camera_elevation,
            camera_azimuth=args.camera_azimuth,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    log.info(f"\nTotal frames rendered: {num_rendered}")
    log.info(f"Output directory: {output_dir}")

    # Optionally create video
    if args.video and num_rendered > 0:
        # Compute video FPS: motion plays at 1/control_dt Hz, we skip frames
        if args.video_fps is not None:
            video_fps = args.video_fps
        else:
            motion_fps = 1.0 / cache["control_dt"]
            video_fps = motion_fps / args.skip_frames
            # Clamp to reasonable range
            video_fps = max(10.0, min(video_fps, 60.0))
        log.info(f"Video FPS: {video_fps:.1f}")
        video_path = create_video(output_dir, fps=video_fps)
        if video_path:
            log.info(f"Video: {video_path}")


if __name__ == "__main__":
    main()

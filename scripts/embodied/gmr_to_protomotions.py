#!/usr/bin/env python3
"""Convert GMR retarget PKL → ProtoMotions MotionPlayer cache format (.pt).

GMR PKL format (from gmr_retarget_headless.py):
    fps:        float       - Motion frame rate (e.g. 30)
    root_pos:   (T, 3)      - Root translation
    root_rot:   (T, 4)      - Root rotation (xyzw quaternion)
    dof_pos:    (T, 29)     - Joint positions

ProtoMotions MotionPlayer cache format:
    dof_pos:      (T', 29)       - Joint positions
    dof_vel:      (T', 29)       - Joint velocities (finite diff)
    body_rot:     (T', 33, 4)    - Body rotations (xyzw quaternion)
    body_pos:     (T', 33, 3)    - Body positions
    body_vel:     (T', 33, 3)    - Body linear velocities (finite diff)
    body_ang_vel: (T', 33, 3)    - Body angular velocities (finite diff)
    control_dt:   float          - 0.02 (50Hz)
    num_frames:   int            - T'

Conversion steps:
1. Load GMR PKL
2. Use MuJoCo FK with ProtoMotions G1 MJCF to compute all 33 body states
3. Resample from source FPS (30Hz) to target control rate (50Hz)
4. Compute velocities via finite differences
5. Save as torch cache

Usage:
    python scripts/embodied/gmr_to_protomotions.py \
        --input /tmp/g1_retarget.pkl \
        --output /tmp/g1_motion_cache.pt \
        --mjcf ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml \
        --control-dt 0.02
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp


def quat_xyzw_to_wxyz(q):
    """Convert quaternion from xyzw to wxyz format."""
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q):
    """Convert quaternion from wxyz to xyzw format."""
    return q[..., [1, 2, 3, 0]]


def _get_gmr_rot_offset():
    """Get GMR's pelvis rot_offset as a scipy Rotation.

    GMR's IK config (smplx_to_g1.json) defines:
        "pelvis": {"rot_offset": [0.5, -0.5, -0.5, -0.5]}  (wxyz)

    This is a 120° rotation mapping Y-up/Z-forward (SMPL-X) to
    Z-up/X-forward (MuJoCo). It maps coordinate axes as:
        X_smplx → Z_mujoco
        Y_smplx → X_mujoco
        Z_smplx → Y_mujoco
    """
    # wxyz [0.5, -0.5, -0.5, -0.5] → xyzw [-0.5, -0.5, -0.5, 0.5]
    rot_offset_xyzw = np.array([-0.5, -0.5, -0.5, 0.5])
    return R.from_quat(rot_offset_xyzw)


def remove_gmr_root_offset(root_rot_xyzw):
    """Remove the Y-up→Z-up frame conversion baked into GMR's pelvis quaternion.

    GMR's IK config applies rot_offset to the pelvis rotation, which gets baked
    into the output quaternion. ProtoMotions expects near-identity pelvis rotation
    for a standing robot (only yaw varies).

    Fix: right-multiply by rot_offset.inv() to undo the frame conversion.
    Since GMR applies: q_out = q_smplx * rot_offset
    We undo:           q_corrected = q_out * rot_offset.inv()

    Args:
        root_rot_xyzw: (T, 4) root rotation quaternions in xyzw format

    Returns:
        corrected_root_rot_xyzw: (T, 4) corrected quaternions in xyzw format
    """
    rot_offset = _get_gmr_rot_offset()
    root_rots = R.from_quat(root_rot_xyzw)
    corrected = root_rots * rot_offset.inv()
    return corrected.as_quat().astype(root_rot_xyzw.dtype)


def convert_root_pos_to_zup(root_pos):
    """Convert GMR root_pos from SMPL-X Y-up frame to MuJoCo Z-up frame.

    GMR's IK solver passes through the input SMPL-X translation without frame
    conversion. SMPL-X uses Y-up (X-right, Y-up, Z-forward), while MuJoCo uses
    Z-up (X-forward, Y-left, Z-up).

    The rot_offset.inv() maps:
        [x, y, z]_smplx → [z, x, y]_mujoco
    Putting the Y (height) into Z (MuJoCo's up axis).

    Args:
        root_pos: (T, 3) root positions in SMPL-X Y-up frame

    Returns:
        root_pos_zup: (T, 3) root positions in MuJoCo Z-up frame
    """
    rot_offset = _get_gmr_rot_offset()
    # Apply rot_offset.inv() to each position vector
    return rot_offset.inv().apply(root_pos).astype(root_pos.dtype)


def _patch_mjcf_xml(xml_path):
    """Patch MJCF for standalone MuJoCo use (strip sensors, add ground).

    Mirrors ProtoMotions' test_tracker_mujoco.py::_patch_mjcf_xml().
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Strip sensors (may reference missing sites)
    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    # Strip contact pairs that reference 'floor' (we add our own)
    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            geom1 = pair.get("geom1", "")
            geom2 = pair.get("geom2", "")
            if "floor" in geom1 or "floor" in geom2:
                contact.remove(pair)

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


def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos,
                         foot_body_indices=None, ground_clearance=0.0,
                         mode="global"):
    """Correct root_pos Z so feet are at ground level based on FK.

    GMR's height scaling can produce a root_pos Z that doesn't match what the
    DOF angles actually produce kinematically. This function:
    1. Runs FK with current root positions
    2. Finds the lowest foot Z from FK per frame
    3. Adjusts root_pos Z based on correction mode

    Modes:
        - "global": Single Z offset for entire motion (most stable, recommended)
        - "smooth": Per-frame Z offset smoothed with Savitzky-Golay filter
        - "perframe": Per-frame independent Z offset (original, causes jitter)

    Args:
        mjcf_path: Path to MuJoCo XML model
        root_pos: (T, 3) root positions (Z-up, will be modified in-place)
        root_rot_xyzw: (T, 4) root rotations in xyzw format
        dof_pos: (T, 29) joint positions
        foot_body_indices: list of body indices to check for ground contact.
            Default: [7, 13] (left_ankle_roll_link, right_ankle_roll_link)
        ground_clearance: target minimum foot Z (default 0.0)
        mode: correction mode - "global", "smooth", or "perframe"

    Returns:
        corrected_root_pos: (T, 3) with adjusted Z values
        foot_min_z_before: (T,) minimum foot Z before correction (for diagnostics)
    """
    import mujoco
    import tempfile
    import os

    if foot_body_indices is None:
        foot_body_indices = [7, 13]  # left_ankle_roll_link, right_ankle_roll_link

    # Load MuJoCo model
    patched_xml = _patch_mjcf_xml(mjcf_path)
    asset_dir = str(Path(mjcf_path).parent)

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

    T = root_pos.shape[0]
    corrected_root_pos = root_pos.copy()
    foot_min_z_before = np.zeros(T, dtype=np.float64)

    # Pass 1: compute per-frame foot min Z
    for t in range(T):
        root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw[t])
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz
        data.qpos[7:] = dof_pos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        min_foot_z = np.inf
        for bi in foot_body_indices:
            foot_z = data.xpos[bi + 1][2]  # +1 for world body offset
            if foot_z < min_foot_z:
                min_foot_z = foot_z

        foot_min_z_before[t] = min_foot_z

    # Pass 2: compute Z offsets based on mode
    per_frame_offsets = ground_clearance - foot_min_z_before

    if mode == "global":
        # Single offset: use median of per-frame offsets (robust to outliers)
        global_offset = np.median(per_frame_offsets)
        corrected_root_pos[:, 2] = root_pos[:, 2] + global_offset
        print(f"    FK ground mode=global: single Z offset = {global_offset:.4f}m")
    elif mode == "smooth":
        # Smoothed per-frame offset using Savitzky-Golay filter
        # Window = 31 frames at 30Hz ≈ 1 second, ensures slow smooth transitions
        win_len = min(31, T if T % 2 == 1 else T - 1)
        if win_len < 5:
            win_len = 5 if T >= 5 else (T if T % 2 == 1 else max(T - 1, 3))
        smooth_offsets = savgol_filter(per_frame_offsets, window_length=win_len, polyorder=3)
        corrected_root_pos[:, 2] = root_pos[:, 2] + smooth_offsets
        print(f"    FK ground mode=smooth: offset range [{smooth_offsets.min():.4f}, {smooth_offsets.max():.4f}]m")
    elif mode == "perframe":
        # Original per-frame independent correction (kept for backward compat)
        corrected_root_pos[:, 2] = root_pos[:, 2] + per_frame_offsets
        print(f"    FK ground mode=perframe: offset range [{per_frame_offsets.min():.4f}, {per_frame_offsets.max():.4f}]m")
    else:
        raise ValueError(f"Unknown fk_ground_mode: {mode}. Use 'global', 'smooth', or 'perframe'.")

    return corrected_root_pos, foot_min_z_before


def mujoco_fk(mjcf_path, root_pos, root_rot_xyzw, dof_pos):
    """Run MuJoCo forward kinematics to get all body states.

    Args:
        mjcf_path: Path to MuJoCo XML model
        root_pos: (T, 3) root translation
        root_rot_xyzw: (T, 4) root rotation in xyzw
        dof_pos: (T, 29) joint positions

    Returns:
        body_pos: (T, num_bodies, 3) body positions
        body_rot_xyzw: (T, num_bodies, 4) body rotations in xyzw
    """
    import mujoco
    import tempfile
    import os

    # Patch the MJCF XML (strip sensors, add ground, remove floor contact pairs)
    patched_xml = _patch_mjcf_xml(mjcf_path)
    asset_dir = str(Path(mjcf_path).parent)

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

    T = root_pos.shape[0]
    num_bodies = model.nbody - 1  # exclude world body

    body_pos_all = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_rot_all = np.zeros((T, num_bodies, 4), dtype=np.float32)  # xyzw

    for t in range(T):
        # Set qpos: [pos(3), quat_wxyz(4), dof(29)] = 36 total
        root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw[t])
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz
        data.qpos[7:] = dof_pos[t]

        # Zero velocities
        data.qvel[:] = 0.0

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Extract body states (skip world body at index 0)
        for b in range(num_bodies):
            body_pos_all[t, b] = data.xpos[b + 1]
            # MuJoCo body quaternions are wxyz
            body_rot_wxyz = data.xquat[b + 1]
            body_rot_all[t, b] = quat_wxyz_to_xyzw(body_rot_wxyz)

    print(f"  MuJoCo FK: {T} frames, {num_bodies} bodies")
    return body_pos_all, body_rot_all


def resample_motion(times_src, dof_pos, body_pos, body_rot_xyzw, control_dt):
    """Resample motion from source timestamps to target control rate.

    Uses linear interpolation for positions and SLERP for quaternions.

    Args:
        times_src: (T,) source timestamps
        dof_pos: (T, 29)
        body_pos: (T, num_bodies, 3)
        body_rot_xyzw: (T, num_bodies, 4)
        control_dt: target time step

    Returns:
        Resampled arrays at target rate
    """
    T_src = len(times_src)
    duration = times_src[-1]
    T_tgt = int(np.floor(duration / control_dt)) + 1
    times_tgt = np.arange(T_tgt) * control_dt

    # Clamp target times to source range
    times_tgt = np.clip(times_tgt, times_src[0], times_src[-1])

    print(f"  Resampling: {T_src} frames ({1.0/(times_src[1]-times_src[0]):.0f}Hz) -> {T_tgt} frames ({1.0/control_dt:.0f}Hz)")

    # Resample dof_pos: linear interp
    dof_interp = interp1d(times_src, dof_pos, axis=0, kind='linear')
    dof_pos_resampled = dof_interp(times_tgt).astype(np.float32)

    # Resample body_pos: linear interp per body
    num_bodies = body_pos.shape[1]
    body_pos_resampled = np.zeros((T_tgt, num_bodies, 3), dtype=np.float32)
    for b in range(num_bodies):
        bp_interp = interp1d(times_src, body_pos[:, b, :], axis=0, kind='linear')
        body_pos_resampled[:, b, :] = bp_interp(times_tgt)

    # Resample body_rot: SLERP per body
    body_rot_resampled = np.zeros((T_tgt, num_bodies, 4), dtype=np.float32)
    for b in range(num_bodies):
        # scipy Rotation expects xyzw (which is what we have)
        # But scipy's Slerp works with Rotation objects
        rots = R.from_quat(body_rot_xyzw[:, b, :])  # xyzw
        slerp_fn = Slerp(times_src, rots)
        rots_resampled = slerp_fn(times_tgt)
        body_rot_resampled[:, b, :] = rots_resampled.as_quat()  # xyzw

    return times_tgt, dof_pos_resampled, body_pos_resampled, body_rot_resampled


def compute_velocities(dof_pos, body_pos, body_rot_xyzw, dt):
    """Compute velocities via Savitzky-Golay smoothed derivatives.

    Uses Savitzky-Golay filter with deriv=1 to compute velocities as the
    analytical derivative of a local polynomial fit. This is inherently
    smoother than raw finite differences and avoids amplifying high-frequency
    noise from the retargeting pipeline.

    Falls back to finite differences for very short sequences (< 9 frames).

    Args:
        dof_pos: (T, 29)
        body_pos: (T, num_bodies, 3)
        body_rot_xyzw: (T, num_bodies, 4)
        dt: time step

    Returns:
        dof_vel: (T, 29)
        body_vel: (T, num_bodies, 3)
        body_ang_vel: (T, num_bodies, 3)
    """
    T = dof_pos.shape[0]
    num_bodies = body_pos.shape[1]

    # Choose window length based on available frames
    # At 50Hz: window=9 ≈ 0.18s, window=7 ≈ 0.14s
    win_len = min(9, T if T % 2 == 1 else T - 1)
    if win_len < 5:
        # Fall back to finite differences for very short sequences
        print(f"    Warning: sequence too short ({T} frames) for Savitzky-Golay, using finite diff")
        dof_vel = np.zeros_like(dof_pos)
        dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
        dof_vel[0] = dof_vel[1]

        body_vel = np.zeros_like(body_pos)
        body_vel[1:] = (body_pos[1:] - body_pos[:-1]) / dt
        body_vel[0] = body_vel[1]
    else:
        # Savitzky-Golay 1st derivative: smooth velocity estimation
        dof_vel = savgol_filter(dof_pos, window_length=win_len, polyorder=3, deriv=1, delta=dt, axis=0)

        # Body linear velocity: per-body Savitzky-Golay derivative
        body_vel = np.zeros_like(body_pos)
        for b in range(num_bodies):
            body_vel[:, b, :] = savgol_filter(body_pos[:, b, :], window_length=win_len, polyorder=3, deriv=1, delta=dt, axis=0)

    # Body angular velocity from quaternion differences
    # (angular velocity doesn't benefit as much from SG since it's already
    #  computed from relative rotations — but we still smooth the result)
    body_ang_vel = np.zeros((T, num_bodies, 3), dtype=np.float32)
    for b in range(num_bodies):
        rots = R.from_quat(body_rot_xyzw[:, b, :])  # xyzw
        for t in range(1, T):
            # Relative rotation from t-1 to t
            drot = rots[t] * rots[t - 1].inv()
            rotvec = drot.as_rotvec()
            body_ang_vel[t, b] = rotvec / dt
        body_ang_vel[0, b] = body_ang_vel[1, b]

    # Smooth angular velocity if sequence is long enough
    if T >= 5:
        ang_win = min(7, T if T % 2 == 1 else T - 1)
        if ang_win >= 5:
            for b in range(num_bodies):
                body_ang_vel[:, b, :] = savgol_filter(
                    body_ang_vel[:, b, :], window_length=ang_win, polyorder=3, axis=0
                )

    # Boundary stabilization: apply cosine ramp to first/last N frames
    # The IK solver and SG boundary effects create artificial transients at
    # sequence edges. A cosine ease-in/ease-out suppresses these gracefully.
    ramp_frames = min(5, T // 4)  # 5 frames at 50Hz = 0.1s ramp
    if ramp_frames >= 2:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_frames) / ramp_frames))  # 0→1
        ramp = ramp.astype(np.float32)
        # Apply to DOF velocity
        dof_vel[:ramp_frames] *= ramp[:, None]
        dof_vel[-ramp_frames:] *= ramp[::-1, None]
        # Apply to body velocity
        body_vel[:ramp_frames] *= ramp[:, None, None]
        body_vel[-ramp_frames:] *= ramp[::-1, None, None]
        # Apply to angular velocity
        body_ang_vel[:ramp_frames] *= ramp[:, None, None]
        body_ang_vel[-ramp_frames:] *= ramp[::-1, None, None]
        print(f"    Applied velocity boundary ramp ({ramp_frames} frames)")

    return dof_vel.astype(np.float32), body_vel.astype(np.float32), body_ang_vel.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert GMR PKL to ProtoMotions cache")
    parser.add_argument("--input", required=True, help="Input GMR PKL file")
    parser.add_argument("--output", required=True, help="Output ProtoMotions .pt cache file")
    parser.add_argument("--mjcf", required=True, help="Path to ProtoMotions G1 MJCF XML")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Target control dt (default 0.02 = 50Hz)")
    parser.add_argument("--fk-ground-correction", action="store_true", default=True,
                        help="Adjust root Z so feet are at ground level based on FK (default: True)")
    parser.add_argument("--no-fk-ground-correction", dest="fk_ground_correction", action="store_false",
                        help="Disable FK-based ground correction")
    parser.add_argument("--ground-clearance", type=float, default=0.0,
                        help="Target minimum foot Z after FK correction (default: 0.0)")
    parser.add_argument("--fk-ground-mode", choices=["global", "smooth", "perframe"],
                        default="global",
                        help="FK ground correction mode: 'global' (single offset, most stable), "
                             "'smooth' (savgol-smoothed per-frame), 'perframe' (original, jittery). Default: global")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable Savitzky-Golay smoothing on resampled dof_pos/body_pos "
                             "(for debugging; smoothing is enabled by default)")
    args = parser.parse_args()

    # 1. Load GMR PKL
    print(f"Loading GMR PKL: {args.input}")
    with open(args.input, "rb") as f:
        gmr_data = pickle.load(f)

    root_pos = gmr_data["root_pos"].astype(np.float64)     # (T, 3)
    root_rot = gmr_data["root_rot"].astype(np.float64)     # (T, 4) xyzw
    dof_pos = gmr_data["dof_pos"].astype(np.float64)       # (T, 29)
    src_fps = float(gmr_data["fps"])

    T = root_pos.shape[0]
    print(f"  Frames: {T}, FPS: {src_fps}")
    print(f"  root_pos range: [{root_pos.min():.4f}, {root_pos.max():.4f}]")
    print(f"  root_rot sample[0]: {root_rot[0]}")
    print(f"  dof_pos range: [{dof_pos.min():.4f}, {dof_pos.max():.4f}]")

    # 2. Convert GMR output from SMPL-X frame (Y-up) to MuJoCo frame (Z-up)
    # GMR passes through SMPL-X conventions:
    #   - root_rot has rot_offset baked in (120° Y-up→Z-up rotation)
    #   - root_pos is in SMPL-X Y-up coordinate frame
    # We need both in MuJoCo's native Z-up frame for ProtoMotions.

    print(f"\nConverting root translation from Y-up to Z-up...")
    root_pos_before = root_pos.copy()
    root_pos = convert_root_pos_to_zup(root_pos)
    print(f"  root_pos[0] before (Y-up): {root_pos_before[0]}")
    print(f"  root_pos[0] after  (Z-up): {root_pos[0]}")
    print(f"  root_pos Z (height) range: [{root_pos[:,2].min():.4f}, {root_pos[:,2].max():.4f}]")

    print(f"\nRemoving GMR root rotation offset (Y-up→Z-up frame conversion)...")
    root_rot_before = root_rot.copy()
    root_rot = remove_gmr_root_offset(root_rot)
    print(f"  root_rot[0] before: {root_rot_before[0]}")
    print(f"  root_rot[0] after:  {root_rot[0]}")
    # Verify: standing pose should now be near-identity (only yaw)
    from scipy.spatial.transform import Rotation as R_check
    r0 = R_check.from_quat(root_rot[0])
    euler0 = r0.as_euler('xyz', degrees=True)
    angle0 = r0.magnitude() * 180 / np.pi
    print(f"  corrected pelvis euler: x={euler0[0]:.1f}, y={euler0[1]:.1f}, z={euler0[2]:.1f}")
    print(f"  corrected pelvis angle from identity: {angle0:.1f} deg")

    # 2b. FK-based ground correction: adjust root_pos Z so feet are at ground level
    if args.fk_ground_correction:
        print(f"\nApplying FK-based ground correction (clearance={args.ground_clearance:.3f}m, mode={args.fk_ground_mode})...")
        root_pos, foot_min_z = fk_ground_correction(
            args.mjcf, root_pos, root_rot, dof_pos,
            ground_clearance=args.ground_clearance,
            mode=args.fk_ground_mode,
        )
        print(f"  Foot min Z before correction: [{foot_min_z.min():.4f}, {foot_min_z.max():.4f}] (mean: {foot_min_z.mean():.4f})")
        print(f"  root_pos Z after correction: [{root_pos[:,2].min():.4f}, {root_pos[:,2].max():.4f}]")
        print(f"  root_pos[0] after correction: {root_pos[0]}")

    # 3. MuJoCo FK to get all body states
    print(f"\nRunning MuJoCo FK with: {args.mjcf}")
    body_pos, body_rot = mujoco_fk(args.mjcf, root_pos, root_rot, dof_pos)
    print(f"  body_pos shape: {body_pos.shape}, range: [{body_pos.min():.4f}, {body_pos.max():.4f}]")
    print(f"  body_rot shape: {body_rot.shape}")

    # 4. Resample to target control rate
    src_dt = 1.0 / src_fps
    times_src = np.arange(T) * src_dt

    print(f"\nResampling from {src_fps:.0f}Hz to {1.0/args.control_dt:.0f}Hz")
    times_tgt, dof_pos_r, body_pos_r, body_rot_r = resample_motion(
        times_src, dof_pos, body_pos, body_rot, args.control_dt
    )

    # 5. Smooth resampled data before velocity computation
    if not args.no_smooth:
        T_r = dof_pos_r.shape[0]
        sg_win = min(11, T_r if T_r % 2 == 1 else T_r - 1)  # ~0.22s at 50Hz
        if sg_win >= 5:
            print(f"  Smoothing resampled dof_pos (Savitzky-Golay, window={sg_win})...")
            dof_pos_r = savgol_filter(dof_pos_r, window_length=sg_win, polyorder=3, axis=0).astype(np.float32)
            # Also smooth body positions
            num_bodies_r = body_pos_r.shape[1]
            for b in range(num_bodies_r):
                body_pos_r[:, b, :] = savgol_filter(
                    body_pos_r[:, b, :], window_length=sg_win, polyorder=3, axis=0
                )
    else:
        print(f"  Skipping post-resample smoothing (--no-smooth)")

    # 6. Compute velocities
    print(f"\nComputing velocities...")
    dof_vel, body_vel, body_ang_vel = compute_velocities(
        dof_pos_r, body_pos_r, body_rot_r, args.control_dt
    )

    T_final = dof_pos_r.shape[0]
    print(f"  Final frames: {T_final}")
    print(f"  dof_pos: {dof_pos_r.shape}, dof_vel: {dof_vel.shape}")
    print(f"  body_pos: {body_pos_r.shape}, body_rot: {body_rot_r.shape}")
    print(f"  body_vel: {body_vel.shape}, body_ang_vel: {body_ang_vel.shape}")

    # 7. Save as ProtoMotions cache
    import torch

    cache = {
        "dof_pos":      dof_pos_r,          # (T', 29)
        "dof_vel":      dof_vel,            # (T', 29)
        "body_rot":     body_rot_r,         # (T', 33, 4) xyzw
        "body_pos":     body_pos_r,         # (T', 33, 3)
        "body_vel":     body_vel,           # (T', 33, 3)
        "body_ang_vel": body_ang_vel,       # (T', 33, 3)
        "control_dt":   args.control_dt,
        "num_frames":   T_final,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(output_path))

    print(f"\nSaved ProtoMotions cache to: {args.output}")
    print(f"  control_dt: {args.control_dt} ({1.0/args.control_dt:.0f}Hz)")
    print(f"  num_frames: {T_final}")
    print(f"  duration: {T_final * args.control_dt:.2f}s")

    # Sanity check
    print("\nSanity check - loading back:")
    loaded = torch.load(str(output_path), weights_only=False)
    for k, v in loaded.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

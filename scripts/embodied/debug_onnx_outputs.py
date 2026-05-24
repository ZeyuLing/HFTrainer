"""
Diagnostic script: Does the ONNX RL policy produce different outputs
when we change the reference motion target?

Tests two cases:
  Case A: future reference = same as current state (standing still)
  Case B: future reference = significantly different (offset position by 1m)

Compares joint_pos_targets, stiffness_targets, damping_targets, and actions.
"""

import numpy as np
import onnxruntime as ort
import mujoco
import yaml
from pathlib import Path


def load_initial_state(mjcf_path: str):
    """Load MuJoCo SMPL model and get initial body state (T-pose)."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Body positions: skip world body (index 0), take 24 bodies
    body_pos = data.xpos[1:25].copy().astype(np.float32)  # (24, 3)
    # Body rotations: MuJoCo uses wxyz, convert to xyzw
    body_quat_wxyz = data.xquat[1:25].copy().astype(np.float32)  # (24, 4) wxyz
    body_quat_xyzw = np.concatenate(
        [body_quat_wxyz[:, 1:4], body_quat_wxyz[:, 0:1]], axis=-1
    )  # (24, 4) xyzw
    # Body velocities: cvel is (num_bodies, 6) = [ang_vel(3), lin_vel(3)]
    body_vel = data.cvel[1:25, 3:6].copy().astype(np.float32)  # (24, 3)
    body_ang_vel = data.cvel[1:25, 0:3].copy().astype(np.float32)  # (24, 3)

    return body_pos, body_quat_xyzw, body_vel, body_ang_vel


def run_onnx_inference(session, inputs: dict):
    """Run ONNX inference and return outputs as a dict."""
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    feed = {name: inputs[name] for name in input_names}
    results = session.run(output_names, feed)
    return {name: result for name, result in zip(output_names, results)}


def print_tensor_stats(name: str, arr: np.ndarray, indent: str = "  "):
    """Print mean, std, min, max of a tensor."""
    print(f"{indent}{name}:")
    print(f"{indent}  shape={arr.shape}, dtype={arr.dtype}")
    print(f"{indent}  mean={arr.mean():.6f}, std={arr.std():.6f}")
    print(f"{indent}  min={arr.min():.6f}, max={arr.max():.6f}")
    print(f"{indent}  L2 norm={np.linalg.norm(arr):.6f}")


def main():
    base_dir = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
    onnx_path = base_dir / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"
    yaml_path = base_dir / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
    mjcf_path = base_dir / "ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"

    # Load YAML metadata
    print("=" * 70)
    print("ONNX RL Policy Diagnostic: Reference Motion Sensitivity")
    print("=" * 70)

    with open(yaml_path, "r") as f:
        metadata = yaml.safe_load(f)

    print(f"\nModel type: {metadata['type']}")
    print(f"Control dt: {metadata['timing']['control_dt']}")
    print(f"Physics dt: {metadata['timing']['physics_dt']}")
    print(f"Decimation: {metadata['timing']['decimation']}")
    print(f"Num bodies: {metadata['robot']['num_bodies']}")
    print(f"Num DOFs: {metadata['robot']['num_dofs']}")

    # Load ONNX model
    print(f"\nLoading ONNX model from: {onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])

    print("ONNX inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
    print("ONNX outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    # Load initial state from MuJoCo
    print(f"\nLoading MuJoCo model from: {mjcf_path}")
    body_pos, body_rot, body_vel, body_ang_vel = load_initial_state(str(mjcf_path))

    print(f"\nInitial state (from MuJoCo T-pose):")
    print_tensor_stats("body_pos", body_pos)
    print_tensor_stats("body_rot (xyzw)", body_rot)
    print_tensor_stats("body_vel", body_vel)
    print_tensor_stats("body_ang_vel", body_ang_vel)

    # Prepare common inputs (batch dimension = 1)
    current_pos = body_pos[np.newaxis, :, :]  # (1, 24, 3)
    current_rot = body_rot[np.newaxis, :, :]  # (1, 24, 4)
    current_vel = body_vel[np.newaxis, :, :]  # (1, 24, 3)
    current_ang_vel = body_ang_vel[np.newaxis, :, :]  # (1, 24, 3)
    ground_heights = np.zeros((1,), dtype=np.float32)
    historical_actions = np.zeros((1, 1, 69), dtype=np.float32)

    # =========================================================================
    # Case A: Future reference = same as current state (standing still)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CASE A: Future reference = SAME as current state (standing still)")
    print("=" * 70)

    inputs_a = {
        "current_rigid_body_ang_vel": current_ang_vel,
        "current_rigid_body_pos": current_pos,
        "current_rigid_body_rot": current_rot,
        "current_rigid_body_vel": current_vel,
        "ground_heights": ground_heights,
        "historical_actions": historical_actions,
        # Future = same as current (no motion desired)
        "mimic_future_ang_vel": current_ang_vel[:, np.newaxis, :, :],  # (1, 1, 24, 3)
        "mimic_future_pos": current_pos[:, np.newaxis, :, :],  # (1, 1, 24, 3)
        "mimic_future_rot": current_rot[:, np.newaxis, :, :],  # (1, 1, 24, 4)
        "mimic_future_vel": current_vel[:, np.newaxis, :, :],  # (1, 1, 24, 3)
    }

    outputs_a = run_onnx_inference(session, inputs_a)

    print("\nCase A outputs:")
    for name, arr in outputs_a.items():
        print_tensor_stats(name, arr)

    # =========================================================================
    # Case B: Future reference = significantly different (offset by 1m in X)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CASE B: Future reference = OFFSET by +1m in X direction")
    print("=" * 70)

    # Offset all body positions by 1m in X
    offset_pos = current_pos.copy()
    offset_pos[:, :, 0] += 1.0  # +1m in X for all bodies

    # Also create a forward velocity to be consistent
    offset_vel = current_vel.copy()
    offset_vel[:, :, 0] += 1.0  # 1 m/s in X direction

    inputs_b = {
        "current_rigid_body_ang_vel": current_ang_vel,
        "current_rigid_body_pos": current_pos,
        "current_rigid_body_rot": current_rot,
        "current_rigid_body_vel": current_vel,
        "ground_heights": ground_heights,
        "historical_actions": historical_actions,
        # Future = offset position (motion target is far away)
        "mimic_future_ang_vel": current_ang_vel[:, np.newaxis, :, :],  # same ang vel
        "mimic_future_pos": offset_pos[:, np.newaxis, :, :],  # (1, 1, 24, 3) offset
        "mimic_future_rot": current_rot[:, np.newaxis, :, :],  # same rotation
        "mimic_future_vel": offset_vel[:, np.newaxis, :, :],  # forward velocity
    }

    outputs_b = run_onnx_inference(session, inputs_b)

    print("\nCase B outputs:")
    for name, arr in outputs_b.items():
        print_tensor_stats(name, arr)

    # =========================================================================
    # Comparison: Case A vs Case B
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Case A vs Case B (difference)")
    print("=" * 70)

    for name in outputs_a:
        diff = outputs_b[name] - outputs_a[name]
        abs_diff = np.abs(diff)
        print(f"\n  {name} difference (B - A):")
        print(f"    mean_abs_diff = {abs_diff.mean():.6f}")
        print(f"    max_abs_diff  = {abs_diff.max():.6f}")
        print(f"    L2 norm diff  = {np.linalg.norm(diff):.6f}")
        print(f"    relative L2   = {np.linalg.norm(diff) / (np.linalg.norm(outputs_a[name]) + 1e-8):.6f}")

        # Check if any elements changed
        changed = np.sum(abs_diff > 1e-6)
        total = abs_diff.size
        print(f"    elements changed (>1e-6): {changed}/{total} ({100*changed/total:.1f}%)")

    # =========================================================================
    # Additional Case C: Only rotation differs
    # =========================================================================
    print("\n" + "=" * 70)
    print("CASE C: Future reference = ROTATED (90 deg yaw on all bodies)")
    print("=" * 70)

    # Create a 90-degree rotation around Z axis in xyzw format
    # q = [sin(θ/2)*axis, cos(θ/2)] = [0, 0, sin(45°), cos(45°)] = [0, 0, 0.7071, 0.7071]
    rotated_rot = current_rot.copy()
    # Apply 90-degree Z rotation to root body via quaternion multiplication
    # For simplicity, just set all body rotations to a 90-degree rotated version
    # q_rot = [0, 0, sin(pi/4), cos(pi/4)] in xyzw
    q_rot = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)

    # Quaternion multiply q_rot * q for each body
    def quat_multiply_xyzw(q1, q2):
        """Multiply quaternions in xyzw format."""
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.stack([x, y, z, w], axis=-1)

    # Broadcast q_rot to match (24, 4) and apply rotation
    q_rot_expanded = np.broadcast_to(q_rot, (24, 4)).astype(np.float32)
    rotated_rot = quat_multiply_xyzw(q_rot_expanded, rotated_rot[0]).astype(np.float32)
    rotated_rot = rotated_rot[np.newaxis]  # (1, 24, 4)

    inputs_c = {
        "current_rigid_body_ang_vel": current_ang_vel,
        "current_rigid_body_pos": current_pos,
        "current_rigid_body_rot": current_rot,
        "current_rigid_body_vel": current_vel,
        "ground_heights": ground_heights,
        "historical_actions": historical_actions,
        # Future = rotated (different orientation target)
        "mimic_future_ang_vel": current_ang_vel[:, np.newaxis, :, :],
        "mimic_future_pos": current_pos[:, np.newaxis, :, :],  # same position
        "mimic_future_rot": rotated_rot[:, np.newaxis, :, :],  # rotated!
        "mimic_future_vel": current_vel[:, np.newaxis, :, :],
    }

    outputs_c = run_onnx_inference(session, inputs_c)

    print("\nCase C outputs:")
    for name, arr in outputs_c.items():
        print_tensor_stats(name, arr)

    print("\n  Comparison: Case A vs Case C (rotation only differs):")
    for name in outputs_a:
        diff = outputs_c[name] - outputs_a[name]
        abs_diff = np.abs(diff)
        print(f"\n  {name} difference (C - A):")
        print(f"    mean_abs_diff = {abs_diff.mean():.6f}")
        print(f"    max_abs_diff  = {abs_diff.max():.6f}")
        print(f"    L2 norm diff  = {np.linalg.norm(diff):.6f}")
        print(f"    relative L2   = {np.linalg.norm(diff) / (np.linalg.norm(outputs_a[name]) + 1e-8):.6f}")
        changed = np.sum(abs_diff > 1e-6)
        total = abs_diff.size
        print(f"    elements changed (>1e-6): {changed}/{total} ({100*changed/total:.1f}%)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    actions_diff_ab = np.linalg.norm(outputs_b["actions"] - outputs_a["actions"])
    actions_diff_ac = np.linalg.norm(outputs_c["actions"] - outputs_a["actions"])
    jpt_diff_ab = np.linalg.norm(outputs_b["joint_pos_targets"] - outputs_a["joint_pos_targets"])
    jpt_diff_ac = np.linalg.norm(outputs_c["joint_pos_targets"] - outputs_a["joint_pos_targets"])

    print(f"\n  Actions L2 diff (A vs B, position offset):  {actions_diff_ab:.6f}")
    print(f"  Actions L2 diff (A vs C, rotation offset):  {actions_diff_ac:.6f}")
    print(f"  Joint pos targets L2 diff (A vs B):         {jpt_diff_ab:.6f}")
    print(f"  Joint pos targets L2 diff (A vs C):         {jpt_diff_ac:.6f}")

    if actions_diff_ab > 0.01 or actions_diff_ac > 0.01:
        print("\n  ✓ CONCLUSION: The ONNX model IS SENSITIVE to reference motion changes.")
        print("    Different future targets produce different policy outputs.")
    else:
        print("\n  ✗ CONCLUSION: The ONNX model is NOT sensitive to reference motion changes.")
        print("    This may indicate a bug in input construction or model export.")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()

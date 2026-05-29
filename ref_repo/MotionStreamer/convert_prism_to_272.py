"""
Convert PRISM predictions to 272-dim MotionStreamer representation.

Pipeline:
1. Load PRISM NPZ (global_orient, body_pose, transl, betas)
2. Apply face_z_transform (rotate first frame heading to face Z+)
3. Run FK using SMPL (NOT SMPLX!) to get 22 joint positions
4. Apply representation_272 logic (heading removal, local positions, velocities, 6D rotations)
5. Save as <original_hml3d_id>.npy with shape (T, 272)

IMPORTANT: The GT 272-dim data was computed with SMPL body model.
Using SMPLX produces physically impossible skeletons due to different body templates.
PRISM outputs 63-dim body_pose (21 SMPLX joints), so we pad to 69-dim (23 SMPL joints)
with zeros for the last 2 hand joints.

Usage:
    python3 convert_prism_to_272.py \
        --pred_dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten \
        --annotation data/annotation/test_hml3d.json \
        --out_dir ref_repo/MotionStreamer/prism_272_predictions
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import smplx
from tqdm import tqdm


# ============ Rotation Utilities (from MotionStreamer's face_z_align_util.py) ============

def expmap_to_quaternion(e):
    """Convert axis-angle (exponential map) to quaternion [w, x, y, z]."""
    assert e.shape[-1] == 3
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)
    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def qmul_np(q, r):
    """Multiply quaternions q and r (numpy)."""
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qmul(q, r):
    """Multiply quaternion(s) q with quaternion(s) r (torch)."""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    original_shape = q.shape
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot_np(q, v):
    """Rotate vector(s) v by quaternion(s) q (numpy)."""
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qrot(q, v):
    """Rotate vector(s) v by quaternion(s) q (torch)."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qbetween_np(v0, v1):
    """Find quaternion to rotate v0 to v1."""
    assert v0.shape[-1] == 3
    assert v1.shape[-1] == 3
    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1, keepdim=True)
    q = torch.cat([w, v], dim=-1)
    return (q / torch.norm(q, dim=-1, keepdim=True)).numpy()


def quaternion_to_matrix_np(quaternions):
    """Convert quaternions [w,x,y,z] to rotation matrices (numpy)."""
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_matrix(quaternions):
    """Convert quaternions [w,x,y,z] to rotation matrices (torch)."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix):
    """Convert rotation matrix to 6D representation (first two ROWS = row-major convention).
    This matches MotionStreamer/pytorch3d convention where rot6d = [row0, row1] of R."""
    # matrix: (..., 3, 3) -> (..., 6)
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle to quaternion [w,x,y,z] (torch)."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle to rotation matrix (torch)."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """Convert rotation matrix to axis-angle (torch)."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix):
    """Convert rotation matrix to quaternion [w,x,y,z] (torch)."""
    def _copysign(a, b):
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)

    def _sqrt_positive_part(x):
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def quaternion_to_axis_angle(quaternions):
    """Convert quaternion [w,x,y,z] to axis-angle (torch)."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


# ============ Face Z Transform ============

def my_quat_rotate(q, v):
    """Rotate vector v by quaternion q in [x,y,z,w] format."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def calc_heading(q):
    """Calculate heading from quaternion in [x,y,z,w] format."""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    """Calculate inverse heading quaternion from [x,y,z,w] quaternion."""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1
    return -heading, axis


def face_z_transform(global_orient_aa, body_pose_aa, trans, betas):
    """
    Apply face-Z transform to SMPLX axis-angle parameters.

    Args:
        global_orient_aa: (T, 3) root orientation in axis-angle
        body_pose_aa: (T, 63) body pose in axis-angle (21 joints)
        trans: (T, 3) translation
        betas: (10,) shape parameters

    Returns:
        smpl_85_face_z: (T, 85) in face_z_transform format
            [0:3] = root_orient_aa, [3:66] = body_pose_aa, [66:72] = zeros, [72:75] = trans, [75:85] = betas
    """
    seq_len = global_orient_aa.shape[0]

    # Get first frame root orientation quaternion [w,x,y,z]
    root_first_quat_wxyz = expmap_to_quaternion(global_orient_aa[0])  # (4,)
    # Convert to [x,y,z,w] for calc_heading
    root_first_quat_xyzw = root_first_quat_wxyz[[1, 2, 3, 0]]
    root_first_quat_xyzw = torch.from_numpy(root_first_quat_xyzw).float().unsqueeze(0)

    # Calculate inverse heading
    heading_inv, axis = calc_heading_quat_inv(root_first_quat_xyzw)
    heading_inv_axis_angle = heading_inv * axis  # (1, 3)
    heading_inv_axis_angle = heading_inv_axis_angle.numpy()

    # Compute heading rotation quaternion [w,x,y,z]
    q_diff = expmap_to_quaternion(heading_inv_axis_angle)  # (1, 4)

    # Apply heading rotation to all root orientations
    root_quats_wxyz = expmap_to_quaternion(global_orient_aa)  # (T, 4)
    result_root_quat = qmul_np(
        q_diff.reshape(1, -1).repeat(seq_len, axis=0),
        root_quats_wxyz
    )  # (T, 4)
    result_root_aa = quaternion_to_axis_angle(
        torch.from_numpy(result_root_quat).float()
    ).numpy()  # (T, 3)

    # Apply heading rotation to translation
    trans_rotated = qrot_np(
        q_diff.reshape(1, -1).repeat(seq_len, axis=0),
        trans
    )  # (T, 3)

    # Pack into smpl_85_face_z_transform format:
    # [0:3] root_orient, [3:66] body_pose(21j), [66:72] zeros(2j placeholder), [72:75] trans, [75:85] betas
    smpl_85 = np.zeros((seq_len, 85), dtype=np.float64)
    smpl_85[:, 0:3] = result_root_aa
    smpl_85[:, 3:66] = body_pose_aa
    smpl_85[:, 66:72] = 0.0  # placeholder for extra joints
    smpl_85[:, 72:75] = trans_rotated
    smpl_85[:, 75:85] = betas[None, :]  # broadcast betas to all frames

    return smpl_85


# ============ Forward Kinematics ============

def run_fk_smpl(smpl_85_face_z, smpl_model, device='cuda'):
    """
    Run forward kinematics using SMPL model (NOT SMPLX!).

    The GT 272-dim representation was computed with SMPL body model.
    Using SMPLX produces physically impossible skeletons due to different body templates.

    PRISM outputs 63-dim body_pose (21 SMPLX body joints). We pad to 69-dim
    (23 SMPL body joints) with zeros for the last 2 hand joints.

    Args:
        smpl_85_face_z: (T, 85) face-Z-transformed parameters
            [0:3] root orient, [3:66] body_pose(21j), [66:72] zeros, [72:75] trans, [75:85] betas
        smpl_model: SMPL model instance (from smplx.create with model_type='smpl')
        device: computation device

    Returns:
        joints: (T, 22, 3) joint positions
    """
    T = smpl_85_face_z.shape[0]

    root_orient = torch.from_numpy(smpl_85_face_z[:, :3]).float().to(device)
    body_pose_21j = torch.from_numpy(smpl_85_face_z[:, 3:66]).float().to(device)  # (T, 63)
    trans = torch.from_numpy(smpl_85_face_z[:, 72:75]).float().to(device)
    betas = torch.from_numpy(smpl_85_face_z[:, 75:85]).float().to(device)

    # Pad body_pose from 63 (21 joints) to 69 (23 joints) for SMPL
    # Last 2 joints (L_Hand, R_Hand) set to zero (neutral hand pose)
    body_pose_23j = torch.zeros(T, 69, device=device)
    body_pose_23j[:, :63] = body_pose_21j

    # Run SMPL FK in batches to avoid OOM
    batch_size = 512
    all_joints = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        with torch.no_grad():
            output = smpl_model(
                global_orient=root_orient[start:end],
                body_pose=body_pose_23j[start:end],
                transl=trans[start:end],
                betas=betas[start:end],
            )
        # output.joints: (batch, 24, 3) for SMPL - we need first 22
        joints_batch = output.joints[:, :22, :].detach().cpu().numpy()
        all_joints.append(joints_batch)

    joints = np.concatenate(all_joints, axis=0)  # (T, 22, 3)
    return joints


# ============ Representation 272 ============

def rot_yaw(yaw):
    """Y-axis rotation matrix."""
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def compute_representation_272(joints_22, smpl_85_face_z):
    """
    Compute 272-dim representation from joint positions and face-Z parameters.

    Args:
        joints_22: (T, 22, 3) joint positions from FK
        smpl_85_face_z: (T, 85) face-Z-transformed parameters

    Returns:
        repr_272: (T, 272) or None if conversion fails
    """
    root_idx = 0
    nfrm = joints_22.shape[0]
    njoint = 22

    position_data = joints_22.copy()

    # Get rotations: first 66 dims = 22 joints x 3 axis-angle
    rotation_smpl_axis_angle = smpl_85_face_z[:, :66].reshape(nfrm, njoint, 3)
    rotations_wxyz = expmap_to_quaternion(rotation_smpl_axis_angle)  # (T, 22, 4)
    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # (T, 22, 3, 3)

    # Put on floor and center root at origin for first frame
    ori = position_data[0, root_idx].copy()
    y_min = np.min(position_data[:, :, 1])
    ori[1] = y_min
    position_data = position_data - ori

    # Root velocities (before removing XZ)
    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]

    # Calculate local position: all frames at XZ origin
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    # Calculate heading from root rotation matrix
    global_heading = -np.arctan2(
        rotations_matrix[:, root_idx, 0, 2],
        rotations_matrix[:, root_idx, 2, 2]
    )
    global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

    # Positions with heading removed
    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
        position_data[..., None]
    ).squeeze(-1)

    # Velocities with heading removed
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    # Root velocity XZ with heading removed
    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1],
        velocities_root[:, :, None]
    ).squeeze()[..., [0, 2]]

    # Remove heading from root rotation
    rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

    # Pack into 272-dim representation
    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6  # = 272
    final_x = np.zeros((nfrm, size_frame))

    # First frame: identity heading rotation (6D)
    final_x[0, 2] = 1  # first column of identity = [1,0,0] but in 6D = cols 0,1 of 3x3
    final_x[0, 6] = 1  # second column of identity

    # Heading angular velocity as 6D rotation (row-major, matching GT matrix_to_rotation_6d)
    # global_heading_diff_rot shape: (T-1, 3, 3)
    # Extract first 2 ROWS: [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]
    # This matches GT representation_272.py line 109:
    #   matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy()
    heading_6d = global_heading_diff_rot[:, :2, :].reshape(-1, 6)  # (T-1, 6)
    final_x[1:, 2:8] = heading_6d

    # Root XZ velocity (heading removed)
    final_x[1:, :2] = velocities_root_xy_no_heading

    # Local positions (heading removed)
    final_x[:, 8:8 + 3 * njoint] = np.reshape(positions_no_heading, (nfrm, -1))

    # Local velocities (heading removed)
    final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = np.reshape(velocities_no_heading, (nfrm - 1, -1))

    # Local rotations as 6D (heading removed from root)
    # rotations_matrix shape: (T, 22, 3, 3)
    # Extract first 2 ROWS (row-major) to match GT representation_272.py line 116:
    #   np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1))
    # Per joint: [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]
    rot6d_row_major = rotations_matrix[:, :, :2, :].reshape(nfrm, -1)  # (T, 132)
    final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = rot6d_row_major

    return final_x


# ============ Main ============

def build_id_mapping(annotation_path):
    """
    Build mapping from MotionHub IDs (humanml3d_XXXXX) to original HumanML3D IDs.

    Returns:
        dict: {motionhub_id: original_hml3d_id}
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Handle both formats: {"data_list": {...}} or direct dict
    if 'data_list' in data:
        data_list = data['data_list']
    else:
        data_list = data

    mapping = {}
    for key, value in data_list.items():
        if not key.startswith('humanml3d_'):
            continue
        # Extract original ID from smplx_path
        # e.g., "motionx/motion_data/smplx_55/humanml/013046.npz" -> "013046"
        # or "motionx/motion_data/smplx_55/humanml/M013654.npz" -> "M013654"
        smplx_path = value.get('smplx_path', '')
        if smplx_path:
            original_id = os.path.splitext(os.path.basename(smplx_path))[0]
            mapping[key] = original_id

    return mapping


def main():
    parser = argparse.ArgumentParser(description='Convert PRISM predictions to 272-dim representation')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing PRISM prediction NPZ files')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Path to test_hml3d.json for ID mapping')
    parser.add_argument('--smpl_model_dir', type=str,
                        default='checkpoints/smpl_models',
                        help='Directory containing SMPL models (with smpl/ subdirectory)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for 272-dim NPY files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for FK computation')
    parser.add_argument('--test_split', type=str, default=None,
                        help='Path to test.txt split file (to filter which IDs to convert)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build ID mapping
    print("Building ID mapping...")
    id_mapping = build_id_mapping(args.annotation)
    print(f"  Found {len(id_mapping)} mapped IDs")

    # Load test split if provided (to only convert needed IDs)
    target_ids = None
    if args.test_split:
        with open(args.test_split, 'r') as f:
            target_ids = set(line.strip() for line in f if line.strip())
        print(f"  Test split has {len(target_ids)} IDs")

    # Build reverse mapping: original_id -> motionhub_id
    reverse_mapping = {}
    for mhub_id, orig_id in id_mapping.items():
        reverse_mapping[orig_id] = mhub_id

    # Initialize SMPL model for FK (NOT SMPLX!)
    print("Loading SMPL model...")
    body_model = smplx.create(
        model_path=args.smpl_model_dir,
        model_type='smpl',
        gender='neutral',
        num_betas=10,
    ).to(args.device)
    body_model.eval()
    for p in body_model.parameters():
        p.requires_grad = False
    print(f"  SMPL model loaded on {args.device}")

    # Find all prediction files
    pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith('.npz')]
    print(f"  Found {len(pred_files)} prediction files")

    # Process each prediction
    success_count = 0
    skip_count = 0
    fail_count = 0

    for pred_file in tqdm(pred_files, desc="Converting"):
        motionhub_id = os.path.splitext(pred_file)[0]  # e.g., "humanml3d_10006"

        # Get original HumanML3D ID
        if motionhub_id not in id_mapping:
            skip_count += 1
            continue
        original_id = id_mapping[motionhub_id]

        # Skip if not in test split
        if target_ids is not None and original_id not in target_ids:
            skip_count += 1
            continue

        # Check if already converted
        out_path = os.path.join(args.out_dir, f"{original_id}.npy")
        if os.path.exists(out_path):
            success_count += 1
            continue

        # Load PRISM prediction
        try:
            pred = np.load(os.path.join(args.pred_dir, pred_file))
            global_orient = pred['global_orient']  # (T, 3) axis-angle
            body_pose = pred['body_pose']  # (T, 63) axis-angle
            transl = pred['transl']  # (T, 3)
            betas = pred['betas']  # (10,)
        except Exception as e:
            print(f"  Failed to load {pred_file}: {e}")
            fail_count += 1
            continue

        T = global_orient.shape[0]
        if T < 2:
            print(f"  Skipping {pred_file}: too short ({T} frames)")
            skip_count += 1
            continue

        # Ensure float64
        global_orient = global_orient.astype(np.float64)
        body_pose = body_pose.astype(np.float64)
        transl = transl.astype(np.float64)
        betas = betas.astype(np.float64)

        # Step 1: Face Z transform
        try:
            smpl_85_fz = face_z_transform(global_orient, body_pose, transl, betas)
        except Exception as e:
            print(f"  face_z_transform failed for {pred_file}: {e}")
            fail_count += 1
            continue

        # Step 2: FK to get joint positions (using SMPL, NOT SMPLX!)
        try:
            joints_22 = run_fk_smpl(smpl_85_fz, body_model, device=args.device)
        except Exception as e:
            print(f"  FK failed for {pred_file}: {e}")
            fail_count += 1
            continue

        # Step 3: Compute 272-dim representation
        try:
            repr_272 = compute_representation_272(joints_22, smpl_85_fz)
        except Exception as e:
            print(f"  representation_272 failed for {pred_file}: {e}")
            fail_count += 1
            continue

        if repr_272 is None:
            fail_count += 1
            continue

        # Save
        np.save(out_path, repr_272)
        success_count += 1

    print(f"\nDone! Converted: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")
    print(f"Output directory: {args.out_dir}")


if __name__ == '__main__':
    main()

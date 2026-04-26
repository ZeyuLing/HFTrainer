"""Transition canonicalization utilities for E14/E15/E16 evaluation.

Training motion is canonical (pelvis at origin at frame 0, heading +Z),
but transition inference feeds two independent motions (or one motion +
a static target pose) to the model. Without matching the training
distribution, the model produces "teleport back to origin" artefacts.

This module provides:

  - extract_yaw_from_root_rot6d: pull Y-axis yaw from row-major rot6d
  - build_yaw_rotation_matrix:   build 3x3 yaw rotation matrix
  - apply_rigid_transform:       apply Y-axis rigid transform to motion
  - place_b_after_a:             world-space placement of motion B
                                 after motion A (forward step + small
                                 yaw offset) — avoids "return to origin"
  - canonicalize_segment:        place segment anchor frame at origin
                                 with heading +Z
  - decanonicalize_segment:      inverse of canonicalize_segment

The motion tensor layout is the 135-dim SMPL-22 representation:

    [0:3]    translation (abs)
    [3:9]    root rotation 6D (row-major, joint 0 = pelvis)
    [9:135]  body rotations 6D (21 joints x 6 dims, local)

Row-major rot6d is the training convention after
``load_smplx.py`` reorders column-major output via ``[0,3,1,4,2,5]``.
We therefore call ``geometry.rot6d_to_rotation_matrix`` (column-major
native) only AFTER reordering 6D back to column-major, OR by using the
fact that geometry's builder reconstructs an orthonormal matrix from the
first two columns — which is what we feed in after reordering.

See docs/temp/m2m_canonical_ood_solution.md for derivation.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 6D <-> matrix helpers (row-major convention used by training pipeline)
# ---------------------------------------------------------------------------

# Empirically verified (see _self_test below): the
# ``geometry.rot6d_to_rotation_matrix`` implementation uses
# ``rot6d.view(..., 3, 2)`` which, under PyTorch's row-major storage,
# maps the flat 6D layout ``[R00, R01, R10, R11, R20, R21]`` to the
# matrix with first column = elements [0, 2, 4] and second column =
# elements [1, 3, 5]. This is exactly the row-major layout used in the
# HyMotion training pipeline after load_smplx.py applies its reorder.
#
# So ``rot6d_to_rotation_matrix`` / ``rotation_matrix_to_rot6d`` already
# operate in the row-major convention — no reorder needed here. The docs
# that describe geometry.py as "column-major" refer to a different axis
# (the math convention for "6D = first two columns of R"), not to the
# memory layout. This led to an easy-to-miss earlier bug; keeping this
# note to discourage re-adding the reorder.


def _rot6d_rowmajor_to_matrix(rot6d_row: Tensor) -> Tensor:
    """Row-major rot6d -> 3x3 rotation matrix via Gram-Schmidt."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix,
    )
    return rot6d_to_rotation_matrix(rot6d_row)


def _matrix_to_rot6d_rowmajor(R: Tensor) -> Tensor:
    """3x3 rotation matrix -> row-major rot6d (first two columns)."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rotation_matrix_to_rot6d,
    )
    return rotation_matrix_to_rot6d(R)


# ---------------------------------------------------------------------------
# Yaw helpers
# ---------------------------------------------------------------------------


def extract_yaw_from_root_rot6d(root_rot6d: Tensor) -> Tensor:
    """Extract yaw (rotation around Y axis) from root rot6d.

    Args:
        root_rot6d: (..., 6) row-major 6D of the root joint.

    Returns:
        (...,) yaw angle in radians in [-pi, pi].

    For Y-up SMPL convention, R_y(theta) has the form:
        [[ cos(t), 0, sin(t)],
         [      0, 1,      0],
         [-sin(t), 0, cos(t)]]
    so yaw = atan2(R[0, 2], R[2, 2]).

    Note: this assumes small pitch/roll. For walking/standing/sitting
    this is a safe approximation; for gymnastics / somersaults, a full
    Euler decomposition would be needed.
    """
    R = _rot6d_rowmajor_to_matrix(root_rot6d)
    yaw = torch.atan2(R[..., 0, 2], R[..., 2, 2])
    return yaw


def build_yaw_rotation_matrix(yaw: Tensor) -> Tensor:
    """Build 3x3 yaw rotation matrix R_y(yaw).

    Args:
        yaw: scalar or (...,) tensor of yaw angles (radians).

    Returns:
        (..., 3, 3) rotation matrix.
    """
    if not torch.is_tensor(yaw):
        yaw = torch.as_tensor(yaw, dtype=torch.float32)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    row0 = torch.stack([c, zero, s], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-s, zero, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# ---------------------------------------------------------------------------
# Rigid transform application
# ---------------------------------------------------------------------------


def apply_rigid_transform_to_motion(
    motion: Tensor,
    R_yaw: Tensor,
    offset: Tensor,
    rotation_space: str = "local",
) -> Tensor:
    """Apply a Y-axis rigid transform (R_yaw, offset) to a 135-dim motion.

    - translation (dims 0:3):   t' = R_yaw @ t + offset
    - root rotation (dims 3:9): R_root' = R_yaw @ R_root
    - body rotations (dims 9:135):
        * local space: unchanged (parent-relative, independent of world yaw)
        * global space: ALL joints are world-referenced, so they must also
          be yaw-rotated: R_j' = R_yaw @ R_j. Failing to do so leaves body
          joints rotated about the original (non-canonical) heading while
          the pelvis claims a new heading — producing a 0.1-0.4 m position
          drift at the cond→gen boundary. (Bug found 2026-04-23 when local
          and global models showed the same canonical/decanonical code
          path but only global had the extra per-joint drift.)

    Args:
        motion: (T, 135) tensor (or (..., T, 135)).
        R_yaw: (3, 3) yaw rotation matrix.
        offset: (3,) translation offset.
        rotation_space: "local" or "global".

    Returns:
        (..., T, 135) transformed motion.
    """
    assert motion.shape[-1] == 135, f"expected motion dim 135, got {motion.shape[-1]}"
    out = motion.clone()

    # Translation: t' = R @ t + offset
    transl = out[..., 0:3]
    transl_new = torch.einsum('ij,...tj->...ti', R_yaw, transl) + offset.view(*([1] * (transl.dim() - 1)), 3)
    out[..., 0:3] = transl_new

    # Root rotation: R_root' = R_yaw @ R_root
    root_rot6d = out[..., 3:9]
    R_root = _rot6d_rowmajor_to_matrix(root_rot6d)
    R_root_new = torch.einsum('ij,...tjk->...tik', R_yaw, R_root)
    out[..., 3:9] = _matrix_to_rot6d_rowmajor(R_root_new)

    # Body rotations: local = unchanged, global = yaw-rotate per joint.
    if rotation_space == "global":
        body_rot6d = out[..., 9:135]                              # (..., T, 126)
        shape = body_rot6d.shape
        body_rot6d_flat = body_rot6d.reshape(*shape[:-1], 21, 6)  # (..., T, 21, 6)
        R_body = _rot6d_rowmajor_to_matrix(body_rot6d_flat)       # (..., T, 21, 3, 3)
        R_body_new = torch.einsum('ij,...tkjl->...tkil', R_yaw, R_body)
        out[..., 9:135] = _matrix_to_rot6d_rowmajor(R_body_new).reshape(shape)

    return out


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def canonicalize_segment(
    motion: Tensor,
    anchor_frame: int = 0,
    rotation_space: str = "local",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Place anchor frame's XZ at origin with heading +Z (yaw = 0). Y is
    preserved so the canonical motion stays in the training pelvis-height
    distribution (v2 mean_Y ≈ 1.09m).

    ⚠️ Bug fix (2026-04-21): Previously subtracted full anchor_pos including
    Y, pushing the canonical pelvis down to near-ground (Y≈0). The model was
    trained on motions with pelvis Y ≈ 1.09m ± 0.14 — a Y=0 input is ~8σ
    OOD and caused transition generation to drift wildly in height
    (E14/E15 floating artifact, 2026-04-21).

    ⚠️ Bug fix (2026-04-23): ``rotation_space`` passed through so that
    global-rot motions also get their body joints yaw-rotated (not just
    pelvis). Before this fix, global models saw a "head rotated one way,
    body rotated another way" input at condition frames, producing a
    0.1-0.4 m boundary jump at cond→gen.

    Args:
        motion: (T, 135) motion in world coordinates.
        anchor_frame: frame index to use as canonical anchor (default 0).
        rotation_space: "local" (body rot is parent-relative, unchanged by
            world yaw) or "global" (body rot is world-referenced, MUST be
            yaw-rotated alongside pelvis).

    Returns:
        motion_canon: (T, 135) canonicalized motion.
        R_canon: (3, 3) applied yaw rotation matrix (world -> canonical).
        offset_canon: (3,) applied translation offset.
    """
    anchor_yaw = extract_yaw_from_root_rot6d(motion[anchor_frame, 3:9])
    R_canon = build_yaw_rotation_matrix(-anchor_yaw)

    anchor_pos = motion[anchor_frame, 0:3]
    rotated = torch.einsum('ij,j->i', R_canon, anchor_pos)
    offset_canon = torch.stack([
        -rotated[0],
        torch.zeros_like(rotated[0]),
        -rotated[2],
    ])

    motion_canon = apply_rigid_transform_to_motion(
        motion, R_canon, offset_canon, rotation_space=rotation_space)
    return motion_canon, R_canon, offset_canon


def decanonicalize_segment(
    motion_canon: Tensor,
    R_canon: Tensor,
    offset_canon: Tensor,
    rotation_space: str = "local",
) -> Tensor:
    """Inverse of canonicalize_segment: map canonical back to world.

    If world = R_canon^T @ (canon - offset_canon), then equivalently
    world = R_canon^T @ canon + (-R_canon^T @ offset_canon).

    ⚠️ Must pass the same ``rotation_space`` as canonicalize_segment,
    otherwise global-rot body joints will be decanonicalized with only
    pelvis-yaw correction (producing cond→gen boundary drift).
    """
    R_decanon = R_canon.transpose(-1, -2)
    offset_decanon = -torch.einsum('ij,j->i', R_decanon, offset_canon)
    return apply_rigid_transform_to_motion(
        motion_canon, R_decanon, offset_decanon,
        rotation_space=rotation_space)


# ---------------------------------------------------------------------------
# World-space placement: avoid "return to origin" in transition tasks
# ---------------------------------------------------------------------------


def place_b_after_a(
    motion_a: Tensor,
    motion_b: Tensor,
    forward_step: float = 1.0,
    yaw_offset_deg: float = 0.0,
    rotation_space: str = "local",
) -> Tensor:
    """Place motion B in the world so that its first frame follows A's last.

    Specifically, B is rotated so that its initial heading equals A's final
    heading plus ``yaw_offset_deg``, and translated so that its first-frame
    pelvis sits at ``A_end + forward_step * A_forward_dir``.

    ⚠️ 2026-04-24: when the caller's model uses ``global`` body rotations,
    the 21 body joints are world-referenced and MUST be yaw-rotated along
    with the pelvis. Forgetting this leaves the body facing the original
    canonical-B heading while the pelvis now points along A's heading —
    producing a visible 0.1-0.4 m drift at the cond→gen boundary after
    decanonicalization. (Same failure mode fixed earlier in
    canonicalize_segment; this call site had the same latent bug.)

    Args:
        motion_a: (T_a, 135) motion A, already in world coordinates.
        motion_b: (T_b, 135) motion B, canonical (starts at origin, heading +Z).
        forward_step: Distance (meters) to continue forward from A's end.
        yaw_offset_deg: Additional yaw turn (degrees) applied to B.
        rotation_space: "local" or "global" — must match the model's
            body-rotation convention.

    Returns:
        motion_b_world: (T_b, 135) motion B transformed to world coordinates.
    """
    # A end heading
    a_end_yaw = extract_yaw_from_root_rot6d(motion_a[-1, 3:9])
    a_end_pos = motion_a[-1, 0:3]

    # B's current (canonical) yaw at frame 0
    b_start_yaw = extract_yaw_from_root_rot6d(motion_b[0, 3:9])

    # Desired B yaw at frame 0
    yaw_offset = torch.as_tensor(yaw_offset_deg * 3.141592653589793 / 180.0,
                                 dtype=a_end_yaw.dtype, device=a_end_yaw.device)
    target_yaw = a_end_yaw + yaw_offset

    # Rotation to apply to B: delta = target - current
    delta_yaw = target_yaw - b_start_yaw
    R_B = build_yaw_rotation_matrix(delta_yaw)

    # Forward direction in world (along A's final heading)
    forward_dir = torch.stack([
        torch.sin(a_end_yaw),
        torch.zeros_like(a_end_yaw),
        torch.cos(a_end_yaw),
    ])  # (3,)
    # Target position for B[0]:
    #   XZ: A_end_XZ + forward_step * A_forward_dir  (walk forward one step)
    #   Y : B[0].Y (preserve B's own pelvis height; otherwise a crouched B
    #       placed after a standing A would be lifted to the standing pelvis
    #       height, visually "floating" above ground. 2026-04-21 bug fix.)
    b0_pos = motion_b[0, 0:3]
    target_b0_xz = a_end_pos + forward_step * forward_dir
    target_b0 = torch.stack([target_b0_xz[0], b0_pos[1], target_b0_xz[2]])

    # offset: we want R_B @ motion_b[0, 0:3] + offset = target_b0
    offset = target_b0 - torch.einsum('ij,j->i', R_B, b0_pos)

    return apply_rigid_transform_to_motion(
        motion_b, R_B, offset, rotation_space=rotation_space)


# ---------------------------------------------------------------------------
# Self-test (run: python3 -m hftrainer.pipelines.motion.transition_utils)
# ---------------------------------------------------------------------------


def _self_test():
    """Round-trip and sanity checks."""
    torch.manual_seed(0)
    T = 20

    # Build a clean motion: fixed translation + identity rotation everywhere.
    # Using torch.randn would produce non-orthonormal rot6d that gets
    # silently re-projected by Gram-Schmidt — which would corrupt the
    # round-trip test. We want a pristine starting point.
    motion = torch.zeros(T, 135)
    motion[:, 0:3] = torch.tensor([1.5, 0.0, 0.3])
    identity_rot6d_row = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    for j in range(22):
        motion[:, 3 + j*6 : 3 + (j+1)*6] = identity_rot6d_row

    # Test 1: extract yaw from identity = 0
    yaw = extract_yaw_from_root_rot6d(motion[0, 3:9])
    assert torch.allclose(yaw, torch.tensor(0.0), atol=1e-5), f"yaw={yaw}"

    # Test 2: yaw rotation round-trip
    target_yaw = torch.tensor(1.234)
    R = build_yaw_rotation_matrix(target_yaw)
    # Apply to +Z should give (sin, 0, cos)
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    rotated = R @ z_axis
    expected = torch.stack([torch.sin(target_yaw), torch.tensor(0.0), torch.cos(target_yaw)])
    assert torch.allclose(rotated, expected, atol=1e-5), f"rotated={rotated}"

    # Test 3: canonicalize -> decanonicalize round-trip
    motion_canon, R_canon, offset_canon = canonicalize_segment(motion)
    # Anchor should be at origin with heading 0
    assert torch.allclose(motion_canon[0, 0:3], torch.zeros(3), atol=1e-5)
    anchor_yaw_after = extract_yaw_from_root_rot6d(motion_canon[0, 3:9])
    assert torch.allclose(anchor_yaw_after, torch.tensor(0.0), atol=1e-5), (
        f"canon yaw={anchor_yaw_after}"
    )
    motion_recovered = decanonicalize_segment(motion_canon, R_canon, offset_canon)
    max_err = (motion - motion_recovered).abs().max().item()
    assert max_err < 1e-4, f"round-trip error {max_err}"

    # Test 4: place_b_after_a produces correct position
    motion_b = motion.clone()
    motion_b[:, 0:3] = 0.0  # B canonical: at origin
    # A end heading = 0 (identity), forward dir = (0, 0, 1)
    motion_b_world = place_b_after_a(motion, motion_b, forward_step=1.0, yaw_offset_deg=0.0)
    expected_b0 = motion[-1, 0:3] + torch.tensor([0.0, 0.0, 1.0])
    actual_b0 = motion_b_world[0, 0:3]
    err = (expected_b0 - actual_b0).abs().max().item()
    assert err < 1e-4, f"B placement error {err}, expected {expected_b0}, got {actual_b0}"

    # Test 5: rigid transform preserves body rot6d (dims 9:135)
    body_before = motion[:, 9:135].clone()
    body_after = motion_b_world[:, 9:135]
    assert torch.allclose(body_before, body_after, atol=1e-5), (
        "body rot6d must be invariant under Y-axis rigid transform"
    )

    print("transition_utils self-test OK")


if __name__ == '__main__':
    _self_test()

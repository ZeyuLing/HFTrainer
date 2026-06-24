# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Compatibility copy for the PhysFlow/OpenTrack CUDA11.4 runtime. JAX 0.4.7 is
# the newest cudnn82 wheel we can use on the V100 pool, but OpenTrack imports
# jax.scipy.spatial.transform.Rotation. This file mirrors the small JAX Rotation
# implementation introduced in later JAX releases, with scalar_first support
# added to as_quat() for current OpenTrack call sites.

import functools
import re
import typing

import jax
import jax.numpy as jnp


class Rotation(typing.NamedTuple):
    """Rotation in 3 dimensions, backed by scalar-last quaternions."""

    quat: typing.Any

    @classmethod
    def concatenate(cls, rotations: typing.Sequence["Rotation"]):
        return cls(jnp.concatenate([rotation.quat for rotation in rotations]))

    @classmethod
    def from_euler(cls, seq: str, angles, degrees: bool = False):
        num_axes = len(seq)
        if num_axes < 1 or num_axes > 3:
            raise ValueError(f"Expected non-empty axis specification up to 3 chars, got {seq}")
        intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
        extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError(f"Expected axes from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {seq}")
        if any(seq[i] == seq[i + 1] for i in range(num_axes - 1)):
            raise ValueError(f"Expected consecutive axes to be different, got {seq}")
        angles = jnp.atleast_1d(angles)
        axes = jnp.array([_elementary_basis_index(x) for x in seq.lower()])
        return cls(_elementary_quat_compose(angles, axes, intrinsic, degrees))

    @classmethod
    def from_matrix(cls, matrix):
        return cls(_from_matrix(matrix))

    @classmethod
    def from_mrp(cls, mrp):
        return cls(_from_mrp(mrp))

    @classmethod
    def from_quat(cls, quat):
        return cls(_normalize_quaternion(quat))

    @classmethod
    def from_rotvec(cls, rotvec, degrees: bool = False):
        return cls(_from_rotvec(rotvec, degrees))

    @classmethod
    def identity(cls, num: typing.Optional[int] = None, dtype=float):
        if num is not None:
            return cls(jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=dtype), (num, 1)))
        return cls(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=dtype))

    @classmethod
    def random(cls, random_key, num: typing.Optional[int] = None):
        raise NotImplementedError

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single rotation is not subscriptable.")
        return Rotation(self.quat[indexer])

    def __len__(self):
        if self.single:
            raise TypeError("Single rotation has no len().")
        return self.quat.shape[0]

    def __mul__(self, other):
        return Rotation.from_quat(_compose_quat(self.quat, other.quat))

    def apply(self, vectors, inverse: bool = False):
        return _apply(self.as_matrix(), vectors, inverse)

    def as_euler(self, seq: str, degrees: bool = False):
        if len(seq) != 3:
            raise ValueError(f"Expected 3 axes, got {seq}.")
        intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
        extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError(f"Expected axes from ['x', 'y', 'z'] or ['X', 'Y', 'Z'], got {seq}")
        if any(seq[i] == seq[i + 1] for i in range(2)):
            raise ValueError(f"Expected consecutive axes to be different, got {seq}")
        axes = jnp.array([_elementary_basis_index(x) for x in seq.lower()])
        return _compute_euler_from_quat(self.quat, axes, extrinsic, degrees)

    def as_matrix(self):
        return _as_matrix(self.quat)

    def as_mrp(self):
        return _as_mrp(self.quat)

    def as_rotvec(self, degrees: bool = False):
        return _as_rotvec(self.quat, degrees)

    def as_quat(self, scalar_first: bool = False):
        if scalar_first:
            return self.quat[..., [3, 0, 1, 2]]
        return self.quat

    def inv(self):
        return Rotation(_inv(self.quat))

    def magnitude(self):
        return _magnitude(self.quat)

    def mean(self, weights: typing.Optional[typing.Any] = None):
        weights = jnp.where(
            weights is None,
            jnp.ones(self.quat.shape[0], dtype=self.quat.dtype),
            jnp.asarray(weights, dtype=self.quat.dtype),
        )
        if weights.ndim != 1:
            raise ValueError(f"Expected weights to be 1 dimensional, got shape {weights.shape}.")
        if weights.shape[0] != len(self):
            raise ValueError("Expected weights to have one value per rotation.")
        k_mat = jnp.dot(weights[jnp.newaxis, :] * self.quat.T, self.quat)
        _, vecs = jnp.linalg.eigh(k_mat)
        return Rotation(vecs[:, -1])

    @property
    def single(self) -> bool:
        return self.quat.ndim == 1


class Slerp(typing.NamedTuple):
    times: typing.Any
    timedelta: typing.Any
    rotations: Rotation
    rotvecs: typing.Any

    @classmethod
    def init(cls, times, rotations: Rotation):
        if not isinstance(rotations, Rotation):
            raise TypeError("rotations must be a Rotation instance.")
        if rotations.single or len(rotations) == 1:
            raise ValueError("rotations must contain at least 2 rotations.")
        times = jnp.asarray(times, dtype=rotations.quat.dtype)
        if times.ndim != 1:
            raise ValueError(f"Expected 1 dimensional times, got {times.ndim}.")
        if times.shape[0] != len(rotations):
            raise ValueError("Expected number of rotations to equal number of timestamps.")
        timedelta = jnp.diff(times)
        new_rotations = Rotation(rotations.as_quat()[:-1])
        return cls(
            times=times,
            timedelta=timedelta,
            rotations=new_rotations,
            rotvecs=(new_rotations.inv() * Rotation(rotations.as_quat()[1:])).as_rotvec(),
        )

    def __call__(self, times):
        compute_times = jnp.asarray(times, dtype=self.times.dtype)
        if compute_times.ndim > 1:
            raise ValueError("times must be at most 1-dimensional.")
        single_time = compute_times.ndim == 0
        compute_times = jnp.atleast_1d(compute_times)
        ind = jnp.maximum(jnp.searchsorted(self.times, compute_times) - 1, 0)
        alpha = (compute_times - self.times[ind]) / self.timedelta[ind]
        result = self.rotations[ind] * Rotation.from_rotvec(self.rotvecs[ind] * alpha[:, None])
        if single_time:
            return result[0]
        return result


@functools.partial(jnp.vectorize, signature="(m,m),(m),()->(m)")
def _apply(matrix, vector, inverse: bool):
    return jnp.where(inverse, matrix.T, matrix) @ vector


@functools.partial(jnp.vectorize, signature="(m)->(n,n)")
def _as_matrix(quat):
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw = x * y, z * w
    xz, yw = x * z, y * w
    yz, xw = y * z, x * w
    return jnp.array(
        [
            [+x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
        ]
    )


@functools.partial(jnp.vectorize, signature="(m)->(n)")
def _as_mrp(quat):
    sign = jnp.where(quat[3] < 0, -1.0, 1.0)
    denominator = 1.0 + sign * quat[3]
    return sign * quat[:3] / denominator


@functools.partial(jnp.vectorize, signature="(m),()->(n)")
def _as_rotvec(quat, degrees: bool):
    quat = jnp.where(quat[3] < 0, -quat, quat)
    angle = 2.0 * jnp.arctan2(_vector_norm(quat[:3]), quat[3])
    angle2 = angle * angle
    small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_scale = angle / jnp.sin(angle / 2)
    scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
    scale = jnp.where(degrees, jnp.rad2deg(scale), scale)
    return scale * jnp.array(quat[:3])


@functools.partial(jnp.vectorize, signature="(n),(n)->(n)")
def _compose_quat(p, q):
    cross = jnp.cross(p[:3], q[:3])
    return jnp.array(
        [
            p[3] * q[0] + q[3] * p[0] + cross[0],
            p[3] * q[1] + q[3] * p[1] + cross[1],
            p[3] * q[2] + q[3] * p[2] + cross[2],
            p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
        ]
    )


@functools.partial(jnp.vectorize, signature="(m),(l),(),()->(n)")
def _compute_euler_from_quat(quat, axes, extrinsic: bool, degrees: bool):
    angle_first = jnp.where(extrinsic, 0, 2)
    angle_third = jnp.where(extrinsic, 2, 0)
    axes = jnp.where(extrinsic, axes, axes[::-1])
    i, j, k = axes[0], axes[1], axes[2]
    symmetric = i == k
    k = jnp.where(symmetric, 3 - i - j, k)
    sign = jnp.array((i - j) * (j - k) * (k - i) // 2, dtype=quat.dtype)
    eps = 1e-7
    a = jnp.where(symmetric, quat[3], quat[3] - quat[j])
    b = jnp.where(symmetric, quat[i], quat[i] + quat[k] * sign)
    c = jnp.where(symmetric, quat[j], quat[j] + quat[3])
    d = jnp.where(symmetric, quat[k] * sign, quat[k] * sign - quat[i])
    angles = jnp.empty(3, dtype=quat.dtype)
    angles = angles.at[1].set(2 * jnp.arctan2(jnp.hypot(c, d), jnp.hypot(a, b)))
    case = jnp.where(jnp.abs(angles[1] - jnp.pi) <= eps, 2, 0)
    case = jnp.where(jnp.abs(angles[1]) <= eps, 1, case)
    half_sum = jnp.arctan2(b, a)
    half_diff = jnp.arctan2(d, c)
    angles = angles.at[0].set(jnp.where(case == 1, 2 * half_sum, 2 * half_diff * jnp.where(extrinsic, -1, 1)))
    angles = angles.at[angle_first].set(jnp.where(case == 0, half_sum - half_diff, angles[angle_first]))
    angles = angles.at[angle_third].set(jnp.where(case == 0, half_sum + half_diff, angles[angle_third]))
    angles = angles.at[angle_third].set(jnp.where(symmetric, angles[angle_third], angles[angle_third] * sign))
    angles = angles.at[1].set(jnp.where(symmetric, angles[1], angles[1] - jnp.pi / 2))
    angles = (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.where(degrees, jnp.rad2deg(angles), angles)


def _elementary_basis_index(axis: str) -> int:
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    raise ValueError(f"Expected axis from ['x', 'y', 'z'], got {axis}")


@functools.partial(jnp.vectorize, signature="(m),(m),(),()->(n)")
def _elementary_quat_compose(angles, axes, intrinsic: bool, degrees: bool):
    angles = jnp.where(degrees, jnp.deg2rad(angles), angles)
    result = _make_elementary_quat(axes[0], angles[0])
    for idx in range(1, len(axes)):
        quat = _make_elementary_quat(axes[idx], angles[idx])
        result = jnp.where(intrinsic, _compose_quat(result, quat), _compose_quat(quat, result))
    return result


@functools.partial(jnp.vectorize, signature="(m),()->(n)")
def _from_rotvec(rotvec, degrees: bool):
    rotvec = jnp.where(degrees, jnp.deg2rad(rotvec), rotvec)
    angle = _vector_norm(rotvec)
    angle2 = angle * angle
    small_scale = 0.5 - angle2 / 48 + angle2 * angle2 / 3840
    large_scale = jnp.sin(angle / 2) / angle
    scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
    return jnp.hstack([scale * rotvec, jnp.cos(angle / 2)])


@functools.partial(jnp.vectorize, signature="(m,m)->(n)")
def _from_matrix(matrix):
    matrix_trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    decision = jnp.array([matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix_trace], dtype=matrix.dtype)
    choice = jnp.argmax(decision)
    i = choice
    j = (i + 1) % 3
    k = (j + 1) % 3
    quat_012 = jnp.empty(4, dtype=matrix.dtype)
    quat_012 = quat_012.at[i].set(1 - decision[3] + 2 * matrix[i, i])
    quat_012 = quat_012.at[j].set(matrix[j, i] + matrix[i, j])
    quat_012 = quat_012.at[k].set(matrix[k, i] + matrix[i, k])
    quat_012 = quat_012.at[3].set(matrix[k, j] - matrix[j, k])
    quat_3 = jnp.empty(4, dtype=matrix.dtype)
    quat_3 = quat_3.at[0].set(matrix[2, 1] - matrix[1, 2])
    quat_3 = quat_3.at[1].set(matrix[0, 2] - matrix[2, 0])
    quat_3 = quat_3.at[2].set(matrix[1, 0] - matrix[0, 1])
    quat_3 = quat_3.at[3].set(1 + decision[3])
    quat = jnp.where(choice != 3, quat_012, quat_3)
    return _normalize_quaternion(quat)


@functools.partial(jnp.vectorize, signature="(m)->(n)")
def _from_mrp(mrp):
    mrp_squared_plus_1 = jnp.dot(mrp, mrp) + 1
    return jnp.hstack([2 * mrp[:3], (2 - mrp_squared_plus_1)]) / mrp_squared_plus_1


@functools.partial(jnp.vectorize, signature="(n)->(n)")
def _inv(quat):
    return quat.at[3].set(-quat[3])


@functools.partial(jnp.vectorize, signature="(n)->()")
def _magnitude(quat):
    return 2.0 * jnp.arctan2(_vector_norm(quat[:3]), jnp.abs(quat[3]))


@functools.partial(jnp.vectorize, signature="(),()->(n)")
def _make_elementary_quat(axis: int, angle):
    quat = jnp.zeros(4, dtype=angle.dtype)
    quat = quat.at[3].set(jnp.cos(angle / 2.0))
    quat = quat.at[axis].set(jnp.sin(angle / 2.0))
    return quat


@functools.partial(jnp.vectorize, signature="(n)->(n)")
def _normalize_quaternion(quat):
    return quat / _vector_norm(quat)


@functools.partial(jnp.vectorize, signature="(n)->()")
def _vector_norm(vector):
    return jnp.sqrt(jnp.dot(vector, vector))

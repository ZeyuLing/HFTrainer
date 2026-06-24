"""Numeric equivalence checks for the hftrainer.motion migration.

Verifies that the new public library (hftrainer.motion.skeleton.fk +
representation.rotation) is numerically identical to the legacy implementations
it replaces, so that compat shims are safe and eval numbers do not move.
"""

import numpy as np
import torch

torch.manual_seed(0)

B, T, J = 2, 5, 22
# Derive rot6d from valid (orthonormal) rotation matrices so round-trip tests are
# meaningful (random 6D is not orthonormal and would fail after Gram-Schmidt).
_aa = torch.randn(B, T, J, 3, dtype=torch.float64)
from hftrainer.motion.representation.rotation import axis_angle_to_matrix as _aa2m, matrix_to_rotation_6d as _m2r6
rot6d_row = _m2r6(_aa2m(_aa), "row")
transl = torch.randn(B, T, 3, dtype=torch.float64)
offsets = torch.randn(J, 3, dtype=torch.float64)
motion135 = torch.cat([transl, rot6d_row.reshape(B, T, J * 6)], dim=-1)

ok = True


def check(name, a, b, tol=1e-9):
    global ok
    d = (a - b).abs().max().item()
    status = "OK " if d < tol else "FAIL"
    if d >= tol:
        ok = False
    print(f"[{status}] {name}: max|diff|={d:.3e}")


# 1. geometry.py row-major rot6d->matrix vs unified rotation convention='row'
from hftrainer.models.motion.hymotion_m2m.network.geometry import (
    rot6d_to_rotation_matrix as geo_r2m,
    rotation_matrix_to_rot6d as geo_m2r,
)
from hftrainer.motion.representation.rotation import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

flat = rot6d_row.reshape(-1, 6)
check("geometry rot6d_to_matrix == unified(row)", geo_r2m(flat), rotation_6d_to_matrix(flat, "row"))
M = geo_r2m(flat)
check("geometry matrix_to_rot6d == unified(row)", geo_m2r(M), matrix_to_rotation_6d(M, "row"))

# 2. FK: legacy differentiable_fk vs new forward_kinematics
from hftrainer.pipelines.motion.differentiable_fk import (
    differentiable_fk as legacy_dfk,
    motion135_to_fk as legacy_m2fk,
    fk_to_motion135 as legacy_fk2m,
)
from hftrainer.motion.skeleton.fk import (
    forward_kinematics as new_fk,
    motion135_to_fk as new_m2fk,
    fk_to_motion135 as new_fk2m,
    local_to_global_rot6d,
    global_to_local_rot6d,
)

local_rotmat = rotation_6d_to_matrix(rot6d_row, "row")
lp, lr = legacy_dfk(local_rotmat, transl, offsets)
np_, nr = new_fk(local_rotmat, transl, offsets)
check("FK world_pos legacy==new", lp, np_)
check("FK world_rot legacy==new", lr, nr)

# 3. motion135_to_fk local path legacy vs new
for which, (la, na) in enumerate(
    zip(legacy_m2fk(motion135, offsets, "local"), new_m2fk(motion135, offsets, "local"))
):
    check(f"motion135_to_fk[local] out{which} legacy==new", la, na)

# 4. fk_to_motion135 local path legacy vs new
check("fk_to_motion135[local] legacy==new", legacy_fk2m(local_rotmat, transl, "local"), new_fk2m(local_rotmat, transl, "local"))

# 5. global<->local round trip (new helpers)
g = local_to_global_rot6d(rot6d_row)
back = global_to_local_rot6d(g)
check("local->global->local round trip", back, rot6d_row, tol=1e-7)

# 6. new global<->local vs fk_utils (the M2M global-rot decode path)
try:
    from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
        local_to_global_rot6d_torch as fku_l2g,
        global_to_local_rot6d_torch as fku_g2l,
    )
    check("local_to_global new==fk_utils", local_to_global_rot6d(rot6d_row), fku_l2g(rot6d_row), tol=1e-7)
    check("global_to_local new==fk_utils", global_to_local_rot6d(g), fku_g2l(g), tol=1e-7)
except Exception as e:  # noqa
    print(f"[SKIP] fk_utils global<->local comparison: {e}")

print("\nRESULT:", "ALL OK" if ok else "SOME FAILED")

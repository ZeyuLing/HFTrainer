# `hftrainer.motion` API Reference

API-level reference for the public motion library, documented in the style of the
PyTorch / PyTorch3D API docs (signature → summary → **Parameters** →
**Returns** → **Shape** → **Example**).

Every symbol below is import-light: set the environment variable
`HFTRAINER_SKIP_AUTOREGISTER=1` to skip the training-stack auto-registration
before importing. For the conceptual conversion map and the rot6d-convention trap
see [`representations.md`](representations.md); for a runnable end-to-end demo see
[`scripts/demo/hml263_multi_repr_demo.py`](../../scripts/demo/hml263_multi_repr_demo.py).

**Global conventions.** Unless stated otherwise, arrays are time-major `(T, ...)`;
positions are in metres and **Y-up**; rotation matrices are right-handed and
applied to *column* vectors (`v' = R @ v`); `motion_135 = [transl(3),
22×rot6d(6)]` uses the **ROW-major** 6D layout; `HML263` is **20 fps**, every
other representation is **30 fps**. Functions accept NumPy `ndarray` or torch
`Tensor` and return the same type/dtype/device they were given, unless a
parameter or the **Return type** says otherwise.

## Package layout

| Subpackage | Module | Purpose |
|------------|--------|---------|
| `representation` | `rotation` | rotation type conversions + 6D conventions |
| | `specs` | per-representation layout metadata (single source of truth) |
| | `convert` | top-level cross-representation conversion API |
| | `humanml` | HML263 decode (`recover_from_ric`) + 272↔263 bridges |
| | `motion272` | MotionStreamer-272 encode/decode |
| `skeleton` | `names` | SMPL-22 joint names / parents |
| | `fk` | forward kinematics + rot6d row/global helpers |
| | `body_models` | SMPL/-X/-H loaders + model-dir resolution |
| `retarget` | `hml263_smpl` | HML263 → SMPL `motion_135` (IK) |
| | `smpl_soma` | SMPL `motion_135` ↔ SOMA30, KIMODO/SOMA → SMPL |
| | `smpl_g1` | SMPL → Unitree G1 29-DOF |

---

# `representation.rotation`

Unified rotation conversions for NumPy **and** torch (dtype/device preserved).
Canonical home for what used to live in
`…components.utils.geometry.rotation_convert` (now a thin shim).

## 6D rotation conventions

The continuous 6D rotation of [Zhou et al. 2019] keeps the first two basis
vectors of the rotation matrix and recovers the third by Gram–Schmidt. Two
in-memory orderings are in circulation; mixing them silently corrupts poses:

| convention | layout | used by |
|---|---|---|
| `COLUMN` | `[R00, R10, R20, R01, R11, R21]` (first two **columns**) | math default; `HML263` rot block; MotionCLIP/MDM helpers |
| `ROW` | `[R00, R01, R10, R11, R20, R21]` (first two **rows**) | data/model I/O; `motion_135/138/198` |

```python
class Rot6DConvention:
    COLUMN = "column"
    ROW    = "row"
```

### `repack_6d`

```python
repack_6d(d6, src, dst)
```

Permute the trailing 6 channels of a 6D rotation between the `COLUMN` and `ROW`
conventions. The rotation itself is unchanged; only the storage order is
permuted (`COLUMN → ROW` is the index permutation `[0, 3, 1, 4, 2, 5]`).

**Parameters:**

- **d6** (*ndarray or Tensor*) – 6D rotations with a trailing dim of 6. Reshape
  `(..., N, 6)` first for per-joint vectors.
- **src** (*str*) – source convention, `"column"` or `"row"`.
- **dst** (*str*) – destination convention, `"column"` or `"row"`.

**Returns:** *(same type as `d6`)* – repacked 6D rotations, identical shape.

**Shape:**

- d6: `(*, 6)`
- output: `(*, 6)`

### Core conversions

All converters take and return the input array type. The `convention` argument
(where present) selects the 6D layout and defaults to `"column"`. Euler helpers
take `order` (e.g. `"XYZ"`) and `deg` (degrees vs radians).

```python
axis_angle_to_matrix(axis_angle)                    # (...,3)   -> (...,3,3)
matrix_to_axis_angle(matrix)                         # (...,3,3) -> (...,3)
quaternion_to_matrix(quaternions)                    # (...,4 wxyz) -> (...,3,3)
matrix_to_quaternion(matrix)                         # (...,3,3) -> (...,4 wxyz)
quaternion_to_axis_angle(quaternions)                # (...,4) -> (...,3)
axis_angle_to_quaternion(axis_angle)                 # (...,3) -> (...,4)
standardize_quaternion(quaternions)                  # enforce w >= 0

euler_to_matrix(e, order="XYZ", deg=False)           # (...,3) -> (...,3,3)
matrix_to_euler(matrix, order="XYZ", deg=False)      # (...,3,3) -> (...,3)
euler_to_quaternion(e, order="XYZ", deg=False)
quaternion_to_euler(quat, order="XYZ", deg=False)
axis_angle_to_euler(axis_angle, order="XYZ", deg=False)

rotation_6d_to_matrix(d6, convention="column")       # (...,6) -> (...,3,3)  (Gram-Schmidt)
matrix_to_rotation_6d(matrix, convention="column")   # (...,3,3) -> (...,6)
quaternion_to_rotation_6d(quat, convention="column")
rotation_6d_to_quaternion(d6, convention="column")
axis_angle_to_rotation_6d(axis_angle, convention="column")
rotation_6d_to_axis_angle(d6, convention="column")
rotation_6d_to_euler(d6, order="XYZ", convention="column")
```

**Parameters (shared):**

- **\*** (*ndarray or Tensor*) – the source rotation in the type named by the
  function (`axis_angle` `(...,3)`, `matrix` `(...,3,3)`, `quaternions`
  `(...,4)` w-first, `d6` `(...,6)`, `e`/euler `(...,3)`).
- **convention** (*str*, optional) – 6D layout for any `*_6d` argument/return.
  Default: `"column"`.
- **order** (*str*, optional) – intrinsic Euler order. Default: `"XYZ"`.
- **deg** (*bool*, optional) – interpret/emit Euler angles in degrees. Default: `False`.

**Returns:** *(same type as input)* – the converted rotation.

> **Warning:** the math default is `COLUMN`. Always pass `convention="row"`
> explicitly when round-tripping `motion_135/138/198` data, otherwise the rot
> block is silently transposed.

### `rot_convert`

```python
rot_convert(x, src_type, dst_type, **kwargs)
```

Generic dispatcher between any two rotation types.

**Parameters:**

- **x** (*ndarray or Tensor*) – source rotation.
- **src_type** (*str*) – one of `"axis_angle" | "matrix" | "quaternion" |
  "euler" | "rotation_6d"` (aliases `aa`, `rotmat`, `quat`, `6d` accepted).
- **dst_type** (*str*) – destination type, same vocabulary as `src_type`.
- **kwargs** – forwarded to the underlying converter (`convention`, `order`, `deg`).

**Returns:** *(same type as `x`)* – `x` expressed in `dst_type`.

**Example:**

```python
>>> from hftrainer.motion.representation.rotation import matrix_to_rotation_6d, repack_6d
>>> col = matrix_to_rotation_6d(R, convention="column")   # (*, 6)
>>> row = repack_6d(col, "column", "row")                 # == matrix_to_rotation_6d(R, "row")
```

---

# `representation.specs`

Single source of truth for representation channel layouts. Pure data, no heavy
imports.

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    start: int
    end: int          # channel block [start, end)
    desc: str

@dataclass(frozen=True)
class MotionRepr:
    name: str
    dim: int
    fps: int
    body_model: str
    num_joints: int
    rot6d_convention: str | None      # "row" | "column" | None
    transl_type: str | None           # "abs" | "abs_rel" | None
    fields: tuple[FieldSpec, ...]
    norm_stats: str
    decode_via: str
    notes: str
    aliases: tuple[str, ...]
```

### `get_spec`

```python
get_spec(name)
```

Look up a representation spec by canonical name or alias (case-insensitive).

**Parameters:**

- **name** (*str*) – canonical name (e.g. `"ms272"`) or alias (e.g. `"272"`,
  `"hymotion_m2m"`).

**Returns:** *MotionRepr* – the matching spec.

**Raises:** `KeyError` – if `name` is unknown.

### `list_specs`

```python
list_specs()
```

**Returns:** *list[MotionRepr]* – all registered specs.

### `infer_spec_from_dim`

```python
infer_spec_from_dim(dim)
```

Infer the representation from an unambiguous channel count.

**Parameters:**

- **dim** (*int*) – last-axis size (e.g. `272`).

**Returns:** *MotionRepr* – the unique spec with that `dim`.

**Raises:** `KeyError` / `ValueError` – if no spec, or more than one, has `dim`.

Registered names: `motion_135`, `motion_138`, `motion_198`, `motion_147`,
`motion_151`, `motion_201`, `hml263`, `ms272`. `REGISTRY` (also exported as
`MOTION_SPECS`) is the `dict` alias → `MotionRepr`.

**Example:**

```python
>>> from hftrainer.motion.representation.specs import get_spec
>>> s = get_spec("ms272")
>>> s.dim, s.fps, s.rot6d_convention
(272, 30, 'row')
>>> [(f.name, f.start, f.end) for f in s.fields][:2]
[('root_data', 0, 4), ('ric_data', 4, 67)]
```

---

# `representation.convert`

Top-level conversion map. **Use these instead of hand-picking low-level
helpers** — each function fixes the convention/fps internally, so the
`HML263 → 272` chain needs *no* manual `repack_6d`.

### `hml263_to_joints`

```python
hml263_to_joints(m263, joints_num=22)
```

Decode HumanML3D-263 features to 3D joint positions via `recover_from_ric`.

**Parameters:**

- **m263** (*ndarray or Tensor*) – un-normalized HML263 features.
- **joints_num** (*int*, optional) – number of joints to recover. Default: `22`.

**Returns:** *(same type as `m263`)* – world-space joint positions, Y-up metres.

**Shape:**

- m263: `(..., T, 263)`
- output: `(..., T, joints_num, 3)`

### `hml263_to_motion135`

```python
hml263_to_motion135(m263, **ik_kwargs)
```

Convert HML263 to SMPL `motion_135` by position-space inverse kinematics (see
[`retarget_hml263_clip`](#retarget_hml263_clip) for the full keyword set).

**Parameters:**

- **m263** (*ndarray*) – `(T, 263)` un-normalized HML263 features.
- **ik_kwargs** – forwarded to `retarget_hml263_clip` (`device`, `source_fps`,
  `target_fps`, `refine_iters`, `rot6d_convention`, …).

**Returns:** *ndarray* – `motion_135`, **ROW-major**.

**Shape:**

- m263: `(T, 263)` @ 20 fps
- output: `(T', 135)` @ 30 fps, where `T' = round(T * target_fps / source_fps)`

### `motion135_to_motion272`

```python
motion135_to_motion272(m135, *, rotation_space="local",
                       skeleton="canon272", bone_offsets=None)
```

Convert SMPL `motion_135` to the MotionStreamer-272 evaluator space (canon-272 FK
+ encode).

**Parameters:**

- **m135** (*ndarray*) – `(T, 135)` **ROW-major** SMPL motion.
- **rotation_space** (*str*, optional) – `"local"` if the stored rot6d are local
  joint rotations, `"global"` if world rotations. Default: `"local"`.
- **skeleton** (*str*, optional) – FK skeleton. Default: `"canon272"`.
- **bone_offsets** (*ndarray or None*, optional) – override `(22, 3)` rest-pose
  bone offsets. Default: `None` (use the bundled canon-272 skeleton).

**Returns:** *ndarray* – `(T, 272)`.

### `motion272_to_hml263`

```python
motion272_to_hml263(m272, **kwargs)
```

Decode MS272 and re-encode to HML263 (includes the 30 → 20 fps resample).

**Parameters:**

- **m272** (*ndarray*) – `(T, 272)` @ 30 fps.
- **kwargs** – forwarded to the decode/encode bridge.

**Returns:** *ndarray* – `(T', 263)` @ 20 fps.

### `motion272_to_joints`

```python
motion272_to_joints(m272)
```

**Parameters:**

- **m272** (*ndarray*) – `(T, 272)`.

**Returns:** *ndarray* – `(T, 22, 3)` joint positions.

### `hml263_to_motion272`

```python
hml263_to_motion272(m263, *, ik_kwargs=None, **enc_kwargs)
```

Full `HML263 → motion_135 → MS272` chain in one call (no manual repack).

**Parameters:**

- **m263** (*ndarray*) – `(T, 263)` un-normalized HML263.
- **ik_kwargs** (*dict or None*, optional) – keyword args for the IK stage
  (`retarget_hml263_clip`). Default: `None`.
- **enc_kwargs** – keyword args for the 135→272 encode stage.

**Returns:** *ndarray* – `(T', 272)` in MS272 evaluator space.

**Example:**

```python
>>> from hftrainer.motion.representation import convert
>>> joints = convert.hml263_to_joints(m263)                       # (T, 22, 3)
>>> m272   = convert.hml263_to_motion272(m263, ik_kwargs={"device": "cuda"})
>>> m272.shape
(T', 272)
```

---

# `representation.humanml`

Native HML263 decoding (pure torch, **no `ref_repo`**) plus lazily-imported
bridges that depend on MoMask / SMPL-H assets.

### `recover_root_rot_pos`

```python
recover_root_rot_pos(data)
```

Recover the per-frame root yaw quaternion and root position from HML263 features.

**Parameters:**

- **data** (*Tensor*) – `(..., T, 263)` HumanML3D features.

**Returns:** *(Tensor, Tensor)* – `(r_rot_quat, r_pos)`.

**Shape:**

- data: `(..., T, 263)`
- r_rot_quat: `(..., T, 4)` (w-first)
- r_pos: `(..., T, 3)`

### `recover_from_ric`

```python
recover_from_ric(data, joints_num=22)
```

Decode HML263 to world-space joint positions (native re-implementation of the
canonical HumanML3D / MoMask `recover_from_ric`).

**Parameters:**

- **data** (*Tensor*) – `(..., T, 263)` un-normalized HML263 features.
- **joints_num** (*int*, optional) – number of joints. Default: `22`.

**Returns:** *Tensor* – `(..., T, joints_num, 3)` world-space positions.

Lazily re-exported from `hftrainer.datasets.motion.representation.humanml_repr`
(require MoMask/SMPL-H assets): `humanml272_to_humanml263`,
`motion198_to_humanml263`, `joints_to_humanml263`, `recover_272_stored_positions`,
`recover_272_to_smplh_joints`, `fk_smplh_joints`,
`recover_local_rotations_and_root`, `linear_resample_positions`.

---

# `representation.motion272`

MotionStreamer-272 encode/decode. Uses the bundled GT-272 canonical skeleton
(`hftrainer/motion/assets/bone_offsets_canon272.npy`), **not** the SMPL-H rest
pose.

### `encode_smpl_to_272`

```python
encode_smpl_to_272(joints_world, local_rotmat)
```

Encode world-space joints + SMPL local rotations into the 272 layout.

**Parameters:**

- **joints_world** (*ndarray or Tensor*) – `(T, 22, 3)`, Y-up metres.
- **local_rotmat** (*ndarray or Tensor*) – `(T, 22, 3, 3)` SMPL local rotations
  (the rot block is emitted **ROW-major**).

**Returns:** *(same type as input)* – `(T, 272)`.

### `motion135_to_272`

```python
motion135_to_272(m135, *, rotation_space="local", skeleton="canon272", bone_offsets=None)
```

Same as [`convert.motion135_to_motion272`](#motion135_to_motion272); see there
for parameters.

**Returns:** *ndarray* – `(T, 272)`.

### `reencode_272_via_stored_positions` / `reencode_272_via_fk`

```python
reencode_272_via_stored_positions(m272)
reencode_272_via_fk(m272, smplh_model=None)
```

Re-encode a 272 clip from its stored positions, or by FK from its rotations
(diagnostics for the FK skeleton).

**Parameters:**

- **m272** (*ndarray*) – `(T, 272)`.
- **smplh_model** (*optional*) – preloaded SMPL-H model for the FK variant.
  Default: `None`.

**Returns:** *ndarray* – `(T, 272)`.

---

# `skeleton.names`

```python
SMPL22_NAMES: list[str]        # 22 canonical joint names (Pelvis, L_Hip, ...)
SMPL22_PARENTS: np.ndarray     # (22,) parent index per joint, root = -1
SMPL22_FOOT_JOINTS, SMPL22_LEG_JOINTS, SMPL22_END_EFFECTORS   # index groups
```

---

# `skeleton.fk`

Differentiable forward kinematics + rot6d row/global helpers (torch).

### `forward_kinematics`

```python
forward_kinematics(local_rotmat, translation, bone_offsets, parents=SMPL22_PARENTS)
```

Propagate local joint rotations along the kinematic tree to world space.

**Parameters:**

- **local_rotmat** (*Tensor*) – `(*, J, 3, 3)` local rotations.
- **translation** (*Tensor*) – `(*, 3)` root translation.
- **bone_offsets** (*Tensor*) – `(J, 3)` rest-pose bone offsets.
- **parents** (*ndarray*, optional) – `(J,)` parent indices. Default: `SMPL22_PARENTS`.

**Returns:** *(Tensor, Tensor)* – `(world_positions (*, J, 3), world_rotations (*, J, 3, 3))`.

**Shape:**

- local_rotmat: `(*, J, 3, 3)`
- translation: `(*, 3)`
- world_positions: `(*, J, 3)`; world_rotations: `(*, J, 3, 3)`

`differentiable_fk(local_rotmat, translation, bone_offsets)` is the SMPL-22 alias.

### `motion135_to_fk` / `fk_to_motion135`

```python
motion135_to_fk(motion_denorm, bone_offsets, rotation_space="local")
fk_to_motion135(local_rotmat, translation, rotation_space="local")
```

Decode `motion_135` to FK outputs, and the inverse.

**Parameters:**

- **motion_denorm** (*Tensor*) – `(*, 135)` un-normalized ROW-major motion.
- **bone_offsets** (*Tensor*) – `(22, 3)` rest offsets (commonly
  `data/hymotion_m2m_data/bone_offsets_22.pt`).
- **rotation_space** (*str*, optional) – `"local"` or `"global"` (world rot6d are
  converted to local before FK). Default: `"local"`.
- **local_rotmat** (*Tensor*) – `(*, 22, 3, 3)` for `fk_to_motion135`.
- **translation** (*Tensor*) – `(*, 3)` for `fk_to_motion135`.

**Returns:**

- `motion135_to_fk` → *(world_pos (*, 22, 3), world_rot (*, 22, 3, 3), transl (*, 3), local_rotmat (*, 22, 3, 3))*
- `fk_to_motion135` → *Tensor* `(*, 135)`

### rot6d row/global helpers

```python
rot6d_to_rotmat_row_major(rot6d)        # (*,6) -> (*,3,3)
rotmat_to_rot6d_row_major(rotmat)       # (*,3,3) -> (*,6)
local_to_global_rot6d(rot6d)            # propagate along SMPL-22 chain (row-major)
global_to_local_rot6d(rot6d)
```

---

# `skeleton.body_models`

### `resolve_smpl_model_dir`

```python
resolve_smpl_model_dir(override=None)
```

Resolve the SMPL model directory by priority.

**Parameters:**

- **override** (*str or None*, optional) – explicit path; wins if set. Default: `None`.

**Returns:** *str* – the first existing dir in the order: `override` →
`$HFTRAINER_SMPL_MODEL_DIR` → `checkpoints/smpl_models` →
`ref_repo/MDM/body_models_nochumpy` → `ref_repo/MDM/body_models`.

`SmplLite`, `SmplxLite`, `SmplxLiteJ24`, `SmplxLiteV437Coco17` are lazy wrappers
(require `smplx`).

---

# `retarget.hml263_smpl`

HumanML3D-263 → SMPL `motion_135` via inverse kinematics. Requires `smplx` and a
SMPL model dir. Output defaults to **ROW-major** (chain-ready for
`motion135_to_272`).

### `retarget_hml263_clip`

```python
retarget_hml263_clip(
    feats_263, *, smpl_rest=None, model_dir=None, device="cpu",
    source_fps=20.0, target_fps=30.0, batch_size=256,
    floor_align=True, foot_height_align=True,
    refine_iters=0, refine_lr=2e-2,
    rotation_init="position", orientation_mode="bone", parent_ref_weight=0.25,
    pose_l2_weight=0.0, angle_prior_weight=0.0,
    smooth_weight=1e-3, joint_accel_weight=0.0,
    joint_fit_weight_preset="uniform",
    gmm_pose_prior=None, gmm_pose_prior_weight=0.0,
    rot6d_convention="row",
)
```

Fit an SMPL pose sequence to the HML263 joint positions. The decoded 22-joint
positions are resampled to `target_fps` (positions linearly, rotation init via
**Slerp**), an analytic per-joint rotation estimate initializes the pose, and an
optional differentiable refinement (`refine_iters > 0`) minimizes the joint fit
error.

**Parameters:**

- **feats_263** (*ndarray*) – `(T, 263)` un-normalized HML263 features.
- **smpl_rest** (*tuple or None*, optional) – preloaded `(model, rest_joints,
  parents)` from `load_smpl_rest` (see [Building blocks](#building-blocks-public));
  avoids reloading the model. Default: `None`.
- **model_dir** (*str or None*, optional) – SMPL model dir; resolved via
  `resolve_smpl_model_dir` if `None`. Default: `None`.
- **device** (*str*, optional) – torch device for IK. Default: `"cpu"`.
- **source_fps** (*float*, optional) – input frame rate. Default: `20.0`.
- **target_fps** (*float*, optional) – output frame rate. Default: `30.0`.
- **batch_size** (*int*, optional) – frames per IK batch. Default: `256`.
- **floor_align** (*bool*, optional) – shift the clip so the lowest contact sits
  on `y = 0`. Default: `True`.
- **foot_height_align** (*bool*, optional) – per-frame foot-height correction
  (fixes catastrophic-tail clips). Default: `True`.
- **refine_iters** (*int*, optional) – differentiable (Adam) refinement steps;
  `0` disables it. Default: `0`.
- **refine_lr** (*float*, optional) – refinement learning rate. Default: `2e-2`.
- **rotation_init** (*str*, optional) – pose initialization, `"position"`
  (analytic bone alignment) or `"hml263"` (use the 263 rot block). Default:
  `"position"`.
- **orientation_mode** (*str*, optional) – bone-alignment frame, `"bone"` or
  `"parent_frame"`. Default: `"bone"`.
- **parent_ref_weight** (*float*, optional) – blend toward the parent frame in
  the analytic estimate. Default: `0.25`.
- **pose_l2_weight**, **angle_prior_weight**, **smooth_weight**,
  **joint_accel_weight** (*float*, optional) – refinement regularizers
  (pose L2, joint-angle prior, temporal smoothness, acceleration). Defaults:
  `0.0`, `0.0`, `1e-3`, `0.0`.
- **joint_fit_weight_preset** (*str*, optional) – per-joint fit weighting,
  `"uniform" | "relaxed_torso" | "relaxed_upper"`. Default: `"uniform"`.
- **gmm_pose_prior** (*optional*) – SMPLify GMM prior from `load_gmm_pose_prior`
  (see [Building blocks](#building-blocks-public)). Default: `None`.
- **gmm_pose_prior_weight** (*float*, optional) – weight of the GMM prior.
  Default: `0.0`.
- **rot6d_convention** (*str*, optional) – output 6D layout. Default: `"row"`.
  Pass `"column"` to reproduce the legacy MotionCLIP layout.

**Returns:** *dict* – with keys:

- **motion_135** (*ndarray*) – `(T', 135)` SMPL motion.
- **transl** (*ndarray*) – `(T', 3)` root translation.
- **global_orient** (*ndarray*) – `(T', 3)` root axis-angle.
- **body_pose** (*ndarray*) – `(T', 63)` body axis-angle (21 joints).
- **target_joints** (*ndarray*) – `(T', 22, 3)` IK targets.
- **fitted_joints** (*ndarray*) – `(T', 22, 3)` recovered joints.
- **fit_mpjpe_mm** (*ndarray*) – `(T',)` per-frame fit error in mm (the main
  quality signal; the mapping is approximate).

**Shape:**

- feats_263: `(T, 263)` @ `source_fps`
- motion_135: `(T', 135)` @ `target_fps`, `T' = round(T * target_fps / source_fps)`

**Example:**

```python
>>> from hftrainer.motion.retarget.hml263_smpl import retarget_hml263_clip
>>> out = retarget_hml263_clip(feats, device="cuda", refine_iters=80, refine_lr=0.02)
>>> out["motion_135"].shape, float(out["fit_mpjpe_mm"].mean())
((T', 135), 31.8)
```

> **Note:** `refine_iters=0` runs the analytic estimate only (fast, ~68 mm mean
> MPJPE on MDM clips with a catastrophic tail); `refine_iters=80` adds Adam
> refinement and drops the mean to ~32 mm. Use the larger value to reproduce the
> paper-grade pipeline.

### `hml263_to_motion135`

```python
hml263_to_motion135(feats_263, **kwargs)
```

Convenience wrapper returning only `motion_135` (all keyword args identical to
`retarget_hml263_clip`).

**Returns:** *ndarray* – `(T', 135)`.

### Building blocks (public)

```python
load_smpl_rest(model_dir=None, device="cpu")          # -> (model, rest_joints (22,3), parents)
estimate_local_rotations(target_joints, rest_joints, parents,
                         orientation_mode="bone", parent_ref_weight=0.25)   # -> (T,22,3,3)
refine_smpl_fit(...)                                   # differentiable Adam refine
make_joint_fit_weights(preset)                         # -> (22,)
load_gmm_pose_prior(device)                            # SMPLify GMM prior (lazy ref_repo/FlowMDM)
resample_linear(x, src_fps, dst_fps)                   # linear resample for positions/features
```

> **Note:** rotation sequences are resampled with per-joint **Slerp**
> (`_resample_rotations`), not linear interpolation of rotation vectors;
> `resample_linear` is for positions/feature channels only.

---

# `retarget.smpl_soma`

SMPL `motion_135` ↔ KIMODO/SOMA. Reads SOMA skeleton assets from
`ref_repo/KIMODO/.../assets/skeletons` (override via `assets_root` or
`$KIMODO_SKELETON_ASSETS`).

### Skeleton constants

```python
SMPL22_NAMES, SMPL22_PARENTS                 # SMPL-22
SOMA30_NAMES, SOMA30_PARENTS                 # SOMA-30 (Hips, Spine1, ... RightToeBase)
SMPL22_TO_SOMA30                             # (22,) SMPL idx -> SOMA idx map
SOMA77_IDX                                   # name -> index in the SOMA-77 layout
```

### Config dataclasses

```python
@dataclass(frozen=True)
class SMPLToSOMAConfig:
    assets_root=None; shoulder_offset_alpha=0.75; smpl_height_mode="source_root"

@dataclass(frozen=True)
class SOMAToSMPLIKConfig:
    model_dir="ref_repo/MDM/body_models"; device=None; batch_size=512
    floor_align=True; foot_height_align=True; refine_iters=5; refine_lr=2e-2
    orientation_mode="parent_frame"; parent_ref_weight=0.25
    pose_l2_weight=0.0; angle_prior_weight=0.0; smooth_weight=0.01; joint_accel_weight=0.001
    joint_fit_weight_preset="relaxed_upper"
    soma_orientation_guides=True; head_guide_weight=0.15; leaf_guide_weight=0.35
```

### `SMPLSOMARetargeter`

```python
class SMPLSOMARetargeter(config=None, **overrides)
```

**Parameters:**

- **config** (*SMPLToSOMAConfig or None*, optional) – configuration. Default: `None`.
- **overrides** – per-field overrides applied on top of `config`.

**Methods:**

- **smpl_to_soma(motion_135)** → *dict* `{soma30_joints (T,30,3),
  soma30_global_rots, soma30_local_rots}`.
- **soma_to_smpl_from_rotations(soma30_global_rots, source_motion_135,
  height_mode=None)** → *dict* `{motion_135, transl, global_orient, body_pose,
  fitted_joints}`.
- **roundtrip_smpl(motion_135)** → *dict* `{**soma, **smpl, source_motion_135}`.

### `KIMODOSOMAToSMPLRetargeter`

```python
class KIMODOSOMAToSMPLRetargeter(config=None, **overrides)
```

**Methods:**

- **retarget_positions(positions22, soma77=None)** → *dict* `{motion_135, transl,
  global_orient, body_pose, target_joints, fitted_joints, fit_mpjpe_mm}`.
- **retarget_file(path)** → same (`.npy` positions, or `.npz` with `positions`
  [+`posed_joints`]).
- **save_npz(path, result, \*\*metadata)** – *staticmethod*.

### One-call helpers

```python
smpl_motion135_to_soma30(motion_135, **kwargs)        # = SMPLSOMARetargeter().smpl_to_soma
smpl_soma30_roundtrip(motion_135, **kwargs)           # = .roundtrip_smpl
kimodo_soma_to_smpl_motion135(positions22, soma77=None, **kwargs)
```

**Example:**

```python
>>> from hftrainer.motion.retarget import smpl_soma30_roundtrip
>>> rt = smpl_soma30_roundtrip(m135)
>>> rt["soma30_joints"].shape, rt["fitted_joints"].shape
((T, 30, 3), (T, 22, 3))
```

> **Note:** SOMA-30 is a reduced skeleton, so the SMPL → SOMA → SMPL round trip
> is lossy (typ. ~0.2 m MPJPE on full-body clips). Use `fit_mpjpe_mm` to gauge
> fidelity.

---

# `retarget.smpl_g1`

> **Warning:** Use **GMR** for any visualization or deployment. The class below
> (`SMPLToG1Retargeter`, analytic per-frame Euler decomposition) is fast but
> low-quality and is kept only for legacy/quick-look use. The correct
> SMPL → Unitree-G1 path is **GMR** (General Motion Retargeting, mink IK),
> vendored at `ref_repo/GMR/` and driven by
> `scripts/embodied/gmr_retarget_headless.py` and
> `scripts/embodied/smpl_g1_compare_demo.py`
> (`gmr_retarget_to_qpos`, `load_g1_model`, `qpos_to_robot_frames`). See
> [`representations.md`](representations.md) for the
> `motion_135 → SMPL-X axis-angle → GMR mink IK → qpos → MuJoCo FK` pipeline, and
> `scripts/demo/hml263_multi_repr_demo.py:G1GMR` for the runnable version used by
> the web viewer.

Maps SMPL 22-joint motion to Unitree G1 29-DOF joint angles (per-frame Euler
decomposition + hardware limits). One-way (angles, not a body model); render via
MuJoCo FK on `g1.xml`.

```python
SMPL_JOINT_NAMES: list[str]                  # 22
G1_JOINT_NAMES: list[str]                    # 29 (leg×12, waist×3, arm×14), g1.xml order
G1_JOINT_LIMITS: dict[str, tuple[float, float]]   # radians, from URDF
```

### `SMPLToG1Retargeter`

```python
class SMPLToG1Retargeter(apply_limits=True, rest_pose_calibration=True, g1_dof=29)
```

**Parameters:**

- **apply_limits** (*bool*, optional) – clamp DOF to `G1_JOINT_LIMITS`. Default: `True`.
- **rest_pose_calibration** (*bool*, optional) – subtract the rest-pose offset.
  Default: `True`.
- **g1_dof** (*int*, optional) – number of actuated DOF. Default: `29`.

**Methods:**

- **retarget(rot6d, transl, fps=30.0)** – `rot6d` is `(T, 22, 6)` ROW or `(T, 132)`.
- **retarget_from_hymotion(motion_135, fps=30.0)**.
- **retarget_from_hymotion_201(motion_201, fps=30.0)**.
- **to_mujoco_qpos(result)** → *ndarray* `(T, 36)` = `[root_pos(3),
  root_quat_wxyz(4), dof(29)]`.
- **to_asap_pkl(result, output_path)** → path.

All `retarget*` methods return a *dict* with keys: `joint_angles (T, 29)`,
`root_pos (T, 3)`, `root_orient_quat (T, 4 wxyz)`, `root_orient_euler (T, 3)`,
`fps`, `joint_names`, `dof`.

> **Note:** MuJoCo is **Z-up** while the SMPL root is Y-up; pre-rotate the root
> by `Rx(+90°)` before FK, then map link positions back `Z-up → Y-up`. See
> `scripts/demo/hml263_multi_repr_demo.py:G1MujocoFK` for the full loop.

---

# End-to-end example

```python
>>> import os; os.environ["HFTRAINER_SKIP_AUTOREGISTER"] = "1"
>>> import numpy as np
>>> from hftrainer.motion.representation import convert
>>> from hftrainer.motion.retarget import smpl_soma30_roundtrip
>>> from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter
>>>
>>> feats = np.load("ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs/000000.npy")  # (T, 263)
>>> hml_joints = convert.hml263_to_joints(feats)                  # (T, 22, 3)
>>> m135       = convert.hml263_to_motion135(feats, device="cuda")  # (T', 135) ROW
>>> soma       = smpl_soma30_roundtrip(m135)                      # SOMA30 + SMPL round trip
>>> g1         = SMPLToG1Retargeter().retarget_from_hymotion(m135)  # legacy quick-look; use GMR for real G1
```

See the runnable demo + web viewer: `scripts/demo/hml263_multi_repr_demo.py` and
`motion_annot_web/repr_convert_demo/app.py`.

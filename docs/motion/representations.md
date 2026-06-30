# Motion Representations

This page documents the repository-level motion representations that can be
saved under `outputs/evaluation/{task}/{dataset}/{motion_representation}/...`.
Use `hftrainer.motion.representation.convert` for cross-representation bridges.

## Summary

| Name | Shape | FPS | Main users | Coordinate / rotation convention |
|---|---:|---:|---|---|
| `motion135` | `(T,135)` | usually 30 | SMPL mesh viewers, HYMotion/PRISM/KIMODO bridges | root translation + 22 row-major local 6D rotations |
| `hml263` | `(T,263)` | 20 | MDM, MoMask, T2M-GPT, MLD, MotionGPT-style baselines | HumanML3D canonical RIC features |
| `ms272` | `(T,272)` | 30 | MotionStreamer, GoToZero, MotionStreamer evaluator | MotionStreamer `humanml3d_272`, Y-up, row-major local 6D block |
| `dart276` | `(T,276)` | 20 | ViMoGen / DART-style models | DART canonical Z-up frame, row-interleaved first-two-column 6D rotations |

## DART276

`dart276` is the DART-style global representation used by ViMoGen. The public
implementation lives in `hftrainer.motion.representation.dart276`.

Per-frame layout:

| Slice | Dim | Meaning |
|---|---:|---|
| `[0:126]` | 126 | 21 SMPL body local rotations as DART/row 6D |
| `[126:192]` | 66 | 22 canonical joint positions |
| `[192:258]` | 66 | 22 canonical joint velocities |
| `[258:264]` | 6 | root/global orientation as DART/row 6D |
| `[264:270]` | 6 | root/global orientation velocity `R[t+1] @ R[t]^T` as DART/row 6D |
| `[270:273]` | 3 | root translation |
| `[273:276]` | 3 | root translation velocity |

Length semantics:

- Encoding a `T`-frame SMPL/joint sequence produces a `(T-1,276)` tensor.
- `dart276_to_smpl_params(..., equal_length=False)` returns `T-1` frames.
- `dart276_to_smpl_params(..., equal_length=True)` integrates the final velocity
  and repeats the final body pose, returning `T` frames.

Coordinate semantics:

- Native DART coordinates are Z-up.
- Canonicalization uses first-frame left/right hips and shoulders to determine
  the body-facing frame.
- The first-frame pelvis joint becomes the translation origin.
- `set_floor=False` by default, matching ViMoGen. This differs from
  HumanML3D/MS272 floor-aligned conventions.
- For ViMoGen/MBench visualization and repository evaluator bridges, use
  `coord_conversion="mbench"`, which applies the official matrix:

```python
[[-1, 0, 0],
 [ 0, 0, 1],
 [ 0, 1, 0]]
```

Rotation convention:

- DART stores `R[..., :, :2].reshape(6)`, i.e.
  `[R00,R01,R10,R11,R20,R21]`.
- In `hftrainer.motion.representation.rotation` this is
  `convention="row"`.
- It is not the standard column-concatenated Zhou-6D vector
  `[R00,R10,R20,R01,R11,R21]`.

SMPL conversion:

```python
from hftrainer.motion.representation.dart276 import (
    dart276_to_smpl_params,
    dart276_to_motion135,
    smpl_params_and_joints_to_dart276,
)

smpl, joints = dart276_to_smpl_params(m276, recover_from_velocity=True, equal_length=True)
m135 = dart276_to_motion135(m276, rotation_convention="row")  # repository motion135
m276_rt = smpl_params_and_joints_to_dart276(smpl, joints)
```

Important caveat: SMPL pose/trans alone is not enough to encode DART276, because
DART276 stores explicit joints. The joints must come from the same body model,
shape, and coordinate frame as the SMPL parameters.

## Motion135 And MotionCLIP135

Repository `motion135` is row-major:

```text
[root_translation(3), 22 * row-major local rot6d(132)]
```

The historical MotionCLIP evaluator expects column-major 6D rotations. Treat
MotionCLIP135 as an evaluator input format, not as the canonical saved
representation. Convert explicitly when needed.

## MS272

`ms272` is MotionStreamer's `humanml3d_272` format. It stores heading-canonical
root velocities, explicit joint positions/velocities, and row-major local
rotations. It is the native input for `MotionStreamer272Evaluator`.

## HML263

`hml263` is the HumanML3D representation used by many T2M baselines. Bridges
from `hml263` to SMPL/MS272 are diagnostic cross-representation conversions and
should be reported as such in model cards and internal run configs.

# KIMODO/SOMA and SMPL Retargeting

This note documents the reusable retargeting API for converting between
KIMODO/SOMA skeleton outputs and SMPL-style `motion_135`. Use this module when
you need to evaluate KIMODO on SMPL/HumanML3D metrics, visualize KIMODO outputs
as SMPL meshes, or measure the loss introduced by a SMPL -> SOMA -> SMPL round
trip.

## Canonical Code Location

Library module:

```text
hftrainer/motion/retarget/smpl_soma.py
```

Public imports:

```python
from hftrainer.motion.retarget import (
    SMPLSOMARetargeter,
    KIMODOSOMAToSMPLRetargeter,
    smpl_motion135_to_soma30,
    smpl_soma30_roundtrip,
    kimodo_soma_to_smpl_motion135,
)
```

The old `hftrainer.models.motion.components.retarget` import path is a thin
compatibility wrapper only. New code should import from
`hftrainer.motion.retarget`.

## Direction 1: SMPL motion_135 to SOMA30

Input:

- `motion_135`: `(T, 135)` = root translation `(3)` plus 22 local rot6d joints.

Output:

- `soma30_joints`: `(T, 30, 3)`.
- `soma30_global_rots`: `(T, 30, 3, 3)`.
- `soma30_local_rots`: `(T, 30, 3, 3)`.

Example:

```python
import numpy as np
from hftrainer.motion.retarget import SMPLSOMARetargeter

motion_135 = np.load("motion.npz")["motion_135"]
retargeter = SMPLSOMARetargeter()
soma = retargeter.smpl_to_soma(motion_135)
```

Default behavior uses the validated shoulder-offset correction. This keeps the
official direct SMPL-to-SOMA rotation transfer but corrects the SOMA shoulder
rest-direction mismatch that previously caused collapsed shoulders.

## Direction 2: SOMA30 Rotations Back to SMPL

This path is intended for round-trip audits where the source SMPL motion is
known:

```python
roundtrip = SMPLSOMARetargeter().roundtrip_smpl(motion_135)
restored_motion = roundtrip["motion_135"]
```

The default height mode is `source_root`, which preserves the original SMPL root
height. Use `height_mode="foot_floor"` only for an explicit ablation or a case
where the source root height is not available.

## Direction 3: KIMODO/SOMA to SMPL motion_135

KIMODO debug/output files usually contain:

- `positions`: `(T, 22, 3)` KIMODO/SOMA body joints.
- `posed_joints`: `(T, 77, 3)` SOMA mesh/skeleton landmarks.
- `global_rot_mats`: `(T, 77, 3, 3)` SOMA global rotations.
- `root_positions`: optional `(T, 3)` root translation.

Use `KIMODOSOMAToSMPLRetargeter`:

```python
import numpy as np
from hftrainer.motion.retarget import KIMODOSOMAToSMPLRetargeter

with np.load("kimodo_output.npz", allow_pickle=True) as data:
    positions = data["positions"]
    global_rot_mats = data["global_rot_mats"]

retargeter = KIMODOSOMAToSMPLRetargeter()
smpl = retargeter.retarget_rotations(global_rot_mats, positions22=positions)
motion_135 = smpl["motion_135"]
```

For NPZ files, prefer:

```bash
python3 scripts/eval/kimodo_soma_to_smpl.py \
  --in-dir outputs/.../debug_npz \
  --out-dir outputs/.../smpl135
```

Position-only fallback is available but degraded:

```python
from hftrainer.motion.retarget import kimodo_soma_to_smpl_motion135

smpl = kimodo_soma_to_smpl_motion135(positions, soma77)
```

This fallback cannot recover twist and upper-body orientation reliably. Do not
use it for KIMODO/SMPL mesh inspection unless the missing-rotation limitation is
the thing being diagnosed.

The fallback IK settings are:

- `orientation_mode="parent_frame"`.
- `parent_ref_weight=0.25`.
- `soma_orientation_guides=True`.
- `head_guide_weight=0.15`.
- `leaf_guide_weight=0.35`.
- `joint_fit_weight_preset="relaxed_upper"`.
- `refine_iters=5`, `refine_lr=0.02`.
- `smooth_weight=0.01`, `joint_accel_weight=0.001`.
- `floor_align=True`, `foot_height_align=True`.

## Dependencies and Paths

SOMA skeleton assets are resolved from:

```text
ref_repo/KIMODO/kimodo/kimodo/assets/skeletons
```

You can override this with:

```bash
export KIMODO_SKELETON_ASSETS=/path/to/KIMODO/kimodo/kimodo/assets/skeletons
```

The SMPL model directory defaults to:

```text
ref_repo/MDM/body_models
```

If `ref_repo/MDM/body_models_nochumpy` exists, the loader prefers it to avoid
legacy chumpy pickle issues.

## Saved Result Fields

KIMODO/SOMA-to-SMPL returns:

- `motion_135`: `(T, 135)` retargeted SMPL motion.
- `transl`: `(T, 3)`.
- `global_orient`: `(T, 3)`.
- `body_pose`: `(T, 63)`.
- `target_joints`: `(T, 22, 3)` after floor alignment.
- `fitted_joints`: `(T, 22, 3)` SMPL forward joints.
- `fit_mpjpe_mm`: per-frame fitting diagnostic.

For batch scripts, save the returned dict with `np.savez_compressed` or call:

```python
retargeter.save_npz("out/000000.npz", smpl, caption="example")
```

## Practical Checks

When adding a new baseline or dataset:

1. Visualize the source KIMODO/SOMA skeleton first.
2. Retarget to SMPL with this library tool.
3. Visualize the SMPL mesh and skeleton together.
4. Inspect `fit_mpjpe_mm`; high values usually indicate wrong coordinates,
   wrong joint order, or missing floor/root alignment.
5. Only then convert to evaluator-specific representations.

Related document:

- [HumanML3D-263 to SMPL Retargeting Pipeline](hml263_to_smpl_retarget_pipeline.md)

# Motion Retargeting Utilities

This directory contains reusable retargeting tools for motion evaluation,
visualization, and embodied-control bridges.

## Public APIs

- `smpl_to_g1.py`
  - `SMPLToG1Retargeter`
  - Converts SMPL/HYMotion motions to Unitree G1 joint space.

- `smpl_soma.py`
  - `SMPLSOMARetargeter`
  - `KIMODOSOMAToSMPLRetargeter`
  - Converts between SMPL `motion_135` and KIMODO/SOMA skeleton outputs.

## KIMODO/SOMA <-> SMPL Quick Use

```python
from hftrainer.motion.retarget import (
    SMPLSOMARetargeter,
    KIMODOSOMAToSMPLRetargeter,
)

# SMPL motion_135 -> SOMA30 -> SMPL motion_135
roundtrip = SMPLSOMARetargeter().roundtrip_smpl(motion_135)

# KIMODO/SOMA positions -> SMPL motion_135
smpl = KIMODOSOMAToSMPLRetargeter().retarget_positions(positions22, soma77)
```

Full documentation:

```text
docs/kimodo_smpl_retargeting.md
```

New scripts should import from `hftrainer.motion.retarget`. The old
`hftrainer.models.motion.components.retarget` path remains as a compatibility
namespace only.

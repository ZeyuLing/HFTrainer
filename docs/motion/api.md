# Motion API

Import the public motion representation helpers from:

```python
from hftrainer.motion.representation import convert
```

Common conversions:

```python
m135 = convert.dart276_to_motion135(m276, rotation_convention="row")
m272 = convert.dart276_to_motion272(m276)
joints = convert.dart276_to_joints(m276, coord="mbench")

joints_hml = convert.hml263_to_joints(m263)
m272_from_135 = convert.motion135_to_motion272(m135)
```

For lower-level DART operations:

```python
from hftrainer.motion.representation.dart276 import (
    dart276_to_smpl_params,
    smpl_params_and_joints_to_dart276,
    canonicalize_smpl_for_dart,
)
```

The canonical `motion135` convention in this repository is row-major local 6D.
Pass `rotation_convention="column"` only for legacy evaluator inputs that
explicitly require MotionCLIP-style column-major rotations.

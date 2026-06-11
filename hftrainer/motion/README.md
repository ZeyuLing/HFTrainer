# HF-Trainer Motion Library

`hftrainer.motion` is the public motion-domain library. It is intended for code
that works with motion data regardless of a specific trainable model.

Use this package for:

- rotation and representation conversion;
- skeleton definitions and FK;
- canonicalization, masking, resampling, and normalization;
- SMPL / SOMA / G1 retargeting;
- task specs and condition patterns;
- reusable motion metrics.

Keep model bundles and neural network layers in `hftrainer.models.motion`.

## Conversions: start here

All cross-representation conversions go through one module:
`hftrainer.motion.representation.convert`.

```python
from hftrainer.motion.representation import convert
joints = convert.hml263_to_joints(m263)        # 263 -> (T,22,3)
m135   = convert.hml263_to_motion135(m263)     # 263 -> SMPL motion_135 (ROW, IK)
m272   = convert.motion135_to_motion272(m135)  # motion_135 -> MS272 (FK+encode)
m272b  = convert.hml263_to_motion272(m263)     # full chain
```

Do NOT pick low-level helpers by hand — the rot6d-convention trap (COLUMN vs
ROW) is the #1 source of silent bugs.

## Docs

- **API reference (every public function/class, signatures + conventions):**
  [`docs/motion/api.md`](../../docs/motion/api.md)
- **Representations & conversions (rot6d trap table, conversion map):**
  [`docs/motion/representations.md`](../../docs/motion/representations.md)
- **Runnable demo + web viewer:** `scripts/demo/hml263_multi_repr_demo.py`,
  `motion_annot_web/repr_convert_demo/app.py`
- Layout source of truth: `hftrainer/motion/representation/specs.py`
- KIMODO/SOMA retarget: [`docs/kimodo_smpl_retargeting.md`](../../docs/kimodo_smpl_retargeting.md)
- Architecture: [`docs/design/motion_library.md`](../../docs/design/motion_library.md)

# Physical Plausibility Metrics

This document describes the physical metrics used by the PRISM paper tables and
the `hftrainer.evaluation.motion.mbench_physics` API.

## Protocol

The Table 1 physical columns follow the non-VLM MBench `motion_quality`
definitions, but they are computed on a shared SMPL-22 joint trajectory instead
of method-specific skeleton dumps.  This avoids cross-method differences caused
by axis conventions, floor conventions, or skeleton definitions.

Supported inputs:

- `motion_135` NPZ: `(T, 135+)`, with `[root_translation(3), 22 * row-major rot6d]`.
- Native MotionStreamer-272 NPY/NPZ: `(T, 272)`, decoded with the stored 272
  positions for Real / conversion controls.

All metrics assume metres, Y-up, and horizontal axes `(X, Z)`.  Finite
differences are frame differences with no fps normalization, matching MBench.

## Metrics

The API returns raw values in metres / frame units:

- `Slide`: average horizontal foot speed on contact frames.
- `Float`: fraction of invalid floating frames.
- `Jitter`: global plus root-relative mean acceleration magnitude.
- `Dynamic`: global plus root-relative mean velocity magnitude.
- `Penet`: average foot depth below the estimated floor.

The paper table scales raw values as:

- `Slide`: `Slide * 1000`, shown in millimetres per frame.
- `Float`: `Float * 100`, shown as percent.
- `Jitter`: `Jitter * 1000`.
- `Dynamic`: `Dynamic * 1000`.
- `Penet`: `Penet * 1000`, shown in millimetres.

`Dynamic` is not a pure error metric.  It is an expressiveness statistic and is
usually interpreted by closeness to Real, while `Slide` and `Float` are physical
error metrics.

## Floor And Contact

For Table 1, the floor is the per-clip minimum foot height after converting the
motion to the shared SMPL-22 FK skeleton.  This is deliberate: some generated
SMPL translations have a constant root-height offset, and using a fixed `Y=0`
floor would mostly measure that representation offset rather than physical
quality.

Contact follows the MBench heuristic:

- a foot is in contact if its 3D velocity is below `0.01`, or
- its height above the clip floor is below `0.02` metres.

Because the floor is the minimum foot height of the same clip, `Penet` is nearly
degenerate under this protocol.  The PRISM main tables therefore report `Slide`,
`Float`, `Jitter`, `Dynamic`, and `PoseQ`, and treat penetration as diagnostic
rather than a primary column.

## Python API

```python
from hftrainer.evaluation.motion.mbench_physics import (
    compute_mbench_physics_for_file,
    evaluate_mbench_physics_dir,
    table_scaled_metrics,
)

raw = compute_mbench_physics_for_file("outputs/evaluation/.../000000.npz", mode="m135")
display = table_scaled_metrics(raw)

summary = evaluate_mbench_physics_dir(
    "outputs/evaluation/ms272_tables_h3d_0607/prep/ours",
    mode="m135",
    workers=16,
)
```

## CLI

Single method:

```bash
python3 scripts/eval/compute_phys_h3d.py \
  --m135-dir outputs/evaluation/ms272_tables_h3d_0607/prep/ours \
  --tag ours \
  --out-json outputs/evaluation/phys_h3d/ours.json
```

Native MotionStreamer-272 Real control:

```bash
python3 scripts/eval/compute_phys_h3d.py \
  --gt272-dir ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data \
  --tag real_smpl
```

Many methods:

```bash
python3 scripts/eval/compute_phys_h3d.py --manifest manifest.tsv --workers 16
```

Each manifest line is:

```text
tag<TAB>m135<TAB>/path/to/repacked_motion135_npz_dir
```

or:

```text
tag<TAB>gt272<TAB>/path/to/native_motion272_dir
```

## Visualization

`motion_annot_web/t2m_compare/app.py` uses the same API to show per-clip metrics
beside the SMPL mesh.  This is the recommended sanity check before trusting
aggregate physical columns: inspect examples where `Slide`, `Float`, or
`Jitter` disagree with visual quality and decide whether the heuristic is
capturing the intended artifact.

## Caveats

- `Slide` depends on the contact detector, so motions with unusual foot height
  offsets can still be misclassified if min-foot floor alignment is not enough.
- `Float` is a heuristic over root and relative foot motion.  It is useful for
  large floating artifacts, but it is not a full dynamics or force-consistency
  metric.
- `Jitter` and `Dynamic` use frame differences, so they are comparable only when
  methods are evaluated at the same fps and frame count policy.
- `PoseQ` is computed separately by `scripts/eval/compute_pose_quality_h3d.py`
  through the MBench NRDF model.  It is not implemented in
  `mbench_physics.py` because it requires the ViMoGen NRDF checkpoint.

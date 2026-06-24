# Motion Library Architecture / Motion 公共库架构

This document defines the target architecture for the motion branch of
HF-Trainer. The current tree places reusable motion-processing utilities under
`hftrainer.models.motion`, which makes the package hard to navigate and blurs
the boundary between model code and general motion infrastructure.

本文定义 HF-Trainer motion branch 的目标架构。当前目录把可复用的动作处理能力放在
`hftrainer.models.motion` 下，导致“模型实现”和“动作公共库”边界混乱，也让 retarget、
representation、FK、task/mask 等能力不容易被自然发现。

## Problem / 当前问题

`hftrainer.models.motion` currently mixes four responsibilities:

当前 `hftrainer.models.motion` 混合了四类职责：

| Responsibility | Current Examples | Problem |
|---|---|---|
| Trainable model implementations | `prism/`, `vermo/`, `hymotion_m2m/`, `motion_clip/` | This is the correct role for `models/`. |
| Reusable motion math and representation | `components/utils/geometry`, `components/body_models`, `components/motion_processor`, `hymotion_m2m/network/geometry.py` | Non-model code is hidden behind model paths. |
| Retargeting and skeleton bridges | `components/retarget`, scripts under `scripts/eval` and `scripts/analysis` | Developers have to know historical scripts to find canonical tools. |
| Task and condition semantics | `vermo/task_utils`, MotionHub transforms, M2M task/mask code | Dataset, model, and task semantics are coupled. |

This leads to imports like:

```python
from hftrainer.models.motion.components.utils.geometry.rotation_convert import ...
from hftrainer.models.motion.hymotion_m2m.network.geometry import ...
from hftrainer.models.motion.vermo.task_utils import ...
```

These imports are semantically wrong for a public library: rotation conversion,
FK, task masks, and retargeting are motion-domain utilities, not model modules.

这些 import 对公共库来说语义不对：旋转转换、FK、任务 mask、retargeting 都是动作领域工具，
不是模型模块。

## Target Principle / 目标原则

The motion branch should expose a domain library first, and models second:

motion branch 应先是动作领域公共库，其次才是模型集合：

```text
hftrainer.motion        # public motion-domain library
hftrainer.models.motion # trainable model bundles and neural networks only
hftrainer.pipelines.motion
hftrainer.trainers.motion
hftrainer.datasets.motion
hftrainer.evaluation.motion
```

Rules:

1. `hftrainer.motion` must be import-light. Importing it should not load
   training frameworks, DeepSpeed, large text encoders, or specific model
   bundles.
2. `hftrainer.models.motion` should contain only model bundles, neural network
   modules, model-local losses, and model-local checkpoint logic.
3. Geometry, body models, representations, retargeting, canonicalization, masks,
   and task specs belong to `hftrainer.motion`.
4. Existing paths should remain as compatibility wrappers during migration.
5. Scripts should call library APIs rather than copying conversion functions.

规则：

1. `hftrainer.motion` 必须轻量 import，不应加载 trainer、DeepSpeed、大模型文本编码器或具体模型 bundle。
2. `hftrainer.models.motion` 只保留可训练模型、网络层、模型局部 loss 和 checkpoint 逻辑。
3. 几何、人体模型、表示、retarget、canonicalization、mask、task spec 归入 `hftrainer.motion`。
4. 迁移期间保留旧路径兼容 wrapper。
5. 脚本调用库 API，不复制转换函数。

## Proposed Package Layout / 推荐目录结构

```text
hftrainer/
├── motion/
│   ├── __init__.py
│   ├── data/
│   │   ├── structures.py          # MotionClip, MotionBatch, Metadata dataclasses
│   │   └── io.py                  # NPZ/NPY/Pickle loading and saving conventions
│   ├── representation/
│   │   ├── specs.py               # motion_135/138/147/151/198/HML263/MS272 specs
│   │   ├── rotation.py            # rot6d/axis-angle/quaternion/matrix conversion
│   │   ├── humanml.py             # HumanML3D/HML263 representation helpers
│   │   ├── motion135.py           # SMPL22 motion_135 utilities
│   │   └── motion272.py           # MotionStreamer 272 utilities
│   ├── skeleton/
│   │   ├── names.py               # SMPL22, SMPLX, SOMA30/SOMA77, G1 names/parents
│   │   ├── fk.py                  # NumPy/Torch forward kinematics
│   │   └── body_models.py         # light SMPL/SMPL-X loaders
│   ├── processing/
│   │   ├── normalize.py           # mean/std, canonical scale helpers
│   │   ├── resample.py            # fps and temporal resampling
│   │   ├── canonicalize.py        # yaw/XZ canonicalization and inverse
│   │   ├── masks.py               # frame/joint/coordinate mask utilities
│   │   └── smoothing.py           # smoothing tools, explicit eval/visual flags
│   ├── tasks/
│   │   ├── specs.py               # T2M, inbetween, repair, edit, control specs
│   │   ├── condition_patterns.py  # rank-k / sampler structures
│   │   └── instructions.py        # task-to-text instruction templates
│   ├── retarget/
│   │   ├── smpl_soma.py           # KIMODO/SOMA <-> SMPL
│   │   ├── smpl_g1.py             # SMPL -> Unitree G1
│   │   └── hml263_smpl.py         # HumanML3D/HML263 -> SMPL IK
│   ├── metrics/
│   │   ├── quality.py             # skating, jitter, acceleration
│   │   └── constraints.py         # mask/trajectory/keypoint satisfaction
│   └── visualization/
│       └── export.py              # viewer-friendly NPZ/layout conversion only
│
└── models/
    └── motion/
        ├── prism/
        ├── vermo/
        ├── hymotion_m2m/
        ├── hymotion_t2m/
        ├── hymotion_umo/
        ├── motion_clip/
        └── physflow/
```

## Current-to-Target Mapping / 当前到目标映射

| Current Path | Target Path | Migration Notes |
|---|---|---|
| `hftrainer.models.motion.components.utils.geometry.rotation_convert` | `hftrainer.motion.representation.rotation` | Keep old module as re-export. |
| `hftrainer.models.motion.hymotion_m2m.network.geometry` | `hftrainer.motion.representation.rotation` and `hftrainer.motion.skeleton.fk` | Split pure geometry from model-local network code. |
| `hftrainer.models.motion.components.body_models.smplx_lite` | `hftrainer.motion.body_models.smplx_lite` / `hftrainer.motion.skeleton.body_models` | Implementation moved; old path is compatibility only. |
| `hftrainer.models.motion.components.motion_processor.smpl_processor` | `hftrainer.motion.processing.smpl_processor` | Implementation moved; old path is compatibility only. |
| `hftrainer.models.motion.components.retarget.smpl_soma` | `hftrainer.motion.retarget.smpl_soma` | Implementation moved; old path is compatibility only. |
| `hftrainer.models.motion.components.retarget.smpl_to_g1` | `hftrainer.motion.retarget.smpl_g1` | Implementation moved and renamed for consistent direction naming. |
| `hftrainer.pipelines.motion.transition_utils` | `hftrainer.motion.processing.canonicalize` | Keep pipeline wrapper for compatibility. |
| `hftrainer.pipelines.motion.differentiable_fk` | `hftrainer.motion.skeleton.fk` | FK is not pipeline-specific. |
| `hftrainer.datasets.motion.representation.humanml_repr` | `hftrainer.motion.representation.humanml` | Dataset can import from the public representation package. |
| `hftrainer.models.motion.vermo.task_utils` | `hftrainer.motion.tasks` | VerMo should depend on task specs, not own them. |
| `hftrainer.evaluation.motion.*` | `hftrainer.motion.metrics` plus evaluator wrappers | Pure metrics move to library; evaluator integration stays in `evaluation/`. |
| `hftrainer.models.motion.physflow.dataset` | `hftrainer.datasets.motion.physflow` | Dataset does not belong under models. |
| `hftrainer.models.motion.physflow.reward` | `hftrainer.evaluation.motion.physflow` or `hftrainer.motion.metrics.physics` | Reward/judge is evaluator-like unless it is trainable model code. |

## Import Policy / Import 规则

Preferred imports after migration:

```python
from hftrainer.motion.representation.rotation import rotation_6d_to_matrix
from hftrainer.motion.skeleton.fk import forward_kinematics
from hftrainer.motion.retarget import KIMODOSOMAToSMPLRetargeter
from hftrainer.motion.processing.canonicalize import canonicalize_segment
```

Avoid in new code:

```python
from hftrainer.models.motion.components.utils.geometry.rotation_convert import ...
from hftrainer.models.motion.hymotion_m2m.network.geometry import ...
from hftrainer.models.motion.vermo.task_utils import ...
```

## Migration Plan / 迁移计划

### Phase 0: Freeze the Design

- Add this document.
- Document which current modules are public APIs versus model-local internals.
- Add `hftrainer/models/motion/components/retarget/README.md` for immediate
  discoverability while migration is not complete.

### Phase 1: Create the New Public Package Without Moving Logic

Create:

```text
hftrainer/motion/
```

This phase is complete. `hftrainer.motion` now exposes representation, skeleton,
retargeting, body-model, and processing entry points without requiring callers to
know the model-zoo package layout.

Important current blocker: `hftrainer/__init__.py` still auto-registers the
full training stack on package import because many historical scripts rely on
`import hftrainer  # trigger auto-imports`. Before `hftrainer.motion` can be
strictly import-light, training and inference entry points must call an explicit
`register_all_modules()` function, then root-package auto-registration can be
disabled by default.

### Phase 2: Move Pure Utilities

Move files that have no model registry side effects:

- rotation conversion.
- FK and differentiable FK.
- skeleton names and parent arrays.
- HumanML3D/HML263 representation helpers.
- SMPL/SOMA/G1 retargeting.
- canonicalization and resampling.

Retargeting, body-model loaders, and SMPL pose processing have moved to
`hftrainer.motion.*`. The old `hftrainer.models.motion.components.*` modules are
kept only as thin compatibility wrappers.

Old paths become thin compatibility wrappers and should emit no warning at
first. Deprecation warnings can be added only after all internal code has moved.

### Phase 3: Decouple Task Semantics

Move general task definitions and condition/mask samplers to
`hftrainer.motion.tasks` and `hftrainer.motion.processing.masks`.

This is especially important because M2M, VerMo, datasets, and evaluators all
need task semantics, but none of those should import from a specific model.

### Phase 4: Update Internal Imports

Update internal code in this order:

1. `hftrainer/datasets/motion`.
2. `hftrainer/evaluation/motion` and quality rules.
3. `hftrainer/pipelines/motion`.
4. `scripts/`.
5. `hftrainer/models/motion/*`.
6. tests.

Run smoke and targeted tests after each layer.

### Phase 5: Clean Models

After imports are migrated:

- `hftrainer.models.motion.components` should either disappear or become a
  deprecated compatibility namespace.
- `hftrainer.models.motion.hymotion_m2m.network` should contain only network
  modules, not generic geometry.
- `hftrainer.models.motion.vermo.task_utils` should be model-specific adapters
  only, not the canonical task library.

## Testing Requirements / 测试要求

Every moved module needs at least one import or behavior test:

| Area | Test |
|---|---|
| Rotation conversion | rot6d/matrix/axis-angle round trip on random tensors |
| FK | SMPL22 local rotations + translation -> stable joint shapes |
| Canonicalization | canonicalize + decanonicalize returns original motion |
| Retarget | SMPL->SOMA->SMPL smoke; KIMODO/SOMA->SMPL smoke |
| Representation | HML263/HumanML conversion shape and scale checks |
| Task masks | deterministic sampler output under fixed seed |

The public package must also pass:

```bash
python3 - <<'PY'
import hftrainer.motion
from hftrainer.motion.retarget import SMPLSOMARetargeter
from hftrainer.motion.representation.rotation import rotation_6d_to_matrix
PY
```

This command should not load model bundles, trainers, or DeepSpeed.

## Non-Goals / 非目标

- Do not move trainable model bundles out of `hftrainer.models.motion`.
- Do not force all scripts to be rewritten in one commit.
- Do not change evaluator numbers while reorganizing imports.
- Do not hide smoothing/post-processing inside retargeting utilities. Retarget
  tools should expose diagnostics; evaluation scripts decide what to report.

## Immediate Next Step / 下一步

The recommended first implementation step is Phase 1:

1. Create `hftrainer/motion/` with public subpackages.
2. Re-export the new KIMODO/SOMA retargeter under `hftrainer.motion.retarget`.
3. Re-export rotation conversion and SMPL/G1 retarget tools.
4. Convert core entry points (`tools/train.py`, `tools/infer.py`) and high-use
   eval scripts from implicit `import hftrainer` side effects to explicit
   `register_all_modules()`.
5. Make `hftrainer/__init__.py` default-light, with an escape hatch for legacy
   scripts if needed.
6. Add import-light tests.

This gives developers a clean public path immediately while preserving all
existing imports and experiments.

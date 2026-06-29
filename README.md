# HF-Trainer

HF-Trainer is a motion-generation model zoo and reproducibility toolkit built
around HuggingFace-style model bundles, validated motion representations,
retargeting operators, and persisted evaluators.

It is intentionally more than a trainer wrapper. The repository provides the
infrastructure needed to reproduce and compare text-to-motion and
kinematic-control methods end to end:

- self-contained `from_pretrained` / `save_pretrained` artifacts for reproduced
  motion models;
- a public motion-domain library under `hftrainer.motion` for representations,
  skeletons, retargeting, and processing;
- persisted HumanML3D-263 and MotionStreamer-272 evaluators;
- paper-facing scripts for generation, conversion, metric collection, and mesh
  inspection.

[Install](#installation) |
[Model Zoo](#model-zoo) |
[Supported Tasks](#supported-tasks) |
[Motion Library](#motion-library) |
[Evaluators](#evaluators) |
[Architecture](#architecture)

---

## Installation

```bash
git clone <repo-url> hf_trainer
cd hf_trainer
pip install -e .
```

For import-light use in scripts and notebooks:

```bash
export HFTRAINER_SKIP_AUTOREGISTER=1
```

Many model cards also require pretrained assets under `checkpoints/` or a
downloaded HuggingFace snapshot. See each card in
[`docs/model_zoo`](docs/model_zoo/README.md).

## Quick Start

Load a reproduced text-to-motion baseline:

```python
from hftrainer.pipelines.momask import MoMaskPipeline

pipe = MoMaskPipeline.from_pretrained("ZeyuLing/hftrainer-momask-humanml3d", device="cuda")
motions = pipe.infer_t2m(["a person walks forward then sits down"], [120])
```

Convert motion between evaluator spaces:

```python
from hftrainer.motion.representation import convert

m135 = convert.hml263_to_motion135(m263, device="cuda")  # HML263 -> SMPL motion_135
m272 = convert.motion135_to_motion272(m135)              # SMPL motion_135 -> MS272
```

Retarget KIMODO/SOMA output to SMPL mesh motion:

```python
from hftrainer.motion.retarget import KIMODOSOMAToSMPLRetargeter

retargeter = KIMODOSOMAToSMPLRetargeter(device="cuda")
smpl = retargeter.retarget_file("kimodo_debug_npz/000000.npz")
motion_135 = smpl["motion_135"]
```

Score a prediction directory:

```bash
python3 scripts/eval/verify_evaluators.py \
  --which hml263 \
  --hml263-pred outputs/evaluation/momask_h3d263_official/momask_263
```

## Model Zoo

Model-Zoo entries expose hftrainer-native bundles and, where available,
self-contained artifacts. Reproduced baselines vendor the model code needed for
inference instead of importing the original repository at runtime.

| Model | Primary task | Native representation | Bundle / Pipeline | Processed Hugging Face artifact |
|---|---|---|---|---|
| [MDM](docs/model_zoo/mdm.md) | Text-to-motion | HumanML3D-263 | `MDMBundle` / `MDMPipeline` | [`ZeyuLing/hftrainer-mdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) |
| [T2M-GPT](docs/model_zoo/t2mgpt.md) | Text-to-motion | HumanML3D-263 | `T2MGPTBundle` / `T2MGPTPipeline` | [`ZeyuLing/hftrainer-t2mgpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) |
| [MoMask](docs/model_zoo/momask.md) | Text-to-motion | HumanML3D-263 | `MoMaskBundle` / `MoMaskPipeline` | [`ZeyuLing/hftrainer-momask-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d) |
| [MoGenTS](docs/model_zoo/mogents.md) | Text-to-motion | HumanML3D-263 | `MoGenTSBundle` / `MoGenTSPipeline` | [`ZeyuLing/hftrainer-mogents-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) |
| [MLD](docs/model_zoo/mld.md) | Text-to-motion | HumanML3D-263 | `MLDBundle` / `MLDPipeline` | [`ZeyuLing/hftrainer-mld-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) |
| [FlowMDM](docs/model_zoo/flowmdm.md) | Text-to-motion / motion composition | HumanML3D-263 | `FlowMDMBundle` / `FlowMDMPipeline` | [`ZeyuLing/hftrainer-flowmdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d) |
| [MotionLab](docs/model_zoo/motionlab.md) | Text-to-motion / motion editing | HumanML3D-263 | `MotionLabBundle` / `MotionLabPipeline` | [`ZeyuLing/hftrainer-motionlab-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlab-humanml3d) |
| [MotionGPT](docs/model_zoo/motiongpt.md) | Text-to-motion / motion-language generation | HumanML3D-263 | `MotionGPTBundle` / `MotionGPTPipeline` | [`ZeyuLing/hftrainer-motiongpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d) |
| [MotionGPT3](docs/model_zoo/motiongpt3.md) | Text-to-motion / motion-language generation | HumanML3D-263 | `MotionGPT3Bundle` / `MotionGPT3Pipeline` | [`ZeyuLing/hftrainer-motiongpt3-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt3-humanml3d) |
| [ViMoGen](docs/model_zoo/vimogen.md) | Text-to-motion | DART276 / SMPL motion_135 | `ViMoGenBundle` / `ViMoGenPipeline` | [`ZeyuLing/hftrainer-vimogen-1.3b-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) |
| [MotionLCM](docs/model_zoo/motionlcm.md) | Text-to-motion | HumanML3D latent / 263 bridge | `MotionLCMBundle` / `MotionLCMPipeline` | [`ZeyuLing/hftrainer-motionlcm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) |
| [MotionStreamer](docs/model_zoo/motionstreamer.md) | Streaming text-to-motion | MotionStreamer-272 | `MotionStreamerBundle` / `MotionStreamerPipeline` | [`ZeyuLing/hftrainer-motionstreamer-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) |
| [Go to Zero / MotionMillion](docs/model_zoo/gotozero.md) | Zero-shot text-to-motion | MotionStreamer-272 | `MotionMillionBundle` / `MotionMillionPipeline` | [`7B-train`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272), [`3B-train`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272) |
| [HY-Motion T2M 1.0](docs/model_zoo/hymotion_t2m.md) | Text-to-motion | HY-Motion 201 / SMPL motion_135 | `HyMotionT2MBundle` / `HyMotionT2MPipeline` | [`full`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0), [`lite`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) |
| [KIMODO](docs/model_zoo/kimodo.md) | Text + kinematic control | SOMA / G1 / SMPL-X | `KIMODOBundle` / `KIMODOPipeline` | [`ZeyuLing/hftrainer-kimodo-soma-rp`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp) (private / license review) |

Research stacks in the repository:

| Stack | Scope | Public entry point |
|---|---|---|
| PRISM / PRISM-MCM | audio/text motion generation and control | `PrismBundle`, `PrismMCMBundle`, `hftrainer.pipelines.motion.prism_*` |
| VerMo | VQ/AR motion generation components | `VermoBundle`, `VermoProcessor` |
| HY-Motion-M2M | motion-to-motion editing / control research pipeline | `HyMotionM2MBundle`, `HyMotionM2MPipeline` |
| HY-Motion-UMO | temporal fusion on HunyuanMotion MMDiT | `HyMotionUMOBundle`, `HyMotionUMOPipeline` |
| MotionCLIP | motion-text contrastive model | `MotionCLIPBundle` |
| PhysFlow | KIMODO-G1 generation with physics reward tooling | `PhysFlowBundle`, `PhysFlowG1Bundle` |

The complete index is in [`docs/model_zoo/README.md`](docs/model_zoo/README.md).
Each card should state the artifact layout, representation, exact evaluator
protocol, metric JSON path, and original paper/code links.
The canonical calling method for each model lives in
`docs/model_zoo/<method>.md` under the loading / pipeline sections; the root
README keeps only cross-model quick starts. Uploaded Hugging Face artifacts
mirror the corresponding `docs/model_zoo` card. Keep them synchronized with
`python3 tools/sync_model_zoo_cards.py --write-local` and, after logging in to
Hugging Face, `python3 tools/sync_model_zoo_cards.py --push`.

## Supported Tasks

HF-Trainer exposes task-specific motion APIs instead of collapsing motion work
into a generic training-task checklist.

| Task family | What is supported | Representative APIs |
|---|---|---|
| Text-to-motion | Prompt-conditioned generation on HumanML3D-263, MS272, HY-Motion 201, or SMPL motion_135 output spaces | `infer_t2m`, `text_to_motion`, model-specific pipelines |
| Length-conditioned T2M | Fixed-frame generation for evaluator parity | `infer_t2m(texts, lengths)` |
| Streaming / autoregressive T2M | token-by-token or segment-wise generation | `MotionStreamerPipeline`, `MotionMillionPipeline` |
| Text + kinematic control | KIMODO prompt generation with multi-prompt stitching, full-body keyframes, end-effectors, root-2D paths, and saved constraint JSON | `KIMODOPipeline.multi_prompt`, `fullbody_keyframe_constraint`, `end_effector_constraint`, `root2d_constraint` |
| Motion representation conversion | HML263, MS272, SMPL `motion_135`, HY-Motion 201, SOMA30/77, InterHuman-262 bridges | `hftrainer.motion.representation.convert` |
| Retargeting | HML263 -> SMPL, SMPL <-> SOMA, KIMODO/SOMA -> SMPL, SMPL -> Unitree G1 | `hftrainer.motion.retarget` |
| Evaluation | persisted retrieval metrics, cross-representation scoring, physical quality checks | `HumanML263Evaluator`, `MotionStreamer272Evaluator`, `hftrainer.evaluation.motion.mbench_physics`, `hftrainer.evaluation.quality_check_rules` |
| Visualization | reusable motion visualization protocols/operators plus viewer consumers with explicit reference / condition-marker / generated roles | `hftrainer.motion.visualization`, `docs/visualization.md` |
| Training / fine-tuning | config-driven training for research stacks and reusable modules | `tools/train.py`, `tools/dist_train.sh`, MMEngine-style configs |

## Motion Library

Reusable motion code lives under [`hftrainer.motion`](hftrainer/motion/README.md).
Model bundles live under `hftrainer.models.motion`; domain logic should not be
hidden behind model-zoo paths.

### Representations

The repository currently uses these motion spaces:

| Representation | FPS | Shape | Used by |
|---|---:|---|---|
| HumanML3D-263 | 20 | `(T, 263)` | MDM, T2M-GPT, MoMask, MoGenTS, MLD, FlowMDM, MotionLab, MotionGPT, MotionGPT3, MotionLCM, HumanML evaluator |
| MotionStreamer-272 | 30 | `(T, 272)` | MotionStreamer, Go to Zero, MS272 evaluator |
| SMPL `motion_135` | usually 30 | root translation + 22 row-major rot6d joints | mesh rendering, HY-Motion scoring bridge, retargeting |
| DART-276 | 20 | `(T, 276)` | ViMoGen / DART-style models; bridge through `dart276 -> motion135 -> ms272` |
| HY-Motion 201 | 30 | SMPL 135 + joint-position features | HY-Motion T2M |
| SOMA30 / SOMA77 | model dependent | KIMODO skeleton rotations and joints | KIMODO |
| Unitree G1 | model dependent | 29-DOF robot qpos/qpos-like output | embodied retargeting |

Always use [`hftrainer.motion.representation.convert`](hftrainer/motion/representation/convert.py)
for cross-representation conversions. The API reference is
[`docs/motion/api.md`](docs/motion/api.md), and the representation guide is
[`docs/motion/representations.md`](docs/motion/representations.md).

### Retargeting

Canonical retargeting code lives in
[`hftrainer.motion.retarget`](hftrainer/motion/retarget):

| Operator | Use case |
|---|---|
| `retarget_hml263_clip` / `hml263_to_motion135` | HumanML3D-263 predictions to SMPL motion_135 |
| `SMPLSOMARetargeter` | SMPL motion_135 <-> SOMA30 rotation transfer |
| `KIMODOSOMAToSMPLRetargeter` | KIMODO SOMA output to SMPL motion_135 |
| `GMRSMPLToG1Retargeter` | SMPL/SMPL-H/SMPL-X -> Unitree G1 via GMR mink IK (visualization / deployment) |

> For SMPL -> Unitree G1, use **`GMRSMPLToG1Retargeter`** (General Motion
> Retargeting, mink IK). (A previous fast analytic Euler-decomposition backend
> was removed — it produced low-quality, broken poses.) It is a first-class
> in-repo API that wraps a **minimal in-tree vendored GMR**
> (`hftrainer/motion/retarget/_gmr/`, no `ref_repo` dependency) and returns a
> ground-aligned, Z-up G1 motion (`dof_pos` + floating-base root) ready for
> MuJoCo:
>
> ```python
> from hftrainer.motion.retarget import GMRSMPLToG1Retargeter
> res = GMRSMPLToG1Retargeter().retarget_smplh(poses, trans, betas=betas, fps=30)
> qpos = GMRSMPLToG1Retargeter().to_mujoco_qpos(res)   # (T, 36)
> ```
>
> See
> [`docs/motion/representations.md` §9](docs/motion/representations.md#9-smpl-motion_135---unitree-g1-gmr-retarget),
> which documents all entrypoints, the lazy GMR runtime deps (`mink daqp smplx
> mujoco`, vendored in-tree so not in `pyproject.toml`), the headless GL backend
> (OSMesa/EGL) needed for offscreen mesh rendering, and the automatic
> ground-alignment (excluding the mjcf floor plane) that keeps the robot's feet
> on the ground instead of floating or sinking.

For KIMODO mesh inspection, the correct path requires `global_rot_mats` in the
KIMODO debug NPZ. Position-only IK is a degraded fallback and should not be used
as a mesh-quality signal.

## Evaluators

The text-to-motion evaluator stack is persisted under
[`hftrainer.evaluation.evaluators`](hftrainer/evaluation/evaluators).

| Evaluator | Input | Metrics |
|---|---|---|
| `HumanML263Evaluator` | unnormalized HML263 `(T,263)` at 20 fps | FID, R-Precision, Matching Score, Diversity |
| `MotionStreamer272Evaluator` | unnormalized MS272 `(T,272)` at 30 fps | FID, R-Precision, Matching Score, Diversity |

Use [`docs/motion/evaluators.md`](docs/motion/evaluators.md) for the operating
contract: prediction layouts, conversion paths, required GT rows, and failure
checklist. Model-card numbers should be copied from generated JSON files, not
typed by hand.

## Artifact Standard

Published model-zoo artifacts should follow the same expectations:

- `from_config`, `from_pretrained`, and `save_pretrained` are available on the
  bundle;
- trainable/generative weights are stored in the artifact, preferably as
  `safetensors`;
- frozen text encoders are included when storage/licensing permits;
- normalization stats and representation metadata live next to the weights;
- `model_index.json` or equivalent metadata identifies the bundle and pipeline;
- conversion scripts provide `--verify` round-trip checks when practical.

## Architecture

The codebase separates three layers:

| Layer | Location | Responsibility |
|---|---|---|
| Motion library | `hftrainer.motion` | representations, skeletons, retargeting, processing, reusable domain APIs |
| Model zoo | `hftrainer.models.motion` + `hftrainer.pipelines` | model bundles, neural networks, samplers, inference APIs |
| Evaluation | `hftrainer.evaluation` + `scripts/eval` | persisted evaluators, conversion jobs, metric collection, paper protocol scripts |

`hftrainer.models.motion.components` is reserved for neural network blocks that
are shared across model-zoo methods. Body models, retargeting, and motion
processors belong to `hftrainer.motion`.

Design notes:

- [`docs/design/motion_library.md`](docs/design/motion_library.md)
- [`docs/motion/api.md`](docs/motion/api.md)
- [`docs/motion/evaluators.md`](docs/motion/evaluators.md)
- [`docs/motion/physical_metrics.md`](docs/motion/physical_metrics.md)

## Documentation

| Guide | Description |
|---|---|
| [Model Zoo](docs/model_zoo/README.md) | reproduced baselines, artifacts, metrics, papers |
| [Motion Representations](docs/motion/representations.md) | HML263/MS272/SMPL/SOMA/G1 layouts and conversion map |
| [Motion API](docs/motion/api.md) | public motion library API reference |
| [Evaluators](docs/motion/evaluators.md) | evaluator protocols and failure checklist |
| [Physical Metrics](docs/motion/physical_metrics.md) | SMPL-22 MBench-style physical plausibility protocol |
| [Visualization](docs/visualization.md) | mesh viewer schemas, KIMODO task protocols, recording workflow |
| [KIMODO Retargeting](docs/kimodo_smpl_retargeting.md) | SOMA/SMPL retargeting contract |
| [Architecture](docs/architecture.md) | training framework structure |
| [Design Docs](docs/design/index.md) | deeper rationale and migration notes |

## License

Project code is intended to use the Apache 2.0 license. Third-party model
weights, datasets, and vendored reference code keep their original licenses;
check the corresponding model card and upstream repository before
redistribution.

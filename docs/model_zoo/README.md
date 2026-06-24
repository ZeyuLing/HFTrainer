# hftrainer Motion Model Zoo

This directory is the public index for reproduced motion models, runtime
wrappers, artifacts, and metric cards.

The publishing standard is:

- bundle-level `from_config`, `from_pretrained`, and `save_pretrained`;
- self-contained generative weights and normalization stats;
- frozen text encoders stored inside the artifact whenever storage/licensing
  permits;
- explicit native representation and conversion path;
- metrics copied from JSON files produced by persisted evaluators;
- links to the processed Hugging Face artifact when one exists;
- links to the original paper and code.

User-facing calling examples live in the individual model cards. This index is
only the table of contents; each `docs/model_zoo/<model>.md` card should include
the canonical `{Method}Pipeline.from_pretrained(...)` example, local
`snapshot_download` form when useful, pipeline calls, representation notes, and
evaluator commands. Uploaded Hugging Face artifacts mirror the same card through
`tools/sync_model_zoo_cards.py`.

## Reproduced / Published Baselines

These entries are expected to be usable as model-zoo baselines.

| Model | Primary task | Native representation | Bundle / Pipeline | Processed Hugging Face artifact | Card |
|---|---|---|---|---|---|
| MDM | Text-to-motion | HumanML3D-263 | `MDMBundle` / `MDMPipeline` | [`ZeyuLing/hftrainer-mdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) | [mdm.md](mdm.md) |
| T2M-GPT | Text-to-motion | HumanML3D-263 | `T2MGPTBundle` / `T2MGPTPipeline` | [`ZeyuLing/hftrainer-t2mgpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) | [t2mgpt.md](t2mgpt.md) |
| MoMask | Text-to-motion | HumanML3D-263 | `MoMaskBundle` / `MoMaskPipeline` | [`ZeyuLing/hftrainer-momask-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d) | [momask.md](momask.md) |
| MoGenTS | Text-to-motion | HumanML3D-263 | `MoGenTSBundle` / `MoGenTSPipeline` | [`ZeyuLing/hftrainer-mogents-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) | [mogents.md](mogents.md) |
| PRISM 1.0 | Text-to-motion | PRISM `motion_138` / SMPL motion_135 | `PrismBundle` / `PrismPipeline` | [`ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000`](https://huggingface.co/ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000) (raw checkpoint package) | [prism.md](prism.md) |
| InterGen | Two-person text-to-motion | InterHuman native-262 | `InterGenBundle` | [`ZeyuLing/hftrainer-intergen-interhuman`](https://huggingface.co/ZeyuLing/hftrainer-intergen-interhuman) | [intergen.md](intergen.md) |
| InterMask | Two-person / InterX text-to-motion | InterHuman native-262 / InterX `(T,56,12)` | `InterMaskBundle` | [`InterHuman`](https://huggingface.co/ZeyuLing/hftrainer-intermask-interhuman), [`InterX`](https://huggingface.co/ZeyuLing/hftrainer-intermask-interx) | [intermask.md](intermask.md) |
| MotionLCM | Text-to-motion | HumanML3D latent / HML263 bridge | `MotionLCMBundle` / `MotionLCMPipeline` | [`ZeyuLing/hftrainer-motionlcm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) | [motionlcm.md](motionlcm.md) |
| MotionStreamer | Streaming text-to-motion | MotionStreamer-272 | `MotionStreamerBundle` / `MotionStreamerPipeline` | [`ZeyuLing/hftrainer-motionstreamer-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) | [motionstreamer.md](motionstreamer.md) |
| Go to Zero / MotionMillion | Zero-shot text-to-motion | MotionStreamer-272 | `MotionMillionBundle` / `MotionMillionPipeline` | [`ZeyuLing/hftrainer-gotozero-7b-humanml272`](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-humanml272) | [gotozero.md](gotozero.md) |
| HY-Motion T2M 1.0 | Text-to-motion | HY-Motion 201 / SMPL motion_135 | `HyMotionT2MBundle` / `HyMotionT2MPipeline` | [`full`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0), [`lite`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) | [hymotion_t2m.md](hymotion_t2m.md) |
| KIMODO | Text + kinematic control | SOMA / G1 / SMPL-X | `KIMODOBundle` / `KIMODOPipeline` | [`SOMA-RP`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp), [`G1-RP`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp), [`G1-SEED`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed), [`SMPLX-RP`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp) (private / license review) | [kimodo.md](kimodo.md) |
| ProtoMotions G1 Tracker | Unitree G1 motion tracking | G1 motion library / robot frames | `PhysicsJudgeReward` / ProtoMotions eval scripts | local bundle under `hftrainer/models/motion/physflow/trackers/protomotions/` | [protomotions_g1_tracker.md](protomotions_g1_tracker.md) |
| Any2Track G1 Tracker | Unitree G1 motion tracking | G1 qpos / MuJoCo body tracking | `Any2TrackJudgeReward` / ONNX MuJoCo rollout | local bundle under `hftrainer/models/motion/physflow/trackers/any2track/` | [any2track_g1_tracker.md](any2track_g1_tracker.md) |
| Humanoid-GPT G1 Tracker | Unitree G1 motion tracking | G1 qpos to keypoint reference | `HgptJudgeReward` / worker server | local bundle under `hftrainer/models/motion/physflow/trackers/humanoid_gpt/` | [humanoid_gpt_g1_tracker.md](humanoid_gpt_g1_tracker.md) |

## Runtime Implementation Audit

This table separates the public artifact inference path from raw-checkpoint
conversion utilities. Paths under `ref_repo/` may still appear in conversion
scripts or optional raw upstream loaders, but the published artifact path should
not require importing upstream implementation code unless explicitly noted.

| Model | Artifact inference implementation | `ref_repo` runtime dependency? | Notes |
|---|---|---|---|
| MDM | `hftrainer.models.motion.mdm.network` native network / diffusion / sampler | No | raw `.pt` conversion uses upstream checkpoint files only |
| T2M-GPT | `hftrainer.models.motion.t2mgpt.network` native VQ-VAE + GPT | No | raw `.pth` conversion uses upstream checkpoint files only |
| MoMask | `hftrainer.models.motion.momask.network` native RVQ-VAE + transformers | No | raw `.tar` conversion uses upstream checkpoint files only |
| MoGenTS | `hftrainer.models.motion.mogents.network` native dual RVQ-VAE + 1D/2D transformers | No | raw `.tar` conversion uses upstream checkpoint files only |
| PRISM 1.0 | `hftrainer.models.motion.prism` + shared Wan/SMPL components | No `ref_repo` runtime import | current Hub artifact is the raw iter15000 checkpoint package; full `save_pretrained` artifact pending |
| InterGen | `hftrainer.models.motion.intergen.network` native denoiser / diffusion / rotation utilities | No | checkpoint and InterHuman-262 stats are packaged in `checkpoints/intergen/hftrainer_interhuman` |
| InterMask | `hftrainer.models.motion.intermask.network` native RVQ-VAE + MaskTransformer sampling | No | InterHuman and InterX VQ/Transformer checkpoints are packaged in `checkpoints/intermask` |
| MotionLCM | `hftrainer.models.motion.motionlcm.network` native MLD VAE + LCM denoiser | No | artifact published as `ZeyuLing/hftrainer-motionlcm-humanml3d` |
| MotionStreamer | `hftrainer.models.motion.motionstreamer.network` native TAE + AR + diffusion head | No | raw checkpoints must be passed explicitly outside artifact inference |
| Go to Zero / MotionMillion | `hftrainer.models.motion.motionmillion.network` native FSQ VAE + LLaMA AR | No | text encoder is packaged in the hftrainer artifact |
| HY-Motion T2M 1.0 | `hftrainer.models.motion.hymotion_t2m` + shared HunyuanMotion MMDiT modules | No | shared component shim preserves HYMotion M2M compatibility; text encoders are packaged in artifacts |
| KIMODO | `hftrainer.models.motion.kimodo.network` native model / motion representation / skeleton runtime | No | four checkpoint variants pass seeded parity against the previous package path |
| ProtoMotions G1 Tracker | packaged ProtoMotions evaluation adapter and released G1 tracker bundle | No for canonical reward, eval, and viewer paths | training/eval still requires the IsaacGym-compatible Python environment |
| Any2Track G1 Tracker | bundled ONNX checkpoint, config, and MuJoCo G1 assets | No | original release name in code is OpenTrack; paper/table name should be Any2Track |
| Humanoid-GPT G1 Tracker | packaged worker adapter and released G1 ONNX checkpoint | No | canonical use is through the long-lived py3.11 worker |

## Model Card Synchronization

`docs/model_zoo/<model>.md` is the source of truth for both repository docs and
Hugging Face model cards. Use the sync helper after editing any card:

```bash
# Check local artifact README files and configured Hub cards.
python3 tools/sync_model_zoo_cards.py --check --remote

# Rewrite README.md inside local artifact directories.
python3 tools/sync_model_zoo_cards.py --write-local

# Upload README.md to configured Hugging Face model repos.
python3 tools/sync_model_zoo_cards.py --push
```

The push command requires an authenticated Hugging Face token with write access.
Private artifacts such as KIMODO are skipped by remote checks unless the token
can read them.

## Research Stacks In This Repository

These modules are important repository entry points, but they are not all
published as model-zoo cards yet.

| Stack | Scope | Entry point | Notes |
|---|---|---|---|
| PRISM | audio/text motion generation | `PrismBundle`, `PrismPipeline` | PRISM 1.0 checkpoint card published; full self-contained artifact pending |
| PRISM-MCM | motion control / editing | `PrismMCMBundle`, `PrismMCMPipeline` | research/training stack |
| VerMo | VQ/AR motion generation components | `VermoBundle`, `VermoPipeline` | includes VQ-VAE, LLaMA/Qwen, processor |
| HY-Motion-M2M | motion-to-motion editing and control | `HyMotionM2MBundle`, `HyMotionM2MPipeline` | paper-eval scripts under `scripts/eval` |
| HY-Motion-V2M | video-to-motion (single tracked person) | `HyMotionV2MBundle`, `HyMotionV2MPipeline` | stage-1 feature→motion verified (`from_pretrained` parity = 0); body-only + with-hand; stage-2 `infer_v2m` (video→motion) implemented, runtime-blocked on gated SAM-3D-Body weights; card [hymotion_v2m.md](hymotion_v2m.md) |
| HY-Motion-UMO | temporal fusion on HunyuanMotion MMDiT | `HyMotionUMOBundle`, `HyMotionUMOPipeline` | research stack |
| MotionCLIP | motion-text contrastive encoder | `MotionCLIPBundle` | evaluator / retrieval utility |
| PhysFlow | KIMODO-G1 + physics reward tooling | `PhysFlowBundle`, `PhysFlowG1Bundle` | embodied-motion research stack |

## Supported Task Surface

| Task family | Model-zoo coverage | Evaluation / conversion |
|---|---|---|
| Text-to-motion | MDM, T2M-GPT, MoMask, MotionLCM, MotionStreamer, Go to Zero, HY-Motion T2M | HumanML3D-263 and/or MotionStreamer-272 evaluators |
| Two-person / interaction T2M | InterGen, InterMask | InterHuman-262 / InterCLIP evaluator; InterX official HHI text-mot-match evaluator |
| Streaming / autoregressive T2M | MotionStreamer, Go to Zero, T2M-GPT | native model cards plus MS272 evaluator where applicable |
| Text + kinematic control | KIMODO | SOMA/G1/SMPL-X runtime; SMPL mesh bridge via `hftrainer.motion.retarget` |
| Motion-to-motion / editing | HY-Motion-M2M, PRISM-MCM, VerMo/PRISM research scripts | paper-specific evaluator scripts |
| Video-to-motion | HY-Motion-V2M (body-only + with-hand) | stage-1 feature→motion + `from_pretrained`; SMPL-H mesh viewer; end-to-end `infer_v2m` (ffmpeg+YOLOX+ByteTrack+SAM-3D-Body) implemented, needs gated SAM-3D-Body weights to run |
| Retargeting / embodiment | SMPL <-> SOMA, KIMODO/SOMA -> SMPL, SMPL -> Unitree G1 | `hftrainer.motion.retarget` |
| Physical motion tracking | ProtoMotions, Any2Track, Humanoid-GPT G1 trackers | `docs/model_zoo/protomotions_g1_tracker.md`, `docs/model_zoo/any2track_g1_tracker.md`, `docs/model_zoo/humanoid_gpt_g1_tracker.md`, and `docs/physflow/tracker_baselines.md` |

## Evaluator Contract

Every new text-to-motion card should document:

1. Native representation and FPS.
2. Prediction directory layout.
3. Exact evaluator command.
4. Metric JSON path.
5. GT/real row used for sanity checking.
6. Whether any cross-representation conversion was used.

See [`../motion/evaluators.md`](../motion/evaluators.md) for the shared
protocol and failure checklist.

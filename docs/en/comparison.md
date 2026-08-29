# Comparison

HFTrainer combines config-driven experiment management with repository-owned
model execution. Its defining difference is code ownership, not a new wrapper
around whatever model package happens to be installed.

| Capability | Script-oriented project | External model-package wrapper | HFTrainer |
| --- | --- | --- | --- |
| Config-driven construction | project-specific | varies | MMEngine config + local registries |
| Distributed runtime | project-specific | usually framework-specific | Accelerate |
| Model math ownership | copied or implicit | delegated to installed package | local `hftrainer/models/<implementation>/network` |
| Training/inference reuse | often duplicated | depends on wrapper | shared `ModelBundle` atomic operations |
| Artifact schema | ad hoc | controlled by package version | owned and validated per implementation |
| LoRA | ad hoc or delegated | adapter-package dependent | local `LoRALinear` implementation |
| Component resolution | imports and dynamic paths | package class lookup | local `MODEL_COMPONENTS` only |
| Dependency drift | manual | model behavior can change with package version | forbidden model-package imports tested |
| Multi-optimizer training | project-specific | often limited | runner/trainer protocol |

General-purpose infrastructure dependencies remain intentional: PyTorch owns
tensor kernels, Accelerate owns distributed orchestration, MMEngine owns config
and registry primitives, and safetensors/NumPy/Pillow support artifacts and
data. They do not supply HFTrainer's model definitions or task algorithms.

## Current validation boundary

- ViT, LLaMA, SD1.5, DMD, StyleGAN2, and Wan have reduced local model tests;
  individual implementation pages/configs describe checkpoint limitations.
- LTX model/trainer/pipeline source is packaged locally from one pinned,
  modified snapshot and remains under its own license. Contract tests and a
  tiny local Gemma path pass, but the repository test environment has not run
  the gated 22B workflow end to end.
- Reference implementations demonstrate framework structure and do not by
  themselves claim benchmark reproduction.

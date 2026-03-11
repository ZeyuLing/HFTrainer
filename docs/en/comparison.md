# Comparison

HF-Trainer sits between MMEngine-style experiment management and HuggingFace-native runtime components.

| Capability | MMEngine-style stack | HF Trainer | HF-Trainer |
| --- | --- | --- | --- |
| Config-driven experiments | strong | limited | strong |
| Native `accelerate` runtime | no | yes | yes |
| Reuse train/infer task logic | weak | weak | strong via `ModelBundle` |
| Per-module freeze/save control | ad hoc | limited | config-driven |
| Multi-optimizer hooks | possible but verbose | limited | supported in runner/trainer API |
| Direct diffusers / transformers usage | often wrapped | transformers-centric | direct |

## Where HF-Trainer Is Still Incomplete

- GAN and DMD now have runnable reference projects, but they are framework-oriented reference implementations rather than benchmark-tuned reproductions.
- Public docs now live under `docs/en/` and `docs/zh-cn/`.

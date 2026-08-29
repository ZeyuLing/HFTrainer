# Task Matrix

## Runnable Today

| Task | Bundle | Trainer | Pipeline | Example Config | Validation status |
| --- | --- | --- | --- | --- | --- |
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | `configs/classification/vit_base_demo.py` | built-in smoke path |
| Text-to-image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | `configs/text2image/sd15_demo.py` | built-in smoke path |
| Causal LM SFT | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_sft_demo.py` | built-in smoke path |
| Causal LM LoRA | `CausalLMBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llm/llama_lora_demo.py` | built-in smoke path |
| Text-to-video | `WanBundle` | `WanTrainer` | `WanPipeline` | `configs/text2video/wan_demo.py` | built-in smoke path |
| LTX-2.5 distilled audio-video inference | `LTXVideoBundle` | n/a | `LTXVideoPipeline` | `configs/ltx_video/infer_ltx_video_2_5_distilled.py` | registry/config + mocked official API contract; no 22B GPU run |
| LTX-2.5 LoRA training + dev inference | `LTXVideoBundle` | `LTXVideoTrainer` (managed) | `LTXVideoPipeline` | `configs/ltx_video/train_ltx_video_2_5_lora.py` | managed trainer/config + mocked official API contract; no 22B GPU run |
| GAN | `StyleGAN2Bundle` | `GANTrainer` | `StyleGAN2Pipeline` | `configs/gan/gan_demo.py` | framework reference |
| DMD one-step distillation | `DMDBundle` | `DMDTrainer` | `DMDPipeline` | `configs/distillation/dmd_demo.py` | framework reference |

`built-in smoke path` means the repository has a reduced startup path for the
task. `framework reference` means the implementation demonstrates integration
structure and does not claim benchmark reproduction. LTX's lightweight tests
validate the pinned API/config contract without allocating the gated 22B
weights; see the [LTX-Video 2.5 guide](models/ltx_video_2_5.md).

## Validation Output Convention

- Classification: `preds`, `scores`, `gts`, optional `metas`
- Text-to-image: `preds`, `prompts`, optional `gts`
- Text-to-video: `preds`, `prompts`
- LLM: `preds`, `gts`, `input_prompts`, optional `loss_lm`

The LTX pipeline returns an output dictionary containing video/audio objects
or an encoded output path, frame count, frame rate, dimensions, seed, mode, and
the resolved tiling configuration. Encoding consumes the video iterator, so a
saved result reports `video=None` rather than exposing an already-consumed
iterator.

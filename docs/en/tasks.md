# Task Matrix

## Runnable Today

| Task | Bundle | Trainer | Pipeline | Example Config | Validation status |
| --- | --- | --- | --- | --- | --- |
| Classification | `ViTBundle` | `ClassificationTrainer` | `ClassificationPipeline` | `configs/vit/vit_base_demo.py` | built-in smoke path |
| Text-to-image | `SD15Bundle` | `SD15Trainer` | `SD15Pipeline` | `configs/sd15/sd15_demo.py` | built-in smoke path |
| Causal LM SFT | `LlamaBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llama/llama_sft_demo.py` | built-in smoke path |
| Causal LM LoRA | `LlamaBundle` | `CausalLMTrainer` | `CausalLMPipeline` | `configs/llama/llama_lora_demo.py` | built-in smoke path |
| Text-to-video | `WanBundle` | `WanTrainer` | `WanPipeline` | `configs/wan/wan_demo.py` | built-in smoke path |
| LTX-2.5 distilled audio-video inference | `LTXVideoBundle` | n/a | `LTXVideoPipeline` | `configs/ltx_video/infer_ltx_video_2_5_distilled.py` | local registry/config/checkpoint contracts; no 22B GPU run |
| LTX-2.5 LoRA training + dev inference | `LTXVideoBundle` | `LTXVideoTrainer` (managed local loop) | `LTXVideoPipeline` | `configs/ltx_video/train_ltx_video_2_5_lora.py` | packaged trainer/preprocess contracts and tiny Gemma path; no 22B GPU run |
| MiniMax-H3 synchronized A/V inference | `MiniMaxH3Bundle` | n/a | `MiniMaxH3Pipeline` | `configs/minimax_h3/infer_h3_base_fl2va.py`, `infer_h3_base_ref2va.py` | local tiny/reference contracts; no full 33B+32B GPU run |
| MiniMax-H3 experimental LoRA | `MiniMaxH3Bundle` | `MiniMaxH3Trainer` | `MiniMaxH3Pipeline` | `configs/minimax_h3/train_h3_base_lora.py`, `infer_h3_base_fl2va_lora.py` | cached-feature loss/backward, checkpoint-to-inference merge, and artifact contracts; no official-recipe claim |
| GAN | `StyleGAN2Bundle` | `StyleGAN2Trainer` | `StyleGAN2Pipeline` | `configs/stylegan2/stylegan2_demo.py` | framework reference |
| DMD one-step distillation | `DMDBundle` | `DMDTrainer` | `DMDPipeline` | `configs/dmd/dmd_demo.py` | framework reference |

`built-in smoke path` means the repository has a reduced startup path for the
task. `framework reference` means the implementation demonstrates integration
structure and does not claim benchmark reproduction. LTX's model, trainer, and
pipeline source is packaged locally. Lightweight tests validate its local
API/config/checkpoint boundary without allocating the gated 22B weights; see
the [LTX-Video 2.5 guide](models/ltx_video_2_5.md).
MiniMax-H3's released 768p stack and its stricter validation boundary are
documented in the [MiniMax-H3 guide](models/minimax_h3.md).

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

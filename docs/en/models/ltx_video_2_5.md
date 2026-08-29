# LTX-Video 2.5

HFTrainer contains a modified, pinned LTX-2 source snapshot reorganized into
the framework's local layers:

```text
hftrainer/models/ltx_video/network/             model math and loaders
hftrainer/pipelines/ltx_video/backend/           inference backend
hftrainer/trainers/ltx_video/native/             training implementation
hftrainer/trainers/ltx_video/preprocess_scripts/ preprocessing
```

No separately installed `ltx-core`, `ltx-pipelines`, or `ltx-trainer` package,
and no second source checkout, supplies executable code at runtime. Internal
imports point to the HFTrainer namespaces. LTX's Gemma text path and LoRA
injection also use repository-local implementations.

## Source and license

- Source: https://github.com/Lightricks/LTX-2
- Pinned revision: `400fd31054597515f47125691032c04b1c3ee24e`
- Modification record: [UPSTREAM.md](https://github.com/ZeyuLing/HFTrainer/blob/main/hftrainer/models/ltx_video/UPSTREAM.md)
- Complete license: [LTX-2.x Community License Agreement](https://github.com/ZeyuLing/HFTrainer/blob/main/hftrainer/models/ltx_video/LICENSE.ltx-2.x)
- Repository notices: [THIRD_PARTY_NOTICES.md](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md)

The LTX license is not an Apache/MIT-style permissive license. It contains use
restrictions, redistribution obligations, modified-file notice requirements,
and commercial-license conditions. Read the complete agreement before use or
redistribution. Modified Python files carry the required notice.

## Supported surfaces and validation boundary

| Surface | Entry point | Validation in this repository |
| --- | --- | --- |
| Distilled text/image-conditioned audio-video generation | `LTXVideoBundle` + `LTXVideoPipeline` | config, checkpoint roles, shape constraints, backend construction and call mapping |
| Dev two-stage generation with official and user LoRAs | same, `mode='dev_two_stage'` | separation/order of both LoRA sources, guidance arguments, call contract |
| LoRA training | `LTXVideoTrainer` | local config parsing, local LoRA wiring, checkpoint/resume mapping, managed-runner dispatch |
| Preprocessing | `hftrainer-ltx-preprocess` | packaged script discovery and full argv/role validation |
| Local Gemma text path | `hftrainer/models/ltx_video/network/text_encoders/gemma` | tiny forward/backward, tokenizer/processor, hidden-state contract, checkpoint names |
| Real 22B generation/training | same public interfaces | **not run in this development machine** |

The final row matters: the test suite does not claim model quality, throughput,
training convergence, or a successful full 22B CUDA allocation.

Image conditioning for video generation is supported by the LTX pipeline.
Image-conditioned *Gemma prompt enhancement* is a different optional feature;
it currently raises a clear error because the Gemma vision tower has not yet
been localized. Text-only prompt enhancement and normal text encoding use the
local Gemma runtime.

## Runtime

Install one of the local-source extras after selecting a suitable CUDA PyTorch
wheel for the machine:

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"
python -m pip install -e ".[ltx-video]"  # both

# Only when those features are enabled:
python -m pip install -e ".[ltx-video-integrations]"  # W&B / Hub publication
python -m pip install -e ".[ltx-video-hdr]"           # EXR / HDR media
```

The extras contain supporting media/scientific libraries, not external LTX or
model-framework packages. The local runtime checks required PyTorch compiler
capabilities and reports them before constructing the 22B graph. Use Linux and
NVIDIA CUDA for the advertised training path, and plan GPU memory based on the
actual resolution, frame count, audio, validation, precision, and offload
settings.

## Checkpoint pack

After accepting the gated model terms, arrange the split pack under one root:

```bash
export LTX25_CHECKPOINT_ROOT="$PWD/checkpoints/LTX-2.5"
```

The supplied configs expect these roles:

| Component | Role |
| --- | --- |
| distilled transformer | fixed-schedule fast generation |
| dev transformer | LoRA training and guided two-stage generation |
| packed Gemma 4 encoder with LTX projection | prompt features; stock Gemma is not interchangeable |
| video VAE | video latent encode/decode |
| audio VAE/vocoder | synchronized audio latent encode/decode |
| spatial upsampler | second-stage latent upsampling |
| duration head | optional prompt-conditioned duration |
| official distilled LoRA | required transition in dev two-stage mode |
| user LoRA | adaptation produced by training |

Checkpoint-role validation runs before the expensive model load. Comfy-specific
INT8-convrot artifacts are rejected by the native local path.

## Distilled inference

```bash
hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

Use `--image frame.jpg` for a conditioning frame and `--auto-duration` when a
duration-head checkpoint is configured. The distilled schedule is fixed;
negative prompts and a user-selected step count are rejected instead of being
silently ignored.

The current two-stage output contract requires height and width divisible by
64 and `num_frames % 8 == 1`.

## Preprocessing

The packaged script accepts CSV, JSON, or JSONL manifests. A minimal JSON item:

```json
{
  "caption": "A handheld shot follows a cyclist through a quiet alley.",
  "video": "videos/cyclist.mp4"
}
```

Compute latents and text features:

```bash
hftrainer-ltx-preprocess data/ltx_video_2_5/dataset.jsonl \
  --resolution-buckets 960x544x49 \
  --model-path "$LTX25_CHECKPOINT_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX25_CHECKPOINT_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --output-dir data/ltx_video_2_5/.precomputed
```

No `--ltx-repo` argument exists. Use `--skip-audio` only when the training
strategy also disables generated audio. Re-run with `--overwrite` after
changing checkpoint roles, buckets, or the text encoder.

## LoRA training

```bash
export LTX25_PREPROCESSED_DATA="$PWD/data/ltx_video_2_5/.precomputed"
export HFTRAINER_WORK_DIR="$PWD/outputs/training/ltx_video_2_5_lora"

hftrainer-train configs/ltx_video/train_ltx_video_2_5_lora.py
```

`LTXVideoTrainer` is a managed trainer because the LTX algorithm has a tightly
coupled data/cache/validation/checkpoint lifecycle. “Managed” does not mean
external delegation: the implementation is packaged at
`hftrainer.trainers.ltx_video.native`, uses HFTrainer's local LoRA, and is
selected through the normal runner builder.

The example is a starting recipe, not a benchmark-optimal configuration.

## Dev inference with a trained LoRA

```bash
export LTX25_USER_LORA="$PWD/outputs/training/ltx_video_2_5_lora/checkpoints/<saved-lora>.safetensors"

hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_dev_lora.py \
  --prompt "A slow dolly shot moves through an art studio while rain taps the windows." \
  --negative-prompt "blurry, distorted, low quality, artifacts" \
  --num-inference-steps 30 \
  --output outputs/ltx_video_2_5/dev_lora.mp4
```

This mode uses the dev transformer, the required official distilled LoRA, and
the user LoRA as three distinct roles. It supports guidance, a negative prompt,
and a configurable step count.

## Troubleshooting

- **Missing supporting library:** install the matching LTX extra; an error
  should name the missing media/runtime dependency.
- **Forbidden model-package import:** this is a source bug. Do not install the
  missing model package as a workaround; report the import path.
- **Checkpoint-role error:** verify dev versus distilled transformer and the
  projected Gemma/LTX encoder filename.
- **CUDA OOM:** reduce buckets/frames/validation, use gradient checkpointing,
  and choose an appropriate precision/offload strategy.
- **Contract tests pass but generation fails:** report the GPU, driver,
  CUDA/PyTorch versions, exact HFTrainer commit, command, checkpoint filenames,
  and first stack trace. Contract tests do not allocate the full model.

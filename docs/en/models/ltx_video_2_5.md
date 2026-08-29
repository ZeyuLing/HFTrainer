# LTX-Video 2.5

HFTrainer integrates LTX-Video 2.5 through a thin, config-driven adapter over
Lightricks' native `ltx-core`, `ltx-pipelines`, and `ltx-trainer` packages. The
official packages continue to own model loading, denoising, preprocessing,
optimization, checkpointing, and audio/video encoding. HFTrainer adds one
registry/config surface, strict split-checkpoint validation, and reproducible
entry points.

The adapter is pinned to Lightricks/LTX-2 commit
`400fd31054597515f47125691032c04b1c3ee24e`. Do not independently upgrade one
of the three LTX packages: their internal APIs move together.

Official references:

- [LTX-2 source](https://github.com/Lightricks/LTX-2)
- [LTX-2.5 model and checkpoints](https://huggingface.co/Lightricks/LTX-2.5)
- [LTX Trainer quick start](https://docs.ltx.io/open-source-model/ltx-trainer/quick-start)
- [LTX-2.5 license](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/LICENSE)

## Support and validation status

| Surface | HFTrainer support | Validation performed in this repository |
| --- | --- | --- |
| Distilled text/image-to-audio-video inference | `LTXVideoBundle` + `LTXVideoPipeline` | config parsing, checkpoint-role checks, registry construction, argument mapping, and mocked official API contract |
| Dev two-stage inference with LoRA | same bundle/pipeline with `mode='dev_two_stage'` | config and official API contract; separate official distilled LoRA and user LoRA are enforced |
| Text-to-audio-video LoRA training | `LTXVideoTrainer` managed trainer | official Pydantic config mapping, preprocessing command construction, checkpoint-role checks, and managed-runner dispatch |
| Full 22B GPU generation/training | executable when weights and supported hardware are provided | **not run in the repository validation environment** |

The final row is intentional. Passing the lightweight tests proves that the
integration matches the pinned Python interfaces; it is not evidence of model
quality, throughput, convergence, or a successful 22B CUDA run.

Inside the compatible official environment, the optional real-source contract
can validate the sample with the actual Pydantic class and public signatures:

```bash
HFTRAINER_LTX_SOURCE_ROOT=third_party/LTX-2 \
  python -m pytest -m upstream \
  tests/integration/test_ltx_video_official_contract.py
```

## Environment and capacity planning

- HFTrainer core requires Python 3.10 or newer. The pinned LTX package metadata
  also accepts Python 3.10+. The model card still mentions Python 3.12+,
  CUDA 12.7+, and a PyTorch 2.7-class runtime, but the pinned source now imports
  `torch.compiler.nested_compile_region` at module load time. That API is absent
  from PyTorch 2.7.x, so HFTrainer enforces a capability floor equivalent to
  PyTorch 2.8+ and reports a clear preflight error instead of the upstream
  `AttributeError`.
- Use Linux with NVIDIA CUDA for training. The official trainer uses a
  CUDA/Triton-oriented stack, and `LTXVideoTrainer` rejects non-Linux training
  by default. The upstream inference stack has fallback paths on other systems,
  but HFTrainer does not advertise them as its validated production path.
- The official trainer recommends an 80 GB or larger GPU for its standard
  recipe. Its low-VRAM recipe targets 32 GB GPUs using INT8 quantization,
  8-bit optimizer state, lower LoRA rank, and gradient checkpointing. Treat
  these as planning references, not universal guarantees: resolution, frame
  count, audio, validation, decoder choice, and software versions all change
  peak memory.
- The official distilled inference download shown below is roughly 66 GiB.
  GPU memory is separate from disk space. Use the official FP8/offload options
  only after checking the pinned upstream documentation and output quality.

## Install a compatible runtime

The recommended production path is to let the pinned official checkout select
its mutually compatible PyTorch/CUDA packages, then add HFTrainer without
letting pip replace that runtime. From the HFTrainer repository root on Linux:

```bash
git clone https://github.com/Lightricks/LTX-2.git third_party/LTX-2
git -C third_party/LTX-2 checkout 400fd31054597515f47125691032c04b1c3ee24e

cd third_party/LTX-2
uv sync
uv pip install --python .venv/bin/python --no-deps -e ../..
uv pip install --python .venv/bin/python "mmengine>=0.7,<1" "PyYAML>=6"
cd ../..
```

This uses the upstream runtime/index decisions and keeps them isolated from
other HFTrainer projects. The `--no-deps` install is intentional: the official
environment already owns PyTorch, Accelerate, Transformers, and the LTX
packages; HFTrainer only needs its lightweight config dependencies added.

If you have already installed a PyTorch/CUDA stack that exposes
`torch.compiler.nested_compile_region`, the HFTrainer extras provide a shorter
convenience path. Install only the surface you need:

```bash
python -m pip install -e ".[ltx-video-inference]"
python -m pip install -e ".[ltx-video-train]"

# Both inference and training
python -m pip install -e ".[ltx-video]"
```

These extras install all LTX packages from the same pinned Git commit. They do
not resolve to the older, incompatible package line currently available from
PyPI. They require `torch>=2.8`, but a Python package extra cannot choose the
correct CUDA wheel/index for every machine. Do not use the convenience command
as a substitute for CUDA runtime planning.

## Accept the license and download gated weights

LTX-2.5 is a gated Hugging Face repository. Visit the
[model page](https://huggingface.co/Lightricks/LTX-2.5), review and accept its
access terms, then authenticate a token that can read gated repositories:

```bash
hf auth login
```

The model uses the **LTX-2.x Community License**, not an Apache/MIT-style open
source license. The official model card currently describes no-cost commercial
and production use for entities below USD 10 million in annual revenue and a
paid agreement above that threshold; transfer of fine-tunes may also require a
paid license. Read the binding license itself before commercial use or
distribution because the short model-card summary is not the legal text.

Set one shared checkpoint root:

```bash
export LTX25_CHECKPOINT_ROOT="$PWD/checkpoints/LTX-2.5"
```

For distilled inference, download the split components expected by the example
config:

```bash
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir "$LTX25_CHECKPOINT_ROOT"
```

For training and guided dev+LoRA inference, also download:

```bash
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --local-dir "$LTX25_CHECKPOINT_ROOT"
```

If `hf download` returns 401/403, confirm both model-page acceptance and the
token's gated-repository read permission.

## What each checkpoint does

| Component | Role | Important constraint |
| --- | --- | --- |
| distilled transformer | fast inference base | fixed official distilled schedule; not a training base |
| dev transformer | trainable full DiT | required for LoRA/full training and guided two-stage inference |
| packed Gemma 4 text encoder | text features plus LTX projection | stock Google Gemma 4 is not interchangeable |
| video VAE | encode/decode video latents | the example uses the higher-quality DiffVAE BF16 artifact |
| audio VAE/vocoder | encode/decode synchronized audio | required by the supplied joint audio-video recipe |
| spatial upsampler | second-stage 2x latent upsampling | required by both supplied inference modes |
| duration head | optional prompt-conditioned duration | the example config includes it and therefore expects the file |
| official distilled LoRA | stage transition for dev two-stage inference | required in addition to, not instead of, a user-trained LoRA |
| user LoRA | task/style adaptation produced by training | pass the exact saved `.safetensors` path through `LTX25_USER_LORA` |

Do not pass `*-comfy-int8-convrot.safetensors` to these native PyTorch
pipelines; those artifacts are for ComfyUI. HFTrainer rejects them before a
22B load begins.

## Fast distilled inference

The distilled route uses the official `DistilledPipeline`. Its denoising
schedule is fixed: HFTrainer deliberately rejects `--num-inference-steps` and
a negative prompt for this mode because the checkpoint uses CFG=1 and the
official predefined sigmas.

```bash
hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

Equivalent source-tree entry point:

```bash
python tools/infer_ltx_video.py \
  configs/ltx_video/infer_ltx_video_2_5_distilled.py \
  --prompt "A paper boat drifts through a rain-filled street at dusk." \
  --output outputs/ltx_video_2_5/distilled.mp4
```

Add `--image path/to/frame.jpg` for first-frame conditioning. The CLI also
accepts `--auto-duration` when the duration-head checkpoint is configured; it
is mutually exclusive with `--num-frames`.

The HFTrainer two-stage adapter currently requires height and width divisible
by 64 and a frame count satisfying `num_frames % 8 == 1`; the example is
768x512 and 121
frames. The official base model/VAE contract is divisible by 32; HFTrainer's
current two-stage final-output path intentionally applies the stricter 64
alignment before loading model weights.

## Prepare a training dataset

The official preprocessor consumes CSV, JSON, or JSONL with `caption` and
`video` fields. For example:

```json
[
  {
    "caption": "A handheld shot follows a cyclist through a quiet alley.",
    "video": "videos/cyclist.mp4"
  }
]
```

Preprocessing is implemented by the pinned official source script. Keep an
exact source checkout because the command-line script is not exposed as a
stable console entry point by the package:

```bash
git clone https://github.com/Lightricks/LTX-2.git third_party/LTX-2
git -C third_party/LTX-2 checkout 400fd31054597515f47125691032c04b1c3ee24e
```

Compute video/audio latents and Gemma features:

```bash
hftrainer-ltx-preprocess data/ltx_video_2_5/dataset.json \
  --ltx-repo third_party/LTX-2 \
  --resolution-buckets 960x544x49 \
  --model-path "$LTX25_CHECKPOINT_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --text-encoder-path "$LTX25_CHECKPOINT_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$LTX25_CHECKPOINT_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --output-dir data/ltx_video_2_5/.precomputed
```

Use `--skip-audio` only when the training strategy also stops generating
audio. Re-run with `--overwrite` after changing checkpoint versions,
resolution buckets, or text encoders; cached LTX-2.3 and LTX-2.5 text features
are not interchangeable.

## Train a LoRA

The example config contains the official `LtxTrainerConfig` schema under
`trainer.native_config`. Set the checkpoint and data roots, then launch it
through the normal HFTrainer command:

```bash
export LTX25_PREPROCESSED_DATA="$PWD/data/ltx_video_2_5/.precomputed"
export HFTRAINER_WORK_DIR="$PWD/outputs/training/ltx_video_2_5_lora"

hftrainer-train configs/ltx_video/train_ltx_video_2_5_lora.py
```

For a distributed launch from the source tree:

```bash
accelerate launch tools/train.py configs/ltx_video/train_ltx_video_2_5_lora.py
```

`LTXVideoTrainer` is a managed trainer: it does not translate the official
algorithm into HFTrainer `train_step` calls. It validates and snapshots the
resolved config, then lets `LtxvTrainer` own Accelerator, optimizer,
checkpoint, validation, and resume semantics. This avoids maintaining a
second training implementation that could drift from upstream.

The supplied config is a starting recipe, not a claim of benchmark-optimal
hyperparameters. For a 32 GB class GPU, port the memory controls from the
official
[low-VRAM config](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-trainer/configs/t2v_lora_low_vram.yaml)
and validate quality for your data.

## Inference with the dev model and a trained LoRA

Set `LTX25_USER_LORA` to the exact LoRA checkpoint produced by the trainer:

```bash
export LTX25_USER_LORA="$PWD/outputs/training/ltx_video_2_5_lora/checkpoints/<saved-lora>.safetensors"

hftrainer-ltx-infer \
  configs/ltx_video/infer_ltx_video_2_5_dev_lora.py \
  --prompt "A slow dolly shot moves through an art studio while rain taps the windows." \
  --negative-prompt "blurry, distorted, low quality, artifacts" \
  --num-inference-steps 30 \
  --output outputs/ltx_video_2_5/dev_lora.mp4
```

This route uses the trainable dev transformer, the official distilled LoRA for
the two-stage workflow, and the user LoRA. Unlike the distilled transformer
route, it supports guidance, a negative prompt, and a configurable step count.
Do not try to load the user LoRA on the distilled-transformer config merely to
make inference faster; the two modes have different contracts.

## Config registration and lightweight imports

LTX dependencies are optional and loaded only when the LTX bundle/trainer
actually builds its native backend. Each config explicitly registers its
vertical slice:

```python
custom_imports = dict(
    imports=[
        'hftrainer.models.ltx_video',
        'hftrainer.pipelines.ltx_video',
    ],
    allow_failed_imports=False,
)
```

Training imports `hftrainer.trainers.ltx_video` instead. Importing `hftrainer`
by itself does not load Transformers, Diffusers, Accelerate, or LTX. Use
`hftrainer.register_all_modules()` only when an application genuinely needs
the complete built-in catalogue; configs should prefer focused
`custom_imports`.

## Troubleshooting

- **Missing `ltx_*` module:** install the matching extra. Use `ltx-video` if
  one environment handles both training and inference.
- **401/403 while downloading:** accept the gated model terms and authenticate
  a token with gated-repository read permission.
- **Checkpoint-role error:** check that training uses the `dev-transformer`,
  distilled inference uses the `distilled-transformer`, and the text encoder
  name includes `gemma4-12b-with-proj-ltx-2.5`.
- **CUDA out of memory:** reduce resolution/frame buckets, keep batch size 1,
  enable gradient checkpointing, and consult the official
  [trainer troubleshooting guide](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-trainer/docs/troubleshooting.md).
- **A config test passes but generation fails:** config and mocked-contract
  tests do not allocate the 22B weights. Record the GPU, driver, CUDA/PyTorch
  versions, exact commit, full command, and first stack trace when reporting a
  runtime issue.

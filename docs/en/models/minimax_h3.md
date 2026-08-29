# MiniMax-H3

HFTrainer contains a repository-owned implementation of the public
MiniMax-H3 Base 768p stack. It covers synchronized video/stereo-audio
generation from text, first/last frames, and ordered multimodal references.
The execution path does not import Diffusers, Transformers, Tokenizers, PEFT,
or a MiniMax checkout at runtime.

```text
hftrainer/models/minimax_h3/network/   transformer, Qwen3-VL, tokenizer,
                                       processor, video/audio VAEs, schedulers
hftrainer/models/minimax_h3/bundle.py  components, artifact boundary, atomic ops
hftrainer/pipelines/minimax_h3/        T2VA, FL2VA, and Ref2VA inference graph
hftrainer/trainers/minimax_h3/         experimental cached-feature RF objective
hftrainer/datasets/synchronized_audio_video/
                                       H3 cached-feature data contract
configs/minimax_h3/                    inference and LoRA-training recipes
```

## Scope

| Surface | Local status |
| --- | --- |
| T2VA | released `transformer/` partition |
| first-, last-, and first/last-frame to audio/video | released `transformer/` partition |
| ordered image/video/audio references | released `transformer_ref/` partition |
| Qwen3-VL-32B conditioner | local vision and text execution; hidden state 50 |
| visual/audio codecs | local 24-channel visual VAE and 32-channel, 32-kHz audio VAE |
| video/audio flow schedules | local shift-12 and shift-3 schedulers |
| fine-tuning | experimental cached-feature full/LoRA transformer training |
| H3-Context-IR, Regenerate-2K, hosted 2K stage | not publicly released locally; rejected as unsupported |
| sparse-attention kernel | not publicly released; local model uses full attention |

H3 is a joint rectified-flow audio/video model. It is not an autoregressive
language model and does not use classifier-free guidance or a negative prompt.
The transformer packs text/vision-conditioning, reference audio/video, and
target rows into one non-causal sequence.

## Source and license

- Official model repository: https://github.com/MiniMax-AI/MiniMax-H3
- Official checkpoint: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Frozen sources and modifications: `hftrainer/models/minimax_h3/UPSTREAM.md`
- Apache-2.0 reference-code license: `hftrainer/models/minimax_h3/LICENSE.apache-2.0`
- Complete model agreement: `hftrainer/models/minimax_h3/LICENSE.minimax-h3`
- Required model notice: `hftrainer/models/minimax_h3/NOTICE.minimax-h3`

MiniMax model materials are governed by the MiniMax H3 Community License
Agreement, not a permissive open-source license. It contains territorial
exclusions, use restrictions, redistribution requirements, and additional
commercial terms. Read and accept the complete agreement before downloading
or using the checkpoint. HFTrainer does not bundle weights, tokenizer assets,
or upstream configs.

The local code was adapted from pinned Apache-2.0 Diffusers and Transformers
references, with external framework machinery replaced by HFTrainer-owned
configuration, loading, preprocessing, and orchestration. See the upstream
record for exact commits.

## Install and download

Select a suitable CUDA PyTorch wheel, install HFTrainer and the media/download
helpers, then download the exact tested revision:

```bash
python -m pip install -e ".[minimax-h3]"

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "model_index.json" "modular_model_index.json" \
    "processor/*" "tokenizer/*" "text_encoder/*" \
    "vae/*" "audio_vae/*" "scheduler/*" "audio_scheduler/*" \
    "transformer/*" "transformer_ref/*" \
  --local-dir checkpoints/MiniMax-H3

export MINIMAX_H3_ROOT="$PWD/checkpoints/MiniMax-H3"
```

The include list downloads only the shared Diffusers-format components used by
HFTrainer and both local transformer partitions. It deliberately excludes the
additional original-format `FL2VA/` and `Ref2VA/` trees hosted in the same
repository, avoiding a second copy of the large checkpoints.

The checkpoint is very large: the two mutually exclusive transformer
partitions are each roughly 66 GB, the Qwen3-VL conditioner roughly 67 GB,
and the codecs add roughly 11 GB. Plan host RAM, accelerator memory, and disk
space before loading it. The local loaders reject missing shards, duplicate
keys, shape mismatches, and low-coverage partial loads.

The inference configs support explicit component placement because the full
transformer, conditioner, and codecs do not fit on a typical single GPU:

```bash
export MINIMAX_H3_TRANSFORMER_DEVICE=cuda:0
export MINIMAX_H3_CONDITIONER_DEVICE=cuda:1
export MINIMAX_H3_CODEC_DEVICE=cuda:2
```

`MINIMAX_H3_LOAD_DEVICE` remains the common fallback. Do not also pass the
generic CLI `--device` when using split placement: that flag moves the complete
bundle to one device. The local public loader deliberately rejects
`device_map` instead of silently pretending to perform layer-wise sharding.
Use the four component placement controls above; layer-level dispatch is not
part of this integration.

## Inference

Text-to-audio/video uses the FL2VA partition without keyframes:

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va.py \
  --prompt "A paper boat drifts down a narrow stream; water and birds are audible." \
  --mode t2va \
  --duration 5 \
  --output outputs/minimax_h3/t2va.mp4
```

First/last-frame conditioning uses the same partition:

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va.py \
  --prompt "The camera slowly circles the subject while wind moves the leaves." \
  --mode fl2va \
  --first-frame assets/start.png \
  --last-frame assets/end.png \
  --output outputs/minimax_h3/fl2va.mp4
```

Reference conditioning uses the separate Ref2VA weights. Reference order is
semantic and is preserved by the CLI:

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_ref2va.py \
  --prompt "Use Picture 1 for the character and Video 1 for motion and camera." \
  --mode ref2va \
  --reference-image assets/character.png \
  --reference-video assets/motion.mp4 \
  --duration 5 \
  --output outputs/minimax_h3/ref2va.mp4
```

The generated video is fixed at 24 FPS and audio at 32 kHz stereo. Generation
duration is 5–15 seconds in the current executable public path. Frame counts
are rounded upward to the visual VAE's legal `17*n+5` form. Height and width
must both be multiples of 32; when omitted, the released 768-short-edge canvas
rule is used. Ref2VA accepts at most 9 images, 3 videos, 3 audio clips, and 12
references total; audio-only reference lists are invalid. A Ref2VA Python call
must pass either `duration` or `num_frames`, because silently using the
124-frame T2VA/FL2VA default could truncate a reference soundtrack.

The Python pipeline supports `output_type="pt"`, `"np"`, `"pil"`, and
`"latent"`. Tensor video is `B,T,C,H,W`; NumPy video is `B,T,H,W,C`; PIL is a
nested batch/frame list. As in the frozen pipeline API, decoded audio remains
a float32 CPU tensor `B,2,S` for every non-latent video output type. It keeps
the codec's complete 800-sample-hop output rather than being truncated to a
fractional video-frame duration. HFTrainer intentionally defaults
`output_type` to `"pt"` so the CLI can mux tensors directly; the frozen
Diffusers modular pipeline defaults to `"pil"`.

For deterministic replay, `latents` accepts pre-generated video noise shaped
`[1,24,T,H,W]` and `audio_latents` accepts channel-major stereo noise shaped
`[2,32,A]`. Supplying one skips only that stream's random draw, preserving the
released condition → video → audio draw order for every draw that remains.
`attention_kwargs` is forwarded unchanged on every transformer evaluation.

## Experimental training

MiniMax released model weights but not a complete training recipe. Therefore
`MiniMaxH3Trainer` implements the data-ward rectified-flow objective implied by
the public scheduler:

```text
x_t = t*x_0 + (1-t)*noise
target = x_0 - noise
```

This is an explicit experimental objective, not a claim of official recipe or
convergence parity. The recommended data path caches frozen features so the
32B conditioner and both VAEs are absent while the 33B transformer trains.

Each JSONL manifest row points to one `.safetensors` or `.pt` file:

```json
{"feature_file":"000001.safetensors","keyframe_anchors":[],"reference_geometries":[]}
```

Each cache contains:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `video_latents` | `[24,T,H,W]` | normalized clean visual latents |
| `audio_latents` | `[2,32,L]` | normalized clean stereo audio latents |
| `prompt_embeds` | `[N,5120]` | local Qwen3-VL hidden state 50 |
| `text_token_tags` | `[N]` | packed text/vision modality tags |
| `condition_video_rows` | optional `[Nv,Dv]` | clean FL2VA/Ref2VA visual rows |
| `condition_audio_rows` | optional `[Na,32]` | clean Ref2VA audio rows |

Run the LoRA recipe after setting the cache root:

```bash
export MINIMAX_H3_CACHE_MANIFEST="$PWD/data/minimax_h3/train.jsonl"
export HFTRAINER_WORK_DIR="$PWD/outputs/training/minimax_h3_lora"
hftrainer-train configs/minimax_h3/train_h3_base_lora.py
```

Training checkpoints keep only the local adapter tensors and their HFTrainer
metadata. To load one for generation, use the matching adapter-aware inference
recipe and merge it after loading:

```bash
hftrainer-infer \
  --config configs/minimax_h3/infer_h3_base_fl2va_lora.py \
  --checkpoint outputs/training/minimax_h3_lora/checkpoint-iter_2000 \
  --merge-lora \
  --mode t2va \
  --prompt "A drummer performs on stage" \
  --duration 5 \
  --output outputs/minimax_h3/lora.mp4
```

`MiniMaxH3Bundle.save_pretrained(...)` exports a standalone artifact by
merging any active local LoRA modules into the transformer. Passing
`merge_lora=False` with active adapters is rejected instead of writing an
artifact that cannot be reloaded. Keep the original HFTrainer checkpoint for
resuming adapter training.

This is deliberately an HFTrainer bundle artifact, not a drop-in Diffusers
top-level artifact. Direct component `from_pretrained(...)` calls accept local
directories and explicit `device`/dtype placement only; Hub resolution,
`device_map`, checkpoint `variant`, and `use_safetensors` selection are
rejected rather than silently approximated. Component loading defaults to the
meta-device low-memory path. HFTrainer also enables gradient checkpointing in
the local audio VAE as a training extension; it does not change the published
audio-VAE checkpoint key schema.

Items in one minibatch must share the same packed row geometry. Bucket by
resolution, latent duration, prompt-presentation length, and reference layout,
or use `batch_size=1`.

## Verification boundary

Repository tests exercise tiny transformer, scheduler, VAE, Qwen/tokenizer,
layout, loss/backward, pipeline, cache, and artifact contracts. They also run
with external model packages actively blocked. Development-time comparisons
use frozen upstream reference implementations, which are not runtime
dependencies.

These checks do **not** demonstrate full-checkpoint generation quality,
throughput, convergence, or a successful 33B+32B allocation on this
development machine. A real release qualification should additionally run all
three public modes with the frozen full checkpoint on the target multi-GPU
system and visually/audibly inspect the muxed outputs.

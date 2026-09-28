# StyleGAN2

Pretrained FFHQ 1024 × 1024 image generation with local generator and discriminator training.

**Verification:** Official NVIDIA full-resolution output and native reload match exactly

**Runtime:** Repository-owned model, trainer and pipeline; infrastructure libraries and model assets remain dependencies.

[All models](index.md) · [Setup](#setup) · [Checkpoint](#checkpoint) · [Train](#train) · [Infer](#infer) · [Evidence](#evidence-and-loss)

## At a glance

| Item | Status |
| :--- | :--- |
| Implementation | Local to HFTrainer |
| Demo data | See [data availability](demo-data.md) |
| Convergence | Not established by the evidence on this card |

## Setup

Run commands from the repository root. Install the environment below, download the checkpoint, and prepare data before training.

```bash
python -m pip install -e .
python -m pip install "huggingface_hub>=0.34,<2"
```

The published artifact contains NVIDIA FFHQ `G_ema` and discriminator weights, converted to safetensors. Runtime loading does not execute pickle.

## Checkpoint

[ZeyuLing/HFTrainer-StyleGAN2-ADA-FFHQ-1024](https://huggingface.co/ZeyuLing/HFTrainer-StyleGAN2-ADA-FFHQ-1024) · [pinned revision](https://huggingface.co/ZeyuLing/HFTrainer-StyleGAN2-ADA-FFHQ-1024/tree/c8ba5e0da3c126780db4be4eab9afe86d1b99db9)

This release packages real pretrained weights in a complete **HFTrainer bundle**. Trainer and Pipeline share the repository-owned `from_pretrained` loader. Conversion, source attribution and validation scope are recorded below.

```bash
hf download ZeyuLing/HFTrainer-StyleGAN2-ADA-FFHQ-1024 \
  --revision c8ba5e0da3c126780db4be4eab9afe86d1b99db9 \
  --local-dir checkpoints/hftrainer/stylegan2
export HFTRAINER_CHECKPOINT="$PWD/checkpoints/hftrainer/stylegan2"
```

[Conversion and validation record](evidence/checkpoints/stylegan2.json). Base weights are separate from demo adapters and datasets. To export a fine-tuned bundle:

```bash
python tools/export_model.py --config configs/pretrained/train_stylegan2.py \
  --checkpoint PATH_TO_TRAINING_CHECKPOINT --output outputs/stylegan2_export
```

## Data

Images under `data/classification/demo/images`. The supplied pretrained recipe fine-tunes at 1024 × 1024 with batch size 1.

## Train

```bash
python tools/train.py configs/pretrained/train_stylegan2.py
```

## Infer

Run the published pretrained model:

```bash
python tools/infer.py --config configs/pretrained/infer_stylegan2.py --device cuda \
  --seed 42 --output outputs/stylegan2.png
```

For fine-tuned weights, pass `--checkpoint` with the training output and a matching training model config (including LoRA settings or a changed classification head).

## Evidence and loss

The native generator matches the official NVIDIA 1024 × 1024 FP32 output exactly with constant noise and seed 42. Export/reload also matches exactly. Tiny adversarial backward and regularization contracts are tested.

The new pretrained training entry point completed one optimizer step and saved a checkpoint on an H200. These synthetic execution checks are recorded with the checkpoint evidence; they do not establish useful fine-tuning.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The default config runs 10,000 iterations. Portable PyTorch reference operators favor compatibility over CUDA-kernel speed. No FID result is claimed.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

The NVIDIA implementation and FFHQ weights are for non-commercial research/evaluation under the included NVIDIA license.

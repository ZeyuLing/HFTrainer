# Wan T2V

Text-to-video training and denoising with a local UMT5 path, VAE and transformer.

**Verification:** Full pretrained component export/reload; tiny numerical reference checks

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

Download the complete HFTrainer bundle below. Model weights are hosted separately from the code.

## Checkpoint

[ZeyuLing/HFTrainer-Wan2.1-T2V-1.3B](https://huggingface.co/ZeyuLing/HFTrainer-Wan2.1-T2V-1.3B) · [pinned revision](https://huggingface.co/ZeyuLing/HFTrainer-Wan2.1-T2V-1.3B/tree/4fdf8c0e2665a48e722134a28882fc030238003e)

This release packages real pretrained weights in a complete **HFTrainer bundle**. Trainer and Pipeline share the repository-owned `from_pretrained` loader. Conversion, source attribution and validation scope are recorded below.

```bash
hf download ZeyuLing/HFTrainer-Wan2.1-T2V-1.3B \
  --revision 4fdf8c0e2665a48e722134a28882fc030238003e \
  --local-dir checkpoints/hftrainer/wan
export HFTRAINER_CHECKPOINT="$PWD/checkpoints/hftrainer/wan"
```

[Conversion and validation record](evidence/checkpoints/wan.json). Base weights are separate from demo adapters and datasets. To export a fine-tuned bundle:

```bash
python tools/export_model.py --config configs/pretrained/train_wan.py \
  --checkpoint PATH_TO_TRAINING_CHECKPOINT --output outputs/wan_export
```

## Data

Four synthetic 5-frame, 32 × 32 samples. This is not the real-video dataset used for H3/LTX.

## Train

```bash
python tools/train.py configs/pretrained/train_wan.py
```

## Infer

Run the published pretrained model:

```bash
python tools/infer.py --config configs/pretrained/infer_wan.py --device cuda \
  --prompt "A cat walking." --num-frames 5 --height 32 --width 32 \
  --num-steps 3 --output outputs/wan.mp4
```

For fine-tuned weights, pass `--checkpoint` with the training output and a matching training model config (including LoRA settings or a changed classification head).

## Evidence and loss

Full 1.3B transformer, UMT5 and causal video VAE weights load strictly. Each component output matches before and after export. Tiny UMT5 and VAE outputs match the pinned reference exactly; tiny transformer FP32 maximum error is 3.58e-7.

The new pretrained training entry point completed one optimizer step and saved a checkpoint on an H200. These synthetic execution checks are recorded with the checkpoint evidence; they do not establish useful fine-tuning.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The published Wan 2.1 T2V 1.3B bundle uses the complete causal VAE. The 32 × 32, five-frame command is an execution check, not a video-quality preset. The local SentencePiece path approximates the precompiled normalizer with NFKC; tokenizer equivalence is not established for arbitrary text. No full-resolution quality comparison or convergence result is claimed.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

# ViT classification

Image classification with a local ViT encoder and image processor.

**Verification:** Tiny numerical and artifact tests

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

[ZeyuLing/HFTrainer-ViT-Base-Patch16-224](https://huggingface.co/ZeyuLing/HFTrainer-ViT-Base-Patch16-224) · [pinned revision](https://huggingface.co/ZeyuLing/HFTrainer-ViT-Base-Patch16-224/tree/5e1eca66e547fb8759fbc66e9cda8579136c8b26)

This release packages real pretrained weights in a complete **HFTrainer bundle**. Trainer and Pipeline share the repository-owned `from_pretrained` loader. Conversion, source attribution and validation scope are recorded below.

```bash
hf download ZeyuLing/HFTrainer-ViT-Base-Patch16-224 \
  --revision 5e1eca66e547fb8759fbc66e9cda8579136c8b26 \
  --local-dir checkpoints/hftrainer/vit
export HFTRAINER_CHECKPOINT="$PWD/checkpoints/hftrainer/vit"
```

[Conversion and validation record](evidence/checkpoints/vit.json). Base weights are separate from demo adapters and datasets. To export a fine-tuned bundle:

```bash
python tools/export_model.py --config configs/pretrained/train_vit.py \
  --checkpoint PATH_TO_TRAINING_CHECKPOINT --output outputs/vit_export
```

## Data

20 images under `data/classification/demo/images`, arranged in four class directories. The config reuses training data for validation, so accuracy is not a held-out estimate.

## Train

```bash
python tools/train.py configs/pretrained/train_vit.py
```

## Infer

Run the published pretrained model:

```bash
python tools/infer.py --config configs/pretrained/infer_vit.py --device cuda \
  --input path/to/image.jpg
```

For fine-tuned weights, pass `--checkpoint` with the training output and a matching training model config (including LoRA settings or a changed classification head).

## Evidence and loss

Full pretrained export/reload produced identical logits. Tiny forward, loss, backward and reference-logit checks also pass.

The new pretrained training entry point completed one optimizer step and saved a checkpoint on an H200. These synthetic execution checks are recorded with the checkpoint evidence; they do not establish useful fine-tuning.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The four-class head is adapted to the demo. No full-data accuracy benchmark is provided.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

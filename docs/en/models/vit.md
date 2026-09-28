# ViT classification

Image classification with a local ViT encoder and image processor.

**Verification:** Tiny numerical and artifact tests

**Runtime:** Repository-owned model, trainer and pipeline; infrastructure libraries and model assets remain dependencies.

[All models](index.md) · [Setup](#setup) · [Train](#train) · [Infer](#infer) · [Evidence](#evidence-and-loss)

## At a glance

| Item | Status |
| :--- | :--- |
| Implementation | Local to HFTrainer |
| Demo data | See [data availability](demo-data.md) |
| Convergence | Not established by the evidence on this card |

## Setup

Run commands from the repository root. Prepare the listed weights and data first; a command is not a self-contained download or a claim of convergence.

```bash
python -m pip install -e .
```

Place the compatible checkpoint and tokenizer/config assets at `checkpoints/vit-base-patch16-224`. Weights are not bundled.

## Data

20 images under `data/classification/demo/images`, arranged in four class directories. The config reuses training data for validation, so accuracy is not a held-out estimate.

## Train

```bash
python tools/train.py configs/vit/vit_base_demo.py
```

## Infer

After training, use the saved checkpoint (adjust the step to a checkpoint that exists):

```bash
python tools/infer.py --config configs/vit/vit_base_demo.py \
  --checkpoint work_dirs/vit_smoke/checkpoint-iter_10 \
  --input path/to/image.jpg
```

## Evidence and loss

Tiny forward, loss, backward, reference-logit and artifact round-trip checks.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The four-class head is adapted to the demo. No full-data accuracy benchmark is provided.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

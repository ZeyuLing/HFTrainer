# StyleGAN2

Unconditional 64 × 64 image generation with local generator and discriminator training.

**Verification:** Tiny adversarial and artifact tests

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

No pretrained checkpoint is required for training from initialization.

## Data

Images under `data/classification/demo/images`. The supplied configuration trains from initialization.

## Train

```bash
python tools/train.py configs/stylegan2/stylegan2_demo.py
```

## Infer

After training, use the saved checkpoint (adjust the step to a checkpoint that exists):

```bash
python tools/infer.py --config configs/stylegan2/stylegan2_demo.py \
  --checkpoint work_dirs/stylegan2_demo/checkpoint-iter_10000 \
  --output outputs/stylegan2.png
```

## Evidence and loss

Tiny adversarial updates and artifact round-trip checks.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The default config runs 10,000 iterations. An untrained generator is not a useful image model; no FID result is claimed.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

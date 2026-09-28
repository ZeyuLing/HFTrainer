# Wan T2V

Text-to-video training and denoising with a local UMT5 path, VAE and transformer.

**Verification:** Tiny contract tests; upstream artifact compatibility limited

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

Place the compatible checkpoint and tokenizer/config assets at `checkpoints/Wan2.1-T2V-1.3B-Diffusers`. Weights are not bundled.

## Data

Four synthetic 4-frame, 32 × 32 samples. This is not the real-video dataset used for H3/LTX.

## Train

```bash
python tools/train.py configs/wan/wan_demo.py
```

## Infer

After training, use the saved checkpoint (adjust the step to a checkpoint that exists):

```bash
python tools/infer.py --config configs/wan/wan_demo.py \
  --checkpoint work_dirs/wan_smoke/checkpoint-iter_5 \
  --prompt "A cat walking." --output outputs/wan.mp4
```

## Evidence and loss

Local tiny train/denoise and artifact tests.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The compact VAE can require conversion for full upstream checkpoints. Low checkpoint coverage is rejected. This recipe is not a verified upstream full-weight launch.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

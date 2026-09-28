# DMD

One-step image generation via distribution matching on the local Stable Diffusion 1.5 core.

**Verification:** Tiny algorithm and artifact tests

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

Place the compatible checkpoint and tokenizer/config assets at `checkpoints/stable-diffusion-v1-5`. Weights are not bundled.

## Data

Captioned images in `data/text2image/demo`. The demo can compute teacher regression targets online.

## Train

```bash
python tools/train.py configs/dmd/dmd_demo.py
```

## Infer

After training, use the saved checkpoint (adjust the step to a checkpoint that exists):

```bash
python tools/infer.py --config configs/dmd/dmd_demo.py \
  --checkpoint work_dirs/dmd_demo/checkpoint-iter_10000 \
  --prompt "A paper boat at dusk." --output outputs/dmd.png
```

## Evidence and loss

Distribution-matching, fake-score and teacher paths have focused tests.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The default config runs 10,000 iterations. No published full training run or quality benchmark is attached.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

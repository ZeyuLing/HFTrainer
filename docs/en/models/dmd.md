# DMD

One-step image generation via distribution matching on the local Stable Diffusion 1.5 core.

**Verification:** Author-trained DMD2 initialization; exact native export/reload

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

[ZeyuLing/HFTrainer-DMD2-SD15-Initialization](https://huggingface.co/ZeyuLing/HFTrainer-DMD2-SD15-Initialization) · [pinned revision](https://huggingface.co/ZeyuLing/HFTrainer-DMD2-SD15-Initialization/tree/f3f9cbe23e677f9b555039b7ff0aa1b0c7e8d20d)

This release packages real pretrained weights in a complete **HFTrainer bundle**. Trainer and Pipeline share the repository-owned `from_pretrained` loader. Conversion, source attribution and validation scope are recorded below.

```bash
hf download ZeyuLing/HFTrainer-DMD2-SD15-Initialization \
  --revision f3f9cbe23e677f9b555039b7ff0aa1b0c7e8d20d \
  --local-dir checkpoints/hftrainer/dmd
export HFTRAINER_CHECKPOINT="$PWD/checkpoints/hftrainer/dmd"
```

[Conversion and validation record](evidence/checkpoints/dmd.json). Base weights are separate from demo adapters and datasets. To export a fine-tuned bundle:

```bash
python tools/export_model.py --config configs/pretrained/train_dmd.py \
  --checkpoint PATH_TO_TRAINING_CHECKPOINT --output outputs/dmd_export
```

## Data

Captioned images in `data/text2image/demo`. The demo can compute teacher regression targets online.

## Train

```bash
python tools/train.py configs/pretrained/train_dmd.py
```

## Infer

Run the published pretrained model:

```bash
python tools/infer.py --config configs/pretrained/infer_dmd.py --device cuda \
  --prompt "A paper boat at dusk." --output outputs/dmd.png
```

For fine-tuned weights, pass `--checkpoint` with the training output and a matching training model config (including LoRA settings or a changed classification head).

## Evidence and loss

The published package contains the author-trained DMD2 SD1.5 non-GAN generator and both score networks, with SD1.5 text/VAE assets. All three UNets load strictly and generator output is identical after export/reload. Distribution-matching, fake-score and teacher paths have focused tests.

The pretrained Trainer entry point completed a generator/fake-score update and saved a checkpoint using synthetic captioned images at 64 × 64. The base inference CLI also generated an image. This is execution evidence, not a quality benchmark.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

This DMD2 checkpoint initializes HFTrainer DMD; it is not an original-DMD-paper reproduction and does not imply the objectives are identical. The default config runs 10,000 iterations. No published full training run or quality benchmark is attached.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

DMD2-derived weights retain upstream non-commercial share-alike terms; SD1.5 components retain CreativeML OpenRAIL-M. The Hub artifact preserves both licenses and the upstream notice discrepancy.

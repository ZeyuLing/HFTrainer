# LLaMA causal LM

Instruction fine-tuning and text generation with local tokenization, KV caching and LoRA.

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

[ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat](https://huggingface.co/ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat) · [pinned revision](https://huggingface.co/ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat/tree/b23a22f572ca8d25e3d10411e90932826b9875f1)

This release packages real pretrained weights in a complete **HFTrainer bundle**. Trainer and Pipeline share the repository-owned `from_pretrained` loader. Conversion, source attribution and validation scope are recorded below.

```bash
hf download ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat \
  --revision b23a22f572ca8d25e3d10411e90932826b9875f1 \
  --local-dir checkpoints/hftrainer/llama
export HFTRAINER_CHECKPOINT="$PWD/checkpoints/hftrainer/llama"
```

[Conversion and validation record](evidence/checkpoints/llama.json). Base weights are separate from demo adapters and datasets. To export a fine-tuned bundle:

```bash
python tools/export_model.py --config configs/pretrained/train_llama.py \
  --checkpoint PATH_TO_TRAINING_CHECKPOINT --output outputs/llama_export
```

## Data

Ten instruction/input/output records at `data/llm/demo/alpaca_sample.json`. A full fine-tuning recipe is also available in `configs/llama/llama_sft_demo.py`.

## Train

```bash
python tools/train.py configs/pretrained/train_llama.py
```

## Infer

Run the published pretrained model:

```bash
python tools/infer.py --config configs/pretrained/infer_llama.py --device cuda \
  --prompt "The capital of France is" --max-new-tokens 32
```

For fine-tuned weights, pass `--checkpoint` with the training output and a matching training model config (including LoRA settings or a changed classification head).

## Evidence and loss

Full pretrained export/reload produced identical logits. Real TinyLlama tokenizer encodings and decodings match the reference for Unicode, whitespace and special-token cases. Tiny numerical and sharded-artifact checks also pass.

The new pretrained training entry point completed one optimizer step and saved a checkpoint on an H200. These synthetic execution checks are recorded with the checkpoint evidence; they do not establish useful fine-tuning.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The ten-step recipe demonstrates wiring, not instruction-following quality.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

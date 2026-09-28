# LLaMA causal LM

Instruction fine-tuning and text generation with local tokenization, KV caching and LoRA.

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

Place the compatible checkpoint and tokenizer/config assets at `checkpoints/TinyLlama-1.1B-Chat-v1.0`. Weights are not bundled.

## Data

Ten instruction/input/output records at `data/llm/demo/alpaca_sample.json`. A full fine-tuning recipe is also available in `configs/llama/llama_sft_demo.py`.

## Train

```bash
python tools/train.py configs/llama/llama_lora_demo.py
```

## Infer

After training, use the saved checkpoint (adjust the step to a checkpoint that exists):

```bash
python tools/infer.py --config configs/llama/llama_lora_demo.py \
  --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \
  --merge-lora --prompt "Name one primary color."
```

## Evidence and loss

Tiny forward, loss, backward, generation, reference-logit and sharded-artifact checks.

**Loss curve: not published.** No sustained-decrease or convergence result is claimed.

## Limitations

The ten-step recipe demonstrates wiring, not instruction-following quality.

## Sources and license

Consult the checkpoint provider’s model card for asset terms. See the [repository notices](https://github.com/ZeyuLing/HFTrainer/blob/main/THIRD_PARTY_NOTICES.md) and repository license; model assets are not relicensed by HFTrainer.

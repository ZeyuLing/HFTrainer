# Quick start

Install HFTrainer, a CUDA-enabled PyTorch build suitable for your hardware, and the Hub download helper:

```bash
python -m pip install -e .
python -m pip install "huggingface_hub>=0.34,<2"
```

## Train on public data

```bash
python tools/run_public_demo.py vit
```

This downloads the processed ViT checkpoint and the public Beans dataset, verifies dataset files, trains for 20 steps and evaluates on the separate validation split. Outputs are under `work_dirs/public_data/vit/`.

The same launcher accepts `llama`, `sd15`, `dmd`, `stylegan2`, `wan`, `ltx_video` and `minimax_h3`. Read the corresponding [model card](models/index.md) for hardware, model-specific extras and access agreements. Video preprocessing needs FFmpeg. The short recipes verify execution and do not establish convergence.

```bash
# Download/prepare only:
python tools/run_public_demo.py llama --prepare-only
# Reuse a complete native checkpoint:
python tools/run_public_demo.py llama --checkpoint checkpoints/hftrainer/llama
```

## Infer from a checkpoint

```bash
python tools/infer.py --model ZeyuLing/HFTrainer-TinyLlama-1.1B-Chat \
  --device cuda --prompt "The butterfly is" --max-new-tokens 48

python tools/infer.py --model ZeyuLing/HFTrainer-Stable-Diffusion-1.5 \
  --device cuda --seed 42 --num-steps 30 \
  --prompt "A butterfly on a flower in natural daylight." --output outputs/butterfly.png
```

A local complete checkpoint directory works in place of the Hub ID. Model cards give pinned `--revision` values for reproducibility. `config.json` selects the registered local model, task and pipeline; no separate inference config or external model repository is needed.

## Fine-tuned artifacts

The trainer checkpoint includes optimizer/resume state. Export a standalone model with the matching config:

```bash
python tools/export_model.py --config configs/public_data/vit.py \
  --checkpoint work_dirs/public_data/vit/checkpoint-iter_20 --output outputs/beans_vit
python tools/infer.py --model outputs/beans_vit --device cuda --input path/to/leaf.jpg
```

Use `configs/pretrained/export_minimax_h3.py` for H3 export, so the conditioner and VAEs omitted during cached training are restored. LTX export accepts the LTX trainer's LoRA `.safetensors` file, not its full optimizer state. If training used a custom base location, set `HFTRAINER_CHECKPOINT` to the same path before exporting or resuming.

[Public datasets](models/demo-data.md) · [Model-card standard](models/standard.md)

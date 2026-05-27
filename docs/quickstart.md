# Quick Start

## Smoke Tests

Each task has a demo config that runs on synthetic or tiny data with no external downloads needed. These verify the full pipeline (data loading, forward, backward, checkpoint save) in under a minute.

**ViT Classification** (no GPU required):

```bash
python tools/train.py configs/classification/vit_base_demo.py
```

**SD1.5 Text-to-Image** (requires GPU with fp16):

```bash
python tools/train.py configs/text2image/sd15_demo.py
```

**TinyLlama SFT** (requires GPU with bf16):

```bash
python tools/train.py configs/llm/llama_sft_demo.py
```

**WAN Text-to-Video** (requires GPU with bf16):

```bash
python tools/train.py configs/text2video/wan_demo.py
```

## Inference

After training, run inference with `tools/infer.py`:

**Text-to-Image:**

```bash
python tools/infer.py \
    --config configs/text2image/sd15_demo.py \
    --checkpoint work_dirs/sd15_smoke/checkpoint-10 \
    --prompt "a beautiful sunset" \
    --output output/image.png
```

**Text-to-Video:**

```bash
python tools/infer.py \
    --config configs/text2video/wan_demo.py \
    --checkpoint work_dirs/wan_smoke/checkpoint-5 \
    --prompt "a cat walking in the park" \
    --output output/video.mp4
```

**Classification:**

```bash
python tools/infer.py \
    --config configs/classification/vit_base_demo.py \
    --checkpoint work_dirs/vit_smoke/checkpoint-10 \
    --input data/classification/demo/images/class_0/0000.jpg
```

**LLM:**

```bash
python tools/infer.py \
    --config configs/llm/llama_sft_demo.py \
    --checkpoint work_dirs/llama_smoke/checkpoint-10 \
    --prompt "What is the capital of France?"
```

## Programmatic Inference

You can also use the Pipeline API directly in Python:

```python
from hftrainer.models.text2video import WanModelBundle
from hftrainer.pipelines.text2video import WanPipeline

# Load bundle from config, then override with finetuned weights
bundle = WanModelBundle.from_cfg(cfg.model)
bundle.load_state_dict_selective(torch.load('work_dirs/wan_exp/checkpoint.pth'))

pipeline = WanPipeline(model=bundle)
video = pipeline("a cat running in the park")
```

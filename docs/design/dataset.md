# Dataset Design / 数据集设计

## Overview / 概述

Different datasets have varying formats even for the same task, so datasets are organized by `{task}/{dataset_name}_dataset.py`.

不同数据集即使任务相同，格式也各异，因此按 `{task}/{dataset_name}_dataset.py` 组织。

---

## Directory Structure / 目录结构

```
hftrainer/datasets/
├── __init__.py
├── text2image/
│   ├── __init__.py
│   ├── base_text2image_dataset.py   # Interface: __getitem__ returns {'image': Tensor, 'text': str}
│   ├── laion_dataset.py             # LAION format
│   ├── webdataset_t2i.py            # WebDataset tar format
│   └── hf_imagefolder_dataset.py    # HuggingFace ImageFolder format
├── text2video/
│   ├── __init__.py
│   ├── base_text2video_dataset.py   # Interface: {'video': Tensor[T,C,H,W], 'text': str}
│   ├── webvid_dataset.py
│   └── internvid_dataset.py
├── llm/
│   ├── __init__.py
│   ├── base_llm_dataset.py          # Interface: {'input_ids': Tensor, 'labels': Tensor, 'attention_mask': Tensor}
│   ├── alpaca_dataset.py
│   ├── sharegpt_dataset.py
│   └── pretrain_jsonl_dataset.py
├── classification/
│   ├── __init__.py
│   ├── base_classification_dataset.py  # Interface: {'image': Tensor, 'label': int}
│   ├── imagenet_dataset.py
│   └── hf_image_classification_dataset.py
└── detection/
    ├── __init__.py
    ├── base_detection_dataset.py    # Interface: {'image': Tensor, 'boxes': Tensor, 'labels': Tensor}
    ├── coco_dataset.py
    └── objects365_dataset.py
```

---

## Design Principles / 设计原则

Each `base_{task}_dataset.py` defines the interface using ABC, provides common transform logic and `collate_fn`. Concrete dataset subclasses only need to implement `_load_meta()` and the data parsing part of `__getitem__`.

每个 `base_{task}_dataset.py` 中用 ABC 定义接口，同时提供公共的 transform 逻辑和 `collate_fn`，具体数据集子类只需实现 `_load_meta()` 和 `__getitem__` 的数据解析部分。

---

## Demo Data / 演示数据

Each task has a small amount of test data under `data/{task}/demo/` with a download script:

每个任务在 `data/{task}/demo/` 下放少量测试数据，并提供下载脚本：

```
data/
├── text2image/demo/
│   ├── images/          # A few test images / 几张测试图片
│   └── metadata.jsonl   # {"image": "images/001.jpg", "text": "a cat"}
├── text2video/demo/
│   ├── videos/
│   └── metadata.jsonl
├── llm/demo/
│   └── alpaca_sample.json   # 10 instruction-tuning samples / 10 条 instruction-tuning 样本
├── classification/demo/
│   ├── images/
│   └── labels.txt
└── detection/demo/
    ├── images/
    └── annotations.json     # COCO format / COCO 格式
```

Download script:

下载脚本：

```python
# tools/download_demo_data.py
# Usage: python tools/download_demo_data.py --task text2image
from datasets import load_dataset
DEMO_SOURCES = {
    'text2image':    ('lambdalabs/pokemon-blip-captions', 10),
    'text2video':    ('BestWishYsh/ConsisID-preview', 5),
    'llm':           ('tatsu-lab/alpaca', 10),
    'classification':('zh-plus/tiny-imagenet', 20),
    'detection':     ('detection-datasets/coco_n_samples', 20),
}
```

Each task's `base_{task}_dataset.py` provides a `demo()` class method that directly returns a demo DataLoader:

每个任务的 `base_{task}_dataset.py` 提供一个 `demo()` 类方法直接返回 demo DataLoader：

```python
# One-line verification that the entire pipeline works
# 一行命令验证整个 pipeline 是否跑通
dataloader = Text2ImageDataset.demo()
```

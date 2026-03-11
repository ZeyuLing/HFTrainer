# 快速开始

## 已验证的 Smoke Test

当前最稳妥的首个运行命令是分类 demo：

```bash
python3 tools/train.py configs/classification/vit_base_demo.py
```

这条路径会覆盖：

- 配置解析
- dataloader 构建
- 模型前向与反向
- checkpoint 保存
- validation

## 其他 Demo

```bash
python3 tools/train.py configs/text2image/sd15_demo.py
python3 tools/train.py configs/llm/llama_sft_demo.py
python3 tools/train.py configs/llm/llama_lora_demo.py
python3 tools/train.py configs/text2video/wan_demo.py
```

这些任务需要对应的 `checkpoints/` 资源和足够的 GPU 显存。

LoRA 快速开始：

```bash
python3 tools/train.py configs/llm/llama_lora_demo.py
python3 tools/infer.py \
  --config configs/llm/llama_lora_demo.py \
  --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \
  --merge-lora \
  --prompt "Name one primary color."
```

## 推理示例

分类推理：

```bash
python3 tools/infer.py \
  --config configs/classification/vit_base_demo.py \
  --checkpoint work_dirs/vit_smoke/checkpoint-iter_10 \
  --input data/classification/demo/images/cat/cat_000.jpg \
  --device cpu
```

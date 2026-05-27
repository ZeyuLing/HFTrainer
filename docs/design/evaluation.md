# Evaluation & Visualization Design / 评估与可视化设计

## Overview / 概述

### The Problem with MMEngine / MMEngine 的痛点

`DataSample` is a heavyweight container. Evaluator and Visualizer both need to know its internal fields (`pred_instances`, `gt_instances`, `metainfo`, etc.), creating deep coupling. It's hard for beginners to understand and nearly impossible to reuse across tasks.

`DataSample` 是一个重量级容器，evaluator 和 visualizer 都要知道它的内部字段（`pred_instances`, `gt_instances`, `metainfo` 等），coupling 极深，初学者难以理解，跨任务复用几乎不可能。

### Our Design Principle / 本框架的设计原则

`val_step` returns a **plain Python dict** with agreed-upon keys. No special container classes. Both Evaluator and Visualizer consume this dict.

`val_step` 返回**纯 Python dict**，约定好 key 即可，无任何特殊容器类。Evaluator 和 Visualizer 都消费这个 dict。

---

## val_step Return Value Convention / val_step 返回值约定

Each task's Base Trainer documents the required keys:

各任务的 Base Trainer 中文档注释定义了必须的 key：

**Text-to-Image:**

```python
{
    'preds':   Tensor[B, C, H, W],   # generated images, float, [0,1]
    'gts':     Tensor[B, C, H, W],   # ground truth images (optional), for FID/LPIPS
    'prompts': list[str],             # corresponding text prompts
    'metas':   list[dict],            # optional: original filename, resolution, etc.
}
```

**Classification:**

```python
{
    'preds':   Tensor[B],             # predicted class ids
    'scores':  Tensor[B, num_cls],    # logits / softmax scores
    'gts':     Tensor[B],             # ground truth class ids
    'metas':   list[dict],            # optional: image paths, etc.
}
```

**Detection:**

```python
{
    'pred_boxes':   list[Tensor[N,4]],  # predicted boxes per image, xyxy
    'pred_scores':  list[Tensor[N]],
    'pred_labels':  list[Tensor[N]],
    'gt_boxes':     list[Tensor[M,4]],
    'gt_labels':    list[Tensor[M]],
    'metas':        list[dict],          # image_id etc., for COCO eval
}
```

**LLM:**

```python
{
    'preds':         list[str],   # generated text
    'gts':           list[str],   # reference text
    'input_prompts': list[str],   # input prompts
}
```

---

## Evaluator Design / Evaluator 设计

Receives the entire epoch's `list[dict]` (each dict is one batch's output), merges and computes metrics:

接收整个 epoch 的 `list[dict]`（每个 dict 是一个 batch 的输出），合并后计算指标：

```python
class BaseEvaluator:
    def process(self, output: dict) -> None:
        """Called once per val batch. Caches results.
        每个 val batch 调用一次，缓存结果。"""
        self._results.append(output)

    def compute(self) -> dict:
        """Called after the entire val epoch. Returns metrics dict.
        整个 val epoch 结束后调用，返回指标 dict。"""
        raise NotImplementedError

# Example: classification task / 示例：分类任务
class AccuracyEvaluator(BaseEvaluator):
    def compute(self) -> dict:
        all_preds = torch.cat([r['preds'] for r in self._results])
        all_gts   = torch.cat([r['gts']   for r in self._results])
        top1 = (all_preds == all_gts).float().mean().item()
        return {'top1_acc': top1}
```

---

## Visualizer Design / Visualizer 设计

Also receives the `val_step` return dict. No need to know any container class:

同样接收 `val_step` 返回的 dict，不需要知道任何容器类：

```python
class BaseVisualizer:
    def visualize(self, output: dict, step: int) -> None:
        """Extract needed fields from output dict, log to wandb/tensorboard.
        从 output dict 中取需要的字段，记录到 wandb/tensorboard。"""
        raise NotImplementedError

# Example: text-to-image / 示例：文生图任务
class Text2ImageVisualizer(BaseVisualizer):
    def visualize(self, output: dict, step: int) -> None:
        images  = output['preds']        # Tensor[B, C, H, W]
        prompts = output['prompts']      # list[str]
        self.logger.log_images('val_samples', images, captions=prompts, step=step)
```

---

## Complete Val Loop / 完整的 Val Loop 流程

In the Runner:

在 Runner 中：

```python
# AccelerateRunner.val_epoch()
all_outputs = []
for batch in val_dataloader:
    output = trainer.val_step(batch)                         # returns plain dict / 返回纯 dict
    output = accelerator.gather_for_metrics(output)          # multi-GPU gather / 多卡聚合
    for ev in evaluators:
        ev.process(output)                                   # cache results / 缓存结果
    all_outputs.append(output)

# Compute metrics / 计算指标
metrics = {}
for ev in evaluators:
    metrics.update(ev.compute())               # {'top1_acc': 0.82, 'fid': 12.3, ...}
runner.log(metrics)

# Visualize (last batch only) / 可视化（只取最后一个 batch）
for vis in visualizers:
    vis.visualize(all_outputs[-1], step=runner.global_step)
```

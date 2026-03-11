# 评估与可视化

HF-Trainer 的约定很简单：`val_step()` 返回普通 Python dict。

Hook 另有专门文档，见 [Hook 系统](hooks.md)。在 HF-Trainer 里，evaluator 和 visualizer 是 validation 组件，不属于 hook。

## 为什么这样设计

- 比框架私有容器更容易理解
- 跨任务复用更直接
- evaluator 和 visualizer 只依赖约定好的 key

## 当前 Evaluator

- `AccuracyEvaluator`
- `PerplexityEvaluator`

## 当前 Visualizer

- `FileVisualizer`
- `TensorBoardVisualizer`

## 常见输出形式

分类：

```python
{'preds': Tensor[B], 'scores': Tensor[B, C], 'gts': Tensor[B]}
```

文生图：

```python
{'preds': Tensor[B, C, H, W], 'prompts': list[str]}
```

LLM：

```python
{'preds': list[str], 'gts': list[str], 'input_prompts': list[str], 'loss_lm': Tensor[]}
```

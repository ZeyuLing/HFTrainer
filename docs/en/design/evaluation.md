# Evaluation and Visualization

HF-Trainer keeps validation outputs simple: `val_step()` returns a plain Python dict.

Hooks are documented separately in [Hook System](hooks.md). In HF-Trainer, evaluators and visualizers are validation-time components, not hooks.

## Why

- easier to understand than framework-specific sample containers
- easier to reuse across tasks
- evaluators and visualizers only depend on agreed keys

## Current Evaluators

- `AccuracyEvaluator`
- `PerplexityEvaluator`

## Current Visualizers

- `FileVisualizer`
- `TensorBoardVisualizer`

## Typical Output Shapes

Classification:

```python
{'preds': Tensor[B], 'scores': Tensor[B, C], 'gts': Tensor[B]}
```

Text-to-image:

```python
{'preds': Tensor[B, C, H, W], 'prompts': list[str]}
```

LLM:

```python
{'preds': list[str], 'gts': list[str], 'input_prompts': list[str], 'loss_lm': Tensor[]}
```

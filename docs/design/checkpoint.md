# Checkpoint Design / Checkpoint 设计

## Overview / 概述

Resume and load-checkpoint are fundamentally a difference in "how much to load," not two independent systems. HF-Trainer unifies them into a single `load_from` field with `load_scope` controlling the loading range.

Resume 和 load_checkpoint 本质上是"加载哪些东西"的程度差异，而非两套独立逻辑。HF-Trainer 统一为一个 `load_from` 字段，用 `load_scope` 控制加载范围。

---

## Usage Scenarios / 使用场景

```python
# Scenario A: Only load model weights (transfer learning / finetune from pretrained)
# 场景 A：只加载 model weights（迁移学习 / 从 pretrained 开始 finetune）
load_from = dict(
    path='work_dirs/wan_exp/checkpoint-10000/',
    load_scope='model',        # only model weights; optimizer/scheduler/meta all reset
)

# Scenario B: Full resume (continue interrupted training)
# 场景 B：完整 resume（中断后续训练）
load_from = dict(
    path='work_dirs/wan_exp/checkpoint-10000/',
    load_scope='full',         # load model + optimizer + scheduler + training meta (epoch/iter)
)

# Scenario C: Load partial modules (e.g., only transformer, skip text_encoder)
# 场景 C：从已有 checkpoint 加载部分模块（例如只加载 transformer，跳过 text_encoder）
# No extra field needed -- ModelBundle.load_state_dict_selective() handles this automatically:
# state_dict keys present get loaded, missing ones are skipped
# 不需要额外字段 —— ModelBundle.load_state_dict_selective() 自动处理
```

---

## `load_scope` Semantics / `load_scope` 的含义

| `load_scope` | Model weights | Optimizer states | Scheduler states | Training meta (epoch/iter) |
|---|---|---|---|---|
| `'model'` | Loaded (selective) | Reset | Reset | Reset (from 0) |
| `'full'` | Loaded | Loaded | Loaded | Loaded (continues) |

---

## AccelerateRunner Logic / AccelerateRunner 处理逻辑

```python
# At runner startup / runner 启动时
if cfg.get('auto_resume'):
    # Auto-detect latest checkpoint in work_dir, full resume
    # 自动检测 work_dir 中最新的 checkpoint，full resume
    last_ckpt = find_latest_checkpoint(cfg.work_dir)
    if last_ckpt:
        runner.load(last_ckpt, load_scope='full')

elif cfg.get('load_from'):
    runner.load(cfg.load_from.path, load_scope=cfg.load_from.get('load_scope', 'model'))
```

- `load_scope='full'`: calls `accelerator.load_state(path)`, automatically handles FSDP / DeepSpeed ZeRO optimizer state formats, and restores `global_step` / `epoch` / 调用 `accelerator.load_state(path)`，自动处理 FSDP / DeepSpeed ZeRO optimizer state 格式，同时恢复 `global_step` / `epoch`
- `load_scope='model'`: only calls `bundle.load_state_dict_selective()`, resets optimizer and meta / 只调用 `bundle.load_state_dict_selective()`，optimizer 和 meta 全部重置

---

## Auto-Resume / 自动 Resume

`auto_resume=True` is the recommended default. No need to manually specify `load_from`. The Runner auto-detects the latest checkpoint in `work_dir` and does a full resume, suitable for cluster jobs that may be preempted.

`auto_resume=True` 是推荐的默认行为，不需要手动指定 `load_from`，Runner 自动检测 work_dir 中最新 checkpoint 并 full resume，适合集群任务被抢占后重启的场景。

Resume produces a clear log message:

Resume 时会输出清晰的日志：

```
============================================================
Resuming from checkpoint: work_dirs/wan_exp/checkpoint-5000
Resumed: global_step=5000, epoch=0. Training will continue from step 5001.
============================================================
```

---

## Save Strategy / 保存策略

`CheckpointHook` calls `bundle.state_dict_to_save()`, which only writes modules with `save_ckpt=True`. `accelerator.save_state()` simultaneously writes optimizer / scheduler states for full resume.

`CheckpointHook` 调用 `bundle.state_dict_to_save()`，只写 `save_ckpt=True` 的模块。`accelerator.save_state()` 同时写 optimizer / scheduler states（用于 full resume）。

Use `max_keep_ckpts` to limit disk usage:

通过 `max_keep_ckpts` 控制磁盘上保留的 checkpoint 数量：

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        max_keep_ckpts=3,     # keep only the 3 most recent / 只保留最近 3 个
    ),
)
```

---

## Inference Loading / 推理加载

`WanPipeline.from_checkpoint(bundle_cfg, ckpt_path)` -- first fully builds the ModelBundle (text_encoder / vae etc. loaded from pretrained), then uses `load_state_dict_selective` to override the transformer's finetuned weights. No special adaptation needed.

`WanPipeline.from_checkpoint(bundle_cfg, ckpt_path)` —— 先完整构建 ModelBundle（text_encoder / vae 等从 pretrained 加载），再用 `load_state_dict_selective` 覆盖 transformer 的 finetune 权重，无需特殊适配。

```python
bundle = WanModelBundle.from_cfg(cfg.model)
bundle.load_state_dict_selective(torch.load('work_dirs/wan_exp/checkpoint.pth'))
pipeline = WanPipeline(model=bundle)
video = pipeline("a cat running in the park")
```

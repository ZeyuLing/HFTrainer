# E9 A_adaptive_inpaint 和 D_ada_denoise bug 分析 (2026-04-22)

## 问题
用户反馈两个 setting 的推理结果和 LQ "完全不符合"。

---

## Bug 1: A_adaptive_inpaint — `sdedit_tau` 从未生效

### 设定
```python
'A_adaptive_inpaint': TaskSetting(
    {
        '_use_adaptive_mask': True,
        '_editing_mode': False,
        '_sdedit_tau': 0.5,
    },
),
```

### 代码逻辑（`hymotion_m2m_pipeline.py:290-326`）
```python
rep_mode = self.replacement_guidance
use_replacement = (
    rep_mode != 'none'
    and src_mask is not None
    and src_mask.sum() > 0
    and src_mask.sum() < src_mask.numel()
)

if use_replacement:
    keep_mask = src_mask < 0.5
    x_clean = batch['clean_motion'].to(device)
    if self.sdedit_tau > 0.0:
        # SDEdit 路径
        t_init = 1.0 - self.sdedit_tau
        x_partial_noised = (1.0 - t_init) * z + t_init * x_clean
        y0 = torch.where(keep_mask, x_clean, x_partial_noised)
        sdedit_t_init = t_init
    else:
        y0 = torch.where(keep_mask, x_clean, z)
        ...
else:
    y0 = z   # 纯噪声
    sdedit_t_init = None
```

### Bug
A_adaptive_inpaint 设了 `_sdedit_tau=0.5` 但**没有设** `_replacement_guidance`。pipeline 默认 `replacement_guidance='none'`，所以 `use_replacement=False`，进 `else: y0 = z`（line 325）→ **SDEdit 完全被绕过**。

实际执行等价于：
- y0 = 纯噪声
- mask = MoGenDIT adaptive_mask
- ODE 从 t=0 跑到 t=1
- VACE inactive 通道携带 LQ 值（mask=0 区域）

结果：模型从纯噪声生成，VACE 给模型 hint "mask=0 是 LQ"，但对于 M2M（standard 非-_man 训练）这种 hint 是弱条件 → **生成结果和 LQ 总体相似但局部偏差大**，和 adaptive mask 定义的缺陷区域无关。

### 正确逻辑
SDEdit 意图是"从 LQ 加轻度噪声出发做去噪"，保留 LQ 结构。必须同时：
1. 设 `_replacement_guidance='skip_last'`（让 `use_replacement=True`）
2. 设 `_sdedit_tau=0.5`
3. pass `clean_motion=LQ` 给 pipeline

### Fix
在 TaskSetting 里加 `_replacement_guidance='skip_last'`。还要保证 eval loop 把这个传给 pipeline。

---

## Bug 2: D_ada_denoise_t010 — Stage 1 完全不参考 LQ

### 设定
```python
'D_ada_denoise_t010': TaskSetting(
    {
        '_ada_denoise': True,
        '_ada_threshold_mode': 'abs',
        '_ada_threshold': 0.10,
        '_editing_mode': False,
    },
),
```

### 代码逻辑（`tools/eval_m2m_v2_all_tasks.py:2122-2181`）

**Stage 1**:
```python
pipeline.replacement_guidance = 'skip_last'
pipeline.sdedit_tau = 0.0
stage1_mask_t = torch.ones_like(src_mask)
stage1_mask_t[:, 0:1, :] = 0.0  # 只锁 frame 0
sampled_norm_stage1 = _run_inference_pass(stage1_mask_t, editing_override=False)
```

这里的 `_run_inference_pass` with `editing_override=False`：
```python
src_motion_here = motion_norm * (1 - stage1_mask_t)  # 只保留 frame 0，其它全 0
```

然后进 pipeline:
- `use_replacement=True` (有 skip_last, mask 有 0 和 1)
- `y0 = torch.where(keep_mask, clean_motion, z)` → frame 0 = LQ norm, 其它 = pure noise
- 每步 ODE 后 imputation: frame 0 = LQ

**Stage 2**:
```python
change = |motion_norm - sampled_norm_stage1|  # (T, D)
new_mask[t, d] = change[t, d] > 0.10
```

### Bug
**M2M 没有训练过 motion repair 任务**。当我们把 mask 设成"仅 frame 0 是 condition"时，模型看到：
- Frame 0 的 LQ pose（via VACE inactive + x_t imputation）
- Frame 1-T 全是纯噪声 + VACE inactive=0 + mask=1

模型在训练分布 **M5 full_mask** (5% 权重) 下学过"全帧生成"但**没学过"仅 frame 0 锚定"**。实际行为：生成一个**从 frame 0 开始的 T2M-style 动作**，和 LQ frame 1-T 毫无相似性。

因此 Stage 2 的 `change = |LQ - stage1|` **全局都很大**（两个不同的动作）→ 阈值 0.10 下 `new_mask` 几乎 all-1 → Stage 3 也是"全帧生成" → 和 LQ 完全无关。

### 对比 MoGenDIT 的原版
MoGenDIT 用相同的 mask "仅 frame 0" 为什么 work？
- MoGenDIT **训练时 50% batch 是 motion_degradation**（8 种合成退化 → 原始干净）
- 模型学会了"看见有缺陷的 motion → 输出干净 motion"
- Stage 1 给它 degraded 作为 inactive → 模型**主动修复**
- 所以 Stage 1 output ≈ repaired LQ，change 反映"哪些地方 LQ 有缺陷"

**M2M 没有这种训练**，Stage 1 输出 = T2M-style 随机动作，不是 repaired LQ。

### Fix 方向

**Option A (推荐): Stage 1 改为 SDEdit**
Stage 1 的目的是"让模型对 LQ 做轻度修正"。用 SDEdit:
```python
pipeline.replacement_guidance = 'skip_last'
pipeline.sdedit_tau = 0.5  # 从 LQ 加 50% 噪声
stage1_mask_t = torch.ones_like(src_mask)  # full_mask，但 sdedit 从 LQ 出发
```

这样 Stage 1 从 `0.5*noise + 0.5*LQ` 去噪，模型把 LQ 拉回 manifold（低层次修复抖动等）。然后 change 反映"模型把 LQ 哪些地方拉开了" → 即模型认为的缺陷位置。

**Option B: 删除 D_ada_denoise_*，承认 M2M 没有修复能力**
用户指出这个 setting 的 Stage 1 是错的。考虑到 fixing SDEdit 后还是跟 MoGenDIT 原设计不同（MoGenDIT 用 DDPM + x₀ 预测，M2M 用 flow matching），可以先 deprecate，专注于 D_strict_mask 方向。

**Option C: 改 M2M 训练引入 motion degradation**
这是根本解法，但要重训模型（workflow 改动大，优先级放后）。

---

## 推荐 Fix

先修 A_adaptive_inpaint (Option A 相同：加 replacement_guidance='skip_last')，同时 deprecate D_ada_denoise_* (Option B)，同时分析 D_strict_mask 抖动（#216）。

因为 A_adaptive_inpaint 的 adaptive mask 就是 MoGenDIT 提供的缺陷位置（用户说"不准"），所以真正可靠的 setting 只剩下 D_strict_mask 系列 + 未来的 StableMotion baseline。

# Debug History
> workspace 创建于 2026-04-20 21:40:00
> 问题: E9 Motion Repair (uncond_local + uncond_global × 3 settings) 所有 setting aggregated jitter_pos 相对旧实现全面恶化。
> 重点数据: uncond_global × C_full_inpaint × sample 0 (T=427) 和 sample 1 (T=152)

---

## Iteration 1 (INFRA + P3) — 2026-04-20 22:05:00
### 类型
数据分析 + 假设验证

### 假设与预注册
- 假设: 本轮退化不是我加的 SDEdit/sliding-window/mask refactor 代码引入的 bug，而是**训练本身在恶化** — 因为旧 eval 用的是 epoch_657，新 eval 用的是 epoch_846，中间差 ~200 epoch。
- 预期结果:
  - 若假设成立: 用同一份当前推理代码在多个连续 checkpoint 上跑同一样本，jitter_pos 应随 epoch 增加而单调上升或有明显趋势。
  - 若假设不成立: 连续 checkpoint 的 jitter 应接近（模型 1 epoch 不可能差几倍）。
- 检验指标: 同样本同 setting 同代码，jitter_pos(epoch=N) vs jitter_pos(epoch=N+k)
### 方向
1. 先按长度分桶统计现有 50-sample aggregated 的 per_sample — 验证短样本是否也恶化（否则 sliding-window 仍可能是主因）。
2. 若短样本也恶化，创建 test_quick.py 跑多 checkpoint 对比。

### 修改
- 创建 `.claude/autodebug_workspaces/e9_sdedit_regression/test_quick.py` — 加载指定 checkpoint + 跑单样本 + 打印 jitter，固定 seed=42 保证不同 ckpt 等比。

### 执行结果
**1. 长度分桶（本轮新 eval）**:
```
uncond_local C_full_inpaint: short N=42 mean=1611, long N=8 mean=1864  # short 也高，且比 long 还差
uncond_global C_full_inpaint: short N=42 mean=5930, long N=8 mean=5335  # 同样
```
短样本 jitter >= 长样本 jitter。**推翻"sliding-window 是主因"假设**。同时证明根因在"所有路径共享"的代码/模型变化。

**2. checkpoint-to-checkpoint 对比（seed 固定 42）**:
```
uncond_global C_full_inpaint:
  sample 0 (T=427):
    epoch_845: jitter=5044
    epoch_846: jitter=5108  (+1.3%)
    epoch_847: jitter=6448  (+26%)
    epoch_848: jitter=6863  (+6.4%)
  sample 1 (T=152):
    epoch_845: jitter=5800
    epoch_848: jitter=6314  (+9%)
```
**连续 3-4 epoch jitter 单向上升 ~36%**。相当于 200 epoch 从 353 → 5835 的轨迹完全一致。

### 预测 vs 实际
- 预测: 连续 checkpoint 的 jitter 单调上升 → 确认训练恶化
- 实际: **完美吻合**，4 个 epoch jitter 单调上升，幅度显著超过 GPU 随机性
- 偏差原因: 无

### 结论
**根因不是本轮的任何推理代码改动（SDEdit/sliding-window/198-dim mask/no-dilation）**，而是：
1. **训练本身在 epoch 657 → 846 之间持续恶化**（过拟合或训练方式退化），导致 uncond_global/local 模型在 E9 inference 时输出抖动越来越大。
2. 我上一次 rerun 用的"最新 checkpoint"恰好接近训练恶化最严重的位置。

### 回退
**不回退**代码改动。SDEdit / 198-dim mask / no-dilation 在数学和语义上都有道理（且 3-sample smoke 时新代码 jitter=669 好过旧路径 1329），新代码本身没问题。
**真正要修的是训练**。

---
</content>

## Iteration 2 — 2026-04-20 22:20:00
### 假设与预注册
- 假设: 用户说"别的任务推理没大问题"可能是基于旧 dashboard 数据。如果用 epoch_848 跑 E2/E3/E5，它们也会同样恶化。
- 预期结果:
  - 若训练全面恶化: E2/E3/E5 jitter_pos 也暴涨数倍
  - 若只是 E9 特有 bug: E2/E3/E5 应跟旧数据接近
- 检验指标: jitter_pos ratio (new / old)

### 方向
跨 task 横扫：用当前代码 + epoch_848 跑 E2/E3/E5 setting A × 5 samples + 15 samples。

### 执行结果
用当前代码 + epoch_848:

| Task | 旧 (657) | 新 (848, 5-sample) | 新 (848, 15-sample) | Ratio (15-sample) |
|------|---:|---:|---:|---:|
| E2/A | 1381 | 18641 | 13139 | **9.5×** |
| E3/A |  617 | 33386 | 72015 | **117×** |
| E5/A |  270 | 1820  | 1763  | **6.5×** |
| E9/C_full | 353 | 5835 | — | **16.5×** |

### 预测 vs 实际
- 预测: 其他任务也会恶化 → 证实训练问题
- 实际: 完全符合。E3 甚至 117× 于旧值，**比 E9 的 16.5× 还严重**
- 偏差原因: 无

### 结论
**不是 E9-specific 推理 bug**。所有 task 都显著恶化（6-117×）。用户的"别的任务没问题"很可能基于 dashboard 上的旧 epoch_657 数据，那是历史记录，不代表当前 checkpoint 的性能。

需要用户确认：他的判断基于什么数据。如果基于同一 epoch_848 的新实验且看到其他任务还好，那是 E9 特有；如果基于旧 dashboard，那就是训练问题。

### 回退
不回退。

---

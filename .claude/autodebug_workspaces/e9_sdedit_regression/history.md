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

## Iteration 4 (INFRA) — 2026-04-20 22:35:00
### 类型
基础设施 + 观测缺口补全：读取训练 loss 曲线，验证 H-T1（训练发散）

### 假设与预注册
- 假设 H-T1: 训练在 epoch 657 → 848 之间发散（loss 上升或震荡）
- 预期结果:
  - 若发散: per-epoch mean loss 在 epoch 657 之后上升或剧烈震荡
  - 若不发散: loss 持续下降，说明恶化不是 loss-level 问题，而是 train/eval objective gap（过拟合）
- 检验指标: `awk` 聚合 `train.log` 的 per-epoch mean(loss, loss_velocity)

### 方向
1. `git log` + `git status` 看最近代码改动
2. 定位 `work_dirs/hymotion_m2m_v2_uncond_global_046b/20260414_194625/train.log`
3. 用 awk 聚合 per-epoch loss，检查 epoch 272→850 全程走势
4. 同时收集 loss_weights config（smoothness=0.5, fk_cons=0.1, trans_dim=5.0）

### 执行结果
**关键事实 1：训练 log 起点 epoch 272**
- 这是 resume 的 run（20260414_194625 启动，从 epoch 272 开始 ≈ 前一个 run 的结束）
- log 一直记到 epoch 851（22:16 出了 checkpoint-epoch_850，训练仍在继续 ~15 min/epoch）

**关键事实 2：per-epoch mean loss 持续下降**
```
epoch 272: mean_loss=0.0235  loss_velocity=0.0233
epoch 511: mean_loss=0.0191  loss_velocity=0.0189
epoch 691: mean_loss=0.0185  loss_velocity=0.0183
epoch 811: mean_loss=0.0185  loss_velocity=0.0183
epoch 848: mean_loss=0.0179  loss_velocity=0.0177
epoch 851: mean_loss=0.0174  loss_velocity=0.0172
```
**无任何上升或发散**。loss 从 0.0235 → 0.0174 单调下降（有 noise 但 trend 显著）。

**关键事实 3：loss weights 配置**
`_base_hymotion_m2m_v2_046b.py`:
- velocity_weight=1.0（主目标 flow-matching velocity MSE）
- motion_smoothness_weight=0.5
- fk_consistency_weight=0.1 (warmup 2000 steps)
- trans_dim_weight=5.0（translation 维度 weighted smooth_l1）
- x1_weight=0.0（直接 x1 监督关）

log 中观察到：
- loss_smoothness ~= 0.0001（smoothness term 几乎完美）
- loss_fk_consistency ~= 1e-7（fk term 早已收敛到 0）
- 主要贡献来自 loss_velocity

### 预测 vs 实际
- 预测: 若发散，能看到 loss 上升 → **不符合**
- 实际: loss 单调下降但 eval 暴涨 → 经典 **train/eval objective gap**
- 偏差原因: 训练目标是 velocity MSE（在 clean training distribution 上的一阶导），**不是** eval 指标关心的 acceleration jitter。Clean data 上 velocity MSE 下降 ≠ LQ inpaint 场景下 acceleration 良好。

### 结论
- **H-T1 被推翻**：训练没有发散，loss 完美下降。
- **根因转向 H-T2 / 新假设 H-T6（train/eval 分布错位）**：
  - 模型过拟合 clean motion 的局部精细结构 → 但 eval 是 LQ + mask 的 inpaint 场景，模型从未见过此 context，输出出现高频抖动
  - 这解释了为什么"**所有 task 同时恶化**"（6–117×）：它们共享同一个 uncond_global backbone，backbone 越过拟合训练分布，对 OOD 场景（eval 的所有 task 都不是纯 clean reconstruction）鲁棒性越差
  - 也解释了为什么 `loss_smoothness` 极小却不能防止 jitter：smoothness loss 只约束**训练数据**的 smoothness（本来就是平滑的），没有任何 signal 约束 OOD inference 下的 smoothness
- Training 不需要动，问题出在 **checkpoint 选择** 和 **inference 时缺少 smoothness prior**

### 回退
不回退（本轮只读分析，无代码改动）。

---

## Iteration 5 (ROOT CAUSE FIX) — 2026-04-21 00:18:00
### 类型
P3 假设驱动修改 — 定位并修复根因 bug

### 假设与预注册
- 假设 H-B1: Pipeline uncond 推理分支把 `ctxt_input` pad 到 128 tokens（用 null embedding 广播），而训练 uncond 分支只给 1 个 null token。sequence length + attention mask 的 OOD 分布偏移导致所有 uncond model 推理输出崩溃。
- 预期结果:
  - 若为根因: 修复后 E3/A 3-sample quickcheck jitter 从 35000+ 回到 <1000（接近或优于旧 epoch_657 baseline ~611）
  - 若非根因: jitter 仍在 10000+ 量级

### 方向
- 做代码交叉审计：trainer.py (uncond branch) vs pipeline.py (uncond branch)
- 发现分布不一致 → 修 pipeline 的 uncond 分支回 (B, 1, -1)

### 修改
`hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:197-203`:

Before (bugged, 2026-04-20 21:05):
```python
ctxt_input = null_ctxt_input.expand(B, pad_len, -1).contiguous()  # (B, 128, 4096)
ctxt_length = torch.zeros(B, dtype=torch.long, device=device)     # length=0
ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)        # all False
```

After:
```python
ctxt_input = null_ctxt_input.expand(B, 1, -1).contiguous()        # (B, 1, 4096) match training
ctxt_length = torch.ones(B, dtype=torch.long, device=device)      # length=1
ctxt_mask_temporal = _length_to_mask(ctxt_length, 1)              # all True on 1 token
```

### 执行结果
E3/A 3-sample quickcheck (同样本 + 同 seed 下对比):

| Model | Before fix (iter 3/4) | After fix (iter 5) | Ratio |
|---|---:|---:|---:|
| uncond_local  | 35624 | **412** | **86×↓** |
| uncond_global | 44951 | **387** | **116×↓** |

对比 baseline:
- 旧 epoch_612/657 baseline: 611/617
- 新 epoch_836/858 + fixed pipeline: 412/387 → 比旧 ckpt **更好**

其他指标同步改善：
- mpjpe_masked: 1.0 → 0.02-0.03 (30-50× 改善)
- foot_skating_ratio: 0.6-0.8 → 0.01-0.02

### 预测 vs 实际
- 预测: jitter < 1000 → ✅ 完美符合
- 实际: jitter 甚至比旧 baseline 还低 (412/387 vs 611/617)
- 偏差原因: 无 — 训练确实在 progress，只是被推理 bug 掩盖了

### 结论
**根因锁定为 Pipeline uncond 分支的 context token OOD bug**：
1. 训练 uncond 时 `ctxt_input=(B,1,4096)` + `ctxt_mask=[True]`
2. 推理 uncond 时（bugged）`ctxt_input=(B,128,4096)` + `ctxt_mask=[False]*128`
3. 这是 2026-04-20 21:05 改 caption 分支 pad 到 128 时引入的回归
4. Uncond 模型从未见过 128-token context（全 False mask），结果在 attention 中产生 OOD 行为
5. 所有 uncond-based task (E2/E3/E5/E9) 同时受影响，与实验观察完全吻合

**训练完全无问题** — loss 单调下降是真的在学进步。bug 是推理代码在本 session 中引入的。

### 回退
不回退，保留修复。下一步是：
- 跑 E3 完整 50-sample 验证（in progress，后台 bp8w7qgg1）
- 再跑 E9 验证 SDEdit/sliding-window 修复后也正常（因为也经过 uncond pipeline）
- 更新 dashboard 全量重跑 E2/E3/E5/E9 × uncond_local/uncond_global
- Caption 推理代码的 CFG 分支（pipeline.py 224-227）也要审：force_mask=True 时 trainer 是否也只广播 null 1 token？

---

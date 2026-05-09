# Debug Insights
> status: done
> iteration: 5
> best_result: **ROOT CAUSE FOUND AND FIXED** — Pipeline uncond ctxt OOD bug
> last_updated: 2026-04-21 00:45:00

## 🎯 根因（已锁定并修复）

**Bug 位置**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:197-203`
**Bug 引入时间**: 2026-04-20 21:05（本 session 改 caption 推理 pad 到 128 时引入）
**修复时间**: 2026-04-21 00:18

### 现象 vs 修复
- **推理时（bugged）**：Uncond 模型的 `ctxt_input` 被 pad 到 128 tokens（广播 null embedding）+ `ctxt_mask_temporal` 全 False（length=0）
- **训练时**：Uncond 模型的 `ctxt_input` 只有 1 token（单个 null）+ `ctxt_mask_temporal` 全 True（length=1）
- **结果**：所有 uncond model 推理被喂了训练从未见过的输入分布，attention 中产生 OOD 行为

### Fix
把 pipeline uncond 分支改回 `(B, 1, -1)` + `length=1` + `mask all True`，与 trainer 第 212-215 行严格对齐。

## 验证结果（E3, 50 samples × 4 setting × 2 model）

| Setting | uncond_local (before→after) | uncond_global (before→after) | vs 旧 epoch_612/657 |
|---|---:|---:|---:|
| A | 35624 → **484** (74×↓) | 44951 → **534** (84×↓) | 比旧 611/617 更好 ✓ |
| B | 25206 → **435** (58×↓) | 29114 → **514** (57×↓) | 持平 ~513/571 |
| C | 53240 → **608** (88×↓) | 66378 → **594** (112×↓) | 比旧 777/774 更好 ✓ |
| D | 43395 → **509** (85×↓) | 55643 → **536** (104×↓) | 比旧 827/828 更好 ✓ |

**训练全程正常**：loss 从 epoch 272 (0.0235) → 850 (0.0174) 单调下降；新 ckpt 在修复 pipeline 下确实比旧 ckpt 更好。

## 结论（推翻之前所有假设）
- ~~H-T1: 训练发散~~ → 训练正常
- ~~H-T2: 数据/config 改动~~ → 无关
- ~~H-T3: Checkpoint save bug~~ → 无关
- ~~H-T6/H-T7: train/eval objective gap~~ → 无关
- ✅ **H-B1 (Iter 5)**: Pipeline uncond ctxt_input 分布 OOD — **根因**

## Dashboard 状态
- DB 备份: `eval_dashboard.db.bak_after_fix_20260421_003xxx`
- E3 所有 8 条 (uncond_{local,global} × A/B/C/D) 已更新到 fixed-pipeline 的结果
- Model 表 epoch: uncond_local=836, uncond_global=858

## 后续建议（另起 TODO，不属于本 autodebug）
1. **重跑所有 uncond-based task（E2, E5, E9, 其他）** — 全都受同一个 bug 影响，需要全面更新 dashboard
2. **审 caption 分支 CFG 的 uncond arm（pipeline.py 224-227）** — 当 `force_mask=True` 时 trainer 的 `mask_text_cond` 行为（bundle.py 213-217）是 `expand(*ctxt.shape)` 即保持 (B, 128, 4096) 形状，这部分其实是对的，但需要确认 ctxt_mask_temporal 是否保持原 caption 的长度 mask 而非全 False
3. **E9 SDEdit / sliding-window** — 现在应该能验证在 fixed pipeline 下 SDEdit 和 sliding-window 是否确实有效（之前的 50-100× jitter 掩盖了它们的真实效果）
4. **把 pipeline uncond 修复 commit** — 这是重要 bug fix，单独一个 commit，附 root cause 分析

## 关键教训
- **train/eval 分布对齐是强约束**：即使修复一个分支（caption）的 bug（12-20 token → 128），不能忘了交叉审另一个分支（uncond）
- **诊断思路**：当训练 loss 正常但 eval 崩坏时，root cause 大概率在"推理路径和训练路径的某个细节不对齐"（shape、mask、dtype、null handling 等）
- **跨 task 同步恶化 = 共享代码 bug**：所有 uncond-based task 同时烂的模式，几乎必然指向 uncond pipeline 的共享路径而非 task-specific 代码

## 基础设施资产（保留供后续使用）
- `test_quick.py` — 单 sample × 多 ckpt 快速扫描
- `parse_log_to_partial_json.py` — 从 run.log 中途提取 partial 结果
- `work_dirs/e3_after_fix_20260421/` — E3 fixed 的完整 50-sample 结果
- `backups/iter_{1..5}/` — 各迭代的 workspace 状态备份

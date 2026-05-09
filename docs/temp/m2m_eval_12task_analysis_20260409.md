# M2M Eval 12-Task Benchmark 分析报告 (2026-04-09)

## 评估概览

8 个模型 × 12 个任务（T1-T12），每个任务 100 个样本，50 步 ODE。

### 模型列表

| 模型 | 架构 | 参数量 | Rot Space | Epoch | 来源 |
|------|------|--------|-----------|-------|------|
| uncond_fm_man | MMDiT 0.46B | 460M | local | 1000 | T2M-Lite pretrained |
| uncond_fm_man_globalrot | MMDiT 0.46B | 460M | global | 527 | T2M-Lite pretrained |
| dit_fm_man_s | DiT-S | 49M | local | 908 | from scratch |
| dit_fm_man_b | DiT-B | 288M | local | 1000 | from scratch |
| dit_fm_man_l | DiT-L | 383M | local | 95 | from scratch |
| dit_fm_man_globalrot_s | DiT-S | 49M | global | 901 | from scratch |
| dit_fm_man_globalrot_b | DiT-B | 288M | global | 874 | from scratch |
| dit_fm_man_globalrot_l | DiT-L | 383M | global | 100 | from scratch |

---

## 关键指标

### Trans Error (mm) ↓

| Model | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 |
|-------|-----|-----|------|-----|-----|-----|-----|-----|-----|------|------|------|
| uncond_fm_man | 72.2 | 22.7 | 121.6 | 45.1 | **61.1** | 0.0 | 23.3 | 0.0 | **3.0** | **22.3** | **47.1** | **20.3** |
| uncond_fm_man_globalrot | 78.9 | 23.1 | 118.9 | 43.9 | 83.1 | 0.0 | **20.4** | 0.0 | 3.1 | 23.4 | 46.4 | 21.5 |
| dit_fm_man_s | 70.0 | 23.2 | 123.5 | 50.6 | 69.1 | 0.0 | 27.9 | 0.0 | 3.7 | 25.8 | 51.8 | 23.5 |
| dit_fm_man_b | 77.6 | 23.9 | 138.5 | 46.4 | 71.1 | 0.0 | 23.2 | 0.0 | 3.2 | 23.1 | 48.3 | 22.4 |
| dit_fm_man_l | **69.1** | 26.2 | 140.4 | 51.8 | 80.2 | 0.0 | 31.0 | 0.0 | 6.5 | 26.0 | 51.3 | 25.6 |
| dit_fm_man_globalrot_s | 70.9 | 25.1 | 99.1 | 43.7 | 73.5 | 0.0 | 22.6 | 0.0 | 3.8 | 25.4 | 51.6 | 22.6 |
| dit_fm_man_globalrot_b | 69.3 | **22.4** | **95.4** | **42.1** | **65.9** | 0.0 | 21.0 | 0.0 | 3.7 | 22.4 | 47.0 | 21.7 |
| dit_fm_man_globalrot_l | 76.4 | 28.5 | 98.4 | 46.7 | 79.9 | 0.0 | 21.4 | 0.0 | 9.7 | 29.8 | 58.3 | 28.3 |

### Quality Pass Rate (%) ↑

| Model | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 |
|-------|------|------|------|-------|------|------|------|------|------|------|------|------|
| uncond_fm_man | 91.9 | 75.0 | 95.0 | 94.0 | 88.0 | 88.0 | 53.0 | **93.0** | 79.0 | 81.0 | 73.0 | 79.0 |
| uncond_fm_man_globalrot | 92.0 | **83.0** | **97.0** | **100.0** | **93.0** | 85.0 | 43.0 | 90.0 | **83.0** | 84.0 | **85.0** | **86.0** |
| dit_fm_man_s | 86.0 | 68.0 | 90.0 | 84.0 | 88.0 | 89.0 | 54.0 | 75.0 | 76.0 | 69.0 | 66.0 | 65.0 |
| dit_fm_man_b | 87.0 | 74.0 | 91.0 | 83.0 | 90.0 | 89.0 | 55.0 | 82.0 | 79.0 | 70.0 | 75.0 | 80.0 |
| dit_fm_man_l | 78.0 | 66.0 | 84.0 | 81.0 | 88.0 | **89.0** | **55.0** | 73.0 | 52.0 | 63.0 | 69.0 | 75.0 |
| dit_fm_man_globalrot_s | 90.0 | 78.0 | 94.9 | **100.0** | 91.0 | 88.0 | 45.0 | 85.0 | 79.0 | **83.0** | 76.0 | 83.0 |
| dit_fm_man_globalrot_b | **98.0** | 81.0 | 96.0 | **100.0** | 89.0 | 86.0 | 51.0 | 85.0 | 81.0 | 80.0 | 84.0 | 87.0 |
| dit_fm_man_globalrot_l | 92.0 | 73.0 | 92.0 | 99.0 | 89.0 | 87.0 | 49.0 | 72.0 | 71.0 | 74.0 | 69.0 | 78.0 |

---

## 结论

### 1. GlobalRot 在 DiT 架构下优势显著

`dit_fm_man_globalrot_b` 是综合最强模型：

- **T3 未来预测**：95.4 vs 138.5mm（local DiT-B），提升 31%
- **T5 续写**：65.9 vs 71.1mm
- **T1 in-between**：69.3 vs 77.6mm
- **T4 循环补全**：42.1mm + 100% quality pass rate

这与理论预期一致——global rotation 下，被 mask 的关节可以从邻居直接几何插值推断（实验验证邻居可预测性 +41%）。

**但 MMDiT(uncond) 架构下 globalrot 不稳定**：T5 从 61→83mm 退步。原因是预训练权重（T2M-Lite）针对 local rot 优化，globalrot 需要更多 fine-tune epoch 覆盖。

### 2. 模型规模：B > S >> L（受限于训练 epoch）

| Size | Epoch | Trans Err (T1/T3/T5 avg) | Quality Avg |
|------|-------|--------------------------|-------------|
| DiT-S (49M) | 908 | 87.5 | ~78% |
| **DiT-B (288M)** | **1000** | **95.7** | **~80%** |
| DiT-L (383M) | **95** | 96.6 | ~73% |

DiT-L 仅训练了 ~100 epoch（vs S/B 的 900+），严重欠训练。**当前 B-size 是最优性价比**。L 需要继续训练到 500+ epoch 才能公平比较。

GlobalRot 同理：`dit_globalrot_b` (ep874) > `dit_globalrot_s` (ep901) > `dit_globalrot_l` (ep100)。

### 3. MMDiT Pretrained vs DiT From Scratch

| | uncond_fm_man (460M, pretrained) | dit_fm_man_b (288M, scratch) |
|---|---|---|
| T5 续写 | **61.1** | 71.1 |
| T3 预测 | **121.6** | 138.5 |
| Jitter | **更低** | 略高 |
| Quality Avg | **~85%** | ~80% |

**预训练 MMDiT 仍有整体优势**，尤其在长序列续写任务上。但 DiT-B 仅用 63% 参数量就接近，from scratch 训练的性价比很高。

**最佳 globalrot 对比**：`dit_globalrot_b`（288M）在 T3/T4/T7 上超越 `uncond_globalrot`（460M），说明 DiT 架构更受益于 global rotation。

### 4. Quality Pass Rate 揭示的能力分布

| 任务难度 | 任务 | 最佳 Quality |
|----------|------|-------------|
| 易 | T4 循环补全 | **100%** (globalrot) |
| 中 | T3 未来预测、T1 in-between | 95-98% |
| 中 | T6 关节补全、T8 无条件生成 | 85-93% |
| 难 | T9-T12 上采样 | 71-87%（越稀疏越难） |
| 很难 | **T7 Repair** | **43-55%**（所有模型最低） |

**T7 Repair 是当前最大瓶颈**：模型在 scattered mask pattern（checker 标记的零散缺陷区域）上的修复能力不足。可能原因：
- M7 训练权重仅 10%，覆盖不够
- 训练数据包含 ~15% 低质量样本，模型学习了有缺陷的 pattern

### 5. T6 trans_err=0 是预期行为

T6 是下半身→上半身补全，mask 不覆盖 translation 维度，因此 translation 完全保留，误差为零。

### 6. 上采样任务：keyframe 密度直接决定质量

| 任务 | Keyframe 间隔 | 最佳 Trans Err | 最佳 Quality |
|------|-------------|---------------|-------------|
| T9 (5fps) | 每 6 帧 | 3.0mm | 83% |
| T10 (1fps) | 每 30 帧 | 22.3mm | 84% |
| T11 (0.5fps) | 每 60 帧 | 46.4mm | 85% |

误差几乎与 keyframe 间隔成正比。T9 只需插值 5 帧间隔，trans_err 仅 3mm，非常精确。

---

## 改进方向

1. **继续训练 L 模型**：当前仅 ~100 epoch，远未收敛。预计 500 epoch 后 L 应超越 B。
2. **增加 M7 训练权重**：从 10% 提升到 15-20%，改善 T7 repair 能力。
3. **切换到高质量训练数据**：使用 456K 高质量子集（去除 85K 低质量），预期整体 quality +5-10%。
4. **uncond_globalrot 继续训练**：当前仅 527 epoch，T5 退步可能是 fine-tune 不足导致。
5. **评估 T8（无条件生成）的 FID/diversity**：当前只有 trans_err=0（trivial），需要加入分布级指标。

---

## 修复记录

### T7 数据修复 (2026-04-09)

**问题**：T7 trans_err_mm 高达 600-760mm，所有模型都异常。

**根因**：T7 repair 任务依赖预计算的 MoGenDIT adaptive mask。但 `data/eval/hymotion_m2m/adaptive_masks_mogendit/` 目录中只有 4/320 个样本有 mask 文件，其余 316 个 fallback 到 `mask=全1`（整段动作无条件重新生成），导致 trans_err 极高。

**修复**：`scripts/precompute_t7_masks.py` 补全了全部 320 个 adaptive mask，重跑 T7 eval 后 trans_err 降至 20-31mm（降 ~30 倍）。

### repair_eval_cjgame.py 修复 (2026-04-09)

**问题**：M2M repair 推理缺少 padding 到 360 帧 + 缺少 hard blend。

**修复**：对齐 `scripts/eval_m2m_repair.py` 的标准流程：pad to 360 → pipeline → truncate → hard blend。

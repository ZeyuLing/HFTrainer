# Repair Benchmark 调试记录

**日期**: 2026-04-02
**状态**: 排查进行中

---

## 关键实验结论

### 1. 无条件生成通过率（模型基线能力）

| Config | 通过率 | 主要失败项 |
|--------|--------|-----------|
| uncond_fm_man (local rot) | 63% (19/30) | foot_sliding(11) |
| uncond_fm_man_globalrot (global rot) | 80% (16/20) | — |

**结论**: 模型本身生成的动作基本能通过 checker，尤其 global rot 80%。

### 2. SDEdit 照抄问题输入

| 方式 | Local Rot | Global Rot |
|------|-----------|------------|
| 无条件生成 | 63% | 80% |
| SDEdit s=1.0 (纯噪声) | 66% | 40% |
| SDEdit s=0.5 | 33% | 25% |

**关键发现**: Global rot 无条件 80% vs SDEdit s=1.0 40%。两者都从纯噪声开始，差异在于 SDEdit 路径有 replacement guidance (skip_last) 注入了原始有问题的帧。

### 3. Bug 定位：replacement guidance 污染

| 方式 | Global Rot 通过率 |
|------|------------------|
| mask=全1, replacement=none | **75%** |
| mask=frame_flag(86%), replacement=skip_last | **40%** |

**根本原因**: `replacement_guidance='skip_last'` 在每步去噪时把 ~14% 的 keep 帧替换回原始值。但这些 keep 帧来自有问题的输入，它们的值被强制注入去噪过程，污染了相邻帧的生成。

### 4. 之前发现的 Bug：global rot 缺少 global→local 转换

无条件生成测试中 global rot 最初只有 20% pass（大量 arm_penetration + candy_wrapper），修正后恢复到 80%。原因：生成结果是 global rot6d 空间，直接按 local rot6d 保存导致畸形。

---

## 修正方向

1. **不用 SDEdit**：M2M 应该用 VACE completion（mask 坏帧，从纯噪声生成），不是 SDEdit（加噪去噪）
2. **不用 replacement guidance**：replacement 会把有问题的原始帧注入去噪过程
3. **合理的 VACE mask**：需要有足够多的好帧做条件引导，但当前 adaptive mask 标记了 86%+ 的帧
4. **或者全帧 mask + 无条件生成**：作为 baseline，通过率就是模型的无条件生成能力

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `scripts/eval_repair_benchmark.py` | 评测主脚本 |
| `output/repair_benchmark/uncond_gen_fm_local/repaired/` | Local rot 无条件生成结果 (600个) |
| `output/repair_benchmark/uncond_gen_fm_globalrot/repaired/` | Global rot 无条件生成结果 |
| `output/repair_benchmark/eval_report.json` | 旧报告 (SDEdit s=0.5, 待更新) |

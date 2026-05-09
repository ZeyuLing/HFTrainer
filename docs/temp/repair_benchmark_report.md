# Low-Quality Motion Repair Benchmark Report

**日期**: 2026-04-02
**数据**: 600 cases（6类问题 × 100），从 `low_quality.json`（85k条）分层采样单一问题类型
**推理方式**: M2M 使用 SDEdit (s=0.5) + frame-level mask + replacement guidance (skip_last)，匹配 `eval_sparse_keyframe_mib.py` 的推理逻辑
**报告 JSON**: `output/repair_benchmark/eval_report.json`
**可视化**: `motion_annot_web/m2m_repair_compare` (port 8081)，选择 `repair_benchmark`

---

## 测评模型（5 个配置）

| 标签 | 说明 | 推理方式 |
|------|------|----------|
| `uncond_fm_man` (fm) | FM scheduler, local rot | SDEdit s=0.5 + skip_last |
| `uncond_jit_man` (jit) | JiT scheduler, local rot | SDEdit s=0.5 + skip_last |
| `uncond_fm_man_globalrot` (fm_gr) | FM scheduler, global rot | SDEdit s=0.5 + skip_last |
| `uncond_jit_man_globalrot` (jit_gr) | JiT scheduler, global rot | SDEdit s=0.5 + skip_last |
| `mogendit_ada_denoise` (mgdit) | MoGenDIT 0.1B | ada_denoise (step=10) |

---

## M2M 推理流程

完全对齐 `scripts/eval_sparse_keyframe_mib.py:run_completion`：

1. **Mask 构建**: 从 MoGenDIT adaptive mask 的 raw `joint_mask (T, 22)` 构建 frame-level mask — 任何关节被标记的帧整帧 mask=1，否则 mask=0
2. **clean_motion**: 原始运动 normalize 后作为 SDEdit 加噪起点
3. **VACE context**: `src_motion = clean_motion * (1 - mask)` — inactive 部分（mask=0）保留原始值，reactive 部分（mask=1）为全 0
4. **Pipeline**: `HyMotionM2MPipeline(sdedit_strength=0.5, replacement_guidance='skip_last')`
5. **后处理**: `combined = original * (1 - mask) + model_output * mask`

注：由于 adaptive mask 通常标记了大部分帧（~86%），实际效果接近全帧 SDEdit，通过加噪强度控制修复力度。

---

## Fix Rate（该类型 checker 修复通过率）

| 问题类型 | FM | JiT | FM_GR | JiT_GR | **MoGenDIT** |
|---|---|---|---|---|---|
| foot_sliding | 23% | **73%** | 24% | **75%** | 63% |
| jitter | 67% | 9% | **77%** | 16% | **96%** |
| candy_wrapper | 6% | 4% | 11% | 14% | **26%** |
| joint_jump | 39% | 41% | 48% | 42% | **97%** |
| rotation_velocity | 13% | 12% | **48%** | **55%** | 71% |
| neck | 0% | 6% | 20% | 32% | **83%** |
| **总体** | 24% | 24% | **37%** | **39%** | **73%** |

## Overall Pass Rate（修复后所有 checker 全通过率）

| 问题类型 | FM | JiT | FM_GR | JiT_GR | **MoGenDIT** |
|---|---|---|---|---|---|
| foot_sliding | 22% | 11% | 24% | 16% | **63%** |
| jitter | 62% | 8% | **70%** | 14% | **91%** |
| candy_wrapper | 6% | 2% | 11% | 4% | **25%** |
| joint_jump | 35% | 30% | **40%** | 32% | **82%** |
| rotation_velocity | 9% | 3% | **38%** | 13% | **66%** |
| neck | 0% | 3% | 18% | 13% | **76%** |
| **总体** | 22% | 10% | **33%** | 15% | **67%** |

## Per-Model Summary

| 模型 | Total | Orig.Fail | Rep.Pass | Improved | Degraded | Pass Rate |
|---|---|---|---|---|---|---|
| **MoGenDIT** | 600 | 600 | 403 | 403 | 0 | **67.2%** |
| **FM_GR** | 589 | 589 | 194 | 194 | 0 | **32.9%** |
| FM | 589 | 589 | 128 | 128 | 0 | 21.7% |
| JiT_GR | 589 | 589 | 91 | 91 | 0 | 15.4% |
| JiT | 589 | 589 | 56 | 56 | 0 | 9.5% |

---

## 关键发现

1. **MoGenDIT 总体最优**（67% pass rate），在 jitter(96%)、joint_jump(97%)、neck(83%) 上表现突出
2. **Global rotation 显著优于 local rotation**：FM_GR 33% vs FM 22%（+50%），JiT_GR 15% vs JiT 10%（+60%）
3. **FM scheduler 优于 JiT**：FM_GR 33% vs JiT_GR 15%
4. **foot_sliding 是 JiT 的强项**：JiT 73%、JiT_GR 75%，甚至超过 MoGenDIT 63%
5. **candy_wrapper 对所有模型都很难**：最高仅 MoGenDIT 26%，M2M 最高 14%
6. **所有模型 Degraded=0**：没有把好的变差，只有修好或没修好
7. **M2M 最佳配置**：FM_GR（global rotation + FM scheduler）以 33% pass rate 领先其他 M2M 变体

---

## 运行方式

```bash
# Phase 1-3: MoGenDIT adaptive mask + ada_denoise（单 GPU）
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_repair_benchmark.py --phase mogendit

# Phase 4: M2M SDEdit（4 GPU 并行）
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_fm_man --sdedit-strength 0.5
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_jit_man --sdedit-strength 0.5
CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_fm_man_globalrot --sdedit-strength 0.5
CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_jit_man_globalrot --sdedit-strength 0.5

# Phase 5-6: Checker + Report（CPU）
python3 scripts/eval_repair_benchmark.py --phase report
```

## 文件索引

| 文件 | 说明 |
|------|------|
| `scripts/eval_repair_benchmark.py` | 评测主脚本 |
| `output/repair_benchmark/sample_list.json` | 600 case 采样列表 |
| `output/repair_benchmark/adaptive_masks/` | MoGenDIT adaptive mask（600 个） |
| `output/repair_benchmark/<config>/repaired/` | 各模型修复结果 NPZ |
| `output/repair_benchmark/eval_report.json` | 完整评测报告 JSON |

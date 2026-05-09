# E9 推理 Bug 修复 - 实测效果 (2026-04-23)

## 测试配置

- 模型: `uncond_local` + `uncond_global` (v2, 0.46B, caption models 不 run 因为 E9 is not caption-aware)
- Baseline: `stablemotion` (BrokenAMASS ckpt, detect-and-fix)
- 数据: E9 Motion Repair datalist (215 samples, 16 defect types)
- 评估指标: jitter_pos (↓), qc_pass rate (↑), foot_skating_ratio (↓)

## 合并结果

| Setting | uncond_local jitter / QC / foot | uncond_global jitter / QC / foot |
|---|---|---|
| A_adaptive_inpaint (fixed) | 1518 / **25%** / 0.19 | 1294 / 1.4% / 0.17 |
| **D_ada_denoise_t010 (fixed)** | **546 / 82% / 0.07** ← 🎯 | **490 / 17% / 0.03** |
| D_strict_mask_d2_b3 (baseline) | 1504 / 61% / 0.22 | 1356 / 1.4% / 0.18 |
| D_strict_mask_d2_b3_edit | 1769 / 42% / 0.24 | 1822 / 0.9% / 0.20 |
| D_strict_mask_d2_b3_smooth1 | 1467 / **64%** / 0.22 | 1207 / 1.4% / 0.17 |
| D_strict_mask_d2_b3_smooth2 | 1525 / 64% / 0.22 | 1240 / 1.9% / 0.17 |
| stablemotion (baseline) | 3082 / 8.8% / 0.27 | - |

## 关键发现

### 1. D_ada_denoise_t010 修复是最大赢家
- Stage 1 旧设计（`mask=frame0_only` + 全帧生成）→ M2M 把这当 T2M 生成，输出与 LQ 无关
- Stage 1 新设计（`mask=all` + SDEdit τ=0.5）→ 把 LQ 拉回 manifold，change = |LQ − stage1| 精确反映 defect
- **在两个模型上都是最优**：uncond_local jitter -64%（1504→546），QC ×1.3（61%→82%）；uncond_global 同样 jitter 从 ~1356→490
- `foot_skating` 也是各自模型里最低（uncond_local 0.07 vs baseline 0.22）

### 2. Global rotation 模型在 QC 下普遍不通过
- uncond_global 几乎所有 setting 的 QC pass rate 都 <2%，而 uncond_local 是 42-82%
- 印证了已知的 task #205 "global rotation 模型推理质量差异"
- 但 uncond_global 的 jitter/foot_skate 数值看起来不差（甚至 jitter 更低），说明问题在 "结构性/anatomical" 而非 "kinematics smoothness"
- 假设: global rotation 空间下每个关节独立预测 6D rotation，缺乏 kinematic chain constraint → 容易出现 joint twist / candy wrapper 等 QC 能检测但 smoothness 指标不敏感的问题
- D_ada_denoise_t010 在 uncond_global 上 QC 只到 17%，但比 local 的 82% 低得多 → global rot 根本问题不是 mask 问题，是 representation 问题

### 3. A_adaptive_inpaint 修复生效但提升有限
- SDEdit τ=0.5 终于活了，uncond_local QC 从 ~20% → 25%，但远低于 D_ada_denoise 的 82%
- MoGenDIT 预缓存 adaptive mask 本身不够准（用户早已反馈）
- 建议: 将来 deprecate，用 D_ada_denoise_t010 替代

### 4. Editing mode 反直觉地加重 jitter
- 实测: uncond_local jitter 上升（1504 → 1769），QC pass rate 下降（61% → 42%）
- Reactive=LQ 告诉模型"这是 LQ 已损坏"，但 MAN training 没学到主动修复 → 输出更野性
- 结论: editing mode 不是好方向

### 5. LQ 预平滑 σ=1 是最佳 D_strict 变体
- σ=1 (5 Hz cutoff @ 30fps): uncond_local jitter 1504 → 1467, QC 61%→64%
- σ=2 过度平滑，收益归零
- 建议: 如果要用 D_strict_mask，默认开 `_presmooth_clean_sigma=1.0`

### 6. StableMotion baseline
- jitter_pos 3082 最差（比 M2M 高 5-6 倍），QC pass 8.8% 最低
- 反映训练域 (BrokenAMASS synthetic corruption) 与 HyMotion real-domain defect 的 gap
- 作为 baseline 有价值：说明 M2M 的 in-domain 训练对 E9 repair 任务是必要的

## 最佳策略推荐

### For uncond_local
- **D_ada_denoise_t010** (新修复) 作为默认 → jitter 546, QC 82%
- D_strict_mask_d2_b3_smooth1 作为 fallback 如果不想跑 2-stage

### For uncond_global
- 虽然 D_ada_denoise_t010 也是最优 (jitter 490, QC 17%)，但根本 QC pass rate 太低
- **应该优先解决 global rotation 的结构性生成问题（task #205）**，mask/SDEdit 的小改动不能弥补

## 代码变更清单

1. `hftrainer/evaluation/motion/m2m_eval_tasks.py`
   - A_adaptive_inpaint / A_adaptive_inpaint_notau: 加 `_replacement_guidance='skip_last'`
   - D_ada_denoise_*: 更新文档注释（Stage 1 算法改动）
   - D_strict_mask_d2_b3_edit / smooth1 / smooth2: 新增 3 个 variant

2. `tools/eval_m2m_v2_all_tasks.py`
   - 加 `_replacement_guidance` per-setting override（line ~2008）
   - D_ada_denoise Stage 1 重写: mask=all-1, replacement_guidance='skip_last', sdedit_tau=0.5（line ~2123-2155）
   - 加 `_gaussian_temporal_smooth` helper 和 `_presmooth_clean_sigma` kwarg（line ~535 helper, ~1971 plug-in）

3. `scripts/run_stablemotion_e9.py` (完整重写)
   - 端到端 M2M ↔ StableMotion 转换 + detect-and-fix
   - 处理 checkpoint tar.gz 格式
   - 24-joint topology 对齐（wrist → hand pad）
   - y-up ↔ z-up 轴转换

4. `scripts/stablemotion_to_dashboard.py` (新建)
   - StableMotion NPZ → 标准 dashboard format
   - 计算 jitter/bone_length/foot/QC metrics
   - 输出 flat JSON 供 data_importer.py

## 入库状态

全部已入库 dashboard (`http://localhost:8081/task/E9`):
- stablemotion × 1 setting: run_id 2762
- uncond_local × 6 settings: run_id 2763-2768
- uncond_global × 6 settings: run_id 2769-2774

可在 dashboard compare 页面直接对比这 13 个 runs。

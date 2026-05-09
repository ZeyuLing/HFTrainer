# StableMotion E9 Baseline 接入完成 (2026-04-23)

## 已完成 ✅

1. **验证 StableMotion 代码导入** — model / diffusion / feats / normalizer 全部可用
2. **解决 checkpoint 格式问题** — `stablemotion_brokenamass.pt` 实际是 `tar.gz` bundle，含：
   - `stablemotion/ema001000000.pt` (145 MB, EMA weights)
   - `stablemotion/model001000000.pt` (145 MB, non-EMA)
   - `stablemotion/args.json` (训练超参)
   加载前需 `tarfile.open(..., 'r:gz').extractall()`，已写入 `load_stablemotion()`。
3. **确定模型架构** — 从 `args.json` + state_dict 反推：
   - `in_channels=233` (= 232 body + 1 label)
   - `num_layers=8`, `num_attention_heads=8`, `attention_head_dim=64` → inner_dim=512
   - `class_cond=True` (working mode indicator as class)
   - Diffusion: DDPM 50 steps, cosine beta, `predict_xstart=1`, `sigma_small=True` → `FIXED_SMALL`
4. **解决 skeleton mismatch** — StableMotion 用 `JOINTS_EXTRACTOR['smpljoints']` 抽取 SMPL+H 的 24 joints（body 22 + left_hand 22 + right_hand 37）。M2M 只有 SMPL-22 body，缺 2 个 hand joint。解法：
   - `m2m135_to_smpldata_24()`：在 FK 得到的 (T, 22, 3) joints 后拼接 `joints[:, 20:21]` (l_wrist) 和 `joints[:, 21:22]` (r_wrist) 作为占位 hand joint。
   - 合理性：StableMotion 的 feats 管线中 hand joint 只进入 `joints_local` 通道，不参与 `foot_global` / rotZ / trajectory 计算。用 wrist 复制引入小偏差但不破坏 canonical frame 和 root trajectory。
5. **轴系转换** — M2M 用 y-up，StableMotion 用 z-up (globsmplrifke_feats.py:38 `joints[:,:,2].min()` 是 gravity）。`smpldata_y_up_to_z_up` / `smpldata_z_up_to_y_up` 做 y↔z 交换并前乘 `R_x(±90°)` 给 global_orient。
6. **Feats 前后转换** — `smpldata_to_alignglobsmplrifkefeats` (232-dim) 和 `globsmplrifkefeats_to_smpldata` 可用；添加 label=0 通道 → 233-dim 送 normalizer → StableMotion。
7. **Detect-and-Fix pipeline** — 在 `run_stablemotion_detect_fix()` 中镜像 `sample/fix_globsmpl.py` 的最小化版本：
   - Batch size 1, no MC averaging, no ensemble, no SITS, no foot-lock guidance
   - Detect: corrupt label channel → p_sample_loop → 二值化 label (阈值 0.5) → ±1 膨胀 → 末帧强制 clean
   - Fix: good frames keep, bad frames regenerate，label=-1 告知模型"要干净"
8. **全量 E9 跑通** — 215 个样本，单样本 ~2s（T=427 时），总耗时约 7 min。
9. **Dashboard-compatible NPZ 输出** — 每样本输出：
   ```
   output/eval_v2_e9_stablemotion_20260423/npz/XXXXX.npz
     ├─ lq_motion_135    (T, 135) LQ 原输入
     ├─ hq_motion_135    (T, 135) StableMotion 修复后
     ├─ stablemotion_label (T,) uint8，模型检测到的 corrupted 帧
     ├─ prompt_id, defect_type, source_path
   ```

## 已验证

Feasibility sample (T=427): 检测到 23 corrupted frames (5.4%)，推理耗时 2.1s，LQ/HQ 差异 mean=0.0047 / max=0.70（合理：大部分帧未被改动，少部分帧重生成）。

## 已知 limitation

- **Domain gap**：StableMotion 训练数据 = BrokenAMASS (AMASS + 合成退化)，与 E9 的 HyMotion 退化分布不同。baseline 结果仅供对比参考，不做为产品方案。
- **Hand joint 近似**：用 wrist 复制填充 23rd/24th joint position。StableMotion 输出的 hand joint 被丢弃（M2M 表示中无 hand），所以只影响推理时 model 看到的 joints_local 通道的信息。
- **No enhancements**：默认关闭 MC averaging (ProbDetNum=0)、ensemble、SITS、foot-lock guidance。如后续需要提升质量，这些选项在 StableMotion 原脚本里都有现成实现，可加 CLI flag 启用。

## 入库 dashboard 步骤

跑完后需要：
1. 从 NPZ 的 `hq_motion_135` 生成 QC 结果（LQ vs HQ 对比）
2. 按 dashboard 期望的格式包装（参考 `tools/eval_m2m_v2_all_tasks.py` 的 NPZ schema，需要加 `sampled_motion_135`、可能的 per_sample metrics 等）
3. `motion_annot_web/eval_dashboard` 的 E9 task 的 model select 里加一个 "StableMotion-baseline" 条目

脚本位置：`scripts/run_stablemotion_e9.py`

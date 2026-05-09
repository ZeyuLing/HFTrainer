# 2026-04-22 Evening Plan — E9 入库、E4 修复、Caption rewrite、Global 抖动

> 用户 2026-04-22 晚指令：
> 1. E2 caption 跑完 rewrite 了吗？现在看到的还是没 rewrite 的 caption
> 2. 剧烈的抖动/跳变和明显的异常发生在 caption_global / uncond_global
> 3. E9 需要更新到网站上
> 4. E4 的问题完全没解决，仍然存在
>
> 加上之前提到但还没全部做完的：
> 5. Part-level control 增加更多 setting
> 6. 所有过渡片段（E8-D, E14/15/16）应按规则动态计算帧数
> 7. E9 跑通 StableMotion 作为对比
> 8. KIMODO E14 推理错误修复 + 补齐其他 task

---

## 0. 执行优先级

按"用户痛点 + 互相依赖"排序：

| P | 任务 | 原因 |
|---|---|---|
| P0 | **E9 入库 dashboard** | 最新全量结果 uncond_local 83%/71%/60% 已出，用户想立刻看 |
| P0 | **E2 caption rewrite 跑起来** | 老 caption "pass a ball" 根本不可能生成合理 E2 结果；新 rewrite 数据已就绪但模型没重跑 |
| P1 | **E4 修复** | 用户 2 次反馈未解决，必须彻底解决 |
| P1 | **诊断 global rot 模型异常** | caption_global/uncond_global 在 E2/E9 均表现差（E9 global <16%, E2 mpjpe 0.34 but jitter 3526） |
| P2 | **E8-D / E14/15/16 过渡帧数动态化** | 涉及几个 task 的质量，影响面广 |
| P2 | **E10 Part-level 加 setting** | 数据集可能也要扩展 |
| P3 | **StableMotion baseline** | 外部对比，独立工作 |
| P3 | **KIMODO E14 修复** | baseline 质量问题 |

---

## 1. E9 入库 Dashboard (P0, 预计 1 小时)

### 数据状态
```
output/eval_v2_e9_20260422/
├── uncond_local/eval_v2_YYYYMMDD_HHMMSS.json  ← 7 settings × 215 samples
├── uncond_global/eval_v2_YYYYMMDD_HHMMSS.json ← 7 settings × 215 samples
└── logs/*.log
```

### 执行步骤

#### 1.1 拆分扁平 JSON
```bash
python3 tools/split_eval_v2_to_flat.py \
    --in-dir output/eval_v2_e9_20260422 \
    --out-dir output/eval_v2_e9_20260422/import_jsons \
    --timestamp "2026-04-22 20:00:00"
```

#### 1.2 备份 DB
```bash
cp motion_annot_web/eval_dashboard/eval_dashboard.db \
   motion_annot_web/eval_dashboard/eval_dashboard.db.bak_before_e9_newsettings_$(date +%Y%m%d_%H%M%S)
```

#### 1.3 删除旧 E9 记录（避免和新 D_* 混合）
```bash
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db "
  DELETE FROM eval_runs WHERE task_id='E9'
      AND model_id IN (SELECT id FROM models WHERE name IN ('uncond_local','uncond_global'));
"
```
（级联通过 CASCADE 删掉 sample_results 和 agg_metrics；如果没 CASCADE 要手动删）

#### 1.4 批量 import
```bash
for j in output/eval_v2_e9_20260422/import_jsons/*.json; do
    python3 motion_annot_web/eval_dashboard/data_importer.py import "$j" \
        --notes "E9 D_* redesign 2026-04-22 (ada_denoise, strict_mask, qc_mask)"
done
```

#### 1.5 验证
- 打开 http://host:8081/task/E9
- 确认 14 个 settings × 2 models = 28 条 run 都在
- 抽查 `D_ada_denoise_t010 × uncond_local` → pass rate 71%
- 打开任一 sample → 3D viewer 能播放（npz_path 有效）

### 风险
- **split_eval_v2_to_flat.py 可能不识别新 metric `qc_pass` / 14 个 `qc_*`**：检查 importer 是否把它们入库
- **dashboard 前端 E9 radar chart 可能没有 `qc_pass` 轴**：eval_task_registry 已加 qc_pass 到 default_metrics，但前端可能还用老列表

---

## 2. E2 Caption Rewrite 重跑 (P0, 预计 30 min eval + 10 min import)

### 诊断
- `data/eval/m2m_v2/eval_e2_inbetween_rewritten.json`：rewrite 已完成（ascii_ratio=1.00，英文完整描述）
- DB 里 E2 caption 记录：`"pass a ball"`, `"soccer"` —— 旧短 caption
- 原因：2026-04-22 16:20 那批 eval 跑时，E2 数据集是**重建前**的（那时 rewritten caption 是老短版本）
- 现在需要：**用新 rewritten 数据重跑 E2 + E5 的 caption 模型**

### 执行
```bash
# 只跑 caption 模型（uncond 不用 rewrite）
python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local caption_global \
    --tasks E2 E5 \
    --max-samples 0 \
    --device cuda:4 \
    --output-dir output/eval_v2_caption_rewrite_20260422 \
    --save-npz \
    --use-rewritten \
    --num-steps 20
# 并行：可以拆成 caption_local 跑 GPU 4, caption_global 跑 GPU 5
```

### 入库
- 删旧 caption × E2/E5 run
- Import 新 JSON

### 验证
- Sample 0 的 `text_caption` 应该是 "A person catches a ball..."，不是 "pass a ball"
- E2 mpjpe/jitter 可能改善（长 caption 给模型更多约束）

---

## 3. E4 修复（P1, 必须彻底解决）

### 现象（用户反馈 2 次）
- 黄色帧（约束锚点 sprite/ring/line）**飘浮**
- 蓝色帧（生成 mesh）在**地面**
- 两者不对齐

### 必要诊断步骤（之前缺失的）

#### 3.1 先读代码搞清楚"黄色帧"的具体来源
- `motion_annot_web/eval_dashboard/templates/task_detail.html` 里 E4 渲染路径
- `/api/source_motions/<task>/<prompt_id>` 还是专门的 E4 端点？
- "黄色" 是 `constraint sprite` / `target ring` / `end-effector trajectory`？
- 坐标是 FK joint 输出还是从 eval datalist 里读的 `target_xyz`？

#### 3.2 对比两边坐标系
- 蓝色 mesh：走 `canonicalizeGround()` 后 `smplMesh.position.y = groundOffset`
- 黄色锚点：可能是 world-space raw 坐标，没加 groundOffset 偏移
- 或者 黄色是 FK 的 bone_offsets 骨架（短 leg）vs 蓝色 mesh (长 leg) —— **就是之前 E9 高度问题的同款 bug**

#### 3.3 可能的其他 bug
- E4 数据集 `eval_e4_end_effector.json` 里 target 位置是否已做 ground-normalization？
- build_end_effector_mask 返回的 constraint_info 坐标系是 world / canonical / local？
- 推理完后 position_constraint solver 把生成的 joint 往 constraint 拉，如果 constraint 坐标错，pull 到错位置

### 实施
1. **读 task_detail.html E4 渲染分支**（估计 100-200 行）
2. **读 tools/eval_m2m_v2_all_tasks.py E4 分支** + `hftrainer/pipelines/motion/position_constraint.py`
3. **localhost 跑一个 E4 sample**，对比：
   - 输出 NPZ 里蓝色 mesh 脚底的世界 Y
   - 黄色约束点的世界 Y
   - 两者差多少 → 对应哪里的 offset 遗漏
4. **修复 offset 应用**（预期：黄色点也需要 `+ groundOffset`，或用和 mesh 同源的 LBS 算法）
5. **在浏览器里验证**（ctrl+shift+R 硬刷新）

### 如果是数据问题
- 如果发现 E4 datalist 里 target_xyz 是 raw world（未 normalize）而 mesh 是 canonical，数据集要修

---

## 4. Global Rotation 模型异常诊断（P1, 预计 1-2 小时）

### 观察到的现象（E9 + E2）
| 模型 | E2_A mpjpe_masked | E2_A jitter_pos | E9 best pass% |
|---|---|---|---|
| uncond_local | 0.34 | 776 | **83%** |
| **uncond_global** | 0.34 | **3526** | **16%** ❌ |
| caption_local | 0.43 | 2156 | -- |
| **caption_global** | 0.40 | **1840** | -- |

**Global rot 模型在 E2 jitter 高 4-5×，在 E9 修复能力差 4-5×**。

### 假设
Global rotation 数值上 std 膨胀 2-6× 倍（CLAUDE.md §Global vs Local）。推理侧可能：
- **归一化不对**：global std 更大，denormalize 后数值漂移累积
- **Pipeline 的 FK 转换精度问题**：global → local 转换可能放大数值误差
- **训练不足**：global 模型 epoch 可能不够（对比 local 的 epoch 1035，global 1031 类似）

### 诊断步骤
1. 对同一 LQ 样本 `output_denorm` 对比：
   - uncond_local：输出值范围、jitter per-frame
   - uncond_global：输出值范围、jitter per-frame
2. 检查 global 模型 `HyMotionM2MBundle.decode_motion_from_latent()`：
   - global rot6d → local rot6d 的转换是否在 `float32`？（CLAUDE.md 警告：float16 会放大 1000×）
   - 是否用了对的 mean/std_dir (`_stats_198dim_global_rot`)？
3. 对比 v1 `_globalrot` 的 E2 数字（mpjpe 0.15, jitter 270）—— 比 v2 local 还好，说明 v2 global 训练出了什么问题

### 可能的 fix
- 确认归一化文件正确
- 确认 decode 路径 dtype
- 若问题不在推理，可能是训练退化 → 建议重训 / 换 checkpoint 时间点

---

## 5. E8-D / E14/15/16 过渡帧数动态化（P2, 预计 3-4 小时）

### 现状
- E8-D: `_loop_append=60` 写死
- E14: `_cond_frames={5,15,30}`，transition = T - 2*cond 写死
- E15: `_cond_head=15`, N_transition 相对固定
- E16: `_cond_tail=15`, N_transition 相对固定

### compute_transition_length 已存在
`hftrainer/evaluation/motion/...` 或 `tools/` 里已有该函数。E14 line ~1513 调用了，E15/16 目前没用。

### 实施
1. 统一所有 transition-type task 使用 `compute_transition_length(tail_state, head_state, fps)`：
   - 基于 **pelvis 位移距离** + 初速度/末速度估算合理帧数（最少 ~30 帧，最多 ~120 帧）
   - E8-D: 基于 loop 首末 pelvis 位置差
2. 在 TaskSetting 里加 `_transition_frames='auto'` 作为默认，保留旧的硬编码值作为 fallback
3. 对每个 task 验证：
   - 跑 5 个样本，确认 transition 长度因样本而异（30-120 的范围）
   - 检查 mpjpe_masked 是否降低（目标：< 老写死的 baseline）

---

## 6. E10 Part-level Control 增加 Setting（P2, 预计 2 小时）

### 现状
3 个 setting: `A_upper`, `B_lower`, `C_spine_only`

### 拟增
- `D_left_arm`: 保留左臂 4 joints (L_Collar, L_Shoulder, L_Elbow, L_Wrist)
- `E_right_arm`: 保留右臂 (R_*)
- `F_both_arms`: 双臂
- `G_head`: 保留 Neck + Head（joint 12, 15）
- `H_trunk`: pelvis + spine1/2/3 + neck + head
- `I_legs_only`: 保留双腿

### 实施
1. 扩展 `build_part_level_mask` / `EvalTask E10 settings`
2. 每个 setting 定义 keep_joints 列表
3. **检查数据集**：`eval_e10_part_control.json` 是否需要对应扩展
4. 跑小规模 smoke（5 samples × 新 settings）
5. 全量 eval 入库

---

## 7. E9 StableMotion Baseline（P3, 预计 4-6 小时）

### 步骤
1. 找源码：
   ```bash
   find ref_repo/ -maxdepth 3 -iname '*stable*motion*'
   find /apdcephfs_cq10 -maxdepth 4 -iname '*stable*motion*' 2>/dev/null
   ```
2. 了解其 repair pipeline（LQ → HQ）
3. 写 wrapper，输出对齐 M2M 格式（198-dim rot6d + trans 或 NPZ）
4. 独立 eval 脚本（不混入 `eval_m2m_v2_all_tasks.py`）
5. 通过 `data_importer.py baseline` 入 DB

---

## 8. KIMODO E14 修复（P3, 预计 2-3 小时）

### 步骤
1. 找 KIMODO 的 E14 推理脚本（`scripts/kimodo_*` 或 `ref_repo/KIMODO/`）
2. 取 1 个 sample 的输出 NPZ，分析异常（trans 范围？poses 范围？mesh 变形？）
3. 定位坐标系或 transition 构造 bug
4. 修复后对比 M2M 同 sample
5. 看其他 KIMODO 缺失的 task，补齐

---

## 9. 验收标准

### 全量 E9 入库
- Dashboard `/task/E9` 显示 14 × 2 = 28 条 run
- B_post_replace × uncond_local QC pass = 83%
- 前端 radar 图有 qc_pass 轴

### E2 Caption 重跑
- DB 中 caption_local × E2_A sample 0 的 text_caption 是长英文，不是 "pass a ball"
- mpjpe_masked 与之前比不明显退化

### E4 修复
- 浏览器打开 E4 任意 case，**黄色约束点和蓝色 mesh 脚底同一地面**
- 切换 sample，现象保持（不是单样本巧合）

### Global 异常
- 找到根因（归一化？dtype？训练？）
- 至少定量解释 E9 uncond_global 16% vs local 83% 差距

### 过渡帧数动态化
- E14_A（5 帧 cond）输出的 transition 段长度随 sample 变化
- Mpjpe_masked（E14/15/16）下降 ≥ 10%

### E10 扩展
- 至少新增 3 个 setting（D_left_arm, E_right_arm, G_head）
- 每个 setting 上所有 v2 模型都能正常跑通

### StableMotion
- 对 E9 215 样本跑通，出 QC pass rate 数字
- 作为 baseline 条目入 DB

### KIMODO
- E14 不再有 mesh 异常，至少 5 个 sample 可视化合理
- 补 1-2 个缺失 task 的 KIMODO 数据

---

## 10. 并行度 & 工作流

1. 先做 #1 (E9 入库) + #2 (E2 caption 重跑，放 GPU 4/5 后台)
2. 并行做 #3 (E4 修复) + #4 (global 诊断)
3. 然后 #6 (E10 扩展) + #5 (过渡帧数)
4. 最后 #7/8（外部依赖）

每完成一项提醒一次，不要静默一串。

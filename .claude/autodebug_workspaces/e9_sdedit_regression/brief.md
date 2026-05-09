# Debug Brief — E9 Motion Repair Regression (SDEdit + 198-dim + sliding-window)

## 项目背景

HyMotion M2M v2 — flow-matching 动作补全/修复模型（198-dim: trans+rot6d+position）。
E9 Motion Repair 任务通过 MoGenDIT 生成的 adaptive joint mask 标注缺陷区域，
用 SDEdit 思想部分加噪再去噪 "只改缺陷" 的理想做法。

本轮修改同时做了 3 件事：
1. Adaptive mask 从 135-dim 改为 198-dim native（直接对应 v2 motion layout）。
2. 删掉了 kinematic + temporal dilation（默认不 dilate，mask 保留原 sparse 区域）。
3. SDEdit-τ=0.5 partial-noise 起点替代 "清零 + 纯噪声起点"。
4. （独立）sliding-window 让 T>360 的长样本不再 crop 到 360，以 2 窗+blend 方式处理全长。

四项修改**一次性全部上线**并 rerun 了 uncond_local + uncond_global × 3 settings × 50 样本。

## 问题描述

新实现 aggregated jitter_pos **全面恶化**（即使是 C_full_inpaint，该 setting 用 mask=all_1，
**不经过** adaptive mask / dilation / SDEdit 逻辑，仍从 353 → 1651/5835）。

对比表（aggregated mean jitter_pos）：

| Model + Setting | 旧 (135 + dilation + zero + full-noise, crop 360) | 新 (198 + no dilation + SDEdit τ=0.5, sliding-window) | 变化 |
|---|---:|---:|---:|
| uncond_local  / A_adaptive_inpaint | 1984 | 2773 | +40% |
| uncond_local  / A_adaptive_edit    | 2143 | 7717 | +260% |
| uncond_local  / C_full_inpaint     |  354 | 1651 | +370% |
| uncond_global / A_adaptive_inpaint | 1446 | 2885 | +100% |
| uncond_global / A_adaptive_edit    | 2014 | 5057 | +150% |
| uncond_global / C_full_inpaint     |  353 | 5835 | +1550% |

关键观察：C_full_inpaint 没被 Q1/Q2/Q3 的任何修改触及（mask=all_1，SDEdit τ=0），
却仍暴涨 ~5x~16x — 说明**根因在 Q4（sliding-window）或数据分布变化**（短样本 vs 长样本）。

## 用户观察

"这个变化显然是细节的错误造成的" — 指向实现 bug，不是算法层面的问题。

## 目标指标

- **主指标**: aggregated mean jitter_pos
- **辅助**: median jitter_pos（避免长尾被少数极端样本主导）、foot_skating_ratio、foot_penetration
- **目标**: 新 C_full_inpaint ≤ 旧 354，或定位并修复使 5 个 setting 全部 ≤ 旧值

## 可能的解决方向

1. **SDEdit / flow-matching edit 实现本身不正确（用户核心怀疑，最高优先级）**
   - 本 session 加的 SDEdit partial-noise start 是凭理解写的，**没有任何数学推导验证、没有参考实现对照**
   - 具体怀疑点：
     - a) **时间约定方向可能反了**：pipeline 用 `t=0 → noise, t=1 → clean`（行 280 注释断言），但 trainer 训练时的时间方向是否真的如此？如果训练是 `t=0→clean, t=1→noise`（标准 DDPM 约定），这套 pipeline 整体 init + ODE 方向都反了，但因为 replacement 每步强拉，所以在全 mask=0（C_full）时碰巧能跑；一旦 mask 区域多（C_full_inpaint 整段被 regen），pipeline 的 flow velocity 会系统性反方向，输出崩坏。
     - b) **`x_t = (1-t)*z + t*clean` 这个插值公式本身可能不是此 pipeline 训练时用的 schedule**，如果 pipeline 用 rectified flow 的别的参数化，SDEdit 插值公式错误 → 初始 `x_init` 不在 model 的训练数据分布上
     - c) **`where(keep_mask, x_clean, x_partial_noised)` 里 `x_clean` 是归一化后的 LQ**，但每步 replacement 用的 `x_clean` 也是 normalized LQ；如果 pipeline 内部在某个地方又 renormalize 了，LQ 值会被二次处理
     - d) **ODE 从 `start_i` 起步，但 `t[start_i]` 并不一定等于 `1-tau`**（只是 `>= 1-tau`），与初始化的 `x_init` 时间戳错配 → model 在错误的 t 下预测 velocity
     - e) **CFG 分支的 SDEdit 对齐**：当前 CFG 时 `x_input` 被 cat 到 2B，但 `y0` 只有 B 维，且 start_i 对双分支是否一致没验证
   - **验证方法**：
     - A. 把 SDEdit 关掉（τ=0），看 A_adaptive_inpaint 是否也回到接近 C_full 水平（说明 SDEdit 是主要退化源）
     - B. 对比 training 时的 time convention：查 `bundle._noise_scheduler_cfg` 和 trainer 中 `q_sample` / loss 的定义，确认 t 方向
     - C. 对单个样本跑 before/after，打印 ODE 每步的 x.mean / x.std，看是否在预期量级

2. **Flow-matching "edit" 模式（is_editing=True）也同样可疑**
   - `_editing_mode=True` 时 `src_motion = LQ（不清零）`，但 pipeline 里 VACE `reactive = src_motion * src_mask = LQ * mask` — 这等价于"告诉模型这块本来应该是 LQ 的值但你需要修"
   - 但训练时 editing 分支是否用这种 reactive 构造？还是另有方式？
   - 训练数据的 mask 分布 vs 这种 editing 场景的 mask 分布是否一致？
   - **验证方法**：查 trainer 里 editing / repair 阶段的 batch 构造，逐通道对齐

3. **Sliding-window 实现问题（次要怀疑）**
   - 2 window linear blend：在 seam 处的 velocity 是否连续？blend 在 normalized space 做的，denorm 后不保证位置平滑
   - 两个 window 各自独立 sample，noise seed 不同，blend = 两个完全不同轨迹的加权平均 → 必然 jitter 增加
   - C_full_inpaint 的 353 → 5835 用 sliding window 完全可以解释：短样本 cropped 只需跑一次；长样本现在跑两次 + 盲目 blend
   - **验证方法**：短样本（<360）metric 是否几乎不变？长样本的 jitter 是否是 seam 周围集中？

4. **基线样本集不一致（排查项）**
   - 两次 run 的 `filter + order` 可能不同 → 50 样本的子集不完全相同
   - 验证：hash `per_sample` 的 prompt_id 列表对比

## 调试策略

用户明确不信任 SDEdit / flow-matching edit 实现 → **首先隔离测试这两条**，而不是接受它们正确再往后推。

迭代优先级：
- P1: 关掉 SDEdit（τ=0），看 A_adaptive_inpaint 是否回归到接近旧 "zero + full-noise" 的水平
- P2: 关掉 sliding-window（crop 到 360），看 C_full_inpaint 是否回到 ~353
- P3: 验证 pipeline 的 time convention + flow parameterization 是否与 init 公式一致
- P4: 其他

## 参考文献

- 本 session 聊天记录 — 改动历史、语义约定、pipeline 细节
- `hftrainer/pipelines/motion/mogendit_pipeline.py` — MoGenDIT 对照
- `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/motion_process/motion_refiner.py` — ada_denoise 参考
- `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` — 目标 pipeline（已加 sdedit_tau）

## 代码入口

- `tools/eval_m2m_v2_all_tasks.py` — 主驱动 (`evaluate_sample` + sliding-window + SDEdit 配置)
- `tools/eval_m2m_v2_all_tasks.py:1049-1101` — **sliding-window 块**（本轮新增，最高怀疑）
- `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:280-305` — SDEdit init + ODE skip
- `hftrainer/evaluation/motion/m2m_eval_tasks.py` — E9 task settings（已去 dilation）

## 运行命令

完整 E9 rerun（~25 min）:
```bash
CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local uncond_global --tasks E9 \
    --max-samples 50 --output-dir work_dirs/e9_windowed_rerun --save-npz
```

单 setting 单 sample 快速验证（为 autodebug 准备）:
```bash
CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E9 --settings C_full_inpaint \
    --max-samples 1 --output-dir /tmp/e9_debug_c --save-npz
```

## 结果解析

结果 JSON 路径：`{out_dir}/eval_v2_{timestamp}.json`
路径：`[model_name]['tasks'][task_key]['aggregated'][metric_name]` → `mean` / `std` / `median`
`[model_name]['tasks'][task_key]['per_sample']` → list of dict，含 `jitter_pos`, `_num_frames`, `_sample_idx`

## 预计运行时间

- 单 sample 单 setting: ~5s
- 1 model × 3 settings × 50 samples: ~7 min（长样本 windowed 2 pass 占大头）
- 2 models × 3 settings × 50 samples: ~15 min

## Observability Mapping

| 问题表现 | 可量化指标 | 获取方式 |
|------|---------|---------|
| Sliding-window seam 不平滑 | 单 sample 内第 T-360 帧附近的 per-frame acc 和 vel | 从 NPZ 加载 positions 计算 |
| 长样本比短样本更差 | jitter_pos 按 `_num_frames` 分桶统计 | per_sample 过滤 |
| 样本集合变化 | 两次 run 的 (prompt_id, _sample_idx) 集合差异 | 比对 per_sample 列表 |
| 纯 SDEdit 相关问题 | C_full_inpaint (sdedit_tau=0) 是否恶化 | 对比同一 setting 的 metric |
| Dilation 影响 | A_adaptive_inpaint 比 C_full_inpaint 的相对增量 | 各 setting 分别看 |

## 数据分析
（首次迭代中填充）

## 代码知识库

### 关键文件索引
- `tools/eval_m2m_v2_all_tasks.py` — 主 eval 驱动。`load_eval_samples` 决定是否 crop 到 360；`evaluate_sample` 做 mask 构造 + inference + metrics。E9 走 sliding-window 分支。
- `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` — FlowMatching ODE。`sdedit_tau` 控制 SDEdit partial-noise 起点。每步 `x[keep_mask] = x_clean`。
- `hftrainer/evaluation/motion/m2m_eval_tasks.py` — 任务 settings 定义。E9 三个 setting: A_inpaint / A_edit / C_full。

### 核心逻辑摘要（sliding-window 路径）

```
load_eval_samples(task_id='E9'):
    T = motion.shape[0]          # 不 crop，保留全长
    motion = motion[:T]

evaluate_sample:
    T = sample['T']
    motion_norm = bundle.normalize_motion(motion)  # (1, T, D)
    src_mask = build mask (T, D)

    if T ≤ T_PAD (360):
        # 短样本路径
        pad motion_norm 到 T_PAD
        batch = {src_motion, src_mask, src_length=[T], tgt_length=[T], clean_motion}
        pipeline(batch) → sampled_norm (1, T_PAD, D)
    else:
        # long 样本 sliding-window
        Window A: frames [0, 360]
        Window B: frames [T-360, T]
        Overlap: [T-360, 360], length = 720 - T
        Run Window A → outA (1, 360, D)
        Run Window B → outB (1, 360, D)
        Blend: prefix A, overlap linear w, suffix B → sampled_full (1, T, D)
        sampled_norm = sampled_full

    output_denorm = bundle.denormalize_motion(sampled_norm)[0][:T]
    cond_mask = (mask_135 < 0.5)
    output_135[cond_mask] = motion_135[cond_mask]  # lock unmasked to LQ
    metrics = compute_all_metrics(output_135, ...)
```

### 领域知识备忘

- Flow matching convention in this repo: **t=0 → noise, t=1 → clean**. y0 at t=0 = where(keep_mask, x_clean, z).
- SDEdit τ 解释: τ=0 → standard (full noise on masked), τ=1 → clean on masked (no change).
- jitter_pos metric (from m2m_eval_metrics): 二阶差分 `||acc||₂`，单位 m/s²，on positions[22,3]。高 jitter = 轨迹抖动。
- E9 旧 crop 到 360: 50 样本里约 29/215 原本 > 360，现在这些样本都跑到 full length。

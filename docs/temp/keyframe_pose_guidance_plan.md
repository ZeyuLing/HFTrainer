# Keyframe Pose Guidance — 方案文档

## 1. 目标

实现 **Ref-Pose Condition**：用户指定一个目标姿态（单帧 target_pose），模型在指定 keyframe 位置生成的动作精确匹配该姿态，同时保持动态自然性。

## 2. 技术路线

基于前期调研（`docs/design/keyframe_pose_guidance_research.md`），当前采用 **方案四（Mask-Aware Flow Matching + Imputation）** 作为主方案。

### 2.1 核心原理

| 组件 | 说明 |
|------|------|
| **训练** | `mask_aware_noise=True`：已知区域在 x_t 中保持 clean，模型学会从 x_t 直接读取已知信息 |
| **推理** | `replacement_guidance='flow_interp'`：ODE 每步替换已知区域为 flow 插值值 |
| **结果** | Keyframe 位置精确保持 target_pose（零误差），周围帧自然过渡 |

### 2.2 推理管线

```
Step 1: 构建输入
    composite_motion[keyframe_idx] = target_pose
    src_mask = build_mask(mode='anchor_inbetween')  # 保留 first + kf + last

Step 2: 归一化 + 清零
    normalized = bundle.normalize(composite) * (1 - mask)

Step 3: VACE 构建
    vace_context = bundle.prepare_vace_input(normalized, mask)

Step 4: ODE 积分 + replacement guidance
    for each step:
        v = model(x_t, vace_context, t)
        x_{t+1} = x_t + v * dt
        x_{t+1}[known] = (1-t_{next})*z_0 + t_{next}*x_clean  # flow_interp

Step 5: Post-hoc blend
    final = composite * (1 - mask) + denorm(x_1) * mask
```

### 2.3 三种 Imputation 策略

| 策略 | Mask 模式 | 使用场景 |
|------|-----------|---------|
| `keyframe_only` | 全 mask，仅 kf=0 | 完全从 keyframe 约束重新生成 |
| `anchor_inbetween` | 全 mask，first + kf + last = 0 | 保留首尾帧，在锚点间补全 |
| `local_edit` | kf 附近 ±30 帧 mask | 局部编辑，最小改动 |

### 2.4 已训练模型

| 模型 | Loss | Text | Rotation | MAN | Epoch |
|------|------|------|----------|-----|-------|
| uncond_fm_man | velocity (FM) | 无 | local | ✅ | 395 |
| uncond_jit_man | JiT | 无 | local | ✅ | 362 |
| caption_fm_man | velocity (FM) | 有 | local | ✅ | 125 |
| caption_jit_man | JiT | 有 | local | ✅ | 253 |
| uncond_fm_man_globalrot | velocity (FM) | 无 | global | ✅ | 77 |
| uncond_jit_man_globalrot | JiT | 无 | global | ✅ | 69 |
| caption_fm_man_globalrot | velocity (FM) | 有 | global | ✅ | 68 |
| caption_jit_man_globalrot | JiT | 有 | global | ✅ | 59 |
| MoGenDIT 0.1B | DDPM x0 | 无 | global (内部) | 天生支持 | pretrained |

### 2.5 与 MoGenDIT 的对比

| 特性 | HyMotion M2M (MAN) | MoGenDIT |
|------|-------------------|----------|
| 架构 | MMDiT + VACE + Flow Matching | DiT + AdaLN + DDPM |
| 表示 | 135-dim (22j rot6d + transl) | 201-dim (22j rot6d + 22j joint + transl) |
| Imputation | replacement guidance (flow_interp) | 原生 obs_mask + 每步替换 |
| Mask 粒度 | 逐帧逐关节 | 逐帧逐关节 |
| 文本条件 | 支持 (caption 变体) | 不支持 |
| 参数量 | 0.46B | 0.1B |

## 3. 评测方案

### 3.1 评测指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **KF L2** | keyframe 帧输出与 GT 的 L2 距离 | 关键帧精度（MAN + blend 应为 0） |
| **KF Trans Error** | keyframe 帧 translation 误差 | 位置精度 |
| **Global MPJPE** | 生成区域与 GT 的平均逐帧 L2 | 整体重建质量 |
| **Boundary Smoothness** | mask 边界处加速度幅度 | 过渡自然性 |
| **Foot Skating** | 脚部关节速度 | 物理合理性 |

### 3.2 评测矩阵

- **模型**：8 个 MAN 变体 + 2 个 Non-MAN baseline + MoGenDIT
- **Imputation 策略**：keyframe_only / anchor_inbetween / local_edit
- **Replacement Guidance**：none / flow_interp
- **Keyframe 位置**：25% / 50% / 75% of motion length
- **测试数据**：20 samples from test_motionhub_recon

## 4. 文件清单

| 文件 | 用途 |
|------|------|
| `scripts/eval_keyframe_pose_guidance.py` | 评测主脚本 |
| `scripts/serve_kf_eval_results.py` | 结果可视化 Web 服务 |
| `docs/design/keyframe_pose_guidance_research.md` | 前期调研报告 |
| `docs/design/keyframe_pose_guidance_plan.md` | 本方案文档 |
| `output/eval_keyframe_pose/` | 评测输出目录 |

## 5. 运行方式

```bash
# 在 debug 机器上运行评测
python3 scripts/eval_keyframe_pose_guidance.py \
    --output-dir output/eval_keyframe_pose/local_rot \
    --num-cases 20 --gpu 0

# 查看结果
python3 scripts/serve_kf_eval_results.py \
    --result-dir output/eval_keyframe_pose --port 8095

# Web 访问
http://<host>:8095/
```

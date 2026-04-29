# StableMotion — 基于不配对损坏数据的动作清洗扩散模型

## 基本信息

| 字段 | 内容 |
|------|------|
| **论文标题** | StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data |
| **作者** | Yuxuan Mu, Hung Yu Ling, Yi Shi, Ismael Baira Ojeda, Pengcheng Xi, Chang Shu, Fabio Zinno, Xue Bin Peng |
| **单位** | Simon Fraser University / Lightspeed Studios (Tencent) / NRC Canada |
| **时间 / 会议** | SIGGRAPH Asia 2025 Conference Papers |
| **arXiv** | [2505.03154](https://arxiv.org/abs/2505.03154) |
| **项目主页** | https://yxmu.foo/stablemotion-page/ |
| **代码** | https://github.com/Murrol/StableMotion |
| **Checkpoint** | `save/stablemotion_brokenamass.pt`（OneDrive, 257MB, 基于 BrokenAMASS 训练 1M 步） |

---

## 论文核心内容

### 问题定位：动作清洗 (Motion Cleanup) 的根本困境

工业级动捕数据（optical mocap、inertial、手工 rigging）中普遍存在：
- **关节抖动 (jitter)**、**骨骼穿插 (joint jump)**、**抽风 (artifact)**
- **脚滑 (foot sliding)**、**漂浮 (floating)**、**根漂移 (root drift)**

现有 motion cleanup 工作（如 DeepMotion, NeMF, HuMoR）依赖两类监督：
1. **配对数据 (clean, corrupted) pair**：手工构造，昂贵且难以覆盖真实噪声分布
2. **合成损坏 (synthetic artifact augmentation)**：手工规则，与真实 artifact 有 domain gap

**StableMotion 的核心主张**：*不需要 clean 数据集*。直接从**同一份原始 mocap 数据（既含干净片段，也含损坏片段）** 中，用扩散模型同时学会"检测 (detect)"和"修复 (fix)"。

### 关键设计：Quality Indicator 作为动作表示的"额外特征通道"

**核心创新点**：把"当前帧是否损坏"这个二值标签 **作为动作特征向量的最后一维**，与身体特征（rotation + translation + ...）一起参与扩散过程。

```
原始表示:  x_body ∈ R^{T × D_body}       （Global SMPL RIFKE = 139 维）
新表示:    x = [x_body, label] ∈ R^{T × (D_body + 1)}
          └──────────────┬────────────┘
                        一起被扩散 / 去噪
```

这个看似微小的改动带来一个优雅的对偶性（训练和推理中的双重角色）：
- **训练时**：`label` 既可作为"输入条件"（inpainting mode），又可作为"生成目标"（detection mode）
- **推理时**：先 corrupt `label` 通道让模型**预测质量标签**（detect），再以预测的 label 为条件让模型**重新生成损坏帧**（fix）

### 两模式联合训练（Two-Mode Training）

每个 batch 按 `args.fraction` 划分为两部分（`train/training_loop_smpl.py::mask_manager`）：

| 模式 | 划分 | 条件维度 (`inpaint_cond=0`) | 生成目标 (`inpaint_cond=1`) |
|------|------|---------------------------|---------------------------|
| **Detection 模式** | `bs // fraction` | 所有 body 通道 `[:, :-1]` 作为条件 | 仅 `label` 通道 `[:, -1]` |
| **Inpainting 模式** | 其余 | `label` 通道作为条件 `[:, -1]` | body 通道按 cosine schedule 随机 mask |

**cosine mask schedule**：`num_masked = round(N · cos(t · π/2))`，`t ~ U(0,1)`，至少保留 5 个 token。与 MaskGIT / PhyMoGen 的思路一致，训练时覆盖从"几乎全遮"到"几乎不遮"的全谱 mask ratio。

注意：`inpaint_cond` 最终还与 `attention_mask` 做 AND，忽略 padding tokens。

### 网络架构：StableMotion-DiT（改自 Stable Audio）

`model/stablemotion.py::StableMotionDiTModel`，默认配置：

| 超参 | 值 | 备注 |
|------|-----|------|
| `in_channels` | 64 | body + label 的特征维度 |
| `num_layers` | 24 | Transformer blocks |
| `num_attention_heads` | 24 | |
| `attention_head_dim` | 64 | |
| `inner_dim` | 24×64 = 1536 | |
| `cond_index` | `[0, -1]` | 从 `inpaint_cond` 抽出 2 个 mode indicator |
| `mode_indicator_dim` | 2 | 对应 body + label 两个"角色" |
| 位置编码 | 1D Rotary (`rotary_embed_dim = 32`) | 复用 diffusers `get_1d_rotary_pos_embed` |
| Attention | `StableAudioAttnProcessor2_0` | FlashAttn-2 friendly |
| FFN | SwiGLU | Stable Audio 风格 |

**AdaLN-single 调制（PixArt-α 风格）**：
- `CombinedTimestepModeEmbeddings`：timestep → 256 → TimestepEmbedding → inner_dim；可选叠加 `LabelEmbedding(vs_modes=2)` 的 mode 向量
- 每个 block 用 `scale_shift_table (1,1,6,dim) + time_hidden_states` 产生 6 个调制向量 → chunk(6)：`shift/scale/gate` × `{MSA, MLP}`
- block 对 MSA 和 FFN 分别做 `x = x + gate · f(norm(x)·(1+scale) + shift)`

**输入融合**：
```python
# cond_index = [0, -1]
hidden = concat([x, inpaint_cond[:, cond_index]], dim=1)    # (B, in+2, T)
hidden = preprocess_conv(hidden) + hidden                    # 1×1 Conv1d 残差
hidden = proj_in(hidden.transpose(1,2))                      # → (B, T, inner)
... transformer blocks ...
out = proj_out(hidden).transpose(1,2)                        # (B, out, T)
out = postprocess_conv(out) + out                            # 1×1 Conv1d 残差
```

可选 `zero_init=True`：`proj_out + postprocess_conv` 零初始化，让模型从恒等映射出发。

### 动作表示：Global SMPL RIFKE

基于 `data_loaders/amasstools/globsmplrifke_feats.py`（改自 STMC 的 SMPL RIFKE）：
- **全局化**（Global）：不是以 pelvis 为 root 的 local representation，而是保留世界坐标（更利于直接 impute 全局位置）
- **RIFKE**（Root-Invariant Forward Kinematics Embedding）：pelvis 朝向 + 世界位移 + 关节 6D rotation
- **BrokenAMASS**：AMASS 原始数据 → 20fps → canonicalize → 叠加 `motion_artifacts_smpl` 合成损坏 → `foot_slidedetect_zup` 标注脚滑标签（作为 ground-truth quality label）→ 保存为 `AMASS_*_globsmpl_corrupted_cano`

### 训练细节

`train/training_loop_smpl.py`：
- **优化器**：AdamW，`lr=1e-4`（默认），`weight_decay`
- **精度**：`torch.GradScaler` + `autocast(fp16)`
- **Loss**：`--l1_loss`（L1），可选 `--snr_gamma`（SNR loss weighting）、可选 `feature_w` 每维权重（`normalizer_dir/feature_w_file`）
- **EMA**：`--model_ema`（默认开，基于 `ema-pytorch`，`include_online_model=False`）
- **梯度裁剪**：`--gradient_clip` + `clip_grad_norm_(1.0)`（`scaler.unscale_` 后）
- **调度采样**：`uniform` schedule sampler（非 importance-weighted）
- **训练规模**：`batch_size=128`，`num_steps=1_000_000`
- **DDPM**：1000 steps 训练

### 推理：Detect-and-Fix Pipeline

`sample/fix_globsmpl.py`：

**Stage 1 — Detect**（`detect_labels`）：
```python
inpaint_motion_detmode[:, -1] = 1.0               # corrupt label 通道
inpainting_mask_detmode[:, -1] = False            # 只预测 label 通道
re_sample = sample_fn(...)                        # DDPM / DDIM 反扩散
for _ in range(ProbDetNum):                       # MC 平均
    re_sample += sample_fn(...)
re_sample /= (ProbDetNum + 1)
label = (re_sample[..., -1] > ProbDetTh)          # 阈值化
```

**Stage 2 — Fix**（`fix_motion`）：
```python
# ±1 帧膨胀检测边界
label[..., 1:] += temp_labels[..., :-1]
label[..., :-1] += temp_labels[..., 1:]
label[..., last_frame_of_each] = 0                # 最后一帧强制为好

# 构建 inpainting mask: True=keep good frame, False=repaint
inpainting_mask[sample_i, ..., good_frames] = True
inpainting_mask[:, -1] = True                     # label 通道全部 keep

inpaint_motion[:, -1] = -1.0                       # 告诉模型"这里要干净"
sample_fix = sample_fn(...)
```

**SITS — Soft-Inpaint Time Schedule**（`--enable_sits`）：不一刀切地用同一 timestep 起始 inpaint，而是**按逐帧的"损坏概率"自适应选择起始 timestep**：
```python
soft_ts = torch.clip((re_sample[..., -1:] + 0.5), 0, 1)   # 把 label 归一到 [0,1]
soft_ts = ceil(sin(soft_ts · π/2) · T)                    # 越脏的帧，起点越靠近完全噪声
```
对干净帧做少步修正，对损坏帧做大步重生成。

**Ensemble Cleanup Selection**（`--ensemble`，`sample/utils.py::run_cleanup_selection`）：
- 生成 `forward_rp_times=5` 个候选
- 每个候选用 `eval_times=25` 次快速 detection 在 t=49（中等 noise level 的单步 denoise）评分
- `score = Σ (predicted_label > 0) · attention_mask`，越低越好
- `argmin` 选出最干净的候选

**Foot-Lock Classifier Guidance**（`--classifier_scale>0`）：
- `prepare_cond_fn` 载入 SMPL 24-joint regressor + parents
- `footlocking_fn`：反算关节 → `compute_foot_sliding_wrapper_torch` → `-loss` 梯度回传到 `x_in`
- 梯度 clip 到 [-10, 10]，`grad[:, 0] = 0`（不改 root），scale 100 注入到每一步去噪

---

## 与 HyMotion M2M（我方）的对比

| 维度 | StableMotion | HyMotion M2M（我方） |
|------|--------------|---------------------|
| **任务定位** | Motion **Cleanup**（输入是损坏动作，输出是干净动作） | Motion Generation / Completion（输入是文本 + 约束，输出是动作） |
| **Backbone** | DiT 改自 Stable Audio，24 层 × 24 头 × 64 dim | HunyuanMotion MMDiT 0.46B/1.5B |
| **生成范式** | DDPM (1000 steps train) + DDIM (inference) | Flow Matching (rectified flow, 50-step Euler) |
| **动作表示** | Global SMPL RIFKE (139-dim body) + **1-dim label** | 135-dim (3 transl + 22 rot_6d)，无 label |
| **质量监督信号** | **把 "corrupted? yes/no" 作为可学特征通道** | 无（假设训练数据全部干净） |
| **训练数据范式** | **Unpaired corrupted data** — 无需 clean 参考 | Paired / 人工标注 quality，过滤掉低质量 |
| **Conditioning 方式** | Inpainting（concat mode indicator 通道 + 部分覆盖输入） | VACE（4× motion_dim channel concat） |
| **训练时 mask 策略** | Detection mode + Inpainting mode 按 fraction 切分；cosine ratio schedule | M1-M6 混合（random cell / block / temporal / joint / full / keyframe） |
| **文本条件** | ❌（纯 motion cleanup，无 T2M） | ✅（通过 PRISM / Caption 注入） |
| **推理策略** | Detect-then-Fix 两阶段 + MC 平均 + SITS + Ensemble + 可选 foot-lock guidance | 直接 one-shot denoising + 可选后处理 |
| **训练规模** | BrokenAMASS（~300h AMASS 派生）× 1M steps × bs=128 | MotionHub × 变长 × 大 bs |
| **EMA** | ✅（`ema-pytorch`，默认开） | ✅（EMAHook） |
| **开源** | ✅ 完整（代码 + checkpoint + 数据生成脚本） | ❌ 内部 |

### 核心设计理念差异

1. **质量是"建模对象"还是"预处理条件"？**
   - StableMotion：把"帧是否损坏"**显式建模为特征维度**，一个扩散模型同时学会检测和修复 → 对训练数据质量要求低
   - M2M：假设训练数据已清洗（或 pretraining 吸收噪声），**不建模**质量标签 → 对训练数据质量要求高

2. **Conditioning 粒度**
   - StableMotion：**帧级别**（label 是 `T × 1`），修复粒度到帧
   - M2M：**帧 × 维度级别**（mask 是 `T × 135`），可做 per-joint 控制

3. **Inference 复杂度**
   - StableMotion：两阶段 + 可选 MC × SITS × Ensemble × guidance，最坏情况下单样本需要 `(ProbDetNum+1) + ensemble×(25+1+25)` 次 denoise pass
   - M2M：一次 50 步 ODE 即可

4. **是否需要 clean ground truth？**
   - StableMotion 训练时不需要 — GT label 来自 `foot_slidedetect_zup` 等规则检测器（自监督）；BrokenAMASS 只是用合成损坏造训练集，**真实部署时 label 可以是弱监督（rule-based 或人工抽检）**
   - M2M 当前流程依赖 `motion_annot_web/m2m_database` 的质量筛选产出"高质量子集"再训练

---

## 可借鉴的点（对 HyMotion M2M / 动作标注）

### P0 — 高优先级

1. **Quality Indicator 作为特征通道**：
   - 可以在 M2M 的 135-dim 或 201-dim 表示后追加 1 个 quality channel（per-frame）
   - 训练时把 `motion_annot_web` 已有的 quality checker 输出（foot_sliding / joint_jump / jitter 等）喂给这个通道
   - 推理时得到一个"自检 + 自修"能力：模型输出动作时顺带输出 per-frame 可信度，甚至可以做闭环细修
   - **关键价值**：缓解 `train_hymotion_400h.json` 中 ~85K 低质量样本稀释能力的问题，不用丢数据也能压制噪声贡献

2. **Detect-then-Fix 用于 Motion Repair（对标 MoGenDIT）**：
   - 当前 MoGenDIT 的 `ada_denoise` 模式**不使用 adaptive mask** 做 imputation（见 `hftrainer/models/motion/CLAUDE.md`）
   - StableMotion 的 detect 阶段输出 per-frame label mask，可以直接喂给 MoGenDIT 作为 adaptive inpaint mask
   - 组合路线：StableMotion-detect → MoGenDIT-fix（或 M2M-fix）

### P1 — 中优先级

3. **SITS（Soft-Inpaint Time Schedule）**：
   - 逐帧自适应起始 timestep：`ceil(sin((label + 0.5) · π / 2) · T)`
   - 对 **部分损坏但大部分干净** 的场景（典型 motion repair）有效率优势：干净帧只走少步 Euler，脏帧才走全程
   - M2M v2 post-train 场景可试

4. **Ensemble Cleanup Selection**：
   - 生成 K 个候选 → 用模型自身作为"评分器"（去噪残差 / label 复检）→ argmin
   - 与 SOAR 的 on-policy rollout 精神一致，但这里是**推理时的 best-of-N**，不改训练
   - 对 M2M soar 后训练是**正交**的补强：训练后仍可用 ensemble 进一步压 artifact

5. **Foot-lock Classifier Guidance**：
   - 可微分 `compute_foot_sliding_wrapper_torch` + batch_rigid_transform
   - M2M 目前脚滑依赖 post-process。在 flow matching 里同样可以嵌入 guidance（把 `classifier_scale · ∇ loss_footslide` 加到 velocity field）

### P2 — 长期研究

6. **Unpaired Corrupted Training Paradigm**：
   - 不用人工造 clean/corrupted pair，只要有 **quality 检测器 + 原始（含噪）数据集** 即可训练 cleanup 模型
   - 与 `motion_annot_web/quality_check_rules/` 的 P0-P2 checker 可直接对接：
     - 用 checker 标注 → 作为 label 通道 → 训练 cleanup model
     - 反过来：cleanup model 预测的 label 可以作为新的 checker（self-distillation）

7. **Global SMPL RIFKE**：
   - 对世界坐标约束友好，`foot_slidedetect_zup`（z-up 世界坐标）可以直接作用
   - 如果 M2M 未来需要支持"桌椅交互 / 路径规划 / 多人相对位置"，Global representation 是必要基础

---

## 文件地图（代码阅读指引）

```
StableMotion/
├── model/
│   └── stablemotion.py             # DiT backbone（改自 Stable Audio）
├── train/
│   ├── train_stablemotion_smpl_glob.py  # 入口
│   └── training_loop_smpl.py       # TrainLoop（mask_manager 核心逻辑在此）
├── diffusion/                      # 从 MDM fork 的 GaussianDiffusion
├── sample/
│   ├── fix_globsmpl.py             # 推理入口：detect → fix
│   └── utils.py                    # run_cleanup_selection (ensemble)
│                                    # footlocking_fn (classifier guidance)
│                                    # prepare_cond_fn / build_output_dir
├── data_loaders/
│   ├── corrupting_globsmpl_dataset.py  # BrokenAMASS 构建脚本
│   └── amasstools/
│       ├── globsmplrifke_feats.py     # Global SMPL RIFKE 表示
│       ├── motion_artifacts_smpl.py   # 合成损坏注入
│       └── foot_slidedetect_zup.py    # 脚滑检测 → quality label
├── eval/                           # 评估脚本（TMR + 物理指标）
├── visualize/                      # SMPL 渲染
└── save/
    └── stablemotion_brokenamass.pt  # 预训练 checkpoint（257MB）
```

---

## 运行备忘

```bash
# 训练（BrokenAMASS）
python -m train.train_stablemotion_smpl_glob \
    --save_dir save/stablemotion \
    --data_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
    --normalizer_dir dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
    --l1_loss --model_ema --gradient_clip \
    --batch_size 128 --num_steps 1_000_000

# 推理（增强模式：ensemble + SITS + foot-lock guidance）
python -m sample.fix_globsmpl \
    --model_path save/stablemotion/ema001000000.pt --use_ema \
    --batch_size 32 \
    --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
    --ensemble --enable_sits --classifier_scale 100 \
    --output_dir ./output/stablemotion_hack
```

**关键 CLI 参数**：`--ProbDetNum`（MC 检测次数）、`--ProbDetTh`（label 阈值）、`--skip_timesteps`（DDIM 跳步起点）、`--ts_respace`（选 DDIM 而非 DDPM）、`--fraction`（训练时 det/inpaint 划分）。

# MotionCanvas (HyMotion M2M) — 实验/表格 ↔ Baseline 对齐矩阵与缺口分析

> 目标：让 NeurIPS 2026 投稿的实验设计与表格，逐项对齐相关工作各自的 **native 评测协议**，
> 并补全 reviewer 会期待的 baseline / 指标列 / 协议维度。
>
> 数据来源：`ref_repo/` 各 baseline 仓库与分析文档（见末尾索引）。本文档是
> `papers/HYMotionM2M_NIPS2026/EXPERIMENT_PLAN.md` 引用的对齐文档。

---

## 0. 方法侧关键事实（用于对齐，2026-05 现状）

- 表示：**198-dim**（3 trans + 132 rot6d(22×6) + 63 position(21×3)），SMPL-22，**30 fps**。
- 条件：**三通道 VACE**（`no_inactive`：x_t + reactive + mask = 3×D = 594-dim 输入）。
- Mask：**7 个 mask 族 M1–M7**（注意：旧 outline 写的 138-dim / 4 通道 / M1–M6 已过时）。
- 训练：mask-consistent flow matching（观测坐标全程保持 clean），FK / position / velocity 辅助损失。
- 训练数据：内部 mocap ~549K clips / ~400h（**HumanML3D test 不参与训练**）。
- 评测主战场：**HumanML3D test**（与 CondMDI/UMO/MotionLab 共享），FK→263-dim→20fps 后用官方 evaluator。

---

## 1. HumanML3D 评测惯例（多 baseline 共用，必须照搬）

| 项目 | 惯例值 | 依据 |
|------|--------|------|
| split | test | 各 baseline `split='test'` |
| 帧率 / 长度 | 20 fps / max 196 帧 | CondMDI, OmniControl, MLD |
| 表示 | 263-dim | MDM 系 / MotionLab |
| 采样规模 | 常限 **1000** 条 | CondMDI/MDM/OmniControl `num_samples_limit=1000` |
| Replication | **20 次**（mean ± 95% CI） | CondMDI/MoMask/MLD |
| Diversity | 300 随机对 | CondMDI/MDM/MotionLab |
| T2M 列 | FID↓, R@1/2/3↑, MM-Dist↓, Diversity→, (MModality↑) | MDM `fixed_results.tex` |

**行动**：在 §exp-setup 明确写出"20× / 1000 samples / 95% CI / Diversity 300"，否则易被判协议不一致。

---

## 2. 各 Baseline native 协议速查（决定它们能进哪张表）

| Baseline | 数据/表示 | native 任务协议 | 主指标 | 进我们的表 |
|----------|-----------|-----------------|--------|-----------|
| **CondMDI** | HumanML3D abs-root 263d, 20fps | `benchmark_sparse`(每T帧1关键帧), `gmd_keyframes`(随机5帧), `benchmark_clip`(中段生成), random_frames/joints | FID, R@k, **Traj/KPS fail@20/50cm, kps_mean_err, Skating** | T2M / IB / keyframe / clip / spatial |
| **UMO** | HML3D+MotionFix+InterX, 201d, 30fps（**闭源**） | 4-protocol temporal(pred/back/in-between/keyframe, **帧级 whole-body**) + instruction edit + trajectory(文本坐标) | FID, **MPJPE, [P]-MPJPE**, R@3, Traj.Err | T2M / IB / temporal-4（**仅引用已发表数**） |
| **KIMODO** | 内部 mocap 333d（非 HML3D 主榜） | joint 级 imputation：keyframe / end-effector / Root2D(XZ) / foot-contact | foot skating, FK 一致, 约束满足 | IB / keyframe / spatial / OOD（需表示转换脚注） |
| **MotionLab** | HumanML3D 263d + MotionFix, 20fps | T2M / traj(joint coords hint) / in-between(text+keyframe poses) / **text&traj editing** / style | FID, **Distance**(traj/kp err), R@1(edit), **SRA**(style), Skating | T2M / IB / keyframe / spatial / coverage |
| **OmniControl** | HumanML3D 263d, 196帧 | 任意 joint×**density∈{1,2,5,25,100}** 空间 hint（6 joint × 5 density = 30 组） | **Control L2, Traj fail@k, Skating**, FID, R@k | spatial/trajectory |
| **GMD** | HML3D abs-root（CondMDI 管线） | `gmd_keyframes`(5 随机帧) + 两阶段 traj→motion | **Traj/KPS fail@20/50cm**, Skating | keyframe / trajectory |
| **StableMotion** | BrokenAMASS 139d（非 HML3D） | motion cleanup（detect+fix） | MPJPE, accel err, skating | **repair only**（coverage 表，不进 trajectory） |
| **MDM/MoMask/MLD** | HumanML3D 263d, 20fps | T2M | FID, R@k, MM-Dist, Div, MModality | **T2M 参照表** |

---

## 3. 逐表对齐状态与缺口（核心）

> 论文主文 9 表 + 附录扩展表。状态：✅结构OK ⚠️结构OK但需补 baseline/列 ❌缺文件/缺数。

| # | 表 (`depds/`) | 当前状态 | 缺口 / 行动 |
|---|---------------|----------|-------------|
| 1 | `tab_t2m.tex` | ❌ **文件缺失**（sec_4 已 \input） | **新建**。行：MDM, MLD, MoMask, HY-Motion-Lite, UMO, MotionLab, **MotionCanvas**。列：FID/R@1/R@2/R@3/MM-Dist/Div/MModality |
| 2 | `tab_temporal_completion.tex`（minimal IB，首尾帧） | ⚠️ | 列 OK；补 MotionLab 脚注（其 IB 协议=text+keyframe，不可直接同栏）；保留 [P]-MPJPE |
| 3 | `tab_temporal_unified.tex`（pred/back/clip） | ❌ **文件缺失**（sec_4 已 \input） | **新建**紧凑版：每协议 FID/MPJPE/[P]-MPJPE。附录 `tab_temporal_extended` 已是全列版 |
| 4 | `tab_keyframe_interpolation.tex` | ⚠️ | 已有 regular + random-5；**补 GMD 行**（gmd_keyframes 是 GMD/CondMDI 共同协议）；列已含 KPS Err/Fail@k/Skating ✅ |
| 5 | `tab_spatial_completion.tex` | ⚠️ | (c)(d) trajectory 子表建议改/补 **Traj fail@20cm, fail@50cm, kps_mean_err(m)**（CondMDI/GMD/OmniControl 标准），现仅 Gen/Pres.Err；**OmniControl 需注明 density 设置**（建议 root@density100 dense + density~5 sparse 对应我们的 dense/sparse） |
| 6 | `tab_coverage.tex` | ✅ | 行列齐（含 StableMotion repair）；与 MotionLab Table 1 任务勾选风格一致 |
| 7 | `tab_ood_masks.tex` | ✅ | diagonal/checkerboard/polygon；只保留支持任意 coordinate mask 的 KIMODO/MotionLab ✅ |
| 8 | `tab_ablation.tex` | ✅ | 覆盖 3通道/mask-consistent/representation/mask-mixture/init/text；与 MotionLab 消融维度可比 |
| 9 | `tab_efficiency.tex` | ⚠️ | 建议补"是否多阶段/多 pass"列（StableMotion ensemble、GMD 两阶段、OmniControl guidance vs 我们 50-step 单 ODE） |
| A1 | `tab_temporal_extended.tex`（附录） | ✅ | 与 #3 对应的全列版 |

---

## 4. 需要补充的 Baseline（用户关切：baseline 不够完善）

### 4.1 必补（reviewer 高概率索要）
1. **MDM / MoMask / MLD** → T2M 参照表（#1）。无此表 T2M 质量无对照。
2. **GMD** → keyframe 表 (#4) + trajectory 表 (#5)。`gmd_keyframes` 是社区标准，且 GMD 是 OmniControl/CondMDI 的共同对比对象。
3. **MotionLab** → 已在多表，但需补 **trajectory Distance(0.0286)** 与 in-between 协议脚注。

### 4.2 强烈建议补的 **对比维度/列**（当前 9 表未覆盖）
| 维度 | 来源 baseline | 处理建议 |
|------|---------------|----------|
| **Expert vs Unified** | UMO Tab.4–8, MotionLab | 在 T2M/IB 表加"是否单模型多任务"或单独一小段：联合训练 ≥ 专家模型（支撑 Q1） |
| **Traj fail@20/50cm + kps_mean_err** | CondMDI/GMD/OmniControl | 进 trajectory 子表（#5）替换/补充 Gen/Pres.Err |
| **OmniControl density 矩阵** | OmniControl | 至少 root joint 的 dense(100) + sparse(~5)；附录可给完整 6×5 |
| **Instruction editing R@1/R@3** | MotionLab(MotionFix), UMO | 主文声明 out-of-scope（HML3D），**附录给一句 MotionFix 定性/小表**，避免"回避编辑"质疑 |
| **Style transfer SRA** | MotionLab(69.21) | 非本方法目标，related work 提一句即可，不必进表 |
| **KIT-ML T2M 列** | MDM/MoMask/MLD | 可选；时间够则 T2M 表加 KIT 副栏增强说服力 |

### 4.3 可选 / 二线
- **StableMotion / MoGenDIT**：repair 表（已在 coverage）。可加一个附录 repair 小表（MPJPE/accel/skating），与 trajectory 严格区分。
- **KIMODO Root2D**：trajectory 子表已含；注意表示转换脚注（soma77→smpl22）。

---

## 5. 最终推荐的"实验内容"清单（与 mask 族对应）

| 实验 | 协议 | 训练 mask | 主指标 | 对齐 baseline |
|------|------|-----------|--------|---------------|
| T2M | 全 mask=1 + text | M5/text | FID,R@k,MM-Dist,Div | MDM,MoMask,MLD,UMO,MotionLab |
| Minimal IB | 仅首尾帧 | M3 特例 | MPJPE,[P]-MPJPE,Skating | UMO,KIMODO,CondMDI |
| Temporal-4 | pred / backcast / clip(+ in-between) | M3 | FID/MPJPE/[P]-MPJPE | UMO(4-protocol),CondMDI(clip) |
| Sparse keyframe | regular(每30帧) / random-5 | M6 | FID,KPS Err,Fail@20/50,Skating | CondMDI,**GMD**,KIMODO,MotionLab |
| Spatial part-edit | upper/lower (joints 6–21 / 0–5) | M4 | FID,Gen.Err,Pres.Err,Skating | CondMDI,KIMODO,MotionLab |
| Trajectory | root XZ dense / sparse(每5帧) | M4 on transl | **Traj fail@20/50cm, kps_mean_err**, FID, Skating | OmniControl,GMD,KIMODO,MotionLab,UMO(引用) |
| Coverage | 6 任务族单模型 | 全 | 支持性 + FID | 全体 |
| OOD masks | diagonal/checkerboard/polygon | 不在 M1–M7 | FID,MPJPE(+相对退化) | KIMODO,MotionLab |
| Ablation | minimal IB | — | FID,MPJPE,Foot,[P]-MPJPE,Jitter | 内部 |
| Efficiency | 196 帧 batch1 A100 | — | 参数/步数/秒/阶段数 | 各方法推理配置 |

**self-comparison（重要补充）**：T2M 与 IB 表内加一行"MotionCanvas (single-task specialist)" vs "MotionCanvas (unified)"，直接回答 Q1（统一训练不降质）。

---

## 6. 评测基础设施缺口（决定能否填表）

| 指标 | 状态 | 备注 |
|------|------|------|
| FID / R-Precision / Div / MM-Dist | ❌ 需对接官方 HumanML3D evaluator | 可复用 CondMDI/MDM evaluator wrapper |
| MPJPE / [P]-MPJPE | 🔧 部分有 | [P]-MPJPE = 观测帧误差，UMO 必备 |
| Foot Skating / Jitter | ❌ | 物理合理性，CondMDI/OmniControl 标配 |
| KPS Err / Fail@20/50cm / kps_mean_err | ❌ | 复用 GMD/CondMDI `metrics.py` |
| Traj ADE (root XZ) | ❌ | KIMODO/OmniControl/GMD |

**P0 行动**：对接 CondMDI evaluator（一次性拿到 FID/R@k/Skating/Traj-fail 全套），避免重复造轮子。

---

## 7. 设计取舍（已决策落实，2026-05-29）
1. ✅ trajectory 已**独立成表** `depds/tab_trajectory.tex`，采用标准 **Traj.Err / Fail@20cm / Fail@50cm + Foot**；`tab_spatial_completion` 收敛为纯 body-part editing。
2. ✅ instruction editing → 附录 `depds/tab_instruction_edit.tex`（MotionFix R@1/R@3，明确 out-of-scope）+ 主文/附录指引。
3. ✅ OmniControl density 矩阵 → 附录 `depds/tab_omnicontrol_density.tex`（ρ∈{1,2,5,25,100}），主文 dense/sparse 对应 ρ=100 / ρ≈20。
4. ⏸ **KIT-ML T2M 副栏**：暂缓（取决于算力/时间），如需再加副栏。

---

## 8. 文件索引（引用路径）
- 总览：`ref_repo/CLAUDE.md`, `ref_repo/TECHNICAL_COMPARISON.md`, `ref_repo/README.md`
- CondMDI/GMD：`ref_repo/CondMDI/README.md`, `.../utils/editing_util.py`, `.../eval/eval_humanml_condmdi.py`, `.../data_loaders/humanml/utils/metrics.py`
- UMO：`ref_repo/UMO/CLAUDE.md`
- KIMODO：`ref_repo/KIMODO/CLAUDE.md`
- MotionLab：`ref_repo/MotionLab/docs/temp/CLAUDE.md`, `.../configs/base.yaml`, `.../rfmotion/models/metrics/`
- OmniControl：`ref_repo/OmniControl/README.md`, `.../eval_omnicontrol_all.sh`, `.../eval/eval_humanml.py`
- StableMotion：`ref_repo/StableMotion/CLAUDE.md`
- MDM/MoMask/MLD：`ref_repo/MDM/assets/fixed_results.tex`, `ref_repo/Momask/momask-codes/eval_t2m_trans_res.py`, `ref_repo/MotionLCM/configs/mld_t2m_infer.yaml`

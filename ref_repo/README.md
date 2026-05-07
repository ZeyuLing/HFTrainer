# ref_repo — 参考工作与代码快照

本目录收录与 **HyMotion M2M** 研发相关的第三方论文实现、内部对比代码与工具快照，用于技术对标、消融设计与快速查阅。**体积较大的二进制/缓存已在仓库 `.gitignore` 中排除**，拉仓后若需完整运行某子项目，请按其上游 README 单独准备环境与权重。

---

## 更深文档

| 文档 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | **主推阅读**：对 KIMODO / UMO / MoGenDiT / SOAR / StableMotion 等的结构化分析与和 M2M 的对照 |
| [m2m_ablation_experiments.md](m2m_ablation_experiments.md) | 借鉴 KIMODO / UMO 的 M2M 消融实验清单 |
| 各子目录下的 `CLAUDE.md` / `README.md` | 单个工作的精读笔记或上游说明 |

---

## 一级子目录索引（含哪些「工作」）

以下为 **`ref_repo/` 下每个一级条目** 的一句话定位（按目录名字母序，便于检索）。

| 目录 | 内容概要 |
|------|----------|
| **BrushNet** | ECCV 2024：可插拔图像 inpainting / 双分支分解扩散（与图像编辑管线相关参考）。 |
| **CondMDI** | 「Flexible Motion In-betweening with Diffusion Models」官方实现：条件扩散做 motion in-between。 |
| **HY-SOAR** | 腾讯混元 **HY-SOAR**：rectified-flow 扩散的免奖励后训练 / 轨迹自校正（与 SOAR 系列思路相关）。 |
| **HumanPlus** | 人形影子模仿与模仿学习（HST/HIT 等），含仿真与硬件侧代码。 |
| **KIMODO** | NVIDIA **Kimodo**：大规模可控人体运动生成，imputation 式约束与两阶段 denoiser（**M2M 核心对标之一**）。 |
| **LODGE** | CVPR 2024：**Lodge** 长序列舞蹈生成，由粗到细的扩散与音乐引导。 |
| **MDM** | 经典 **Human Motion Diffusion Model**（HumanML3D 等基准常用基线）。 |
| **MoGenDiT** | 内部 **MoreDiff** 路线：基于 DiT 的 3D 动作修复/去噪（与 `hftrainer` 内 MoGenDIT 管线对接）。 |
| **Momask** | CVPR 2024 **MoMask**：RVQ + masked-modeling T2M。`momask-codes/`（GitHub）+ `paper/MoMask_CVPR2024.pdf` + `weights/{t2m,kit}/`（预训练权重，~460MB）。供 PRISM TMM 重新评测使用。 |
| **MotionStreamer** | ICCV 2025 **MotionStreamer**：因果 TAE + 自回归扩散流式生成。`MotionStreamer/`（GitHub）+ `MotionStreamer_HF/{Causal_TAE,Causal_TAE_t2m_babel,Evaluator_272,Experiments}/` + `humanml3d_272/`（272-dim HumanML3D 数据集）+ `272-dim-Motion-Representation/`（SMPL ↔ 272-dim 转换工具）。**为 PRISM TMM 用 MotionStreamer 的 TMR-272 evaluator 重评所需的全部资产**。 |
| **MotionLCM** | **MotionLCM**：潜在一致性模型，强调实时与可控的人体运动生成。 |
| **OmniControl** | **OmniControl**：任意关节、任意时刻的空间/轨迹控制生成。 |
| **OmniH2O** | 人形 **H2O / OmniH2O** 全身遥操作与学习（人→人形）。 |
| **PHC** | ICCV 2023 **PHC**：物理仿真中的人形实时模仿与稳健控制。 |
| **SOAR** | 扩散 **post-training / exposure bias** 相关论文材料（与 HY-SOAR 并列参考，详见 [CLAUDE.md](CLAUDE.md)）。 |
| **StableMotion** | SIGGRAPH Asia：**StableMotion** 损坏动作检测与修复（cleanup / detect-and-fix）。 |
| **TeSMo** | ECCV 2024：**TeSMo** 场景中人物交互动作 + 文本控制。 |
| **UMO** | **UMO** 统一运动操作（preserve/generate/edit 等），Temporal Fusion 注入上下文（**M2M 核心对标之一**）。 |
| **UnderPressure** | 足部接触、地面反力与 footskate cleanup 的深度学习方法（足部质量相关参考）。 |
| **WanVACE** | **Wan2.1 + VACE** 视频生成与编辑一体化参考实现（`code/` 内为上游结构）。 |

### 根目录其他文件

| 文件 | 说明 |
|------|------|
| **DiffNet补丁.rar.zip** | 历史补丁/资源归档（非代码树；按需手动解压使用）。 |
| **CLAUDE.md** / **m2m_ablation_experiments.md** | 见上文「更深文档」。 |

---

## 使用说明

1. **版权与授权**：各子目录遵循其各自开源协议；内部工程（如 **MoGenDiT**）仅限约定范围内使用。  
2. **与主仓库的关系**：主训练框架在仓库根目录 `hftrainer/`；`ref_repo` **不参与默认训练依赖**，仅作参考与实验对照。  
3. **更新策略**：新增对标工作时，请在本 README 表中补充一行，并在必要时更新 [CLAUDE.md](CLAUDE.md)。

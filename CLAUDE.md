# HF-Trainer：基于 HuggingFace 的统一训练框架

本仓库是 **config 驱动 + Registry 注册** 的人体运动生成训练/推理框架，核心抽象为 `ModelBundle`（Trainer 与 Pipeline 共享同一套 forward 逻辑）。当前主线方法见下表；框架还支持 HyMotion T2M、UMO、MotionCLIP 等，见各 `configs/` 子目录。

---

## 文档边界：CLAUDE.md vs README.md

- `CLAUDE.md` 是**对内工程协作手册**：写给仓库维护者、Agent、实验同学。这里记录目录约定、运行纪律、结果归档规则、临时目录清理规则、集群/凭证/调试注意事项，可以包含尚未公开的 WIP 规划。
- `README.md` 是**对外发布入口**：写给开源用户。只放稳定定位、安装、快速开始、Model Zoo、公开 API、公开评测协议和文档索引；不要放内部凭证、临时实验记录、未验证结论、集群私有路径。
- 新增或更新文档时先判断受众：内部执行规则放 `CLAUDE.md` 或子目录 `CLAUDE.md`；可公开复现说明放 `README.md`、`docs/model_zoo/`、`docs/motion/`、`docs/design/`。

---

## 仓库方法一览

| 方法 | 论文/代号 | 任务 | ModelBundle | Trainer | Pipeline | 配置目录 |
|------|-----------|------|-------------|---------|----------|----------|
| **PRISM** | PRISM (TMM/ECCV) | 文本/姿态条件 Text-to-Motion；latent diffusion + VAE | `PrismBundle` | `PrismTrainer` | `PrismPipeline` | `configs/prism/` |
| **MCM** | PRISM-MCM（音乐条件运动） | 音频条件运动生成；WanVACE 式稀疏 control 分支 | `PrismMCMBundle` | `PrismMCMTrainer` | `PrismMCMPipeline` | `configs/prism/prism_mcm_*.py` |
| **VersatileMotion** | VerMo | 多模态离散 token 运动 LM（T2M、M2T、预训练等） | `VermoBundle` | `VermoTrainer` | `VerMoPipeline` | `configs/vermo/` |
| **HYMotion M2M** | HyMotion M2M | 通用运动补全/编辑（VACE + Flow Matching） | `HyMotionM2MBundle` | `HyMotionM2MTrainer` / `HyMotionM2MSoarTrainer` | `HyMotionM2MPipeline` | `configs/hymotion_m2m/` |
| **PhysFlow** | PhysFlow | KIMODO-G1 在线对抗微调；冻结物理 judge + reward-weighted SFT | `PhysFlowBundle` | `PhysFlowTrainer` | —（训练内 rollout，无独立 Pipeline） | `configs/physflow/` |

### 方法要点（读代码前速览）

- **PRISM**：`hftrainer/models/motion/prism/` — DiT + motion VAE + T5 文本；138-dim 表示（`abs_rel` transl + rot6d）。
- **MCM**：继承 `PrismBundle`，主 transformer 冻结，仅训练稀疏 audio control 分支（`PrismVACEControlTransformer`）。
- **VersatileMotion (VerMo)**：`hftrainer/models/motion/vermo/` — 将运动量化为 token，用因果 LM（LLaMA/Qwen 等）做多任务生成。
- **HYMotion M2M**：`hftrainer/models/motion/hymotion_m2m/` + [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) — 135-dim、VACE 四通道输入、7 种 mask 策略；详尽的 eval/canonical/rot6d 约束见子文档。
- **PhysFlow**：`hftrainer/models/motion/physflow/` — 包装 `ref_repo/KIMODO` 的 G1 生成器；预提取 `text_feat`（不加载 8B 编码器）；`scripts/embodied/` 含 IsaacGym/MuJoCo 打分与 Taiji 提交脚本。

### 运动表示对照（跨方法）

| 方法 | motion_dim | 说明 |
|------|------------|------|
| PRISM / MCM / VerMo | 138 | `abs_rel` transl(6) + rot6d(132) |
| HYMotion M2M | 135 | `abs` transl(3) + rot6d(132) |
| PhysFlow (KIMODO-G1) | 外部 | 经 KIMODO `motion_rep`，非本框架统一 135/138 |

---

## 目录结构及用途

```
hf_trainer/
├── CLAUDE.md                 # 本索引（Agent 入口）
├── README.md                 # 对外简介与 MkDocs 入口
├── pyproject.toml            # 包安装 (pip install -e .)
│
├── hftrainer/                # 核心 Python 包
│   ├── registry.py           # MMEngine Registry：Bundle / Trainer / Pipeline / Dataset …
│   ├── runner/               # AccelerateRunner：分布式、checkpoint、训练循环
│   ├── models/               # ModelBundle 与各任务子模块
│   │   └── motion/           # PRISM / VerMo / M2M / PhysFlow …（见 motion/CLAUDE.md）
│   ├── trainers/             # 训练逻辑（按任务分子目录 motion/）
│   ├── pipelines/            # 推理逻辑（按任务分子目录 motion/）
│   ├── datasets/             # 数据集与 transform（motionhub、classification 等）
│   ├── hooks/                # LoggerHook、CheckpointHook、EMAHook
│   ├── evaluation/           # 评测器；quality_check_rules/ 为动作质检规则
│   ├── visualization/        # TensorBoard / 文件可视化
│   └── utils/                # checkpoint、分布式等工具
│
├── configs/                  # 可运行训练配置 (.py，支持 _base_ 继承)
│   ├── _base_/               # 公共 runtime、optimizer、hook 模板
│   ├── prism/                # PRISM T2M / TP2M / MCM
│   ├── vermo/                # VersatileMotion 预训练与 SFT
│   ├── hymotion_m2m/         # M2M 主线（含 soar/ 后训练）
│   ├── hymotion_t2m/         # HyMotion 文本生成运动（T2M）
│   ├── hymotion_umo/         # UMO 风格适配（外部论文复现）
│   ├── physflow/             # PhysFlow 在线对抗微调
│   ├── motion_clip/          # MotionCLIP
│   └── experiments/          # 实验性/一次性配置（含 PhysFlow 语料等）
│
├── tools/                    # 核心 CLI（保持精简）
│   ├── train.py              # 训练入口 → AccelerateRunner
│   ├── infer.py              # 推理入口
│   ├── dist_train.sh         # accelerate launch 封装
│   ├── taiji_submit.py       # 太极集群提交（必须用此脚本，勿手写 taiji_client）
│   ├── taiji_dist_train.sh   # 集群内分布式训练
│   └── taiji_*.py / taiji_template.json
│
├── scripts/                  # 按功能划分的工具脚本（非框架 API）
│   ├── eval/                 # M2M/KIMODO 全任务评测、指标、multiseed
│   ├── data/                 # 数据构建、转换、统计
│   ├── repair/               # MoGenDIT 等修复管线
│   ├── embodied/             # PhysFlow：G1 仿真、reward、在线对抗、Taiji 提交
│   ├── kimodo/               # KIMODO baseline 评测与可视化拼接
│   ├── guidance/             # 关键帧/关键姿态引导实验
│   ├── caption/              # Caption 改写
│   ├── analysis/             # 统计与画图
│   ├── debug/                # 诊断、padding/rot6d 校验
│   ├── submit/               # 集群任务辅助提交
│   ├── inference/            # 批量推理
│   └── patch/, e9/, e14/, …  # 历史实验与一次性补丁
│
├── tests/
│   ├── smoke/                # pytest -m smoke：各任务 1 step train + infer
│   └── unit/, integration/   # 单元与集成测试
│
├── data/                     # 数据与统计（体量大，部分路径为 symlink）
│   ├── annotation/           # 训练列表 JSON（如 train_hymotion_400h.json）
│   ├── motionhub/            # MotionHub 管线数据
│   ├── hymotion_m2m_data/     # M2M mean/std 等
│   ├── kimodo_text_feature/    # PhysFlow/KIMODO 预提取文本特征
│   └── statistic/            # PRISM/VerMo 等归一化统计
│
├── checkpoints/              # 预训练权重与 symlink（PRISM、HY-Motion、KIMODO、MoGenDIT…）
├── work_dirs/                # 训练输出：config 快照、log、checkpoint
├── outputs/                  # 规范推理/评测/可视化产物；目录契约见下文
├── output/                   # 历史遗留输出目录；新脚本默认不要再写入
├── logs/                     # 部分任务的运行日志
│
├── motion_annot_web/         # 动作标注与评测 Web（见 motion_annot_web/CLAUDE.md）
│   ├── m2m_database/         # 质量分层与修复进度
│   ├── eval_dashboard/       # 评测看板（需 --save-npz）
│   ├── score_m2m/            # M2M 打分与对比
│   └── …                     # repair viewer、keypose_eval 等
│
├── docs/
│   ├── design/CLAUDE.md      # 框架设计（checkpoint、多优化器等）
│   ├── temp/                 # **临时**方案/评测/调研（WIP 必须放这里）
│   └── *.md, en/, zh-cn/     # MkDocs 公开文档
│
├── papers/                   # 各方法 LaTeX 工程（Overleaf 同步见 README_OVERLEAF_SYNC.md）
│   ├── PRISM_TMM2026, PRISM_ECCV2026, VerMo_*, HYMotionM2M_*, PhysFlow, lzy_thesis, …
│
├── ref_repo/                 # 外部 baseline 源码与分析（KIMODO、UMO、MoGenDIT、SOAR…）
├── assets/                   # 文档/演示用静态资源
└── .claude/skills/           # 项目内 Agent skills（taiji、autodebug 等）
```

---

## outputs/ 目录结构契约

`outputs/` 是推理、评测、转换、可视化导出的主目录。所有新脚本默认写 `outputs/`，不要再新增 `output/` 路径；`output/` 只作为历史结果目录保留。`data/`、`checkpoints/`、`outputs/` 保持在仓库根目录，不收进 `artifacts/`。

### 顶层分区

```text
outputs/
├── evaluation/                # 正式评测与可复现推理结果，默认长期保留
├── inference/                 # 非评测集的普通批量推理结果，按项目/方法归档
├── visualization/             # viewer/web/论文图导出的可视化资产
├── conversion/                # 表示转换、retarget、repack 的中间或最终结果
├── diagnostics/               # debug/ablation/health-check 结果，短中期保留
├── tmp/                       # 临时文件，可随时清理；禁止存唯一重要结果
└── _archive/                  # 手动归档的旧结构结果，只读参考，不作为新输出目标
```

### 正式评测路径

所有测试集上的正式推理和评测结果统一放：

```text
outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/
```

命名规则：

- `{task}` 使用稳定任务族名：`t2m`、`m2m`、`semantic_edit`、`repair`、`control`、`interaction_t2m`、`retarget`、`embodied_tracking`、`physics_eval`。
- `{test_dataset}` 使用数据集与 split/protocol：`humanml3d_test`、`humanml3d_official_test`、`motionstreamer_h3d272_test`、`motionfix_test`、`interhuman_test`、`interx_test`、`kimodo_hml3d_test`、`amass_g1_test`。
- `{motion_representation}` 使用当前目录中**直接存储的动作表示**：`motion135`（SMPL motion_135）、`hml263`（HumanML3D-263）、`ms272`（MotionStreamer-272）、`motion201`、`soma`、`g1`、`interhuman262`、`interx_hhi`。正式结果不要再用 `multi_rep`；如果一个 run 产生多种表示，必须拆到多个 representation 目录下。
- 每个 T2M method 至少要保存一份 SMPL 结果：`motion135/{method}/`。为了方便 MotionStreamer Evaluator / HumanML3D evaluator，可以额外保存 `ms272/{method}/` 和 `hml263/{method}/`。这些是同一个 method 的不同表示，不是不同 method。
- `{method}` 是 debug 完成后的稳定方法名，越短越好，使用 lower snake case。只有真实长期版本差异才进入方法名，例如 `hymotion_1b`、`hymotion_lite`；普通运行设置不进入目录名。
- 禁止在正式 method 目录名中使用这些后缀：`selected_caption`、`official_test`、`exactlen`、日期、`from_motion135`、`prep`、`predictions`、`mdmstats`、`fix`、`smoke`、`debug`、`rerun`、`vermo`。这些信息写入 `run_config.json` / `command.txt` / `metrics/*.json`。
- Caption protocol 属于 `{test_dataset}` 的评测协议和 `run_config.json`，不是 method 后缀。当前 `humanml3d_official_test` 的正式语义评测默认使用 selected GT caption，目录名不再写 `selected_caption`。
- `motionclip135` 不是正式动作表示，而是面向 MotionCLIP evaluator 的 SMPL `motion_135` 重映射/评测输入。正式结果仍存 `motion135/{method}/`；MotionCLIP 的派生输入放 `metrics/` 或 `_suites/`，不要提升成 canonical representation。
- 同一个目录内必须能自解释：至少保留 `run_config.json` 或 `command.txt`、必要的预测文件、`metrics/` 和转换日志。

推荐正式结构（注意预测文件直接平铺在 representation/method 目录下，不再包一层 `predictions/motion135/` 或 `prep/`）：

```text
outputs/evaluation/t2m/humanml3d_official_test/
├── motion135/
│   ├── gt_0beta/
│   │   ├── 000000.npz
│   │   ├── ...
│   │   ├── run_config.json
│   │   └── metrics/
│   ├── prism/
│   ├── hymotion_1b/
│   ├── hymotion_lite/
│   ├── motionstreamer/
│   ├── mdm/
│   ├── momask/
│   ├── t2mgpt/
│   ├── motiongpt3/
│   └── kimodo/
├── ms272/
│   ├── gt_0beta/
│   ├── prism/
│   ├── hymotion_1b/
│   ├── hymotion_lite/
│   ├── motionstreamer/
│   ├── mdm/
│   ├── momask/
│   └── kimodo/
├── hml263/
│   ├── gt/
│   ├── mdm/
│   ├── momask/
│   ├── t2mgpt/
│   └── motiongpt3/
├── captions/
├── _suites/                   # evaluator batch outputs only, never canonical predictions
├── _runs/                     # before-confirmation experimental runs
└── _tmp/                      # temporary, safe to delete
```

目录内推荐内容：

```text
outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/
├── 000000.npz                  # 或 .npy；该目录的主表示文件，直接平铺
├── 000019.npz
├── ...
├── run_config.json             # ckpt、seed、steps、CFG、caption protocol、conversion source
├── command.txt                 # 完整命令、环境变量、Taiji task/instance
├── metrics/                    # evaluator JSON，不手抄指标
│   ├── motionstreamer.json
│   ├── motionclip.json
│   ├── hml263.json
│   ├── physics.json
│   └── summary.json
├── logs/                       # shard/Taiji/失败样本日志
└── visualization/              # 可选；正式 viewer 派生产物
```

轻量任务可以省略空子目录。debug 期间可以把带设置名的输出放到 `_runs/{method}_{setting}/`；一旦该 method 被确认，必须提升/迁移到上述 canonical `{representation}/{method}/`，旧 debug 目录删除或移入 `outputs/diagnostics/`，避免后续评测拿错。

### HumanML3D official T2M caption protocol

HumanML3D official test 的原始 `texts/<id>.txt` 往往有多条 full-clip caption。已经确认部分样本的第一条 full-clip caption 与 GT motion 完全不匹配，因此正式 T2M 语义评测不要再默认使用 first caption。

当前可信协议：

- 正式 HumanML3D official test T2M 语义评测使用 motionclip-selected official caption：
  `outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json`
- 该 annotation 覆盖 4042 个 official test id；每个 entry 的 `hierarchical_caption_path` 指向同目录下 `annotation_captions/<id>.json`，并带有：
  - `caption_source = motionclip_selected_official_humanml3d_caption`
  - `caption_selection_policy = best_motionclip_distance_over_official_full_captions`
- 配套文件：
  - `caption_map.json`：每个 id 的全部候选、选中 caption、first-full 对比和选择原因。
  - `prompt_map.json`：`{id: selected_caption}`，供只接受 flat prompt map 的推理脚本使用。
  - `changed_from_first_full.tsv`：与旧 first-full caption 不同的样本。
  - `needs_review.tsv`：MotionCLIP margin 较小、建议人工复查的样本。

旧文件 `data/annotation/test_hml3d_official272_gtlen.json` 仍保留为原始 official-272 split/length 描述，其中 caption path 是 first-full derived captions；它不能作为当前 T2M 语义指标的默认 caption source。任何新推理、评测或 leaderboard 结果必须在 `run_config.json` 或 `command.txt` 中显式记录使用上面的 selected annotation / prompt map。若脚本只支持 `--caption_protocol original`、`first` 或随机 caption，先改脚本或传入 explicit caption map，不要静默 fallback。

### Evaluator normalization TODOs

- HML3D-based 方法在计算 SMPL-based evaluator（例如 MotionCLIP / MotionStreamer-272）时，必须区分两类 GT reference：
  - `raw_gt`：原始 SMPL / MS272 GT，衡量端到端输出与真实 SMPL 分布的差距，会包含 `HML263 -> SMPL` 桥接误差。
  - `hml_roundtrip_gt`：`GT SMPL -> HML263 -> SMPL -> evaluator representation` 后的 GT，使用与方法预测完全相同的 root-restore、IK refine、fps 和编码路径，用于估计去除表示转换偏差后的模型质量。
- MotionCLIP-135 的当前正式 leaderboard 协议使用 raw projection embedding（`l2_normalize=false`）计算 R-Precision / MM-Dist / FID / Diversity；旧的 L2-normalized MotionCLIP FID/MM 数值只能作为历史诊断，不要混入正式表格。
- 不要把 `raw_gt` FID 和 `hml_roundtrip_gt` FID 混成同一列；leaderboard 或 paper 表格必须在列名/脚注里明确 reference。建议同时保留 `GT raw -> hml_roundtrip` 的 bridge calibration row，量化转换本身带来的 FID/几何损失。
- TODO：训练一个不受 SMPL 多解性（尤其 joint-position -> SMPL twist / leaf orientation ambiguity）影响的 motion quality evaluator。目标是直接评价动作语义、时序自然性、物理合理性和关节轨迹质量，而不是依赖某一种 SMPL local rotation 解。

### 普通推理路径

非正式 benchmark、demo、用户 prompt、批量生成放：

```text
outputs/inference/{project_or_task}/{method}_{setting}/{run_id}/
```

`run_id` 建议用 `YYYYMMDD_HHMMSS` 或简短实验名。若后来被纳入正式评测，应迁移或复制到 `outputs/evaluation/...`，不要只在 inference 目录里报指标。

### 可视化路径

可视化导出放：

```text
outputs/visualization/{viewer_or_figure}/{dataset_or_task}/{method}_{setting}/
```

例如 T2M viewer 标准样例可放 `outputs/visualization/t2m_compare/humanml3d_test/momask_ts10_cfg4/`。若 viewer 文件是正式评测目录的派生产物，也可以放在对应 `outputs/evaluation/.../visualization/` 下，并在 viewer 文档中指向该路径。

### 转换与诊断路径

- 表示转换或 retarget 独立任务放 `outputs/conversion/{source_repr}_to_{target_repr}/{dataset}/{method}_{setting}/`。
- debug、ablation、质量排查放 `outputs/diagnostics/{topic}/{YYYYMMDD}_{short_name}/`。
- 诊断结果如果后来被用于论文或 model card，必须提升到 `outputs/evaluation/...` 或 `outputs/visualization/...`，并补齐命令与配置。

### 临时目录

所有可清理临时文件统一放：

```text
outputs/tmp/{YYYYMMDD}_{owner_or_topic}/
```

规则：

- `outputs/tmp/` 下内容默认可随时删除；不要把唯一 checkpoint、唯一指标 JSON、唯一可视化结果放在这里。
- 临时目录最多保留短期 debug 上下文；若超过一周仍有价值，迁移到 `outputs/diagnostics/` 或正式目录。
- 脚本中的 scratch/cache/temp 参数一律默认指向 `outputs/tmp/...`，避免散落到 `scripts/`、`docs/temp/`、源码目录或仓库根目录。

---

## 目标重构架构（WIP，逐步迁移）

> 本节是**重构方向**，不是当前现状。会"慢慢重构"：旧路径保留为 re-export 薄 wrapper，
> 现有 import 与实验零改动；新代码走新路径。每阶段跑 `pytest -m smoke` 守住可用性。

### 设计目标

把当前"领域能力埋在 `models/motion/components/` 下、模型/任务/数据语义耦合"的结构，
重组为**分层、按职责命名、偏深**的开源 motion 库。库名仍为 `hftrainer`，所有实现都在其下。

一条贯穿规则：**目录名即"职责类型"（表示？人体？网络？推理？），"属于哪个方法/任务"
由目录内的文件/子目录名回答**。因"整库皆 motion"，不再设 `motion/` 命名空间，
`models/motion`、`datasets/motion` 等的 `motion/` 子层一并取消（与 `configs/<method>` 对齐）。

### 分层

```
L1 领域库（import-light，不依赖 models/trainers/runner）
   representation / body / processing / io / tasks / visualization
L2 数据 & 评测   datasets / evaluation（metrics 并入此处）
L3 模型          models（网络 + ModelBundle）
L4 训练 & 推理   trainers / pipelines（按 method 建子目录）
```

依赖单向：L1 不依赖上层；`evaluation` 依赖其内 `metrics`；`models` 只依赖 L1。

### 目标目录树

```text
hftrainer/
│  registry.py   runner/   hooks/   utils/         # 框架基础设施
│
├─ representation/                  # motion 表示：定义 / 互转 / 规范态载体
│  ├─ clip.py                       #   MotionClip / MotionBatch（内存规范动作对象）
│  ├─ rotation.py                   #   rot6d/aa/quat/matrix 原子转换
│  ├─ channels/                     #   可复用通道：root/rot6d/joint_pos/velocity/contact/ric
│  ├─ specs.py                      #   声明式注册：每种表示 = 通道列表
│  └─ convert.py                    #   中心辐射：任意 repr ⇄ MotionClip 规范态 ⇄ 任意 repr
│
├─ body/                            # "人体/骨架"域
│  ├─ body_models/                  #   base / smpl / smplh / smplx / soma / unitree_g1
│  ├─ skeletons/                    #   关节名·parents·offsets·T-pose：smpl smplx soma g1 humanml
│  ├─ kinematics/                   #   fk(含可微) / ik_ccd / ik_solver / contact
│  └─ retarget/                     #   base / smpl_soma / smpl_g1 / hml_smpl / cross_skeleton
│
├─ processing/                      # 处理算子：canonicalize / normalize / resample / masks / smoothing
├─ io/                              # 格式读写：npz / bvh / fbx / smpl_io / convert(SMPL↔BVH↔FBX)
├─ tasks/                           # 任务语义：specs / condition / instructions
├─ visualization/
│  ├─ render/                       #   skeleton / mesh / video / threejs/
│  └─ backends/                     #   现有 base_visualizer / file / tensorboard 日志器
│
├─ datasets/
│  ├─ base_dataset.py
│  ├─ {dataset}/ …                  #   humanml3d / motionhub / hymotion_m2m …
│  └─ transforms/                   #   dataset 级 formatting（to_tensor/collate）
│
├─ evaluation/                      # metrics 并入
│  ├─ base_evaluator.py
│  ├─ evaluators/                   #   humanml3d_t2m(263) / motionstreamer_272 / physics
│  ├─ metrics/                      #   quality / constraints / distribution（纯算子）
│  └─ quality_check_rules/
│
├─ models/                          # 网络 + ModelBundle（去掉 motion/ 子层，方法平铺）
│  ├─ base_model_bundle.py   peft_utils.py
│  ├─ _shared/                      #   跨方法网络块（WanVACE blocks 等，收编自 components）
│  └─ {method}/                     #   prism vermo hymotion_m2m hymotion_t2m physflow
│                                   #   mdm momask mogents motionstreamer gotozero kimodo mogendit
│
├─ trainers/
│  ├─ base_trainer.py
│  └─ {method}/{method}_trainer.py  #   prism/prism_trainer.py …
│
└─ pipelines/
   ├─ base/                         #   pipeline 继承体系（见下）
   ├─ mixins/                       #   task 接口：t2m / m2t / inbetween / edit / control
   └─ {method}/{method}_pipeline.py #   prism/prism_pipeline.py …
```

### representation：通道组合式 schema（不用 flags 大文件）

表示拆成可复用**通道（channel）**，每种命名表示是通道列表的声明式拼装，重复在通道层消除：

```python
register("rot6d_smpl22",       [RootTransl("abs"),     Rot6DBody(22)])                 # 135
register("rot6d_abs_rel",      [RootTransl("abs_rel"), Rot6DBody(22)])                 # 138 (PRISM/VerMo)
register("rot6d_smpl22_pos21", [RootTransl("abs"),     Rot6DBody(22), JointPos(21)])   # 198 (HYMotion M2M)
register("rot6d_smpl22_pos22", [RootTransl("abs"),     Rot6DBody(22), JointPos(22)])   # 201 (HYMotion T2M)
register("hml3d_263",          [LocalRootVel(), RICJointPos(22), Rot6DBody(22), JointVel(22), FootContact()])
register("hml_272",            [...])                                                  # MotionStreamer/GoToZero
register("soma",               [RootTransl("abs"), Rot6DBody(N_SOMA)])                 # KIMODO/SOMA（补齐）
```

要点：t2m_201 与 m2m_198 仅差一个 `JointPos(22)` vs `JointPos(21)` 通道；命名按结构本质
（`rot6d_smpl22_pos22`），不绑方法；新增表示=加一条声明，新增通道=加一个类；任意两表示经
`MotionClip` 规范态自动互转。

### pipeline 继承体系（base 只放基础设施）

`base_pipeline` 只封装**通用基础设施 + 抽象钩子**，不放任何 task 级 `infer_*`：

```
BasePipeline                     # 设备/精度移动、from_config/pretrained/checkpoint、eval/seed；抽象 generate()
   ├─ BaseDiffusionPipeline      # 去噪循环 + CFG + scheduler；抽象 _predict()
   └─ BaseFlowMatchingPipeline   # 速度场 ODE 积分(Euler/Heun) + sigma；抽象 _velocity()
          └─ {Method}Pipeline + TaskMixin   # 具体方法装配 + infer_* 经 mixin 注入
```

- **当前只抽 `BaseDiffusionPipeline` 与 `BaseFlowMatchingPipeline`**；token 自回归（VerMo/
  MotionStreamer）与 masked（MoMask）暂不强抽公共层，直接在各自方法 pipeline 内实现。
- task 级接口（`infer_t2m / infer_m2t / infer_inbetween / infer_edit / infer_control`）放在
  `pipelines/mixins/`，由多方法复用，内部统一调用范式基类的 `generate(...)`。调用方始终用
  `pipeline.infer_t2m(...)`，职责分明：基础设施 / 范式采样 / 方法装配 / 任务接口。

### 方法 → 范式映射

| 方法 | 表示 | 范式基类 | task |
|------|------|----------|------|
| PRISM / MCM | 138 | `BaseDiffusionPipeline` | T2M / TP2M |
| MDM / MoGenDIT | 263 / 201 | `BaseDiffusionPipeline` | T2M / repair |
| HYMotion M2M | 198 | `BaseFlowMatchingPipeline` | inbetween / edit / control |
| HYMotion T2M / KIMODO | 201 / SOMA | `BaseFlowMatchingPipeline` | T2M / control |
| VerMo | 138 | （方法内实现，token AR） | T2M / M2T / 预训练 |
| MotionStreamer / GoToZero | 272 | （方法内实现，流式 AR） | T2M / stream |
| MoMask | 263 | （方法内实现，masked token） | T2M |

### 开源方法接入

除自研（PRISM/VerMo/M2M/T2M/PhysFlow）外，纳入开源方法 **MDM / Momask / MotionStreamer /
GoToZero / KIMODO / MoGenDIT** 等：**至少**实现网络（`models/{m}/`）+ 推理 pipeline
（`pipelines/{m}/`，优先复用官方权重）；trainer 视需要再补。

### 冲突消解（重构时注意）

| 新增 | 与现有冲突 | 处理 |
|------|-----------|------|
| 领域可视化 | `hftrainer/visualization/`（训练日志器） | 合一：`render/`(画帧) + `backends/`(现有日志器) |
| 领域处理算子 | `datasets/transforms/`（formatting） | 领域算子叫 `processing/`，dataset transforms 不动 |
| 纯指标 | `evaluation/` | `metrics/` 降为 `evaluation/metrics/`，evaluator 编排在外 |
| `body/body_models/` | 顶层 `models/`（神经网络） | 全路径区分：`hftrainer.body.body_models.smpl` vs `hftrainer.models.prism` |

---

## 子文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| 框架设计 | [`docs/design/CLAUDE.md`](docs/design/CLAUDE.md) | 分模块控制、checkpoint、多优化器、评测/可视化 |
| Motion 公共库设计 | [`docs/design/motion_library.md`](docs/design/motion_library.md) | `hftrainer.motion` 目标架构；从 `models.motion` 迁移公共动作能力 |
| 运动任务栈 | [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) | **HYMotion M2M** 主文档：VACE、mask、eval canonical、历史 bug |
| 动作标注 Web | [`motion_annot_web/CLAUDE.md`](motion_annot_web/CLAUDE.md) | 质检、修复、评测看板 |
| Baseline 调研 | [`ref_repo/CLAUDE.md`](ref_repo/CLAUDE.md) | KIMODO、UMO、MoGenDIT |
| KIMODO/SMPL 重定向 | [`docs/kimodo_smpl_retargeting.md`](docs/kimodo_smpl_retargeting.md) | 常用 `hftrainer.models.motion.components.retarget` API；KIMODO/SOMA ↔ SMPL `motion_135` |
| 质检规则 | [`hftrainer/evaluation/quality_check_rules/CLAUDE.md`](hftrainer/evaluation/quality_check_rules/CLAUDE.md) | Checker 与优先级 |
| 博士论文 | [`papers/lzy_thesis/THESIS_OVERVIEW.md`](papers/lzy_thesis/THESIS_OVERVIEW.md) | 论文总览 |
| 临时文档 | [`docs/temp/`](docs/temp/) | 方案草案、评测计划、调试笔记（**禁止**散落到源码旁） |

---

## Agent 工作规范

1. **先读代码再下结论**：不熟悉模块先读源码，勿凭“常见做法”猜测。
2. **修复须跑通实验**：除 smoke 外，应跑相关训练/评测脚本并看到合理指标后再称“已修复”。
3. **环境可用**：`python3 -m pytest -m smoke tests/smoke/` 做快速校验。
4. **措辞**：未跑实验时用“根据代码分析，可能因为…”；确认后再写“已确认修复”。
5. **评测入库须 `--save-npz`**：`scripts/eval/eval_m2m_v2_all_tasks.py` 供 dashboard 使用时必须带 `--save-npz`；caption 模型还须 `--use-rewritten`。见 [`motion_annot_web/eval_dashboard/CLAUDE.md`](motion_annot_web/eval_dashboard/CLAUDE.md)。

---

## 太极集群提交

```bash
python3 tools/taiji_submit.py <任务名> <config路径> --host_num <节点数>

# 示例
python3 tools/taiji_submit.py m2m_train configs/hymotion_m2m/hymotion_m2m_smoke.py --host_num 2
python3 tools/taiji_submit.py physflow_adv configs/physflow/physflow_online_adv_smoke.py --host_num 1
```

**禁止**手动拼接 `taiji_client start -scfg`（缺少 `template_flag` 等字段）。状态：`taiji_client trl` / `il` / `td` / `stop`。

### 常驻实验实例（训练/推理统一在此进行）

以后**所有实验（训练、推理、调试）默认都在下面这个常驻实例上跑**，不再为每个实验单独提交太极任务：

| 项 | 值 |
|------|------|
| task_flag | `train_keyframe-A100PRO-8x8-2604301946` |
| instance_id | `8b1d891a9dd9859d019dde36a2770336` |
| 规格 | 8 节点 × 8 卡 A100-SXM4-40GB（共 64 卡），每节点 192 核 / ~1003GB 内存 |
| 仓库路径 | `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`（cephfs 已挂载，与本地同路径） |
| 运行环境 | `/usr/local/bin/python3`（Python 3.10.16），`torch 2.5.0+cu118`，每节点可见 8 卡 |
| 节点列表 | host 0 = launcher，host 1~7 = worker-0~worker-6 |

**方式 A — 原生 `taiji_client exec`（在真实终端里交互登录，推荐人工使用）：**

```bash
export TOKEN=<太极token>

# 交互式登录，进入实例的一个 shell（会提示选 host 0~7，输入编号回车）
taiji_client exec train_keyframe-A100PRO-8x8-2604301946 8b1d891a9dd9859d019dde36a2770336 bash

# 直接在某个 host 上跑单条命令（同样会先提示选 host）
taiji_client exec train_keyframe-A100PRO-8x8-2604301946 8b1d891a9dd9859d019dde36a2770336 \
    bash -c "cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer && nvidia-smi"
```

提示选 host 时：`0` = launcher，`1~7` = worker-0~worker-6。

**方式 B — `taiji_exec.py`（非交互环境必须用，如 Cursor/Claude Code；`taiji_client exec` 需要 TTY 会阻塞）：**

```bash
# 用法：python3 tools/taiji_exec.py <task_flag> <instance_id> "<命令>" [timeout秒] [host_index]
# host_index 默认 0（launcher）；也可用环境变量 TAIJI_EXEC_HOST_INDEX 指定

export TOKEN=<太极token>

# 单条命令（默认 60s 超时，跑在 launcher）
python3 tools/taiji_exec.py train_keyframe-A100PRO-8x8-2604301946 8b1d891a9dd9859d019dde36a2770336 \
    "cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer && nvidia-smi" 40

# 指定在 worker-0（host_index=1）上执行
python3 tools/taiji_exec.py train_keyframe-A100PRO-8x8-2604301946 8b1d891a9dd9859d019dde36a2770336 \
    "hostname" 30 1

# 长任务：容器内 nohup 后台跑，再定期 tail 日志
python3 tools/taiji_exec.py train_keyframe-A100PRO-8x8-2604301946 8b1d891a9dd9859d019dde36a2770336 \
    "cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer && nohup bash tools/dist_train.sh <config> > /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/<run>.log 2>&1 &" 20
python3 tools/taiji_exec.py train_keyframe-A100PRO-8x8-2604301946 8b1d891a9dd9859d019dde36a2770336 \
    "tail -50 /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/<run>.log" 20
```

> 注意：`il` 显示该 instance_id 的 `IsSuccess=false` 是历史字段，不影响使用；`State=TRAINING_RUNNING` 即实例存活可登录。访问失败若报 `No permissions / get project info failure`，说明 token 无该项目权限（与本实例无关）。

---

## 设计原则与 API

1. **Config + Registry**：MMEngine `Config` + 自研 `Registry`
2. **Accelerate 原生**：DDP/FSDP/混合精度/梯度累积
3. **HuggingFace 优先**：diffusers / transformers / peft 直用
4. **ModelBundle 共享**：`from_config` / `from_pretrained` / `save_pretrained`（若可导出 HF 产物）
5. **分模块控制**：`trainable`、`save_ckpt`、`module_dtype` 等由 config 指定

**训练入口**：`tools/train.py` → `AccelerateRunner.from_cfg(cfg)` → `runner.train()`  
**分布式**：`tools/dist_train.sh`（内部 `accelerate launch`）

---

## Smoke 测试

```bash
python3 -m pytest -m smoke tests/smoke/test_task_startup.py
```

当前覆盖：`prism`、`prism_mcm`、`vermo`、`hymotion_m2m`、`hymotion_t2m`（各 1 iter train + infer）。新增任务栈须加 smoke case。

---

## 关键依赖

`torch>=2.0`、`accelerate`、`transformers`、`diffusers`、`peft`、`mmengine`（仅 Config/Registry）、`safetensors`、`wandb`/`tensorboard`

---

## 凭证（非高敏，写入文档避免频繁询问）

| 服务 | Token / 说明 |
|------|------|
| 太极平台 | `TOKEN=HzrPZC3djhwaU9HPdEA_Bg`（详见 `.claude/skills/taiji/CLAUDE.md`） |
| HuggingFace | `HF_TOKEN=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`（用户 `ZeyuLing`，write 权限，用于 Model Zoo artifact 上传） |

---

## 框架级注意事项（简表）

| 主题 | 说明 |
|------|------|
| Bundle 级参数 | 2026-03-27 已修：`nn.Parameter` / `register_buffer` 直接挂在 Bundle 上会进入优化器与 checkpoint；见 `docs/design/CLAUDE.md` |
| M2M 训练数据质量 | 多数 config 仍用未过滤的 `train_hymotion_400h.json`；高质量子集见 `motion_annot_web/m2m_database` / `data/hymotion_m2m_refine_data/data_quality_list/high_quality.json` |
| MoGenDIT 修复 | 外部仓库 + `MoGenDITRepairPipeline`；`ada_denoise` **不**用 adaptive mask 做 imputation，见 motion/CLAUDE.md |
| MoGenTS T2M | 本地 native runtime：`hftrainer/models/motion/mogents/`；官方 raw 权重转 artifact 用 `scripts/eval/convert_mogents_checkpoint.py`；正式 HML263 结果放 `outputs/evaluation/t2m/humanml3d_official_test/hml263/mogents/`，采样步数、CFG、seed 等写入 `run_config.json` |
| PhysFlow 环境 | 物理 judge 可能需独立 Python（如 IsaacGym）；见 `PhysFlowBundle` 与 `scripts/embodied/` |

---

## 外部集成（非一等公民）

| 组件 | 位置 | 说明 |
|------|------|------|
| MoGenDIT | `ref_repo` + `mogendit_pipeline.py` | 201-dim 修复 diffusion，`scripts/repair/` |
| KIMODO | `ref_repo/KIMODO` | PhysFlow 与 `scripts/kimodo/` 评测共用 |
| HyMotion T2M / UMO | `configs/hymotion_t2m`, `hymotion_umo` | 相关论文复现与 M2M 预训练来源 |

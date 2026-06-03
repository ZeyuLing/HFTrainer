# HF-Trainer：基于 HuggingFace 的统一训练框架

本仓库是 **config 驱动 + Registry 注册** 的人体运动生成训练/推理框架，核心抽象为 `ModelBundle`（Trainer 与 Pipeline 共享同一套 forward 逻辑）。当前主线方法见下表；框架还支持 HyMotion T2M、UMO、MotionCLIP 等，见各 `configs/` 子目录。

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
├── output/, outputs/         # 推理与实验产物（NPZ、指标 JSON 等）
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

## 子文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| 框架设计 | [`docs/design/CLAUDE.md`](docs/design/CLAUDE.md) | 分模块控制、checkpoint、多优化器、评测/可视化 |
| 运动任务栈 | [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) | **HYMotion M2M** 主文档：VACE、mask、eval canonical、历史 bug |
| 动作标注 Web | [`motion_annot_web/CLAUDE.md`](motion_annot_web/CLAUDE.md) | 质检、修复、评测看板 |
| Baseline 调研 | [`ref_repo/CLAUDE.md`](ref_repo/CLAUDE.md) | KIMODO、UMO、MoGenDIT |
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

## 框架级注意事项（简表）

| 主题 | 说明 |
|------|------|
| Bundle 级参数 | 2026-03-27 已修：`nn.Parameter` / `register_buffer` 直接挂在 Bundle 上会进入优化器与 checkpoint；见 `docs/design/CLAUDE.md` |
| M2M 训练数据质量 | 多数 config 仍用未过滤的 `train_hymotion_400h.json`；高质量子集见 `motion_annot_web/m2m_database` / `data/hymotion_m2m_refine_data/data_quality_list/high_quality.json` |
| MoGenDIT 修复 | 外部仓库 + `MoGenDITRepairPipeline`；`ada_denoise` **不**用 adaptive mask 做 imputation，见 motion/CLAUDE.md |
| PhysFlow 环境 | 物理 judge 可能需独立 Python（如 IsaacGym）；见 `PhysFlowBundle` 与 `scripts/embodied/` |

---

## 外部集成（非一等公民）

| 组件 | 位置 | 说明 |
|------|------|------|
| MoGenDIT | `ref_repo` + `mogendit_pipeline.py` | 201-dim 修复 diffusion，`scripts/repair/` |
| KIMODO | `ref_repo/KIMODO` | PhysFlow 与 `scripts/kimodo/` 评测共用 |
| HyMotion T2M / UMO | `configs/hymotion_t2m`, `hymotion_umo` | 相关论文复现与 M2M 预训练来源 |

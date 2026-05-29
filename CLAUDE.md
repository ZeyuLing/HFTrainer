# HF-Trainer: A Unified Training Framework Built on HuggingFace Ecosystem

## Sub-Documents

| Document | Location | Content |
|----------|----------|---------|
| **Framework Design** | [`docs/design/CLAUDE.md`](docs/design/CLAUDE.md) | Per-module control, Trainer/Pipeline sharing, multi-optimizer, checkpoint, evaluator/visualizer |
| **Motion Task Stack** | [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) | HyMotion M2M: VACE conditioning, motion representation, rot6d conventions, mask patterns, bug history |
| **Motion Annotation Web** | [`motion_annot_web/CLAUDE.md`](motion_annot_web/CLAUDE.md) | Web tools for motion quality management, repair, scoring, keypose editing evaluation |
| **Baseline Research** | [`ref_repo/CLAUDE.md`](ref_repo/CLAUDE.md) | KIMODO, UMO, MoGenDIT analysis |
| **PhD Thesis** | [`papers/lzy_thesis/THESIS_OVERVIEW.md`](papers/lzy_thesis/THESIS_OVERVIEW.md) | 博士论文：面向智能化动作创作的可控人体运动生成关键技术研究。Overleaf synced via `papers/lzy_thesis/project/` |
| **Quality Checkers** | [`hftrainer/evaluation/quality_check_rules/CLAUDE.md`](hftrainer/evaluation/quality_check_rules/CLAUDE.md) | Checker review, per-checker bug analysis, mask accuracy issues, P0-P2 priority matrix |
| **Temp Solutions** | [`docs/temp/`](docs/temp/) | Temporary documents, solution proposals, evaluation plans. All WIP/draft docs go here. |

---

## Temporary Documents Policy

All temporary documents, solution proposals, evaluation plans, and research drafts **must be stored in `docs/temp/`**. This includes:
- Feature design proposals (pre-implementation)
- Evaluation plans and reports
- Research surveys and comparisons
- Bug analysis and debugging notes

Do NOT place temporary/draft documents in other locations (e.g., alongside source code, in `ref_repo/`, etc.).

---

## Agent Working Norms

These norms apply to all AI agents working in this repo. **Priority over any default behavior**.

1. **Read code first, don't guess**: For unfamiliar modules, read source before making judgments. No guessing based on "common practice".
2. **Must run to confirm fix**: Actually run experiment scripts (not just smoke tests), observe correct results before claiming "fixed". Loss curve must decline, metrics must improve.
3. **Environment is ready**: Training dependencies configured. Use `python3 -m pytest -m smoke tests/smoke/` for quick validation.
4. **Wording discipline**: If not yet run, say "based on code analysis, likely because...". Only use "confirmed fixed" after observing expected behavior.
5. **Eval runs must save NPZ**: Any `scripts/eval/eval_m2m_v2_all_tasks.py` invocation meant for dashboard ingestion **must pass `--save-npz`** — otherwise the dashboard 3D viewer 404s and metrics-only data can't be recovered without rerunning inference. Caption models **must also pass `--use-rewritten`**. See [`motion_annot_web/eval_dashboard/CLAUDE.md`](motion_annot_web/eval_dashboard/CLAUDE.md).

---

## Taiji 集群训练提交

提交训练任务到 Taiji 集群时，**必须使用提交脚本**：

```bash
# 直接提交（推荐，自动处理所有配置）
python3 tools/taiji_submit.py <任务名> <config路径> --host_num <节点数>

# 示例：2节点16卡V100
python3 tools/taiji_submit.py my_train_v1 configs/hymotion_umo/hymotion_umo_201dim_046b.py --host_num 2

# 示例：4节点32卡V100（默认）
python3 tools/taiji_submit.py my_train_v2 configs/hymotion_m2m/my_config.py
```

`taiji_submit.py` 自动处理：模板配置、RDMA/同模块调度、认证、JSON-RPC API 调用。

**禁止手动拼 `taiji_client start -scfg` 参数** — 它不支持 `template_flag` 等关键字段。

查看任务状态：
```bash
taiji_client trl                      # 运行中的任务列表
taiji_client il <task_flag>           # 实例状态
taiji_client td <task_flag>           # 任务详情
taiji_client stop <task_flag>         # 停止任务
```

---

## Design Principles

1. **Config-Driven, Registry-Based**: MMEngine `Config` + `Registry`
2. **Accelerate-Native**: All distributed, mixed precision, gradient accumulation via `Accelerator`
3. **HuggingFace-First**: Use diffusers / transformers native classes directly
4. **Per-Module Control**: Each sub-module independently controls trainable/checkpoint/precision
5. **ModelBundle = Shared Core**: Trainer and Pipeline share same `ModelBundle`, forward functions written once

## Public API Contract

1. `ModelBundle.from_config(...)` — unified entry for all bundles
2. `ModelBundle.from_pretrained(...)` — HF-native bundles implement `_bundle_config_from_pretrained(...)`
3. `save_pretrained(...)` — only when bundle can export official HF artifacts
4. Memory/precision control is config-driven (`module_dtype`, `gradient_checkpointing`, etc.)

## Smoke Test Policy

- Entry: `python3 -m pytest -m smoke tests/smoke/test_task_startup.py`
- Covers: prism, prism_mcm, vermo, hymotion_m2m, hymotion_t2m
- Each case: generate temp config -> train 1 step -> infer
- New task stacks must add a smoke case

---

## Repository Structure

```
hf_trainer/
|-- CLAUDE.md                        # This file (index)
|-- configs/                         # Runnable demo configs (.py)
|-- hftrainer/                       # Core framework package
|   |-- registry.py                  # Registries + HF model construction
|   |-- runner/                      # AccelerateRunner
|   |-- models/                      # ModelBundle base + task bundles
|   |   +-- motion/CLAUDE.md         # Motion task stack docs
|   |-- trainers/                    # Training logic
|   |-- pipelines/                   # Inference logic
|   |-- datasets/                    # Task-specific datasets
|   |-- hooks/                       # LoggerHook, CheckpointHook, EMAHook
|   |-- evaluation/                  # AccuracyEvaluator, PerplexityEvaluator
|   +-- visualization/              # TensorBoardVisualizer, FileVisualizer
|-- tools/                           # Core CLI only (train/infer/dist_train/taiji)
|   |-- train.py                     # Training entry point
|   |-- infer.py                     # Inference entry point
|   |-- dist_train.sh                # Distributed training launcher
|   |-- taiji_dist_train.sh          # Taiji cluster distributed training
|   |-- taiji_submit.py              # Taiji job submission
|   |-- taiji_exec.py                # Taiji task executor
|   |-- taiji_exec_host.py           # Taiji host executor
|   |-- download_checkpoints.sh      # Checkpoint download helper
|   +-- taiji_template.json          # Taiji job template
|-- scripts/                         # All utility scripts (organized by function)
|   |-- eval/                        # Evaluation (M2M, repair, metrics, multiseed)
|   |-- data/                        # Data building, conversion, preprocessing
|   |-- repair/                      # Repair & post-processing pipelines
|   |-- debug/                       # Diagnostics, smoke tests, validation
|   |-- analysis/                    # Visualization, statistics, plots
|   |-- submit/                      # Taiji job submission helpers
|   |-- kimodo/                      # KIMODO baseline integration
|   |-- guidance/                    # Keypose/keyframe guidance experiments
|   |-- caption/                     # Caption rewriting
|   |-- patch/                       # One-off data patches & rebuilds
|   |-- e14/                         # E14 transition experiments
|   |-- e9/                          # E9 repair experiments
|   |-- cjgame/                      # CJGame dataset processing
|   |-- globalrot/                   # Global rotation experiments
|   |-- inference/                   # Batch inference scripts
|   +-- misc/                        # Uncategorized (robot_sim, etc.)
|-- docs/
|   |-- design/CLAUDE.md             # Framework design details
|   |-- temp/                        # WIP docs, proposals, plans
|   +-- en/, zh-cn/                  # Public docs
|-- motion_annot_web/                # Web tools (port 8085/8080/8090)
+-- data/                            # Demo data for smoke tests
```

---

## ⚠️ Critical Framework Notes

### Bundle-Level Parameter Bug (fixed 2026-03-27)

`nn.Parameter` and `register_buffer` defined **directly on a ModelBundle** (not inside any sub-module) were silently excluded from training, checkpoint save/load, and DDP gradient sync. This caused inference failures in M2M/T2M/UMO. See [`docs/design/CLAUDE.md`](docs/design/CLAUDE.md) §Checkpoint Strategy for technical details and [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) §Historical Bug Record for full debugging story.

**Rule**: any new bundle-level `nn.Parameter` will now be auto-included in optimizer/save/sync. Use `requires_grad=False` for frozen params (e.g. null embeddings from pretrained checkpoint).

### Training Data Quality Issue (2026-04)

All HyMotion M2M configs currently train on **unfiltered** `data/annotation/train_hymotion_400h.json` (549K samples), which includes ~85K low-quality motions (jitter, foot_sliding, joint_jump, etc.). This limits model quality ceiling. Should use quality-filtered data from `motion_annot_web/m2m_database` (high_quality: 456K samples). See [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) §Training Data Quality Issue for details.

---

## Supported Tasks

| Task | ModelBundle | Trainer | Pipeline | Dataset | Config |
|------|-------------|---------|----------|---------|--------|
| Motion (PRISM T2M) | `PrismBundle` | `PrismTrainer` | `PrismPipeline` | `MotionhubMultiTaskMultiAgentDataset` | `prism/prism_smoke.py` |
| Motion (PRISM MCM) | `PrismMCMBundle` | `PrismMCMTrainer` | `PrismMCMPipeline` | `RandomMotionAudioDataset` | `prism/prism_mcm_smoke.py` |
| Motion (VerMo) | `VerMoBundle` | `VerMoTrainer` | `VerMoPipeline` | `MotionhubMultiTaskMultiAgentDataset` | `vermo/vermo_smoke.py` |
| Motion (HyMotion M2M) | `HyMotionM2MBundle` | `HyMotionM2MTrainer` | `HyMotionM2MPipeline` | `MotionhubMultiTaskMultiAgentDataset` | `hymotion_m2m/` |
| Motion (HyMotion M2M SOAR post-train) | `HyMotionM2MBundle` | `HyMotionM2MSoarTrainer` | `HyMotionM2MPipeline` | `MotionhubMultiTaskMultiAgentDataset` | `hymotion_m2m/soar/` |
| Motion Repair (MoGenDIT) | — (external) | — | `MoGenDITRepairPipeline` | — | `scripts/repair/mogendit_repair.py` |

### MoGenDIT Integration (External)

External diffusion repair framework at `/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/`. Not migrated — uses `sys.path` + wrapper pipeline.

| Property | Value |
|----------|-------|
| Architecture | DiT + AdaLN + RoPE + sliding window attention (window=90) |
| Representation | 201-dim: pose(22x6 rot6d) + joint(22x3) + trans(3) |
| Models | 0.1B (recommended), 0.03B, 0.3B |
| Checkpoint | `checkpoints/mogendit/MoreDiff-0.1B/` (symlink) |
| SMPL body model | `checkpoints/motion_process/body_model/` (symlink to MoGenDIT) |
| Repair modes | denoise, ada_denoise, trans_regen |
| Pipeline | `hftrainer/pipelines/motion/mogendit_pipeline.py` |

Key differences from M2M: column-major rot6d (same as `rotation_convert.py`), internal normalization (no external mean/std).

⚠️ **`ada_denoise` mode does NOT use adaptive mask for imputation** — only protects the first frame during denoise. Translation is freely regenerated. See [`hftrainer/models/motion/CLAUDE.md`](hftrainer/models/motion/CLAUDE.md) §Repair Pipeline Comparison for full analysis.

---

## Key Dependencies

- `torch >= 2.0`, `accelerate`, `transformers`, `diffusers`, `peft`
- `mmengine` — Config (.py parsing, `_base_` inheritance), Registry only
- `safetensors`, `wandb` / `tensorboard`

## Development Notes

- Package: `hftrainer`
- Entry: `tools/train.py` -> `AccelerateRunner.from_cfg(cfg)` -> `runner.train()`
- Launch: `tools/dist_train.sh` wraps `accelerate launch`
- Config: `mmengine.Config` for `.py` parsing + `_base_` inheritance
- Registry: own root registries (not inheriting MMEngine's tree)

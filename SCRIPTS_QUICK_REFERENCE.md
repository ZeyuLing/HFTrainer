# HFTrainer Scripts - Quick Reference Guide

## Quick Facts
- **Total Scripts**: 204 (.py + .sh files)
- **Total LoC**: ~80,000+ lines of code
- **Main Dirs**: `tools/` (127), `scripts/` (68), ROOT (9)
- **Key Subdirs**: `tools/analysis_tools/`, `tools/robot_sim/`, `scripts/eval/`

---

## 🚀 Most Important Scripts (Start Here)

### Training
```bash
# Single GPU training
python tools/train.py configs/your_config.py

# Distributed training (8 GPUs)
accelerate launch tools/train.py configs/your_config.py

# Or with Taiji platform
python tools/taiji_submit.py --config taiji_template.json
```
**File**: `tools/train.py` (Main entry point)

### Inference
```bash
python tools/infer.py --config config.json --checkpoint model.ckpt
```
**File**: `tools/infer.py`

### Comprehensive Evaluation
```bash
python scripts/eval_m2m_all_tasks.py
```
**File**: `scripts/eval_m2m_all_tasks.py` (Single entry point for all M2M evaluation)

---

## 📊 Scripts by Functional Category

### Category 1: Training & Infrastructure (12 files)
| When to Use | Key Files |
|-------------|-----------|
| **Single GPU training** | `tools/train.py` |
| **Distributed training** | `tools/dist_train.sh`, `tools/taiji_dist_train.sh` |
| **Job submission (Taiji)** | `tools/taiji_submit.py`, `tools/taiji_submit_eval.py` |
| **Quick testing** | `_test_train_textfree.sh`, `_test_dit.sh` |

### Category 2: Format Conversion (16 files)
**Purpose**: Convert motion between HumanML3D-263, H3D272, SMPL-85, SMPL-135, FBX
| Format Conversion | Key Files |
|-------------------|-----------|
| **MoMask → H3D272** | `tools/convert_momask263_to_h3d272.py` |
| **HumanML3D → H3D272** | `tools/convert_hml263_to_h3d272.py` |
| **SMPL-85 → 272** | `tools/smpl85_to_repr272.py` |
| **Distributed SMPL fit** | `tools/momask263_to_smpl85_sharded.py`, `_parallel_momask263_to_smpl85.sh` |
| **MotionHub → H3D272** | `tools/convert_motionhub_to_h3d272.py` |
| **Validate conversion** | `tools/validate_smpl85_to_272.py` |

### Category 3: Data Pipeline (25 files)
**Purpose**: Build & prepare evaluation datasets for different tasks (E1-E15)
| Task | Building Script | Validation |
|------|-----------------|-----------|
| **E2 (Inbetween)** | `tools/build_e2_inbetween_v2_data.py` | `scripts/rebuild_e2_inbetween_datalist.py` |
| **E3 (Keyframe)** | `tools/build_e3_keyframe_v2_data.py` | — |
| **E8 (Loop)** | `tools/build_e8_loop_v2_data.py` | — |
| **E9 (Repair)** | `tools/build_e9_repair_v2.py` | — |
| **E14 (Transition)** | `tools/build_e14_hq400h_data.py` | — |
| **E15 (Prepend)** | `tools/build_e15_prepend_v2_data.py` | — |
| **M2M v2 Eval** | `tools/build_m2m_v2_eval_data.py` | — |

### Category 4: Evaluation (48 files)
**Purpose**: Test models across different motion completion tasks

#### Entry Points (Start here)
| What to Evaluate | Script | GPU Count |
|------------------|--------|-----------|
| **M2M (all tasks)** | `scripts/eval_m2m_all_tasks.py` | 8 GPUs |
| **M2M ablation** | `scripts/eval_m2m_ablation.py` | 8 GPUs |
| **M2M repair benchmark** | `scripts/eval_repair_benchmark.py` | Multi |
| **M2M completion** | `scripts/eval_m2m_completion.py` | 8 GPUs |
| **M2M transitions** | `scripts/eval_m2m_transition.py` | — |
| **MoGenDIT repair** | `scripts/eval_mogendit_repair.py` | — |
| **Global rotation repair** | `scripts/eval_globalrot_repair_parallel.py` | 8 GPUs |
| **Keypose guidance** | `scripts/eval_keypose_guidance.py` | — |

#### Parallel Evaluation (Distributed)
| Task | Parallel Script |
|------|-----------------|
| **M2M repair (parallel)** | `scripts/eval_m2m_repair_parallel.py` |
| **GlobalRot repair (parallel)** | `scripts/eval_globalrot_repair_parallel.py`, `_v2.py`, `_v3.py` |
| **CJGame repair** | `scripts/eval_cjgame_repair.py` |
| **MoGenDIT multi-GPU** | `scripts/mogendit_multigpu_repair.py` |

#### Comprehensive Reports
| What to Generate | Script |
|------------------|--------|
| **M2M checkpoint report** | `scripts/eval_m2m_checkpoint_report.py` |
| **Manual repair report** | `tools/eval_manual_repair.py` |
| **Quality check results** | `scripts/run_quality_check_m2m.py` |

### Category 5: Post-Processing & Refinement (20 files)
**Purpose**: Improve motion quality after model generation

#### E14-Specific Polishing (Transition Stitching)
| Issue | Fix Script |
|-------|-----------|
| **Foot sliding** | `scripts/postprocess_e14_antislide.py` |
| **Foot contact** | `scripts/postprocess_e14_foot_contact.py` |
| **Pelvis pinning** | `scripts/postprocess_e14_footpin.py` |
| **Velocity damping** | `scripts/postprocess_e14_veldamp.py` |

#### General Pipelines
| Pipeline | Script |
|----------|--------|
| **Pure blend baseline** | `scripts/run_pure_blend_baseline.py` |
| **Blend then polish** | `scripts/run_blend_then_polish.py` |
| **Hybrid approach** | `scripts/run_hybrid_blend_polish.py` |
| **HyMotion + MoGenDIT** | `scripts/postprocess_hymotion_with_mogendit.py` |

### Category 6: Quality Diagnostics (11 files)
**Purpose**: Find and analyze motion defects

| Analysis | Script |
|----------|--------|
| **Original quality check** | `tools/check_original_quality.py` |
| **SMPL-85 fit quality** | `tools/diag_smpl85_fit_quality.py` |
| **KIMODO boundary jumps** | `tools/diagnose_kimodo_e14_boundary_jumps.py` |
| **Null embedding issues** | `tools/diag_null_embed_pipeline.py`, `_fix_verify.py`, `_e2e_train.py` |
| **Audit orphan params** | `tools/diag_audit_orphan.py` |
| **CJGame pair analysis** | `tools/analyze_cjgame_pairs.py` |
| **FBX pair analysis** | `tools/analyze_fbx_pairs.py` |

### Category 7: Adaptive Mask Computation (2 files)
| Use Case | Script |
|----------|--------|
| **Light-weight masking** | `scripts/compute_adaptive_masks.py` |
| **Precompute for eval** | `scripts/compute_adaptive_masks_for_eval.py` |

### Category 8: Taiji Job Scheduling (15 files)
**Purpose**: Submit and manage distributed jobs on Taiji platform

| Task | Submission Script |
|------|-------------------|
| **Training submission** | `tools/taiji_submit.py` |
| **Eval submission** | `tools/taiji_submit_eval.py` |
| **Global rot tasks** | `tools/submit_globalrot_tasks.py` |
| **M2M v2 eval** | `tools/submit_v2_eval_taiji.py` |
| **Custom tasks** | `scripts/submit_custom_task.py` |
| **Ablation experiments** | `scripts/submit_ablation_taiji.sh` |

### Category 9: Monitoring & Reports (13 files)
| Task | Script |
|------|--------|
| **Live progress monitor** | `scripts/live_progress_monitor.py` (中文) |
| **Full repair monitor** | `scripts/monitor_full_repair.py` (中文) |
| **Result watcher** | `tools/eval_result_watcher.py` |
| **Repair eval monitoring** | `scripts/monitor_repair_eval.sh` |
| **Multi-seed results** | `scripts/merge_multiseed_results.py` |
| **Dashboard integration** | `scripts/stablemotion_to_dashboard.py`, `relabel_dashboard_json.py` |

### Category 10: KIMODO/Robot Simulation (2 files)
| Use Case | Script |
|----------|--------|
| **KIMODO eval** | `tools/run_kimodo_all_tasks.py` |
| **Text-to-G1 robot** | `tools/robot_sim/text_to_g1.py` |

---

## 📁 Directory Structure At-a-Glance

```
hf_trainer/
├── tools/                           # 127 scripts
│   ├── train.py                     # ⭐ Main training entry
│   ├── infer.py                     # ⭐ Main inference entry
│   ├── convert_*.py                 # (16) Format conversion pipelines
│   ├── build_e*_data.py             # (13) Task-specific data builders
│   ├── eval_*.py                    # (18) Evaluation scripts
│   ├── postprocess_e14_*.py         # (4) E14 polishing
│   ├── compute_*stats.py            # (4) Statistics computation
│   ├── diag_*.py                    # (11) Diagnostics
│   ├── taiji_*.py                   # (9) Taiji job management
│   ├── analysis_tools/              # (2) Model analysis
│   │   ├── get_flops.py
│   │   └── print_config.py
│   └── robot_sim/                   # (3) Robot simulation
│       ├── setup_asap.py
│       └── text_to_g1.py
│
├── scripts/                         # 68 scripts
│   ├── eval_m2m_all_tasks.py       # ⭐ Main eval entry
│   ├── eval_m2m_*_parallel.py      # Parallel evaluation
│   ├── postprocess_*.py            # (8) Post-processing pipelines
│   ├── mogendit_*.py               # (3) MoGenDIT repair
│   ├── run_*.py                    # (12) Pipeline runners
│   ├── *_repair*.py                # Repair evaluation
│   ├── monitor_*.py                # Progress tracking
│   ├── preprocess_*.py             # Data preprocessing
│   ├── submit_*.py                 # Job submission
│   └── eval/                        # (4) High-level orchestration
│       ├── build_motion_master_list.py
│       ├── compare_baseline_vs_latest.py
│       └── run_e3_e8d_e14_e15_latest_v2.sh
│
├── ROOT/                            # 9 scripts
│   ├── tools/train.py              # ⭐ (alias from tools/)
│   ├── _test_train_textfree.sh      # Quick test
│   ├── run_setup.sh                # Setup
│   └── E14_DEBUG_VERIFICATION.py    # Debug
```

---

## 🔧 Common Workflows

### Workflow 1: Train a Model
```bash
cd hf_trainer/
python tools/train.py configs/hymotion_m2m_v2/config.py --work-dir output/exp1
```

### Workflow 2: Full Evaluation Suite
```bash
# Run all M2M tasks on 8 GPUs
python scripts/eval_m2m_all_tasks.py --config config.json --output-dir eval_results/

# Generate comprehensive report
python scripts/eval_m2m_checkpoint_report.py --eval-dir eval_results/
```

### Workflow 3: Repair Low-Quality Motions
```bash
# Option A: Pure blend (no model)
python scripts/run_pure_blend_baseline.py --input low_quality.json

# Option B: M2M repair
python scripts/eval_m2m_repair_parallel.py --num-gpus 8 --input low_quality.json

# Option C: MoGenDIT + M2M blend
python scripts/postprocess_hymotion_with_mogendit.py --input low_quality.json

# Post-process: Fix foot sliding, contact, etc
python scripts/postprocess_e14_antislide.py --input repaired.json
python scripts/postprocess_e14_footpin.py --input repaired.json
```

### Workflow 4: Distributed Job on Taiji
```bash
# Submit training job
python tools/taiji_submit.py --config taiji_template.json

# Submit evaluation job  
python tools/taiji_submit_eval.py --config eval_config.json

# Monitor progress
python scripts/live_progress_monitor.py --task-id <task_id>
```

### Workflow 5: Format Conversion
```bash
# MoMask output → H3D272
python tools/convert_momask263_to_h3d272.py --input momask_output/ --output h3d272/

# SMPL-85 → 272-dim
python tools/smpl85_to_repr272.py --input smpl85/ --output repr272/

# Validate
python tools/validate_smpl85_to_272.py --input repr272/
```

### Workflow 6: Build Evaluation Dataset
```bash
# Build E14 (Transition Stitching) dataset
python tools/build_e14_hq400h_data.py --output data/eval/e14/

# Run evaluation on it
python scripts/eval_m2m_all_tasks.py --task e14 --data data/eval/e14/
```

---

## 📌 Notes for New Developers

### Understanding the E-Tasks
- **E1**: ? (not in inventory)
- **E2**: Inbetween generation (generate frames between source and target)
- **E3**: Keyframe-based generation
- **E5**: Root trajectory following
- **E8**: Motion looping
- **E9**: Motion repair from low-quality data
- **E13**: Multi-prompt autoregressive generation
- **E14**: Transition stitching (different motion pairs) ⭐ Heavy optimization here
- **E15**: Motion prepending

### Key Representations
- **HumanML3D-263**: MoMask's standard representation
- **H3D272**: MotionStreamer's 272-dim representation (standard in project)
- **SMPL-85**: Joint positions in world space
- **SMPL-135**: Full SMPL representation (35 joints × 3 + rotation)
- **201-dim / 198-dim**: Alternative representations used in some models

### Quality Metrics
- **Foot Skating**: % of frames where foot is sliding on ground
- **Jitter**: Frame-to-frame joint velocity variance
- **Joint Jumps**: Sudden large position changes (anomalies)
- **Frozen Frames**: Frames with no motion for multiple frames
- **QC Pass**: Motion passes quality check (> threshold across metrics)

### Architecture Patterns
1. **Modular evaluation**: Single-worker scripts called by parallel orchestrators
2. **Config-driven**: Most scripts take YAML/JSON configs
3. **Multi-format support**: Converter pipelines are fundamental
4. **Quality-centric**: Extensive post-processing (polishing, damping, pinning)
5. **GPU scaling**: Support for 1 GPU → 8 GPUs → 100s via Taiji

---

## 🆘 When to Use Each Script

| Need | Script |
|------|--------|
| **"I want to train a model"** | `tools/train.py` |
| **"I need to run inference"** | `tools/infer.py` |
| **"I want to evaluate everything"** | `scripts/eval_m2m_all_tasks.py` |
| **"My motions have foot sliding"** | `scripts/postprocess_e14_antislide.py` |
| **"I need to fix null embeddings"** | `tools/diag_null_embed_pipeline.py` |
| **"Convert my motion format"** | `tools/convert_*.py` (find matching source/target) |
| **"I want to diagnose quality"** | `tools/check_original_quality.py` |
| **"Submit job to Taiji"** | `tools/taiji_submit.py` or `scripts/submit_custom_task.py` |
| **"Monitor my repair job"** | `scripts/live_progress_monitor.py` |
| **"Build eval dataset"** | `tools/build_e*_data.py` (find your task) |

---

Generated: 2026-05-10
Total Scripts Documented: 204
Total LoC Analyzed: ~80,000+


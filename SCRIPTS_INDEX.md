# HFTrainer Scripts & Tools - Complete Index

**Last Updated**: 2026-05-10  
**Total Scripts**: 204 files  
**Total Code**: ~80,000+ lines  
**Status**: ✅ Fully Documented

---

## 📚 Documentation Files (In This Directory)

### 1. **SCRIPTS_QUICK_REFERENCE.md** (14 KB)
**START HERE** - Your quick lookup guide to the most important scripts

**Covers**:
- 3 most critical scripts (train.py, infer.py, eval_m2m_all_tasks.py)
- 10 functional categories with examples
- Common workflows (train, evaluate, repair, convert, etc.)
- "When to use" decision tree
- E-task definitions and quality metrics

**Best for**: Quick lookup, "What script should I use for X?"

---

### 2. **HF_TRAINER_SCRIPTS_INVENTORY.md** (30 KB)
**COMPREHENSIVE REFERENCE** - Complete documentation of all 204 scripts

**Covers**:
- **Section 1**: Root directory (9 files)
- **Section 2**: tools/ directory (127 files organized in 15 categories)
  - Training & Inference (3)
  - Data Conversion (16)
  - Data Building (13)
  - Quality Checking (11)
  - Evaluation & Testing (18)
  - Merging & Aggregation (4)
  - Post-Processing (7)
  - Checkpoint Analysis (4)
  - Taiji/Distributed (9)
  - Job Launchers (10)
  - KIMODO/Robot (2)
  - Caption Management (3)
  - Data Pair Analysis (2)
  - FBX Processing (1)
  - Subdirectories (5 files total)
  
- **Section 3**: scripts/ directory (68 files organized in 17 categories)
  - Evaluation Workers (4)
  - Parallel Evaluation (13)
  - Comprehensive Suites (10)
  - Quality & Completion (5)
  - Post-Processing (8)
  - Data Building (8)
  - Adaptive Masks (2)
  - Repair Pipelines (5)
  - A/B Testing (5)
  - Monitoring (5)
  - Reports & Consolidation (8)
  - Baseline Methods (3)
  - Taiji Management (6)
  - Local Launchers (12)
  - Specialized Eval (4)
  - Testing (2)
  - Subdirectory eval/ (4)

- **Section 4**: Summary by functional domain
- **Section 5**: Statistics and file counts
- **Section 6**: Critical files for new developers
- **Section 7**: Execution environment

**Best for**: Deep dive, understanding all available scripts, finding niche tools

---

## 🎯 How to Use These Documents

### Scenario 1: "I'm new and want to understand the project structure"
1. Start with **SCRIPTS_QUICK_REFERENCE.md** → Section: "Most Important Scripts"
2. Read the directory structure overview
3. Skim the "When to use each script" decision tree

### Scenario 2: "I need to train a model"
1. **SCRIPTS_QUICK_REFERENCE.md** → Workflow 1
2. Run: `python tools/train.py`

### Scenario 3: "I need to evaluate my model"
1. **SCRIPTS_QUICK_REFERENCE.md** → Workflow 2
2. Run: `python scripts/eval_m2m_all_tasks.py`

### Scenario 4: "I have low-quality motions to repair"
1. **SCRIPTS_QUICK_REFERENCE.md** → Workflow 3
2. Choose between pure_blend, M2M repair, or MoGenDIT+M2M

### Scenario 5: "I need to convert motion formats"
1. **SCRIPTS_QUICK_REFERENCE.md** → Workflow 5
2. Find the appropriate `tools/convert_*.py` script

### Scenario 6: "I'm looking for a specific diagnostic tool"
1. **HF_TRAINER_SCRIPTS_INVENTORY.md** → Section 2 → Quality Diagnostics
2. Find the tool matching your symptoms

### Scenario 7: "I want to understand everything about E14 (Transition Stitching)"
1. **HF_TRAINER_SCRIPTS_INVENTORY.md** → Section 2 → Job Launchers (search "e14")
2. **HF_TRAINER_SCRIPTS_INVENTORY.md** → Section 3 → Post-Processing (E14-specific section)
3. Review: build_e14_hq400h_data.py, run_e14_hq400h_eval.sh, postprocess_e14_*.py files

---

## 📊 Script Breakdown by Category

| Category | # Files | Key Purpose | Location |
|----------|---------|-------------|----------|
| **Training & Core Infra** | 12 | Model training, job submission | tools/ + ROOT |
| **Format Conversion** | 16 | Bridge motion formats | tools/ |
| **Data Pipeline** | 25 | Build eval datasets (E1-E15) | tools/ + scripts/ |
| **Evaluation** | 48 | Test models, generate reports | scripts/ main |
| **Post-Processing** | 20 | Quality improvement after gen | scripts/ |
| **Quality Diagnostics** | 11 | Find & analyze defects | tools/ |
| **Adaptive Masks** | 2 | MoGenDIT mask precompute | scripts/ |
| **Repair Pipelines** | 5 | Motion quality repair | scripts/ |
| **A/B Testing** | 5 | Comparison studies | scripts/ |
| **Monitoring** | 5 | Progress tracking | scripts/ + tools/ |
| **Reports** | 8 | Consolidate results | scripts/ |
| **Taiji Platform** | 15 | Distributed GPU scheduling | tools/ + scripts/ |
| **KIMODO/Robot** | 5 | Humanoid simulation | tools/ |
| **Other** | 25 | Analysis, utilities, etc. | Various |
| **TOTAL** | **204** | — | — |

---

## 🔗 Quick Links to Common Tasks

### Training
- **Single GPU**: `tools/train.py`
- **Distributed (8 GPUs)**: `tools/dist_train.sh` or accelerate launch
- **Taiji cluster**: `tools/taiji_submit.py`
- **Quick test**: `_test_train_textfree.sh`

### Evaluation
- **Everything**: `scripts/eval_m2m_all_tasks.py` ⭐
- **Repair evaluation**: `scripts/eval_m2m_repair_parallel.py`
- **Ablation study**: `scripts/eval_m2m_ablation.py`
- **Report generation**: `scripts/eval_m2m_checkpoint_report.py`

### Motion Repair
- **No model (baseline)**: `scripts/run_pure_blend_baseline.py`
- **M2M repair**: `scripts/eval_m2m_repair_parallel.py`
- **MoGenDIT repair**: `scripts/mogendit_repair.py`
- **Fix foot sliding**: `scripts/postprocess_e14_antislide.py`
- **Fix foot contact**: `scripts/postprocess_e14_foot_contact.py`

### Data Management
- **Build E14 data**: `tools/build_e14_hq400h_data.py`
- **Build E9 data**: `tools/build_e9_repair_v2.py`
- **Build any E-task**: `tools/build_e<N>_*_v2_data.py`
- **Process CJGame**: `scripts/process_cjgame_npz.py`
- **Quality check**: `tools/check_original_quality.py`

### Format Conversion
- **MoMask → H3D272**: `tools/convert_momask263_to_h3d272.py`
- **SMPL-85 → 272**: `tools/smpl85_to_repr272.py`
- **FBX → JSON**: `tools/fbx_to_motion_json.py`
- **Validate conversion**: `tools/validate_smpl85_to_272.py`

### Diagnostics
- **Null embeddings**: `tools/diag_null_embed_pipeline.py`
- **Padding issues**: `tools/debug_m2m_padding_real_data.py`
- **SMPL fit quality**: `tools/diag_smpl85_fit_quality.py`
- **Boundary jumps**: `tools/diagnose_kimodo_e14_boundary_jumps.py`
- **FBX pair analysis**: `tools/analyze_fbx_pairs.py`

### Taiji Jobs
- **Submit training**: `tools/taiji_submit.py`
- **Submit eval**: `tools/taiji_submit_eval.py`
- **Monitor progress**: `scripts/live_progress_monitor.py`
- **Custom task**: `scripts/submit_custom_task.py`

---

## 🏗️ Project Architecture

```
Entry Points (Pick One):
├─ Training → tools/train.py (or taiji_submit.py for cluster)
├─ Inference → tools/infer.py
├─ Evaluation → scripts/eval_m2m_all_tasks.py
└─ Repair → scripts/eval_m2m_repair_parallel.py

Data Pipeline:
├─ Raw Data → tools/build_e*_data.py (Task-specific builders)
├─ Format Conversion → tools/convert_*.py
├─ Quality Check → tools/check_original_quality.py
└─ Evaluation Ready

Model Training:
├─ Config File → tools/train.py
├─ Single GPU / Distributed / Taiji
└─ Checkpoint Output

Model Evaluation:
├─ Checkpoint → scripts/eval_m2m_all_tasks.py
├─ 4 Task Types (E2, E3, E8, E9, E14, E15, etc.)
├─ Quality Metrics
└─ Report + JSON Results

Post-Processing (Quality Improvement):
├─ Raw Output → scripts/postprocess_e14_*.py (Task-specific)
├─ Fix: foot_sliding, contact, velocity, etc.
└─ Enhanced Output

Monitoring & Reporting:
├─ Progress → scripts/live_progress_monitor.py
├─ Results → scripts/eval_m2m_checkpoint_report.py
├─ Dashboard → scripts/stablemotion_to_dashboard.py
└─ Visualization → scripts/visualize_mogendit_results.py
```

---

## 💡 Key Insights for New Developers

### The "Motion Representation" Challenge
This project heavily invests in format conversion because different models use different representations:
- **HumanML3D-263**: MoMask's native format (30 dims, 8 fps or similar)
- **H3D272**: Project standard for MotionStreamer compatibility
- **SMPL-85**: Intermediate format (joint positions)
- **SMPL-135**: Full SMPL with rotations

**Key scripts**: `tools/convert_*.py` (16 files dedicated to this)

### The "E-Task" System
Rather than generic motion generation, this project breaks tasks into specific "E" scenarios:
- **E2**: Generate frames between two keyframes (inbetweening)
- **E3**: Generate from sparse keyframe guidance
- **E8**: Loop motion (seamless cycling)
- **E9**: Repair low-quality motion (the core repair task)
- **E14**: Stitch different motion pairs together ⭐ **Heavy optimization here**
- **E15**: Prepend motion to existing sequence

Each has dedicated builders, evaluators, and post-processors.

### The "Quality-Centric" Approach
After generation, extensive post-processing is applied:
- **Foot skating detection & fixing** (E14 focus)
- **Velocity damping** when feet are grounded
- **Joint continuity** checking
- **Canonical form enforcement** (ensure first frame faces Z+)

### The "Modular Evaluation" Pattern
For scalability, evaluation is split into:
1. **Worker scripts** (`_eval_m2m_single.py`) - process 1 case on 1 GPU
2. **Orchestrator scripts** (`eval_m2m_repair_parallel.py`) - coordinate 8 workers
3. **Report generators** (`eval_m2m_checkpoint_report.py`) - consolidate results

### The "Taiji Integration"
For large-scale work, jobs can run on:
- **Local GPU** (1-8 GPUs)
- **Taiji cluster** (100s of GPUs via job queue)

Corresponding scripts exist for both.

---

## 🚀 Your Next Steps

1. **Read SCRIPTS_QUICK_REFERENCE.md** for the 30-second overview
2. **Identify your use case** from the "When to use each script" table
3. **Run the appropriate entry point**:
   - Training: `python tools/train.py configs/your_config.py`
   - Evaluation: `python scripts/eval_m2m_all_tasks.py --config config.json`
   - Repair: `python scripts/eval_m2m_repair_parallel.py --input low_quality.json`
4. **Check HF_TRAINER_SCRIPTS_INVENTORY.md** for deeper understanding
5. **Read the docstrings** in the actual .py file for full details

---

## 📝 File Inventory by Directory

### ROOT Directory (9 files)
- `tools/train.py` ⭐
- `tools/infer.py` ⭐
- `setup.py`
- `E14_DEBUG_VERIFICATION.py`
- `_test_train_textfree.sh`
- `_test_dist_train_textfree.sh`
- `_test_dit.sh`
- `_taiji_run.sh`
- `_remote_cmd.sh`

### tools/ Directory (127 files)
- **Core Training**: `train.py`, `infer.py`, `dist_train.sh`
- **Format Conversion (16)**: `convert_*.py`, `smpl85_to_repr272.py`, etc.
- **Data Building (13)**: `build_e2-e15_*.py`, `rebuild_*.py`
- **Evaluation (18)**: `eval_*.py`
- **Post-Processing (7)**: `patch_*.py`, `render_*.py`
- **Quality Analysis (11)**: `diag_*.py`, `check_*.py`
- **Taiji (9)**: `taiji_*.py`, `submit_*.py`
- **Subdirs**: `analysis_tools/` (2), `robot_sim/` (3)
- **Merging & Aggregation (4)**: `merge_*.py`
- **Monitoring (3)**: `eval_result_watcher.py`, etc.
- **Utilities**: Caption rewrites, pair analysis, misc

### scripts/ Directory (68 files)
- **Evaluation Workers (4)**: `_eval_*.py`
- **Parallel Launchers (13)**: `eval_*_parallel.py`
- **Comprehensive Suites (10)**: `eval_m2m_*.py`
- **Post-Processing (8)**: `postprocess_*.py`
- **Repair & Blending (5)**: `run_*blend*.py`, `repair_and_evaluate.py`
- **Taiji & Monitoring (11)**: Job submission and progress tracking
- **Data Prep (8)**: `build_*.py`, `process_*.py`, `precompute_*.py`
- **Reports & Consolidation (8)**: Report generation, dashboard integration
- **A/B Testing (5)**: `ab_test_*.py`, testing frameworks
- **Subdirectory**: `eval/` (4 high-level orchestrators)

---

## 📞 Quick Help

**Q: Where do I start?**
A: Read `SCRIPTS_QUICK_REFERENCE.md` (5 min) then `tools/train.py` for training or `scripts/eval_m2m_all_tasks.py` for eval.

**Q: How many scripts are there?**
A: **204 total** (127 in tools/, 68 in scripts/, 9 in root + subdirs)

**Q: What's E14 and why is it everywhere?**
A: E14 is the "Transition Stitching" task (stitching different motion pairs). It's heavily optimized with 4 dedicated post-processing scripts because foot-sliding is a major issue.

**Q: How do I repair motion quality?**
A: See Workflow 3 in `SCRIPTS_QUICK_REFERENCE.md` - choose between pure_blend, M2M repair, or MoGenDIT+M2M, then apply post-processing.

**Q: How do I run on Taiji (cluster)?**
A: Use `tools/taiji_submit.py` for training or `scripts/submit_custom_task.py` for custom jobs.

---

**Documentation Status**: ✅ Complete (All 204 scripts documented)  
**Last Review**: 2026-05-10  
**Contact**: See project README for maintainer info


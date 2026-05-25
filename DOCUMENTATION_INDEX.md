# Documentation Index — May 26, 2026

This file catalogs all documentation and investigation files for the HyMotion training framework.

---

## Active Documentation (For Current Work)

### PhysFlow Experimental Framework
- **[PHYSFLOW_STATUS_2026-05-26.md](PHYSFLOW_STATUS_2026-05-26.md)** — Current status, phase plan, action items (READ THIS FIRST)
- **[docs/temp/physflow_experiment_spec.md](docs/temp/physflow_experiment_spec.md)** — Complete experimental specification (5 phases, config matrix, metrics)

### ProtoMotions Integration
- **[PROTOMOTIONS_TRAINING_GUIDE.md](PROTOMOTIONS_TRAINING_GUIDE.md)** — Command-line guide for training RL trackers with custom motion data
- **[PROTOMOTIONS_QUICK_REFERENCE.md](PROTOMOTIONS_QUICK_REFERENCE.md)** — Single-page reference for common tasks
- **[DIRECTION_B_IMPLEMENTATION.md](DIRECTION_B_IMPLEMENTATION.md)** — Step-by-step guide for implementing Direction B (Gen→RL)

### PRISM Model Fixes (Completed May 26)
- **[docs/temp/IMPLEMENTATION_COMPLETE_2026-05-26.md](docs/temp/IMPLEMENTATION_COMPLETE_2026-05-26.md)** — FP32 upcast attention bf16 support (commits: 0bef779)
- **[docs/temp/fp32_upcast_activation_fix_2026-05-26.md](docs/temp/fp32_upcast_activation_fix_2026-05-26.md)** — Technical deep-dive on FP32 upcast fix

### Model Configuration & Dtype
- **[DTYPE_CONFIGURATION_SUMMARY.md](DTYPE_CONFIGURATION_SUMMARY.md)** — Data type configuration across models
- **[DTYPE_QUICK_REFERENCE.md](DTYPE_QUICK_REFERENCE.md)** — Quick reference for dtype settings

---

## Investigation Files (Reference/Archive)

### PRISM Attention Processor
- `PRISM_ATTENTION_CODE_REFS.md` — Code reference for attention processor
- `PRISM_ATTENTION_INDEX.md` — Index of attention-related files
- `PRISM_ATTENTION_TRACE.md` — Detailed trace of attention flow
- `PRISM_QUICK_SUMMARY.txt` — Quick summary of PRISM architecture
- `PRISM_V3_FIX_STATUS.txt` — Status of PRISM v3 fixes

### PRISM Caption Path Analysis
- `PRISM_CAPTION_PATH_FLOW.txt` — Flow diagram of caption processing
- `PRISM_CAPTION_PATH_QUICK_REFERENCE.md` — Quick reference for caption path
- `PRISM_CAPTION_PATH_SUMMARY.md` — Summary of caption path changes
- `PRISM_CAPTION_PATH_TRACE.md` — Detailed trace of caption path
- `PRISM_CODE_REFERENCES.md` — Code references for caption path
- `PRISM_DOCUMENTATION_INDEX.md` — Index of PRISM documentation

### Data & Infrastructure Research
- `docs/temp/data_flywheel_research.md` — Analysis of bidirectional data flow

---

## Archived Session Documentation

### rot6d Validation (May 21)
- `docs/temp/FINAL_DELIVERY_SUMMARY_2026-05-21.md` — Completion status
- `docs/temp/ROT6D_FRAMEWORK_STATUS_2026-05-21.md` — Framework status
- `docs/temp/rot6d_validation_integration_2026-05-21.md` — Integration guide
- `docs/temp/rot6d_convention_verification_2026-05-20.md` — Verification results

### MuJoCo Self-Collision Fix (May 18-19)
- `docs/temp/prism_padding_fix_2026_05_25.md` — Padding fix details

### PRISM Analysis
- `docs/temp/predict_flow_and_pred_type_analysis.md` — Analysis of prediction flow
- `docs/temp/predict_flow_usage_examples.md` — Usage examples

---

## Earlier Session References (In docs/temp/)

These are from prior sessions and serve as reference material:

| File | Date | Topic |
|------|------|-------|
| ANALYSIS_COMPLETE_ACTION_PLAN.md | May 12 | Action planning |
| EMBODIED_PIPELINE_*.md | May 12 | Embodied pipeline debugging |
| CHECKPOINT_LOADING_*.md | May 15 | Checkpoint loading |
| EXECUTIVE_SUMMARY_PRISM_ROOT_CAUSE.md | May 19 | PRISM root cause analysis |
| HYMOTION_M2M_V2_BASELINES_EVALUATION_INFRASTRUCTURE.md | May 18 | M2M evaluation setup |
| survey_motion_gen_embodied_v2_20260508.md | May 8 | Survey of motion generation |
| text_encoder_configuration_guide.md | May 18 | Text encoder configuration |

---

## Debug Scripts (scripts/debug/)

These are one-off debugging scripts from investigation sessions:

- `diagnose_nan_fp16.py` — FP16 NaN diagnosis
- `launch_v3_bg.sh` — Background launch script for v3
- `run_fp16_debug*.sh` — FP16 debugging scripts
- `run_v3_test.sh` — v3 test script

---

## How to Use This Index

### For New Contributors
1. Read **[PHYSFLOW_STATUS_2026-05-26.md](PHYSFLOW_STATUS_2026-05-26.md)** for current project status
2. Read **[docs/temp/physflow_experiment_spec.md](docs/temp/physflow_experiment_spec.md)** for full experiment plan
3. For specific tasks:
   - Training RL tracker: see PROTOMOTIONS_TRAINING_GUIDE.md
   - Direction B implementation: see DIRECTION_B_IMPLEMENTATION.md

### For Researchers Reproducing Results
1. See PHYSFLOW_STATUS_2026-05-26.md for baseline setup (Phase 0)
2. See DTYPE_CONFIGURATION_SUMMARY.md for model configuration
3. See PRISM_QUICK_SUMMARY.txt for model architecture

### For Debug/Troubleshooting
1. Check docs/temp/ for investigation notes
2. See PRISM_ATTENTION_*.md for attention processor issues
3. See IMPLEMENTATION_COMPLETE_2026-05-26.md for known issues and fixes

---

## Quick Links by Topic

### Training & Experiments
- PHYSFLOW_STATUS_2026-05-26.md (← START HERE)
- physflow_experiment_spec.md
- PROTOMOTIONS_TRAINING_GUIDE.md

### Models & Configuration
- DTYPE_CONFIGURATION_SUMMARY.md
- PRISM_QUICK_SUMMARY.txt
- text_encoder_configuration_guide.md

### Fixes & Debugging
- IMPLEMENTATION_COMPLETE_2026-05-26.md
- fp32_upcast_activation_fix_2026-05-26.md
- prism_padding_fix_2026_05_25.md

### Architecture & Design
- PRISM_ATTENTION_TRACE.md
- PRISM_CAPTION_PATH_TRACE.md
- EMBODIED_PIPELINE_*.md

---

## File Organization

This repository contains several documentation formats:

- **Root directory** (*.md, *.txt) — Primary documentation and status files
- **docs/temp/** — Session investigation and analysis files
- **scripts/debug/** — Debugging and utility scripts
- **hftrainer/models/motion/CLAUDE.md** — Model-specific documentation

---

## Version History

| Date | Status | Key Changes |
|------|--------|-------------|
| 2026-05-26 | 🟢 Ready for Phase 0 | FP32 upcast bf16 support, M2M motion dropout, trainer cleanup |
| 2026-05-21 | 🟢 Complete | rot6d validation framework delivery |
| 2026-05-18 | 🟡 Ongoing | PRISM analysis, MuJoCo fixes |
| 2026-05-12 | 🟡 Active | Embodied pipeline debugging |

---

**Last Updated**: May 26, 2026  
**Maintained By**: Claude Code  
**Next Review**: After Phase 0 completion


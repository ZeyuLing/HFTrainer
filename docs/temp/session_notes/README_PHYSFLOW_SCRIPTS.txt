╔══════════════════════════════════════════════════════════════════════════════╗
║                     PhysFlow Scripts Documentation Index                      ║
║                                                                              ║
║              Complete reference for PhysFlow eval demo generation             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 DOCUMENTATION FILES GENERATED
═══════════════════════════════════════════════════════════════════════════════

START HERE:
  → EXECUTIVE_SUMMARY.txt (4KB)
    Quick overview, main findings, key insights

QUICK REFERENCE:
  → PHYSFLOW_QUICK_REF.md (5.2KB)
    One-page guide to main scripts and data formats

COMPREHENSIVE REFERENCE:
  → PHYSFLOW_SCRIPTS_SUMMARY.md (17KB)
    Detailed documentation of 28 core PhysFlow scripts
    Usage examples, API reference, dependencies

TECHNICAL DEEP DIVE:
  → DATA_PRODUCTION_MAP.txt (13KB)
    Detailed data flow, output structure, file producers
    Production statistics and execution timeline

COMPLETE INVENTORY:
  → ALL_SCRIPTS_INVENTORY.txt (17KB)
    All 71 scripts categorized by purpose
    Complete execution flow and dependency graph

═══════════════════════════════════════════════════════════════════════════════
⭐ ANSWER TO YOUR QUESTION
═══════════════════════════════════════════════════════════════════════════════

Q: Which scripts generate the PhysFlow eval demo data?

A: physflow_eval_and_export.py (517 lines)

   Primary purpose: Generate output/physflow/eval_demo/ with:
   - motion_135 NPZ files (T, 135 format)
   - SMPL mesh JSON (kinematic, web-viewable)
   - SMPL mesh JSON (physics-corrected)
   - Metrics summary
   - Optional robot motion retargeting

   Supporting scripts:
   1. physflow_curriculum.py - Curriculum prompts (8 levels, 40+ prompts)
   2. physflow_trainer.py - T2M model fine-tuning
   3. physflow_physics_oracle.py - Physics correction via MuJoCo
   4. run_smpl_physics_sim.py - Core physics simulation engine

═══════════════════════════════════════════════════════════════════════════════
🗂️ DIRECTORY STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

scripts/embodied/
├── physflow_*.py               (6 main PhysFlow scripts)
├── run_*.py                    (Physics simulation & tracking)
├── batch_*.py                  (Batch processing pipelines)
├── motion135_*.py              (Format conversion)
├── pipeline_*.py               (End-to-end pipelines)
├── test_*.py & debug_*.py      (20+ debugging/testing scripts)
└── *.md & *.txt                (Documentation files)

output/physflow/eval_demo/
├── data/
│   ├── npz/                    ← motion_135 NPZ files
│   ├── smpl_mesh/              ← SMPL mesh JSON (kinematic)
│   ├── smpl_mesh_physics/      ← SMPL mesh JSON (physics-corrected)
│   ├── robot_motion/           ← PyRoki retargeted motions (optional)
│   ├── meta/                   ← Per-motion metadata
│   └── (other optional dirs)
├── metrics.json                ← Summary statistics
├── batch_retarget*.sh          ← Retargeting scripts
└── retarget_log*.txt           ← Logs

═══════════════════════════════════════════════════════════════════════════════
🔍 QUICK SCRIPT LOOKUP
═══════════════════════════════════════════════════════════════════════════════

Need to... find the script that:

Generate demo data?
  → physflow_eval_and_export.py

Define curriculum prompts?
  → physflow_curriculum.py

Correct motion physics?
  → physflow_physics_oracle.py

Run physics simulation?
  → run_smpl_physics_sim.py

Train PhysFlow model?
  → physflow_trainer.py

Evaluate models?
  → physflow_evaluate.py

Convert motion_135 to SMPL mesh JSON?
  → batch_npz_to_smpl_mesh_json.py

Convert motion_135 to robot keypoints?
  → motion135_to_pyroki_keypoints.py

Retarget to robot motion?
  → pipeline_motion_to_robot.py

Parallel retargeting?
  → batch_retarget_parallel.py

Debug physics issues?
  → debug_root_transform.py, debug_pose_diagnostic.py, debug_sim_stability.py

═══════════════════════════════════════════════════════════════════════════════
📊 QUICK FACTS
═══════════════════════════════════════════════════════════════════════════════

Total Scripts: 71 Python + 4 shell + 7 documentation files

Main Demo Generator:
  physflow_eval_and_export.py (517 lines)

Total PhysFlow-related scripts:
  6 core (precompute_text, curriculum, trainer, oracle, evaluate, eval_and_export)
  + 3 physics simulation
  + 4 format conversion
  + 4 robot/web export
  + 20+ debug/test/utility

Output Formats:
  - motion_135 NPZ (binary)
  - SMPL mesh JSON (web-viewable, Three.js)
  - PyRoki keypoints (robot format)
  - .motion files (ProtoMotions format)

Data Sizes (per motion):
  - NPZ: 200KB-500KB
  - SMPL mesh JSON: 2-5MB
  - Metadata: 1-2KB

Execution Time:
  - Quick mode demo: 2-5 hours (12 motions, GPU)
  - Full mode demo: 8-12 hours (40-50 motions, GPU)
  - Robot retargeting: ~60 min per motion (CPU, parallelizable)

═══════════════════════════════════════════════════════════════════════════════
🚀 QUICK START
═══════════════════════════════════════════════════════════════════════════════

Generate PhysFlow eval demo data (quick mode):

  python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick

Output will be saved to: output/physflow/eval_demo/

═══════════════════════════════════════════════════════════════════════════════
📖 HOW TO READ THE DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════════

1. First reading (5 min):
   Read EXECUTIVE_SUMMARY.txt
   → Understand main components and data flow

2. Implement/debug (reference):
   Use PHYSFLOW_QUICK_REF.md
   → Find script names, basic usage, data formats

3. Deep technical work:
   Refer to PHYSFLOW_SCRIPTS_SUMMARY.md
   → Full API reference, dependencies, examples

4. Data flow analysis:
   Check DATA_PRODUCTION_MAP.txt
   → Understand which script produces which files

5. Inventory lookup:
   See ALL_SCRIPTS_INVENTORY.txt
   → Find any script by name or purpose

═══════════════════════════════════════════════════════════════════════════════
✅ VERIFICATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

All required scripts exist:
  ✓ physflow_eval_and_export.py (MAIN)
  ✓ physflow_trainer.py
  ✓ physflow_physics_oracle.py
  ✓ physflow_curriculum.py
  ✓ physflow_precompute_text.py
  ✓ physflow_evaluate.py
  ✓ run_smpl_physics_sim.py
  ✓ 60+ supporting scripts

Output directories exist:
  ✓ output/physflow/eval_demo/
  ✓ output/physflow/eval_demo/data/npz/
  ✓ output/physflow/eval_demo/data/smpl_mesh/
  ✓ output/physflow/eval_demo/data/smpl_mesh_physics/

Demo data verified:
  ✓ 20+ motion_135 NPZ files (original_* and physflow_*)
  ✓ SMPL mesh JSON files in multiple directories
  ✓ Metrics.json with summary statistics
  ✓ Metadata and logs

═══════════════════════════════════════════════════════════════════════════════
💬 SUPPORT
═══════════════════════════════════════════════════════════════════════════════

For questions about:

  - Main demo generator → See: physflow_eval_and_export.py docstring
  - Data formats → See: DATA_PRODUCTION_MAP.txt or PHYSFLOW_QUICK_REF.md
  - Physics correction → See: physflow_physics_oracle.py
  - Curriculum prompts → See: physflow_curriculum.py and PHYSFLOW_LEVELS
  - Robot retargeting → See: pipeline_motion_to_robot.py
  - Debugging physics → See: debug_*.py scripts
  - Full reference → See: PHYSFLOW_SCRIPTS_SUMMARY.md

═══════════════════════════════════════════════════════════════════════════════
Last Updated: 2026-05-20
Comprehensive: All 71 scripts in scripts/embodied/ have been catalogued

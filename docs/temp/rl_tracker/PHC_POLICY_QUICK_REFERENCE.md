# PHC & SMPL Humanoid Policies - Quick Reference

## TL;DR

| What | Status | Location | Format |
|-----|--------|----------|--------|
| **G1 Robot Tracker** | ✅ Ready | `ref_repo/ProtoMotions/.../g1-bones-deploy/` | ONNX (22MB) + PyTorch (228MB) |
| **SMPL Humanoid Tracker** | ⏳ Trained | `ref_repo/ProtoMotions/.../smpl/` | PyTorch (121MB) only |
| **PHC G1 Config** | ✅ Available | `ref_repo/PHC/phc/data/cfg/env/env_im_g1_phc.yaml` | YAML |
| **SMPL Robot Library** | ✅ Available | `ref_repo/OmniH2O/phc/phc/smpllib/` | Python |

---

## G1 Robot - Ready to Use

### ONNX Model
```
ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx
```

**Specs**:
- **Input**: Current state (dof_pos, dof_vel, anchor_rot, future references)
- **Output**: Joint targets + adaptive stiffness/damping
- **Control**: 29 DOF @ 50 Hz
- **Metadata**: `unified_pipeline.yaml`

**Used by**:
```python
# scripts/embodied/run_tracker_export.py
_DEFAULT_ONNX = "ref_repo/ProtoMotions/.../g1-bones-deploy/compiled_models/unified_pipeline.onnx"
```

### Usage
```bash
python scripts/embodied/run_tracker_export.py \
    --motion <reference_motion.pt> \
    --output <tracked_motion.pt>
```

---

## SMPL Humanoid - Needs ONNX Export

### Checkpoint Location
```
ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt (121 MB)
```

### Export Command
```bash
cd ref_repo/ProtoMotions
python deployment/export_bm_tracker_onnx.py \
    --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt
```

### After Export
```bash
python scripts/embodied/run_tracker_export.py \
    --motion <smpl_motion.pt> \
    --output <tracked_motion.pt> \
    --onnx ref_repo/ProtoMotions/.../smpl/compiled_models/unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml
```

---

## PHC Training Configs

### G1 PHC Config
```
ref_repo/PHC/phc/data/cfg/env/env_im_g1_phc.yaml
```

**Key Settings**:
- Task: `HumanoidIm` (imitation)
- Envs: 3072 parallel
- Episode: 300 steps
- PNN: 3 primitives
- Mass: 51.4 kg (H1 robot)

### Other PHC Configs
```
ref_repo/OmniH2O/phc/phc/data/cfg/env/
├── phc_kp_mcp_iccv.yaml       (keypoint + MCP)
├── phc_shape_pnn_iccv.yaml    (shape + PNN)
└── h1_im_*.yaml               (H1 humanoid variants)
```

---

## SMPL Robot Library

### Location
```
ref_repo/OmniH2O/phc/phc/smpllib/
```

### Capabilities
- ✅ Auto-generate humanoid from SMPL body model
- ✅ Support SMPL, SMPL-H, SMPL-X
- ✅ Export to MuJoCo XML
- ✅ Capsule or mesh-based
- ✅ Gender + shape control

### Core Module
```python
from uhc.smpllib.smpl_local_robot import SMPLLocalRobot
robot = SMPLLocalRobot(...)
```

---

## Policy Comparison

| | G1 | SMPL |
|---|----|----|
| DOF | 29 | ~56 |
| ONNX | ✅ | ❌ |
| Status | Production | Training done |
| Robot | Unitree G1 | Generic humanoid |

---

## What Exists vs Gaps

### ✅ What You Have
1. G1 ONNX tracker (ready for inference)
2. SMPL PyTorch trainer (trained but needs ONNX)
3. End-to-end motion tracking script (`run_tracker_export.py`)
4. SMPL robot generation library
5. PHC training configs (but no exported policies)

### ❌ What's Missing
1. SMPL ONNX export (easy fix: run `export_bm_tracker_onnx.py`)
2. PHC trained weights (only configs available)
3. Policy zoo / model registry
4. SMPL multi-task policies (only motion tracking available)

---

## Next Steps

### Priority 1: Export SMPL ONNX
```bash
cd ref_repo/ProtoMotions
python deployment/export_bm_tracker_onnx.py \
    --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt
```

### Priority 2: Test SMPL Tracking
```bash
python scripts/embodied/run_tracker_export.py \
    --motion <test_motion.pt> \
    --output <tracked.pt> \
    --onnx ref_repo/ProtoMotions/.../smpl/compiled_models/unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/.../smpl_humanoid.xml
```

### Priority 3: Create Policy Registry
Document all available policies in a central location (e.g., `POLICY_REGISTRY.md`)

---

## File Reference

| File | Purpose | Size |
|------|---------|------|
| `run_tracker_export.py` | Motion tracking with physics sim | Script |
| `unified_pipeline.onnx` (G1) | Policy network (G1) | 22 MB |
| `last.ckpt` (G1) | Full checkpoint (G1) | 228 MB |
| `last.ckpt` (SMPL) | Full checkpoint (SMPL) | 121 MB |
| `smpl_humanoid.xml` | Robot model | MJCF |
| `g1_holo_compat.xml` | Robot model | MJCF |


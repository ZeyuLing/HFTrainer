# SMPL Humanoid Tracker Implementation Guide

## Quick Reference: System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRACKER EXPORT PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

INPUT STAGE (Reference Motion)
  ├─ Motion Cache (.pt file)
  │  └─ dof_pos: (T, 29) joint angles from SMPL retargeting
  │
  └─ MotionPlayer
     └─ Loads cache, provides frame-by-frame references

PHYSICS SIMULATION LOOP (Per-frame, 50 Hz)
  ├─ Record Current State (from MuJoCo)
  │  ├─ DOF positions & velocities
  │  ├─ Body positions & rotations (quaternions)
  │  └─ Fall detection (root height < 0.3m)
  │
  ├─ Get Future References
  │  ├─ Peek ahead [1, 2, 4, 8] steps
  │  ├─ Align heading with current orientation
  │  └─ Query from motion cache
  │
  ├─ ONNX Policy Inference
  │  ├─ Inputs: current state + future references (8 tensors)
  │  └─ Outputs: PD target joint angles (29-dim)
  │
  ├─ Apply PD Control
  │  ├─ τ = kp·(target - q) - kd·qvel
  │  └─ MuJoCo implicit PD controller
  │
  └─ Simulate 20 substeps (0.02s total)
     └─ mj_step() at 1000 Hz

OUTPUT STAGE (Tracked Motion)
  ├─ Record Physics Output
  │  ├─ Body positions from data.xpos[]
  │  ├─ Body rotations from data.xquat[]
  │  ├─ Velocities from data.cvel[]
  │  └─ DOF states from data.qpos[]/qvel[]
  │
  └─ Export Cache (.pt file)
     ├─ Same format as input
     ├─ Physically realistic (gravity, contacts)
     └─ Foot contact respects ground plane

STATUS DETERMINATION
  ├─ success: root_h > 0.4m throughout
  ├─ unstable: root_h ∈ [0.3, 0.4m]
  └─ fell: root_h < 0.3m at any frame
```

---

## Step-by-Step Adaptation for SMPL

### Phase 1: Configuration Setup (30 min)

#### 1.1 Verify SMPL XML Exists
```bash
ls -la ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml
# Should exist, ~500 lines
```

#### 1.2 Count SMPL Joint & Body Indices
```bash
# Extract joint names from XML
grep -E 'name="[LR]_[A-Z]' ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml | grep 'joint' | wc -l
# Expected: ~23 joints

# Find anchor body (torso)
# Search for <body name="Torso"> or <body name="Chest">
# Count its position in tree traversal (body index)
```

#### 1.3 Create SMPL YAML Metadata
```python
# smpl_unified_pipeline.yaml (template)

type: unified_pipeline
dt: 0.02

# Count from smpl_humanoid.xml
joint_names: [
  L_Hip_x, L_Hip_y, L_Hip_z,        # 0-2
  L_Knee_x, L_Knee_y, L_Knee_z,    # 3-5
  L_Ankle_x, L_Ankle_y, L_Ankle_z, # 6-8
  L_Toe_x, L_Toe_y, L_Toe_z,       # 9-11
  R_Hip_x, R_Hip_y, R_Hip_z,       # 12-14
  R_Knee_x, R_Knee_y, R_Knee_z,    # 15-17
  R_Ankle_x, R_Ankle_y, R_Ankle_z, # 18-20
  R_Toe_x, R_Toe_y, R_Toe_z,       # 21-23
  # ... add torso, arms, etc.
]

body_names: [Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe, Torso, ...]

robot:
  num_bodies: 24  # Count in smpl_humanoid.xml
  num_dofs: 23    # Number of joints
  anchor_body_name: Torso
  anchor_body_index: 9  # COMPUTE THIS CAREFULLY
  root_body_name: Pelvis
  root_body_index: 0

control:
  stiffness: [800, 800, 800, 800, ...]  # SMPL values from smpl.py
  damping: [80, 80, 80, 80, ...]        # SMPL values from smpl.py

timing:
  control_dt: 0.02
  physics_dt: 0.001
  decimation: 20
```

**Critical**: Count bodies in XML tree order. Write a helper:
```python
import xml.etree.ElementTree as ET

def count_bodies(mjcf_path):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    
    body_list = []
    def traverse(element):
        for child in element:
            if child.tag == "body":
                body_list.append(child.get("name"))
                traverse(child)
    traverse(worldbody)
    return body_list

bodies = count_bodies("smpl_humanoid.xml")
for idx, name in enumerate(bodies):
    print(f"{idx}: {name}")
```

### Phase 2: ONNX Policy Preparation (2-4 weeks)

**This is the critical bottleneck.**

#### 2.1 Training Data
You need motion capture data retargeted to SMPL:
- **Option A**: Use existing SMPL training data from ProtoMotions
- **Option B**: Retarget G1 motion data to SMPL using ISE/IK
- **Option C**: Use public datasets (AMASS, HumanML3D) pre-converted to SMPL

#### 2.2 ONNX Export Procedure
Follow `deployment/export_bm_tracker_onnx.py`:

1. Train ProtoMotions model on SMPL data
2. Export to ONNX via `torch.onnx.export()` or Lightning's exporter
3. Save as `smpl_unified_pipeline.onnx`
4. Generate YAML metadata

**Likely timeline**: 
- Data prep: 1 week
- Training: 1-2 weeks (depends on GPU availability)
- Export & validation: 2-3 days

#### 2.3 Validation
Test ONNX on single motion:
```bash
python scripts/embodied/run_tracker_export.py \
    --motion data/smpl_test_motion.pt \
    --output output/smpl_tracked_test.pt \
    --onnx path/to/smpl_unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml
```

Check output:
- `output/smpl_tracked_test.pt` contains 50Hz tracked motion
- `status` in log is "success" or "fell"
- Body rotations/positions are physically consistent

### Phase 3: Motion Retargeting (1-2 weeks)

#### 3.1 Prepare Reference Motion Dataset
Convert motion source (SMPL mocap, BVH, etc.) to cache format:
```python
# Your retargeting pipeline should output:
cache = {
    "dof_pos": np.array((T, 23), dtype=np.float32),      # SMPL joint angles
    "dof_vel": np.array((T, 23), dtype=np.float32),      # Time derivatives
    "body_pos": np.array((T, num_bodies, 3), dtype=np.float32),
    "body_rot": np.array((T, num_bodies, 4), dtype=np.float32),  # xyzw
    "body_vel": np.array((T, num_bodies, 3), dtype=np.float32),
    "body_ang_vel": np.array((T, num_bodies, 3), dtype=np.float32),
    "control_dt": 0.02,
    "num_frames": T,
}
torch.save(cache, "path/to/smpl_reference_motion_000.pt")
```

#### 3.2 Use Existing Retargeting Tools
From the repo:
- `batch_t2m_to_embodied.py` (if compatible)
- `gmr_retarget_headless.py` (check if works with SMPL)
- Or adapt `batch_retarget_parallel.py`

#### 3.3 Validate Retargeting
```bash
# Check a sample motion visually
python scripts/embodied/render_tracker_headless.py \
    --motion output/smpl_reference.pt \
    --output-dir /tmp/render_smpl \
    --mode reference \
    --video
```

Should show smooth humanoid motion, no jittering/IK errors.

### Phase 4: Integration & Testing (1 week)

#### 4.1 Single Motion Test
```bash
python scripts/embodied/run_tracker_export.py \
    --motion data/smpl_ref_motion_000.pt \
    --output output/smpl_tracked_000.pt \
    --onnx path/to/smpl_unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml
```

Expected:
- Runs for ~5-20s (depends on motion length)
- Output status: "success" or "fell"
- Logs show ~1-10x realtime speed

#### 4.2 Batch Processing
```bash
python scripts/embodied/run_tracker_export.py \
    --motion-dir data/smpl_reference_motions/ \
    --output-dir output/smpl_tracked_motions/ \
    --onnx path/to/smpl_unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml \
    --pattern 'smpl_ref_*.pt' \
    --max-motions 10
```

Check `output/smpl_tracked_motions/tracker_summary.json` for stats.

#### 4.3 Visualization
```bash
# Render tracked motion
python scripts/embodied/render_tracker_headless.py \
    --motion output/smpl_tracked_000.pt \
    --output-dir /tmp/render_smpl_tracked \
    --mode tracked \
    --video

# Compare with reference
# Side-by-side comparison of reference vs tracked videos
```

#### 4.4 Metrics Collection
```bash
# Extract summary stats
python -c "
import json
with open('output/smpl_tracked_motions/tracker_summary.json') as f:
    results = json.load(f)
    
n_success = sum(1 for r in results if r['status'] == 'success')
n_fell = sum(1 for r in results if r['status'] == 'fell')
avg_height = sum(r.get('root_height_min', 0) for r in results) / len(results)

print(f'Success: {n_success}/{len(results)}')
print(f'Fell: {n_fell}/{len(results)}')
print(f'Avg min height: {avg_height:.3f}m')
"
```

---

## Code Changes Needed

### Option A: Minimal (Use as-is)
```python
# Just point to SMPL config in run_tracker_export.py
smpl_yaml = "path/to/smpl_unified_pipeline.yaml"
smpl_mjcf = "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml"

run_tracker_and_export(
    motion_cache_path="smpl_ref_motion.pt",
    output_path="smpl_tracked_motion.pt",
    onnx_path=smpl_yaml.replace(".yaml", ".onnx"),
    mjcf_path=smpl_mjcf,
)
```

**No code changes required!** The YAML metadata is all that matters.

### Option B: Add SMPL Support Explicitly
Create wrapper script:
```python
# scripts/embodied/run_smpl_tracker_export.py

import argparse
from run_tracker_export import run_tracker_and_export

SMPL_DEFAULTS = {
    "mjcf": "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml",
    "onnx": "path/to/smpl_unified_pipeline.onnx",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    run_tracker_and_export(
        motion_cache_path=args.motion,
        output_path=args.output,
        onnx_path=SMPL_DEFAULTS["onnx"],
        mjcf_path=SMPL_DEFAULTS["mjcf"],
    )

if __name__ == "__main__":
    main()
```

---

## Troubleshooting Guide

### Issue 1: ONNX Model Input Shape Mismatch
```
Error: Expected input shape [1, 29] but got [1, 23]
```

**Fix**: Check YAML `policy_inputs` shape matches actual DOF count.
- G1: `current_dof_pos` shape [1, 29]
- SMPL: `current_dof_pos` shape [1, 23]

### Issue 2: Robot Falls Immediately
```
FALL detected at frame 0 (root_h=0.100)
```

**Causes**:
1. Wrong initial pose → robot not standing
2. Stiffness too low → can't hold position
3. Control targets out of range → joint limits

**Debug**:
```python
# Check initial pose in frame 0
cache = torch.load("smpl_ref_motion.pt")
print("Frame 0 DOF positions:", cache["dof_pos"][0])
print("Frame 0 root height:", cache["body_pos"][0, 0, 2])  # Z coord of Pelvis
```

**Fix**:
- Increase stiffness/damping in YAML
- Verify retargeted motion is in valid SMPL range
- Check MJCF joint limits (`<joint ... range="-π π"`)

### Issue 3: Heading Misalignment
```
Robot rotates weirdly, doesn't follow motion
```

**Cause**: Anchor body index wrong or initial heading offset not computed.

**Fix**:
```python
# In run_tracker_export.py, line 355
# Verify anchor_body_index matches XML
print(f"Anchor body index: {anchor_body_index}")
print(f"Body name: {body_names[anchor_body_index]}")
```

### Issue 4: Motion Cache Dimension Mismatch
```
AssertionError: num_dofs mismatch: expected 23, got 29
```

**Cause**: Using G1 motion cache with SMPL ONNX (or vice versa).

**Fix**: Retarget motion to correct skeleton format.

### Issue 5: ONNX Export Failed
```
RuntimeError: Model has 29 parameters but ONNX expects 23
```

**Cause**: ONNX was exported from G1 checkpoint, not SMPL.

**Fix**: Export ONNX from SMPL-trained checkpoint using correct export script.

---

## Performance Expectations

| Metric | G1 | SMPL | Notes |
|--------|----|----|-------|
| **Simulation speed** | 5-10x realtime | 10-20x realtime | SMPL fewer DOF = faster |
| **Fall rate** | ~10-15% | ? | Depends on motion dataset |
| **Success rate** | ~80-85% | ? | Depends on stiffness tuning |
| **Memory per motion** | ~10 MB | ~6 MB | Smaller DOF = smaller export |
| **Export time per 10s motion** | ~1-2s | ~0.5-1s | Fewer physics steps |

---

## Success Criteria

✓ **Phase 1 (Config)**: YAML loads without errors, all indices verified

✓ **Phase 2 (ONNX)**: Model exports, can run single inference on dummy input

✓ **Phase 3 (Motion)**: Reference cache loads, shape matches YAML

✓ **Phase 4 (Integration)**:
- Single motion exports without error
- Status is "success" on valid motion
- Status is "fell" on invalid motion
- Batch mode processes 10+ motions
- Output shapes match expected (T, 24, 4) etc.

✓ **Final**: Tracked motion visually plausible (render & inspect)

---

## Estimated Timeline

| Phase | Task | Duration | Blocker? |
|-------|------|----------|----------|
| 1 | Config setup | 30 min | ✅ No |
| 2 | ONNX retraining | 2-4 weeks | ⚠️ **Critical** |
| 3 | Motion retargeting | 1-2 weeks | ✅ No |
| 4 | Integration & testing | 1 week | ✅ No |
| **Total** | | **4-7 weeks** | Phase 2 determines schedule |

**If ONNX already exists**: Only phases 3-4 needed = **2-3 weeks**.

---

## Key References in Codebase

- **ONNX inference**: Lines 255-256 in `run_tracker_export.py`
- **PD control setup**: Lines 161-174 in `run_tracker_export.py`
- **Physics loop**: Lines 319-436 in `run_tracker_export.py`
- **SMPL config**: `ref_repo/ProtoMotions/protomotions/robot_configs/smpl.py`
- **YAML loading**: Lines 228-235 in `run_tracker_export.py`
- **Deployment utilities**: `ref_repo/ProtoMotions/deployment/state_utils.py`

---

## Next Steps

1. **Immediately**: Count SMPL bodies/joints in XML, draft YAML
2. **This week**: Verify SMPL MJCF exists, test loading
3. **Next sprint**: Prepare SMPL training data
4. **Following sprint**: Train ONNX model or obtain from team
5. **After ONNX**: Retarget reference motions and test export

**Point of contact**: Check if SMPL ONNX model exists in the team's pretrained models directory first!

# ProtoMotions RL Training Analysis: Complete Documentation Index

## 📚 Document Overview

This folder contains comprehensive documentation for integrating Text-to-Motion (T2M) models with ProtoMotions' reinforcement learning training pipeline. The analysis covers motion file formats, data flow, implementation details, and complete integration examples.

## 📄 Documents

### 1. **PROTOMOTIONS_RL_TRAINING_ANALYSIS.md** (Main Reference)
**Size:** 664 lines | **Format:** Comprehensive Technical Reference

Thorough analysis of ProtoMotions' RL training architecture covering:
- Experiment configuration (MLP motion tracker)
- Motion data format and loading pipeline
- PPO agent implementation
- Environment base class API
- Robot state data structure
- Complete T2M → RL data flow
- Packaged motion files (.pt)
- Reward components
- Integration checklist

**Best for:** Understanding the complete architecture and motion format requirements

---

### 2. **PROTOMOTIONS_QUICK_REFERENCE.md** (Quick Lookup)
**Size:** ~200 lines | **Format:** Quick Reference Cards

Concise reference guide with visual summaries including:
- Motion file format at a glance
- Data flow architecture diagrams
- Key classes and files reference
- MotionLib internal structure
- Motion file loading modes (4 options)
- Training configuration parameters
- RobotState conversion API
- Motion sampling during training
- Critical numerical constraints
- Integration checklist

**Best for:** Quick lookups, remembering API signatures, finding file paths

---

### 3. **PROTOMOTIONS_T2M_INTEGRATION_EXAMPLES.md** (Implementation Guide)
**Size:** ~400 lines | **Format:** 7 Complete Code Examples

Practical, runnable code examples:
1. **Minimal T2M → Motion File Integration**
   - Generate motions from T2M model
   - Save as .motion files
   - Create YAML manifest

2. **Loading T2M Motions**
   - Load YAML manifests
   - Validate motion data
   - Error checking

3. **RL Training Integration**
   - Setup environment with motions
   - PPO agent creation
   - Training loop

4. **Packaged Motion Libraries**
   - Convert individual .motion files to .pt
   - Verify packaged files
   - Fast reloading

5. **Advanced Features**
   - Contact detection
   - Foot contact patterns
   - Contact-aware rewards

6. **Distributed Training**
   - Split motions across ranks
   - Multi-GPU support
   - Rank-specific loading

7. **Validation**
   - Comprehensive motion validation
   - Shape checking
   - Quaternion normalization
   - Sampling tests

**Best for:** Implementing the integration, copying/adapting code

---

## 🔍 Quick Navigation

### By Task

#### "I need to understand the motion file format"
→ See **QUICK_REFERENCE.md** Section "🎯 Motion File Format at a Glance"

#### "I need to save my T2M output"
→ See **EXAMPLES.md** Example 1: "Minimal T2M → Motion File Integration"

#### "I need to load motions into ProtoMotions"
→ See **EXAMPLES.md** Example 2: "Loading T2M Motions into ProtoMotions"

#### "I need to integrate with RL training"
→ See **EXAMPLES.md** Example 3: "Integrating with RL Training"

#### "I need the complete architecture"
→ See **ANALYSIS.md** Sections 1-7

#### "I need to understand RobotState"
→ See **ANALYSIS.md** Section 5 or **EXAMPLES.md** Example 2

#### "I need distributed training setup"
→ See **EXAMPLES.md** Example 6: "Multi-GPU Training"

#### "I need to validate my motions"
→ See **EXAMPLES.md** Example 7: "Validation and Testing"

### By Class/File

#### MotionLib
- **ANALYSIS.md** Section 2: Comprehensive MotionLib overview
- **QUICK_REFERENCE.md** "🏗️ MotionLib Internal Structure"
- **EXAMPLES.md** Examples 1-7: All use MotionLib

#### RobotState
- **ANALYSIS.md** Section 5: RobotState dataclass
- **QUICK_REFERENCE.md** "🔄 RobotState Conversion"
- **EXAMPLES.md** Example 2: RobotState validation

#### PPO Agent
- **ANALYSIS.md** Section 3: PPO implementation
- **EXAMPLES.md** Example 3: PPO integration

#### BaseEnv
- **ANALYSIS.md** Section 4: Environment API
- **EXAMPLES.md** Example 3: Environment creation

#### Motion File Formats
- **ANALYSIS.md** Section 2.2: 4 supported formats
- **QUICK_REFERENCE.md** "📋 Motion File Loading Modes"
- **EXAMPLES.md** Example 1: Save motions

---

## 🎯 Key Concepts at a Glance

### Motion Data Format
**Required fields** (from T2M model):
```
dof_pos           [num_frames, num_dofs]
dof_vel           [num_frames, num_dofs]
rigid_body_pos    [num_frames, num_bodies, 3]
rigid_body_rot    [num_frames, num_bodies, 4]  ← XYZW format!
rigid_body_vel    [num_frames, num_bodies, 3]
rigid_body_ang_vel [num_frames, num_bodies, 3]
fps               scalar
```

### MotionLib Internal Structure
After loading, all motions concatenated:
```
MotionLib.gts     [total_frames, num_bodies, 3]
MotionLib.grs     [total_frames, num_bodies, 4]
MotionLib.dps     [total_frames, num_dofs]
MotionLib.dvs     [total_frames, num_dofs]
... + metadata tensors for each motion ...
```

### Loading Modes
1. **Single .motion file** → Direct load
2. **YAML manifest** → List of motions with weights
3. **.pt package** → Pre-packaged all motions (fastest)
4. **Directory** → Auto-load all .motion files

### Sampling During Training
```
1. Sample motion_id from motion_weights
2. Sample time_t uniformly in [0, motion_length]
3. Get interpolated state at (motion_id, time_t)
4. Use as target for reward computation
```

---

## 🔧 File Paths Reference

| Component | Path |
|-----------|------|
| Experiment Config | `examples/experiments/mimic/mlp.py` |
| MotionLib | `protomotions/components/motion_lib.py` |
| RobotState | `protomotions/simulator/base_simulator/simulator_state.py` |
| BaseEnv | `protomotions/envs/base_env/env.py` |
| PPO Agent | `protomotions/agents/ppo/agent.py` |

---

## ⚠️ Critical Constraints

| Constraint | Why It Matters |
|-----------|----------------|
| **Quaternions XYZW** | ProtoMotions standard, not WXYZ |
| **Quaternions normalized** | Required for SLERP interpolation |
| **Units in meters/radians** | SI units for physics simulation |
| **Proper tensor shapes** | Shape mismatches cause silent errors |
| **FPS must be set** | Used to compute motion_dt and motion_length |

---

## 🚀 Integration Workflow

```
Step 1: Generate Motion with T2M Model
        ↓
Step 2: Format as Dict with Required Fields
        ↓
Step 3: Save as .motion File (torch.save)
        ↓
Step 4: Create YAML Manifest (optional)
        ↓
Step 5: Load into MotionLib
        ↓
Step 6: Validate Motions
        ↓
Step 7: Create RL Environment with MotionLib
        ↓
Step 8: Launch Training with mlp.py Config
```

---

## 📊 Data Flow Diagram

```
T2M Model
   ↓
Output: torch.Tensor [frames, dofs/bodies, ...]
   ↓
RobotState Dict {dof_pos, dof_vel, rigid_body_*, fps}
   ↓
torch.save() → .motion file
   ↓
MotionLib loads all motions
   ├─ Concatenates all fields
   ├─ Computes metadata (motion_lengths, fps, weights)
   └─ Ready for sampling
   ↓
During Training:
   ├─ Sample motion_ids (weighted)
   ├─ Sample motion_times (uniform)
   ├─ Get interpolated RobotState
   ├─ Compute reward: ||current_state - target_state||²
   └─ PPO learns policy to minimize error
```

---

## 💾 Motion Library Sizes

| Format | Load Time | Use Case |
|--------|-----------|----------|
| `.motion` | Slow | Individual motion inspection |
| YAML + `.motion` | Slow | Development |
| `.pt` packaged | Fast | Training, deployment |
| Directory | Slow | Auto-discovery |

**Recommendation:** Use `.pt` packaged files for training

---

## 🔬 Validation Checklist

Before training, ensure:
- [ ] Motion dict has all 7 required fields
- [ ] Quaternions are normalized (norm ≈ 1.0)
- [ ] Quaternions are in xyzw format (not wxyz)
- [ ] FPS is set correctly
- [ ] DOF count matches robot config
- [ ] Body count matches robot config
- [ ] Tensor shapes are correct
- [ ] MotionLib loads without errors
- [ ] Sampling returns valid states
- [ ] Quaternions remain normalized after interpolation

---

## 📞 Common Issues and Solutions

### Issue: "Quaternion not normalized after interpolation"
**Solution:** Use torch.nn.functional.normalize() before saving
```python
rigid_body_rot = F.normalize(torch.randn(...), dim=-1)
```

### Issue: "Shape mismatch in motion loading"
**Solution:** Check DOF/body counts match robot config
```python
robot_config.kinematic_info.num_dofs  # Get expected DOF count
```

### Issue: "Slow motion loading"
**Solution:** Convert to packaged .pt file
```python
motion_lib.save_to_file("packaged.pt")
```

### Issue: "Memory error with large motion library"
**Solution:** Use distributed loading (Example 6)
```python
setup_distributed_training_with_t2m(yaml_file, num_chunks=4)
```

---

## 📖 Reading Recommendations

**For Implementation:**
1. Start with QUICK_REFERENCE.md (5 min read)
2. Review EXAMPLES.md Example 1 (10 min)
3. Adapt Example 1 for your T2M model
4. Reference ANALYSIS.md as needed

**For Deep Understanding:**
1. Read ANALYSIS.md Section 1-3 (30 min)
2. Study motion_lib.py source code (30 min)
3. Review EXAMPLES.md Example 2 (15 min)
4. Trace through MotionLib.get_motion_state() (30 min)

**For Distributed Training:**
1. Read QUICK_REFERENCE.md "🏗️ MotionLib Internal Structure"
2. Study EXAMPLES.md Example 6 (20 min)
3. Configure distributed training setup

---

## 🎓 Learning Path

```
Beginner → Quick Reference (5m)
        → Example 1 (10m)
        → Adapt code for your T2M model

Intermediate → Full Analysis Sections 1-3 (45m)
            → Examples 1-3 (30m)
            → Implement and test

Advanced → Source code study (60m)
        → Distributed training setup (30m)
        → Optimization and scaling
```

---

**Generated:** 2026-05-20  
**ProtoMotions Version:** Latest (2025-2026)  
**Documentation Scope:** MuJoCo CPU-only, single environment RL training


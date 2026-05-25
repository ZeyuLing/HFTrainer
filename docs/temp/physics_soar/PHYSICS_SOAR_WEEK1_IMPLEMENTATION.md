# Physics-SOAR Week 1 Implementation Progress

**Start Date:** 2026-05-18  
**Target End Date:** 2026-05-24  
**Status:** IN PROGRESS

---

## Week 1 Goals

- [x] **Day 1-2:** Create FastPhysicsEvaluator skeleton with SMPL + MuJoCo integration
- [ ] **Day 3-5:** Integrate into SOAR training loop (trainer_physics_soar.py)
- [ ] **Day 6-7:** Test and debug on small batch
- [ ] **Deliverable:** Working Physics-SOAR trainer (validation phase)

---

## Daily Progress Log

### Day 1 (2026-05-18) — Project Setup & Physics Evaluator Skeleton

#### ✅ Completed:
1. **Reviewed all research documents**
   - PHYSICS_SOAR_DECISION_SUMMARY.md (confirmed Physics-SOAR decision)
   - PHYSICS_SOAR_QUICK_START.md (implementation plan)
   - SOAR_PHYSICS_INTEGRATION_ANALYSIS.md (technical details)
   - ref_repo/SOAR/CLAUDE.md (SOAR framework specifics)
   - ref_repo/HY-SOAR/README.md (open-source reference)

2. **Verified project structure**
   - Location: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`
   - HYMotion M2M model: `hftrainer/models/motion/hymotion_m2m/`
   - Target files to create:
     - `hftrainer/models/motion/physics_evaluator.py` (NEW)
     - `hftrainer/models/motion/trainer_physics_soar.py` (NEW)
     - `hftrainer/scripts/train_physics_soar.py` (NEW)
     - `hftrainer/scripts/generate_and_evaluate.py` (NEW)
     - `hftrainer/scripts/ablation_study.py` (NEW)

3. **Analyzed existing trainer code**
   - Found: `hymotion_m2m/bundle.py` and `checkpoint_loading.py`
   - Architecture: MMDiT (dual-stream) with VACE conditioning
   - Input format: [x_t, inactive, reactive, src_mask] (4× motion_dim)
   - Flow matching: velocity prediction (v = x1 - x0)

#### ⏭️ Next Step (Day 2):
- Create `physics_evaluator.py` with:
  - SMPL model loading
  - MuJoCo environment setup
  - Batch evaluation infrastructure
  - 4 metrics: collision_penalty, com_stability, energy_eff, smoothness

---

## Architecture Overview

### Physics Evaluator Location
```
hftrainer/models/motion/physics_evaluator.py
├── FastPhysicsEvaluator (main class)
│   ├── __init__(smpl_model_path, mjcf_path)
│   ├── evaluate_batch(motions) → Dict[str, Tensor]
│   ├── suggest_correction(motion) → Tensor
│   ├── _compute_collision_penalty()
│   ├── _compute_com_stability()
│   ├── _compute_energy_efficiency()
│   ├── _compute_smoothness()
│   └── _forward_kinematics(rotations)
```

### SOAR Trainer Location
```
hftrainer/models/motion/trainer_physics_soar.py
├── TrainerPhysicSOAR (extends existing trainer)
│   ├── __init__(model, physics_evaluator, config)
│   ├── compute_physics_soar_loss()
│   ├── compute_base_loss()
│   ├── train_step(batch)
│   └── validate()
```

### Training Script Location
```
hftrainer/scripts/train_physics_soar.py
├── Setup config
├── Load model + physics evaluator
├── Training loop with Physics-SOAR loss
└── Checkpoint saving
```

---

## Key Implementation Details

### Physics Metrics (Week 1 Target)

1. **Collision Penalty** (0-1, lower is better)
   - Detect self-collisions in MuJoCo simulation
   - Count collision frames / total frames
   - Smooth via moving average

2. **COM Stability** (0-1, higher is better)
   - Simulate motion in MuJoCo
   - Extract center-of-mass trajectory
   - Compute variance → inverse to get stability score

3. **Energy Efficiency** (0-1, higher is better)
   - Compute work done by joints (torque × angular velocity)
   - Normalize to 0-1 scale
   - Lower energy = more efficient

4. **Smoothness** (0-1, higher is better)
   - Joint velocity continuity (jerk as proxy)
   - L2 norm of acceleration differences
   - Normalize to 0-1

5. **Overall Score**
   - Weighted average of 4 metrics
   - Default weights: [0.3, 0.3, 0.2, 0.2]
   - Adjustable in config

### Physics Evaluator Optimization

Target performance: **< 50ms per motion** (ideally < 1s for batch of 32)

Optimization strategies:
- Batch MuJoCo evaluation (32+ parallel instances)
- GPU-accelerated forward kinematics (optional)
- Caching FK results
- Approximate collision detection (voxel-based optional)
- Async evaluation (queue-based processing)

### Training Loop Integration

```python
# Base loss (existing, unchanged)
v_pred = model(x_t, caption, t)
loss_base = L1(v_pred, v_gt)

# NEW: SOAR correction loss with physics
with torch.no_grad():
    # 1. Single-step rollout
    v_rollout = model(x_t, caption, t)
    x_hat = x_t + dt * v_rollout
    
    # 2. Re-noise to create auxiliary points
    for n in range(N):
        t_prime = uniform(t, 1.0)
        x_prime = blend(x_hat, x1_noise, t_prime)
        
        # 3. Quick denoise to get x0_candidate
        x0_candidate = quick_denoise(x_prime, num_steps=5)
        
        # 4. Physics evaluation
        physics_score = evaluator.evaluate(x0_candidate)
        
        # 5. Physics-guided correction target
        if physics_score < threshold:
            x_corrected = evaluator.suggest_correction(x0_candidate)
            x_target = blend(x_corrected, x0_clean, blend_ratio)
        else:
            x_target = x0_clean
        
        # 6. Correction velocity
        v_corr = (x_prime - x_target) / t_prime
        
        # 7. Model prediction on off-trajectory point
        v_off = model(x_prime, caption, t_prime)
        
        # 8. Correction loss
        loss_soar += L1(v_off, v_corr)

loss_total = loss_base + lambda_soar * loss_soar
```

---

## Dependencies & Requirements

### Python Packages
```
torch >= 2.0.0
numpy
mujoco >= 3.0.0
dm_control (optional, for reference)
smpl_utils (for SMPL forward kinematics)
scipy (for filtering/smoothing)
```

### Model Files Needed
```
SMPL model: 
  - Either: official SMPL_NEUTRAL.pkl (requires registration)
  - Or: Use existing SMPL from HYMotion codebase if available
  
MJCF/XML files:
  - SMPL humanoid rig in MuJoCo format
  - Collision geometry definitions
  - Contact properties
```

### Recommended Setup
```bash
# If not already installed:
pip install mujoco dm-control
# Copy SMPL model from ref location or download
# Ensure MJCF files are in: hftrainer/models/motion/physics_assets/
```

---

## Timeline & Deliverables

### Week 1 Deliverables

**Day 1-2 (Complete by 2026-05-19):**
- [ ] `physics_evaluator.py` created with:
  - [ ] FastPhysicsEvaluator class skeleton
  - [ ] SMPL model loading
  - [ ] MuJoCo environment setup
  - [ ] Batch evaluation framework
  - [ ] Unit tests for each metric

**Day 3-5 (Complete by 2026-05-22):**
- [ ] `trainer_physics_soar.py` created with:
  - [ ] TrainerPhysicSOAR class
  - [ ] compute_physics_soar_loss() method
  - [ ] Integration tests with mock data

**Day 6-7 (Complete by 2026-05-24):**
- [ ] End-to-end testing on small batch
- [ ] Debug and fix issues
- [ ] Validation checkpoint saved

### Success Criteria (Week 1)
- [ ] Physics evaluator runs without errors
- [ ] Batch evaluation speed acceptable (< 1s per 32 motions)
- [ ] SOAR training loop integrates cleanly
- [ ] First training step completes without NaN
- [ ] Loss curves logged and monitored
- [ ] No VRAM issues on 1x GPU (or specified hardware)

---

## Known Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| SMPL model not available | Medium | Check ref_repo, fallback to generic humanoid if needed |
| MuJoCo simulation slow | Medium | Implement parallel evaluation, cache FK, approximate metrics |
| Physics evaluation NaN | Low | Add error handling, clipping, numerical stability checks |
| Memory overflow | Low | Batch physics eval with queue system |
| Integration bugs | Medium | Extensive unit tests, mock data validation |

---

## Resources & References

### Code References
- **HY-SOAR Open Source:** ref_repo/HY-SOAR/ (for SOAR implementation details)
- **SOAR Paper:** ref_repo/SOAR/SOAR_paper.pdf
- **HYMotion M2M:** hftrainer/models/motion/hymotion_m2m/ (architecture)

### Documentation
- **PHYSICS_SOAR_QUICK_START.md** — Implementation guide
- **SOAR_PHYSICS_INTEGRATION_ANALYSIS.md** — Technical deep-dive
- **PHYSICS_SOAR_DECISION_SUMMARY.md** — Decision rationale

### External Resources
- MuJoCo Docs: https://mujoco.readthedocs.io/
- SMPL Model: https://smpl.is.tue.mpg.de/
- Diffusers Docs (HYMotion base): https://huggingface.co/docs/diffusers/

---

## Questions & Decision Points

### Q1: Which SMPL variant to use?
- [ ] SMPL-H (hand articulation) — more realistic but slower
- [ ] SMPL+H (deprecated)
- [ ] SMPL (neutral) — fastest, sufficient for physics
- **Recommendation:** SMPL (neutral) for speed, upgrade if needed

### Q2: Physics evaluator backend?
- [ ] Direct MuJoCo (Python API)
- [ ] dm_control wrapper
- [ ] Custom C++ bindings
- **Recommendation:** Direct MuJoCo Python API for simplicity

### Q3: Collision detection method?
- [ ] Contact-based (ground truth from MuJoCo)
- [ ] Distance-based (approximate)
- [ ] Voxel-based (fast approximate)
- **Recommendation:** Contact-based (native MuJoCo)

### Q4: Parallel evaluation strategy?
- [ ] ThreadPool (GIL issues)
- [ ] ProcessPool (heavy overhead)
- [ ] Queue + async processing
- [ ] SIMD via numpy (vector ops)
- **Recommendation:** Vector ops + batch processing for speed

---

## Next Session Plan

**Next session (Day 2-3):**
1. Finalize physics_evaluator.py skeleton
2. Add unit tests for each metric
3. Performance profiling
4. Begin trainer_physics_soar.py integration

**Expected by end of Week 1:**
- Fully functional Physics-SOAR trainer ready for validation tests
- Hyperparameter configuration locked
- Training data pipeline tested

---

**Last Updated:** 2026-05-18  
**Implemented By:** Claude (AI)  
**Status:** Week 1 Day 1 Complete → On Track

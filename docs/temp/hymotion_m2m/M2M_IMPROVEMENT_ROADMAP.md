# HyMotion M2M Improvement Roadmap: Based on KIMODO Analysis

**Status:** Technical Recommendations (2026-05-19)  
**Purpose:** Specific improvements for M2M to close capability gap with KIMODO

---

## Priority Levels

- **P0 (Critical):** Must-have for feature parity with KIMODO
- **P1 (Important):** Significant quality improvements, moderate implementation effort
- **P2 (Nice-to-have):** Marginal gains, high effort or experimental

---

## P0: CRITICAL IMPROVEMENTS

### 1. Add Global Position Constraints to Representation

**Current Issue:**
- M2M: 135 dims (3 abs translation + 22×6 local rotations) → no position dims
- KIMODO: 333 dims with explicit joint positions [5:86]
- **Gap:** Cannot directly constrain end-effector world positions

**Recommendation: Option A - Minimal Change (Recommended)**

```python
# Current: [abs_transl(3), rot_6d(22×6)] = 135 dims
# New: [abs_transl(3), rot_6d(22×6), end_effector_pos(3×4)] = 147 dims
#      Add: L_hand, R_hand, L_foot, R_foot positions

new_dims = 135 + 12

# During training:
# - Compute via FK from rotations
# - Include in loss: ||FK(predicted_rot) - predicted_pos||_L1
```

**Expected Benefit:**
- ✅ Can now do end-effector IK like KIMODO
- ✅ Direct trajectory control

**Estimated Effort:** Medium (1-2 weeks)

---

### 2. Add Explicit Foot Contact Modeling

**Current Issue:**
- KIMODO: 4-dim foot contact with explicit loss
- M2M: No foot contact modeling

**Recommendation:**

```python
# Add to representation: [motion_135, foot_contacts_4] = 139 dims
# Training: BCE loss with weight γ=3
# Inference: Post-process foot lock
```

**Expected Benefit:**
- ✅ Better foot ground interaction
- ✅ Reduces foot skating artifact

**Estimated Effort:** Low (3-5 days)

---

### 3. Add FK Consistency Loss

**Current Issue:**
- KIMODO: Explicit FK loss (weight 5)
- M2M: No explicit FK loss

**Recommendation:**

```python
# During training:
L_fk = smooth_L1(FK(rot_pred) - pos_gt)
total_loss = L_old + γ_fk * L_fk  # γ_fk = 5
```

**Expected Benefit:**
- ✅ Better rotation-position coherence
- ✅ More physically plausible motion

**Estimated Effort:** Low (2-3 days)

---

## P1: IMPORTANT IMPROVEMENTS

### 4. Optional Two-Phase Training

**Current:** M5 only 5% of training (possible T2M dilution)

**Recommendation:**

```
Option A: Partial two-phase (cost-effective)
├─ Phase 1a: 50% on M5 only
├─ Phase 1b: 10% on M1-M4, M6
└─ Phase 2: 40% on all M1-M6

Option B: Full two-phase (like KIMODO)
├─ Phase 1: 50% on M5 only
└─ Phase 2: 50% on M1-M6
```

**Expected Benefit:**
- ✅ Stronger T2M foundation
- ✅ More stable constraint learning

**Estimated Effort:** Medium (1 week tuning + retraining)

---

### 5. Smooth Root Trajectory Post-Processing

**Recommendation:** Post-process root XZ with Gaussian smoothing

```python
smoothed_positions = gaussian_filter_1d(root_positions[:,[0,2]], 
                                         sigma=2, axis=0)
```

**Expected Benefit:**
- ✅ Smoother trajectories
- ✅ Animator-friendly
- ✅ No training change needed

**Estimated Effort:** Low (1-2 days)

---

## IMPLEMENTATION PRIORITY

**Weeks 1-2:** Add EE positions (P0 #1)  
**Weeks 2-3:** Add foot contact (P0 #2)  
**Weeks 3-4:** Add FK loss (P0 #3)  
**Weeks 4-5:** Root smoothing post-process (P1 #5)  
**Weeks 5+:** Two-phase curriculum (P1 #4)

**Total for P0: ~2-3 weeks**

---

## VALIDATION METRICS

For each improvement:

- Constraint satisfaction: position error, FK coherence
- Generation quality: FID, foot skating, jitter
- Task-specific: completion accuracy, T2M diversity

---

## CONCLUSION

Implementing P0 items will give M2M feature parity with KIMODO for:
- End-effector IK
- Trajectory control  
- Foot ground interaction

While maintaining M2M's advantage on temporal flexibility.


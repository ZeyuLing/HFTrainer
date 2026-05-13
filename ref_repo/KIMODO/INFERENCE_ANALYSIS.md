# KIMODO Motion Representation & Inference Analysis

**Analysis Date:** 2026-05-12  
**Source:** Direct code inspection of KIMODO open-source repo  
**Focus:** Inference-time behavior, NOT training

---

## Executive Summary

### Direct Answers to Your Questions

1. **Does KIMODO use heading channel during inference?**
   - ✅ **YES**, the heading channel (`global_root_heading`) is **ALWAYS OUTPUT** by the model
   - ❌ **NOT discarded** — it's part of the final motion representation

2. **Does KIMODO predict both smooth_trans AND trans_residual during inference?**
   - ✅ **smooth_trans (smooth_root_pos)** — ALWAYS predicted
   - ❌ **NO trans_residual** — KIMODO doesn't decompose translation into "smooth + residual"
   - **Note:** Unlike some decomposition methods, KIMODO uses a single smoothed trajectory

3. **How does KIMODO convert motion back to SMPL format?**
   - Detailed in `inverse()` function
   - **Step 1:** Unpack predicted features (smooth_root_pos, heading, joint positions/rotations)
   - **Step 2:** Convert 6D rotations → rotation matrices → local rotations via `global_rots_to_local_rots()`
   - **Step 3:** Use FK (forward kinematics) with local rotations + root positions to get final joint positions
   - **Step 4:** Output dictionary with `local_rot_mats`, `posed_joints`, `root_positions`, etc.

---

## Part 1: KIMODO Motion Representation During INFERENCE

### 1.1 Complete Feature Layout (333 dims, per frame)

KIMODO's motion representation is defined in `KimodoMotionRep.__init__()`:

```python
size_dict = {
    "smooth_root_pos":        [3],       # dims [0:3]
    "global_root_heading":    [2],       # dims [3:5]
    "local_joints_positions": [27×3],   # dims [5:86]
    "global_rot_data":        [27×6],   # dims [86:248]
    "velocities":             [27×3],   # dims [248:329]
    "foot_contacts":          [4],      # dims [329:333]
}
```

**Total: 3 + 2 + 81 + 162 + 81 + 4 = 333 dims per frame**

(Note: Uses native 27-joint Bones Rigplay skeleton; SOMA-30 retarget has same structure)

---

### 1.2 What Gets PREDICTED vs. COMPUTED

| Channel | Size | **Predicted?** | Inference Role |
|---------|------|---|---|
| `smooth_root_pos` | 3 | ✅ **YES** | Model predicts smoothed root trajectory (x,z smooth, y absolute) |
| `global_root_heading` | 2 | ✅ **YES** | Model predicts heading as `[cos(ψ), sin(ψ)]` |
| `local_joints_positions` | 81 | ✅ **YES** | Model predicts joint positions relative to smooth root |
| `global_rot_data` | 162 | ✅ **YES** | Model predicts 6D global rotations for all 27 joints |
| `velocities` | 81 | ❌ **NO** | Computed **after** inverse, not used for anything during diffusion |
| `foot_contacts` | 4 | ❌ **NO** | Computed **after** inverse from position+velocity, not predicted |

**Key Insight:** During diffusion sampling (50 DDIM steps), the model PREDICTS exactly **248 dimensions** (smooth_root_pos + heading + local_joints_positions + global_rot_data). The remaining 85 dimensions (velocities + foot_contacts) are **NEVER predicted** — they're purely computed post-hoc for output.

---

### 1.3 The Heading Channel Is NOT Discarded

**Inference path** (code: `kimodo_model.py:505-510`):

```python
output = self.motion_rep.inverse(
    motion,  # [B, T, 333] with heading at dims [3:5]
    is_normalized=True,
    return_numpy=False,
)
```

The `inverse()` method (lines 161-215 of kimodo_motionrep.py) does:

```python
[
    smooth_root_pos,
    global_root_heading,              # ← EXTRACTED
    local_joints_positions,
    global_rot_data,
    velocities,
    foot_contacts,
] = einops.unpack(features, self.ps, "batch time *")

# ... (rotation conversion) ...

output_tensor_dict = {
    "local_rot_mats": local_rot_mats,
    "global_rot_mats": global_rot_mats,
    "posed_joints": posed_joints,
    "root_positions": root_positions,
    "smooth_root_pos": smooth_root_pos,
    "foot_contacts": foot_contacts,
    "global_root_heading": global_root_heading,  # ← RETURNED IN OUTPUT
}
```

**Result:** The heading is **returned in the final output dictionary** and available for downstream use (visualization, constraint application, etc.). It is NOT discarded.

---

## Part 2: SMOOTH ROOT (Not trans_residual)

### 2.1 KIMODO's Decomposition Strategy

KIMODO **does NOT** decompose trajectory into smooth + residual. Instead:

- **Single smoothed trajectory:** `smooth_root_pos` is pre-computed during data encoding
- **Method:** Heavy ADMM-based smoothing applied to pelvis X-Z coordinates during training preprocessing
- **Purpose:** Stabilize trajectory to match animator workflow (straight lines, curves) rather than noisy mocap joints

### 2.2 How Smooth Root Is Computed (Training/Inference-Agnostic)

Code: `smooth_root.py:201-234`

```python
def get_smooth_root_pos(hip_translations):
    """Smooth root trajectory in ground plane, preserve height."""
    root_translations_xz = hip_translations[..., [0, 2]]  # Extract X-Z
    root_translations_y = hip_translations[..., [1]]      # Extract Y
    
    # ADMM-based smoothing with 6cm margin
    margins = np.full(nframes, 0.06)  # Allow ±6cm deviation
    
    # Multigrid smoothing (scales coarse→fine)
    root_translations_smoothed_xz = smooth_signal(
        root_translations_xz, 
        margins,
        pos_weight=0,      # Don't penalize deviation from original
        admm_iters=500
    )
    
    # Recombine: smoothed XZ + original Y
    result = [smoothed_xz_x, original_y, smoothed_xz_z]  # [x, y, z]
    return result
```

**Key design:**
- X-Z are smoothed aggressively (6cm tolerance for acceleration)
- Y (height) is **NOT smoothed** — remains true to mocap
- Margins allow 6cm deviation from original trajectory for smoothness

### 2.3 Is There a "Residual" Component?

**NO.** The only trajectory stored is the **single smooth_root_pos**:

```python
# During inference decoding (inverse):
hips_offset = root_positions - smooth_root_pos
hips_offset[..., 1] = root_positions[..., 1]  # Y stays unchanged
local_joints_positions = local_joints_positions_origin_is_pelvis + hips_offset[:, :, None]
```

The offset between raw pelvis and smooth root is **NOT stored**. It's computed on-the-fly during FK.

---

## Part 3: INFERENCE-TIME CONVERSION FROM KIMODO FORMAT TO SMPL

### 3.1 Step-by-Step Decoding

**Input:** Model output `motion` of shape `[B, T, 333]`

**Step 1: Unnormalize** (if needed)
```python
if is_normalized:
    features = self.unnormalize(features)
    # Uses per-part stats: global_root_stats, local_root_stats, body_stats
```

**Step 2: Unpack features**
```python
[
    smooth_root_pos,         # [B, T, 3]
    global_root_heading,     # [B, T, 2]
    local_joints_positions,  # [B, T, 81]
    global_rot_data,         # [B, T, 162] = 27 joints × 6D
    velocities,              # [B, T, 81] (unused)
    foot_contacts,           # [B, T, 4]  (unused)
] = einops.unpack(features, self.ps, "batch time *")
```

**Step 3: Convert 6D rotations → matrices**
```python
global_rot_mats = cont6d_to_matrix(global_rot_data)  # [B, T, 27, 3, 3]
```

**Step 4: Convert global rotations → LOCAL rotations**
```python
local_rot_mats = global_rots_to_local_rots(global_rot_mats, self.skeleton)
# Applies skeleton hierarchy: R_local = R_parent^T @ R_global
```

**Step 5: Reconstruct root position from smooth root + local offsets**
```python
posed_joints_from_pos = local_joints_positions.clone()
posed_joints_from_pos[..., 0] += smooth_root_pos[..., None, 0]  # Add X offset
posed_joints_from_pos[..., 2] += smooth_root_pos[..., None, 2]  # Add Z offset
# Y stays unchanged (no offset applied)

root_positions = posed_joints_from_pos[..., self.skeleton.root_idx, :]
```

**Step 6: Forward Kinematics to get final joint positions**
```python
_, posed_joints, _ = self.skeleton.fk(local_rot_mats, root_positions)
# posed_joints: [B, T, 27, 3] in world space
```

**Step 7: Return output dictionary**
```python
output_tensor_dict = {
    "local_rot_mats": local_rot_mats,           # [B, T, 27, 3, 3]
    "global_rot_mats": global_rot_mats,         # [B, T, 27, 3, 3]
    "posed_joints": posed_joints,               # [B, T, 27, 3] world space
    "root_positions": root_positions,           # [B, T, 3]
    "smooth_root_pos": smooth_root_pos,         # [B, T, 3]
    "foot_contacts": foot_contacts > 0.5,       # [B, T, 4] boolean
    "global_root_heading": global_root_heading,  # [B, T, 2]
}
```

---

### 3.2 Differences from SMPL Format

| Aspect | KIMODO (Native) | SMPL (For comparison) |
|--------|---|---|
| **Rotation coordinate frame** | **Global** (world-space) | Local (parent-relative) |
| **Rotation representation** | 6D continuous | Axis-angle or rotation matrix |
| **Root translation** | Smoothed (separate field) | Absolute translation |
| **Joint positions** | Stored separately (for FK) | Not explicitly stored (computed via FK) |
| **Foot contacts** | Explicit boolean [4] | Not in standard SMPL |
| **Velocities** | Computed post-hoc | Not in standard SMPL |
| **Heading** | Explicit [cos(ψ), sin(ψ)] | Implicit in root rotation |

---

## Part 4: WHICH CHANNELS ARE ACTUALLY USED AT INFERENCE?

### 4.1 Per-Denoising-Step Usage

During the 50-step DDIM sampling (code: `kimodo_model.py:589-605`):

```python
for i in progress_bar(indices):
    t = torch.tensor([i] * cur_mot.size(0), device=self.device)
    
    # Denoising step: model predicts clean x0 from noisy x_t
    cur_mot = self.denoising_step(
        cur_mot,          # [B, T, 333] ← ALL 333 dims passed
        pad_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle,
        motion_mask,      # [B, T, 333] constraint mask
        observed_motion,  # [B, T, 333] if constraints present
        num_denoising_steps,
        cfg_weight,
    )
    
    # Optional: Hard-paste constraints back (if env var set)
    if KIMODO_REPAINT_CONDITION == "1":
        cur_mot = cur_mot * (1 - motion_mask) + observed_at_t * motion_mask
    
    # Final step: hard-paste all constraints
    if KIMODO_FINAL_HARD_PASTE == "1":
        cur_mot = cur_mot * (1 - motion_mask) + observed_motion * motion_mask
```

**All 333 channels are:**
- ✅ Predicted by the model each step
- ✅ Subject to imputation if constrained
- ✅ Evolved through full diffusion process

### 4.2 Inside the Transformer

Code: `twostage_denoiser.py:98-103`

```python
# If constraints provided:
if self.motion_mask_mode == "concat":
    # Direct imputation in noisy space
    x = x * (1 - motion_mask) + observed_motion * motion_mask
    
    # Extend with mask as auxiliary input channel
    x_extended = torch.cat([x, motion_mask], axis=-1)  # [B, T, 666]
```

The transformer sees **666 dimensions total:**
- 333 from motion (after imputation)
- 333 from binary mask (which dims are constrained)

The transformer's task: "Denoise the motion, treating masked dims as GT anchors."

---

## Part 5: THE COMPLETE INFERENCE PIPELINE

```
┌─────────────────────────────────────────────────────┐
│ INPUT: Text prompt + constraints (optional)         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 1. ENCODE TEXT                                      │
│    - LLM2Vec (LLaMA-based) → 4096-dim vector       │
│    - Project to latent: 1024-dim                    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 2. BUILD CONSTRAINT TENSORS (if any)               │
│    - create_conditions() → observed_motion [B,T,333]│
│    - Create motion_mask [B,T,333] binary mask      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 3. INITIALIZE RANDOM NOISE                          │
│    - x_t ~ N(0,1), shape [B, T, 333]              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 4. DDIM SAMPLING LOOP (50 steps)                    │
│    for i in [49, 48, ..., 0]:                       │
│      4.1. Impute constraints:                       │
│           x_t = x_t * (1-mask) + observed * mask   │
│      4.2. Extend input with mask:                   │
│           x_ext = cat(x_t, mask) → [B,T,666]      │
│      4.3. Transformer denoises (TwoStageDenoiser)  │
│           pred_x0 = model(x_ext, text, t)         │
│      4.4. Apply CFG (separated):                    │
│           pred_x0 = cfg_pred + w_text*(t_pred-cfg) │
│                    + w_constr*(c_pred-cfg)         │
│      4.5. DDIM sampler: compute x_{t-1}           │
│      4.6. [Optional] Re-impute if env var set     │
│                                                    │
│    After loop:                                     │
│      4.7. [Optional] Hard-paste all constraints   │
│           cur_mot = cur_mot*(1-mask) + obs*mask   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 5. DENORMALIZE                                      │
│    - Undo per-part normalization (3 components)    │
│    - Output: [B, T, 333]                           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 6. INVERSE TRANSFORM (Decode to SMPL-compatible)  │
│    - Unpack 333 dims → components                  │
│    - 6D rotations → matrices → local via hierarchy │
│    - FK: local_rot_mats + root_positions → joints  │
│    - Compute velocities (unused)                   │
│    - Compute foot_contacts from position+velocity │
│    - Output dict: local_rot_mats, posed_joints,   │
│                   root_positions, heading, etc.    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 7. [OPTIONAL] POST-PROCESSING                       │
│    - Foot lock: IK corrections for foot skating    │
│    - Constraint enforcement: exact matching        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 8. [OPTIONAL] SKELETON CONVERSION                   │
│    - SOMA30 → SOMA77 (for mesh rendering)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ OUTPUT: Motion dict                                 │
│ - local_rot_mats [B,T,27,3,3]                      │
│ - global_rot_mats [B,T,27,3,3]                     │
│ - posed_joints [B,T,27,3]                          │
│ - root_positions [B,T,3]                           │
│ - smooth_root_pos [B,T,3]                          │
│ - foot_contacts [B,T,4]                            │
│ - global_root_heading [B,T,2]  ← KEPT              │
└─────────────────────────────────────────────────────┘
```

---

## Part 6: SUMMARY TABLE — WHAT IS PREDICTED VS. COMPUTED

| Feature | Dims | **Predicted by Model?** | **Output in Final Dict?** | **Used for Downstream Tasks?** |
|---------|------|---|---|---|
| `smooth_root_pos` | 3 | ✅ YES | ✅ YES | ✅ YES (trajectory, visualization) |
| `global_root_heading` | 2 | ✅ YES | ✅ YES | ✅ YES (orientation, constraints) |
| `local_joints_positions` | 81 | ✅ YES | ❌ NO | ✅ YES (internal, for FK) |
| `global_rot_data` | 162 | ✅ YES | ❌ NO (converted to matrices) | ✅ YES (for local rot via hierarchy) |
| `velocities` | 81 | ❌ NO (computed post-hoc) | ❌ NO (computed from positions) | ❌ NO (not used) |
| `foot_contacts` | 4 | ❌ NO (computed post-hoc) | ✅ YES (in output dict) | ✅ YES (for post-proc, visualization) |

---

## Part 7: KEY FINDINGS

### 7.1 Heading Is NOT Discarded

The `global_root_heading` (2D: [cos(ψ), sin(ψ)]) is:
1. **Predicted** by the transformer at every diffusion step
2. **Unpacked** during inverse transformation
3. **Returned** in the final output dictionary
4. **Available** for downstream applications

### 7.2 No Decomposition into Smooth + Residual

KIMODO stores:
- **Single smoothed trajectory** (smooth_root_pos)
- **No explicit residual** (residual is implicit in `local_joints_positions`)

The "smooth" part is achieved via heavy ADMM smoothing during data preprocessing, not via learned decomposition.

### 7.3 Inference-Only Channels

Two channels are **NOT predicted** by the model:
- **Velocities:** Computed as finite differences of positions
- **Foot contacts:** Computed from position+velocity thresholds

These are for output/visualization only; they never feedback into generation.

### 7.4 All 333 Dims Are Evolved During Diffusion

Unlike some methods that separate root/body conditioning:
- **KIMODO predicts all 333 dimensions** in each denoising step
- The transformer architecture (TwoStageDenoiser) is abstracted, but the full feature vector is evolved
- Constraints are applied via imputation (replace dims in noisy motion)

### 7.5 Constraints Use Direct Imputation

Unlike some methods that use additional conditioning inputs:
- **KIMODO:** Replace noisy motion dims directly with GT values
- **Mask:** 333-dim binary mask tells which dims are constrained
- **Result:** Constrained dims become frozen (GT) at each diffusion step

---

## Part 8: REFERENCES IN CODE

### Key Files Analyzed
1. `kimodo/motion_rep/reps/kimodo_motionrep.py` — Motion representation and inverse
2. `kimodo/model/kimodo_model.py` — Inference pipeline and __call__
3. `kimodo/motion_rep/smooth_root.py` — Smooth root computation
4. `kimodo/motion_rep/reps/base.py` — Base class, normalization

### Key Methods
- `KimodoMotionRep.__call__()` — Encode SMPL → KIMODO features (training-time)
- `KimodoMotionRep.inverse()` — Decode KIMODO features → SMPL-compatible dict (inference-time)
- `Kimodo._generate()` — Main diffusion loop
- `Kimodo.denoising_step()` — Single denoising step with CFG
- `get_smooth_root_pos()` — ADMM-based trajectory smoothing

---

## Conclusion

**Answer to all three questions:**

1. ✅ **Heading IS used during inference** — predicted at every step, returned in output
2. ❌ **NO trans_residual decomposition** — KIMODO uses single smoothed trajectory
3. **Conversion is via standard FK pipeline:**
   - 6D rotations → matrices → local via skeleton hierarchy
   - FK with local rotations + root positions → final joints
   - Velocities and foot_contacts computed post-hoc from positions


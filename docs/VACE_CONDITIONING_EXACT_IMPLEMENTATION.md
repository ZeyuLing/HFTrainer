# HyMotion M2M v2 VACE Conditioning Implementation Report

## 1. INPUT_ENCODER CHANNEL DIMENSIONS AND PROCESSING

### A. Input Architecture (hymotion_mmdit.py, line 704)

```python
# Input encoder: projects VACE-concatenated motion to feat_dim
self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)
```

**Configuration:**
- `input_dim = 594` (from _base config: `_motion_dim * 3 = 198 * 3`)
- `feat_dim = 1024` (hidden dimension)
- Maps: (B, L, 594) → (B, L, 1024)

### B. Multi-Channel Input Structure (594 dimensions)

The input to `input_encoder` is a **concatenation of three channels**, each 198 dimensions:

```
Channel Layout (594 = 198 + 198 + 198):
┌─ Dims [0:198]    ─ x_t: noisy motion (flow matching input)
├─ Dims [198:396]  ─ reactive: pre-edit values or zeros
└─ Dims [396:594]  ─ mask: binary mask (0=keep, 1=generate)
```

### C. Motion Dimension Breakdown (198 dims per channel)

```
Per-channel structure (e.g., x_t in [0:198]):
┌─ [0:3]       ─ translation (absolute XYZ)
├─ [3:135]     ─ 22 joints × 6D rot6d (SMPL local rotation)
│              │ Layout:
│              │ [3:9]      joint 0 (Pelvis)
│              │ [9:15]     joint 1 (L_Hip)
│              │ [15:21]    joint 2 (R_Hip)
│              │ ... (continuing for 22 joints total)
│              │ [129:135]  joint 21 (R_Wrist)
└─ [135:198]   ─ 21 joints × 3D position (XZ relative, Y absolute)
               │ Pelvis position excluded; only 21 joints
               │ [135:138]   joint 0 position
               │ [138:141]   joint 1 position
               │ ... (continuing for 21 joints)
               └ [195:198]   joint 20 position
```

**Total:** 3 + 132 + 63 = 198 dimensions per channel

## 2. VACE CONDITIONING MODES

### A. Current Configuration (M2M v2)

**File:** `_base_hymotion_m2m_v2_046b.py`, line 100

```python
vace_condition_mode='no_inactive'
```

### B. Three Supported Modes

#### Mode 1: `'split_reactive'` (Legacy/Full VACE)

```python
# From bundle.py lines 465-466
if self.vace_condition_mode == 'split_reactive':
    reactive = src_motion * src_mask
```

**Output VACE context:** (B, L, 3*D) = (B, L, 594)
- Channel 0 [0:198]: `inactive = src_motion * (1 - src_mask)`
  - Known regions (mask=0): contain full motion values
  - Generation regions (mask=1): zero
- Channel 1 [198:396]: `reactive = src_motion * src_mask`
  - Known regions (mask=0): zero
  - Generation regions (mask=1): pre-edit values (editing) or zero (completion)
- Channel 2 [396:594]: `mask` as-is
  - 0 where keep, 1 where generate

**Model Input:** `[x_t, inactive, reactive, mask]` = 4D input (594 + 198 = 792 dims)

#### Mode 2: `'clean_zero_mask'`

```python
# From bundle.py lines 467-468
elif self.vace_condition_mode == 'clean_zero_mask':
    reactive = torch.zeros_like(src_motion)
```

**Output VACE context:** (B, L, 3*D) = (B, L, 594)
- Channel 0: `inactive = src_motion * (1 - src_mask)`
- Channel 1: `reactive = all zeros` (always zero regardless of mode)
- Channel 2: `mask`

**Use case:** Completion-only mode (no editing)

#### Mode 3: `'no_inactive'` (Current Default for v2)

```python
# From bundle.py lines 469-487
elif self.vace_condition_mode == 'no_inactive':
    reactive = src_motion * src_mask
    vace_context = reactive  # (B, L, D) — single channel
    vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 2*D)
    return vace_context
```

**Output VACE context:** (B, L, 2*D) = (B, L, 396)
- Channel 0 [0:198]: `reactive = src_motion * src_mask`
  - Known regions (mask=0): zero
  - Generation regions (mask=1): pre-edit values (editing) or zero (completion)
- Channel 1 [198:396]: `mask` as-is

**Model Input:** `[x_t, reactive, mask]` = 3D input (594 dims, as specified in config)
- `input_dim = 594` matches 3 channels × 198 dims

**Rationale (from code comments):**
```
"v2 slim VACE — drops the `inactive` channel. Rationale:
under mask-aware noise (MAN), `x_t[known] = clean_motion` already
carries known-region values into the model, so `inactive` becomes
redundant. VACE then only needs to signal: (a) what the pre-edit
value was in mask=1 regions (`reactive`, 0 in completion, LQ in
editing), and (b) where the mask is. Total vace_context = 2*D.
Model input = x_t + reactive + mask = 3*D."
```

## 3. PREPARE_VACE_INPUT METHOD (bundle.py, lines 450-503)

### Full Implementation for `no_inactive` Mode

```python
def prepare_vace_input(
    self,
    src_motion: Tensor,
    ref_pose: Optional[Tensor] = None,
    src_mask: Optional[Tensor] = None,
) -> Tensor:
    """Build VACE conditioning context.
    
    Returns tensor of shape (B, L, 3*D) where D is the motion dim.
    """
    B, L_src, D = src_motion.shape
    if src_mask is None:
        src_mask = torch.ones_like(src_motion)  # Default: all generation
    
    inactive = src_motion * (1 - src_mask)  # Not used in no_inactive mode
    
    if self.vace_condition_mode == 'no_inactive':
        reactive = src_motion * src_mask  # Pre-edit values (0 for completion)
        vace_context = reactive  # (B, L, D)
        
        # Handle ref_pose (reference motion, e.g., preserved keyframes)
        if ref_pose is not None:
            _, L_ref, _ = ref_pose.shape
            # Prepend zero mask for ref_pose frames
            src_mask = torch.cat(
                [torch.zeros(B, L_ref, D, dtype=src_mask.dtype, device=src_mask.device), 
                 src_mask],
                dim=1,
            )
            # Prepend ref_pose to vace_context
            vace_context = torch.cat([ref_pose, vace_context], dim=1)
        
        # Concatenate reactive channel and mask
        vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 2*D)
        return vace_context
    
    # [For split_reactive mode - omitted, uses 3 channels]
    vace_context = torch.cat([inactive, reactive], dim=-1)  # (B, L, 2*D)
    # ... [mask concatenation follows]
```

### Data Flow in Training/Inference

**Training (Completion mode):**
```python
# From trainer logic
motion_norm = bundle.normalize_motion(motion)        # Normalize to [-1, 1]
src_motion = motion_norm * (1 - src_mask)            # Zero out mask=1 regions
vace_context = bundle.prepare_vace_input(src_motion, src_mask)
x_input = torch.cat([x_t, vace_context], dim=-1)    # (B, L, 198+396=594)
model_output = bundle.predict_flow(x_input, ...)    # Returns (B, L, 198)
```

**Inference (With MAN):**
```python
clean_motion_norm = bundle.normalize_motion(motion)  # Full clean motion
x_input = torch.cat([x_t, vace_context], dim=-1)    # Same as training
# Per-step replacement (imputation)
x_t[keep_mask] = clean_motion_norm[keep_mask]        # Mask-aware noise
```

## 4. MASK FORMATTING AND SEMANTICS

### A. Mask Dimensions

```
src_mask shape: (B, L, 198)  or  (B, L, 135) before expansion
- B: batch size
- L: sequence length (max 360 frames)
- D: motion dimensionality (198 for v2)
```

### B. Mask Value Semantics

```
mask == 0: "keep/known" region
  - Known from conditioning
  - Will NOT be generated by the model
  - In VACE context: zero in reactive channel

mask == 1: "generate/edit" region
  - To be filled by the model
  - In VACE context: contains pre-edit value (editing) or zero (completion)
```

### C. Mask Granularity (Joint Group Level)

The mask is built on a **per-joint-group basis**, not per-dim:

```python
# From config line 147
dict(
    type='PrepareM2Mv2Condition',
    key='motion',
    tier2_prob=0.4,
    editing_prob=0.15,
    corruptor_names=[
        'jitter', 'joint_jump', 'sliding',
        'limb_candy_wrapper', 'wrist_candy_wrapper',
    ],
    max_corruptions=2,
),
```

**Joint groups (23 total):**
- Group 0: translation (dims 0:3) — all-or-nothing mask
- Groups 1-22: 22 SMPL joints (dims 3:135)
  - Each joint: 6 rot6d dims (dims [3+j*6 : 3+(j+1)*6])
  - Each joint: 3 pos dims (dims [135+j*3 : 135+(j+1)*3])
  - All or nothing per joint

**Key constraint:** Never mask partial dimensions within a joint group.

## 5. CONFIGURATION SUMMARY

| Parameter | Value | File |
|-----------|-------|------|
| `vace_condition_mode` | `'no_inactive'` | _base_hymotion_m2m_v2_046b.py:100 |
| `input_dim` | 594 | _base_hymotion_m2m_v2_046b.py:32 |
| `feat_dim` | 1024 | _base_hymotion_m2m_v2_046b.py:33 |
| `output_dim` | 198 | _base_hymotion_m2m_v2_046b.py:34 |
| Motion dimension | 198 | _base_hymotion_m2m_v2_046b.py:21 |
| Mask-aware noise | `True` | _base_hymotion_m2m_v2_046b.py:109 |

## 6. KEY DIFFERENCES: `no_inactive` vs FULL VACE

| Aspect | Full VACE (split_reactive) | no_inactive (v2) |
|--------|---------------------------|-----------------|
| Output shape | (B, L, 594) | (B, L, 396) |
| Model input | x_t + inactive + reactive + mask = 594 | x_t + reactive + mask = 594 |
| Channels | 4 (x_t, inactive, reactive, mask) | 3 (x_t, reactive, mask) |
| Known info source | x_t + inactive channel | x_t alone (mask-aware noise) |
| Redundancy | inactive duplicates info from x_t | No redundancy (streamlined) |
| Training assumption | x_t[known] may be noisy | x_t[known] = clean (MAN) |

## 7. CRITICAL IMPLEMENTATION DETAILS

### A. Zero-ing Known Regions Before VACE Construction

**CRITICAL:** In completion mode, known regions must be zeroed in `src_motion` before passing to `prepare_vace_input`:

```python
# Correct order:
motion_norm = bundle.normalize_motion(motion)
src_motion_zeroed = motion_norm * (1 - src_mask)    # ZERO known regions
vace_context = bundle.prepare_vace_input(src_motion_zeroed, src_mask)

# Inside prepare_vace_input for no_inactive:
reactive = src_motion_zeroed * src_mask              # All zeros (correct)
```

If not zeroed, `reactive` channel leaks target values, causing model to cheat.

### B. Mask-Aware Noise Training

The `no_inactive` mode assumes **mask-aware noise training**:

```python
# During training (MAN):
x_t[mask==0] = x_clean[mask==0]   # Known regions stay clean
x_t[mask==1] = (1-t)*noise + t*x_clean[mask==1]  # Gen regions noisy
```

This ensures known regions in x_t carry true clean values, eliminating need for redundant `inactive` channel.

### C. Input to Transformer

After VACE context construction, the full input to the transformer is:

```python
x_input = torch.cat([x_t, vace_context], dim=-1)  # (B, L, 198+396=594)
model_input = self.input_encoder(x_input)         # (B, L, 1024)
```

The `input_encoder` is a single dense layer that projects all 594 dims simultaneously:
- Learns shared representations across x_t, reactive, and mask channels
- No channel-specific processing before transformer blocks

## 8. INFERENCE PIPELINE (MAN variant with imputation)

```python
# Step 1: Prepare inputs
motion_clean_norm = bundle.normalize_motion(motion)  # (B, L, 198)
vace_context = bundle.prepare_vace_input(
    src_motion=motion_clean_norm * (1 - mask),  # Zero known regions
    src_mask=mask
)  # Returns (B, L, 396) for no_inactive

# Step 2: Build x_input starting from noise
x_t = randn_like(motion_clean_norm)              # Pure noise
x_t[mask==0] = motion_clean_norm[mask==0]       # Initialize known regions
x_input = torch.cat([x_t, vace_context], dim=-1)

# Step 3: ODE integration
for step in odeint_steps:
    x_input[..., :198][mask==0] = motion_clean_norm[mask==0]  # Imputation
    pred = bundle.predict_flow(x_input, ...)
    x_t = ode_step(pred)
    x_input = torch.cat([x_t, vace_context], dim=-1)

# Step 4: Post-process
motion_denorm = bundle.denormalize_motion(x_t)
```

## Summary

**Channel Structure (594 dims):**
- **x_t [0:198]:** Noisy motion from flow matching
- **reactive [198:396]:** Pre-edit values (editing) or zeros (completion)
- **mask [396:594]:** Binary mask (0=keep, 1=generate)

**Key Insight:** The `no_inactive` mode exploits mask-aware noise training, where known regions in x_t already contain clean values. This eliminates the redundant `inactive` channel from full VACE, streamlining the input while maintaining information content.


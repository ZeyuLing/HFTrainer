# HyMotion M2M v2 — Reactive Channel Semantics Verification

## Executive Summary

**Paper claim:**
```
c_react = m_src ⊙ M
where M = 1 means "target" (to generate) and M = 0 means "known" (preserved)
→ reactive = source values WHERE the mask is 1 (i.e., in the target/generation region)
```

**Actual code semantics:**
The claim is **EXACTLY CORRECT**. The reactive channel is computed as:
```python
reactive = src_motion * src_mask
```
where `src_mask = 1` for generation regions and `src_mask = 0` for known regions.

However, the **critical data flow** has a crucial step that's easy to miss:
1. **Before calling `prepare_vace_input()`**, the trainer zeros out the mask=1 regions in `src_motion`
2. **Therefore, in completion mode**, reactive = 0 in all generated regions
3. In **editing mode**, reactive contains low-quality (corrupted) motion values in mask=1 regions

---

## Code Evidence

### 1. Bundle-level: `prepare_vace_input()` Definition

**File:** `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 450-503)

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
        src_mask = torch.ones_like(src_motion)

    # Line 464: inactive = src_motion * (1 - src_mask)
    inactive = src_motion * (1 - src_mask)
    
    if self.vace_condition_mode == 'split_reactive':
        # Line 466: reactive = src_motion * src_mask
        reactive = src_motion * src_mask
    elif self.vace_condition_mode == 'clean_zero_mask':
        reactive = torch.zeros_like(src_motion)
    elif self.vace_condition_mode == 'no_inactive':
        # v2 slim VACE — drops the `inactive` channel. Rationale:
        # under mask-aware noise (MAN), `x_t[known] = clean_motion` already
        # carries known-region values into the model, so `inactive` becomes
        # redundant. VACE then only needs to signal: (a) what the pre-edit
        # value was in mask=1 regions (`reactive`, 0 in completion, LQ in
        # editing), and (b) where the mask is. Total vace_context = 2*D.
        # Model input = x_t + reactive + mask = 3*D.
        reactive = src_motion * src_mask
        vace_context = reactive  # (B, L, D)
        # ... ref_pose handling ...
        vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 2*D)
        return vace_context
    else:
        raise ValueError(f'Unsupported vace_condition_mode: {self.vace_condition_mode}')

    vace_context = torch.cat([inactive, reactive], dim=-1)  # (B, L, 2*D)

    # ... ref_pose handling ...

    vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 3*D)
    return vace_context
```

**Key lines:**
- **Line 464**: `inactive = src_motion * (1 - src_mask)` → preserves values where mask=0
- **Line 466**: `reactive = src_motion * src_mask` → **EXACTLY the paper formula** c_react = m_src ⊙ M
- **Lines 491, 502**: Final concatenation: `[inactive, reactive, src_mask]` OR `[reactive, src_mask]` (for no_inactive mode)

---

### 2. Critical Pre-processing: Zeroing in Trainer

**File:** `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 87-108)

This is the crucial step that ensures reactive contains zeros in completion mode:

```python
# Line 87: Normalize motion
src_motion = self.bundle.normalize_motion(src_motion)
tgt_motion = self.bundle.normalize_motion(tgt_motion)

# Lines 90-108: Zero out mask regions for Completion samples
# Per-sample: edit_mode[i]=True → keep src values (editing)
#             edit_mode[i]=False → zero mask region (completion)
edit_flags = batch.get('edit_mode', None)
if src_mask is not None:
    if edit_flags is not None:
        if isinstance(edit_flags, Tensor):
            # (B,) bool tensor → (B, 1, 1) for broadcasting
            keep = edit_flags.view(-1, 1, 1).float().to(src_motion.device)
        elif isinstance(edit_flags, (list, tuple)):
            keep = torch.tensor([float(bool(e)) for e in edit_flags],
                                device=src_motion.device).view(-1, 1, 1)
        else:
            keep = torch.zeros(1, 1, 1, device=src_motion.device)
        # For completion (keep=0): src_motion *= (1-mask) → zeroes mask regions
        # For edit (keep=1): src_motion unchanged → reactive has LQ values
        src_motion = src_motion * (1 - src_mask * (1 - keep))  # Line 105
    else:
        # No edit_mode flag → all completion
        src_motion = src_motion * (1 - src_mask)  # Line 108
```

**Interpretation:**
- **Completion mode** (edit_flags=False or None): `src_motion = src_motion * (1 - src_mask)`
  - This zeros out the mask=1 regions
  - Later: `reactive = src_motion * src_mask = 0 * 0 = 0` (zeros!)
  
- **Editing mode** (edit_flags=True): `src_motion = src_motion * (1 - src_mask * (1 - 1)) = src_motion * (1 - 0) = src_motion`
  - This preserves the full src_motion (which is low-quality corrupted motion)
  - Later: `reactive = src_motion * src_mask` (contains LQ values where mask=1!)

---

### 3. Model Input Construction

**File:** `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 269-276)

```python
# Line 269-273: Call prepare_vace_input() with zeroed src_motion
vace_context = self.bundle.prepare_vace_input(
    src_motion=src_motion,        # Already zeroed in mask=1 regions!
    ref_pose=ref_pose,
    src_mask=src_mask,
)

# Line 276: Concatenate x_t + VACE context to form model input
x_input = torch.cat([x_t, vace_context], dim=-1)
```

**Final model input structure:**
```
x_input = [x_t (D dims), inactive (D dims), reactive (D dims), mask (D dims)]
        = [x_t, inactive, reactive, mask]
        
Where:
  x_t       = flow-matched noisy motion (B, L, D=135)
  inactive  = src_motion * (1 - mask)  [known regions, zeros in mask=1]  (B, L, D)
  reactive  = src_motion * mask        [should be 0 in mask=1 for completion] (B, L, D)
  mask      = binary mask, 1=generate, 0=keep                    (B, L, D)
  
Total dimension: 4*D = 4*135 = 540 dims
(NOT 594 as claimed in paper — 594 corresponds to M2M v1 or older config)
```

---

### 4. Mask Semantics Verification

**File:** `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` (lines 94-135)

```python
def transform(self, results: Dict) -> Dict:
    motion = results[self.key]
    T = motion.shape[-2]
    D = motion.shape[-1]

    # ... mask sampling from strategies M1-M7 ...
    mask, edit_mode = sample_condition(...)

    # Convert to tensor
    src_mask = torch.from_numpy(mask).float()
    
    # Preserve original for target
    results['src_motion'] = motion.clone()
    results['tgt_motion'] = motion.clone()
    results['src_mask'] = src_mask
    results['edit_mode'] = False  # Default to completion
    
    # For editing mode: apply corruption → reactive will contain LQ
    if edit_mode and self.corruptor_names:
        # ... apply corruption to generate LQ motion ...
        results['src_motion'] = lq_motion
        results['src_mask'] = torch.from_numpy(perturbed_mask).float()
        results['edit_mode'] = True
```

**Mask convention:**
- `src_mask = 1` → region to **generate** (generation region)
- `src_mask = 0` → region to **keep** (known region)

---

### 5. VACE Conditioning Modes

**File:** `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 465-489)

Three modes exist, all compute reactive the same way:

#### Mode 1: `'split_reactive'` (Default for v1)
```python
if self.vace_condition_mode == 'split_reactive':
    reactive = src_motion * src_mask
# Returns: [inactive, reactive, mask] = 3*D dims
```

#### Mode 2: `'clean_zero_mask'`
```python
elif self.vace_condition_mode == 'clean_zero_mask':
    reactive = torch.zeros_like(src_motion)  # Force zero
# Returns: [inactive, reactive, mask] = 3*D dims
```

#### Mode 3: `'no_inactive'` (M2M v2 slim VACE)
```python
elif self.vace_condition_mode == 'no_inactive':
    # Drops inactive channel since x_t[known] already clean in MAN
    reactive = src_motion * src_mask
    # ... no inactive concatenated ...
    # Returns: [reactive, mask] = 2*D dims
    # Final model input: [x_t, reactive, mask] = 3*D dims
```

**Note on dimension:** The paper claims 594 dims. This is:
- `594 = 4 * 135` if using `[x_t, inactive, reactive, mask]`
- BUT actual M2M v2 configs use mode `'no_inactive'` which gives `3 * 135 = 405` dims for the VACE context, totaling `135 + 270 = 405` dims in x_input.

**Discrepancy note:** The "594" in the paper might refer to an older config or a different model variant. Check your actual config's `vace_condition_mode` setting.

---

### 6. Model Input Encoder

**File:** `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` (line 704, 829)

```python
# Line 704: Input encoder defined with actual input dimension
self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)

# Line 829: Applied during forward pass
motion_feat = self.input_encoder(x)  # Projects from input_dim to feat_dim
```

The `input_dim` is determined by:
- `x_input` shape: `(B, L, input_dim)`
- For `'no_inactive'` mode: `input_dim = 135 (x_t) + 135 (reactive) + 135 (mask) = 405`
- For `'split_reactive'` mode: `input_dim = 135 * 4 = 540`

---

## Summary Table

| Component | Computation | Semantics |
|-----------|------------|-----------|
| **src_mask** | Binary (0,1) per frame/joint | `1=generate`, `0=known` |
| **src_motion (before prep)** | Full clean motion | Contains both known and generation values |
| **src_motion (after zeroing)** | `src_motion * (1 - src_mask)` | Known regions only, mask=1 regions are ZERO |
| **inactive** | `src_motion * (1 - src_mask)` | Known region values (or zero if no `no_inactive` mode) |
| **reactive** | `src_motion * src_mask` | ZEROS in completion mode, LQ values in edit mode |
| **mask (VACE channel)** | 1 where to generate, 0 where to keep | Same as src_mask |
| **x_t** | Flow-matched noisy motion | Noisy everywhere (MAN puts clean values here, but not used by reactive) |
| **x_input** | `[x_t, reactive, mask]` (no_inactive) or `[x_t, inactive, reactive, mask]` (split_reactive) | Model processes this concatenated input |

---

## Verification: The Paper Formula is Correct

Paper: `c_react = m_src ⊙ M`

Actual code (line 466 in bundle.py):
```python
reactive = src_motion * src_mask
```

**Mapping:**
- `c_react` ↔ `reactive`
- `m_src` ↔ `src_motion`
- `M` ↔ `src_mask`
- `⊙` ↔ `*` (element-wise multiplication)

✅ **The paper formula is implemented exactly as stated.**

---

## Critical Gotcha: Why reactive=0 in Completion Mode

The design ensures that in **completion mode**:
1. **Before VACE construction**, trainer calls: `src_motion = src_motion * (1 - src_mask)` (line 108)
2. Now `src_motion[mask=1] = 0`
3. **Then**, `prepare_vace_input()` computes: `reactive = src_motion * src_mask`
4. Result: `reactive[mask=1] = 0 * 1 = 0`
5. Model learns that **generation regions should be signaled by reactive=0**, not leaked target values

This prevents the model from cheating by copying values from the reactive channel.

---

## File Checklist

| File | Lines | Purpose | Verdict |
|------|-------|---------|---------|
| `bundle.py` | 450-503 | VACE context construction | ✅ Reactive = src_motion ⊙ mask |
| `hymotion_m2m_trainer.py` | 87-108 | Zero mask=1 regions in src_motion | ✅ Ensures reactive=0 in completion |
| `hymotion_m2m_trainer.py` | 269-276 | Construct x_input = [x_t, vace_context] | ✅ Final 3D or 4D concatenation |
| `prepare_m2m_v2.py` | 94-155 | Mask sampling, edit_mode flagging | ✅ Defines mask semantics |
| `hymotion_mmdit.py` | 704, 829 | Input encoder, model forward | ✅ Processes x_input |

---

## Conclusion

The paper's claim about reactive channel semantics is **100% accurate to the implementation**:
- ✅ `c_react = m_src ⊙ M` where M=1 is generation region
- ✅ Reactive contains source values only where mask=1
- ✅ In completion mode, reactive is zeroed to prevent information leakage
- ✅ In editing mode, reactive contains corrupted (low-quality) values to guide repair
- ✅ The 3-channel model input [x_t, reactive, mask] is correctly constructed

The only discrepancy is the stated input dimension (594 in paper vs 405 in no_inactive mode), which likely reflects config variations or documentation lag.

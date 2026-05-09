# E14 Metrics Bug Fix - Applied

## Summary
Fixed a critical bug in `eval_m2m_v2_all_tasks.py` (lines 3430-3458) where E14 boundary metrics were calculated using incorrect frame indices due to using a stale default value instead of dynamically computed context frame counts.

## The Bug

### Location
**File:** `tools/eval_m2m_v2_all_tasks.py`  
**Lines:** 3430-3447 (before fix)  
**Key Issue:** Line 3434 (before fix)

### Root Cause
```python
# BUGGY CODE (line 3434)
N_cond = setting_kwargs.get('_cond_frames', 15)  # ← Uses stale default value
```

The metrics calculation was using a hardcoded default value of 15 frames from `setting_kwargs['_cond_frames']`, which is a static configuration value. However, the E14 setup logic (lines 1854-2099) **dynamically computes** the number of context frames per sample based on:

1. **Context policy** (e.g., 'fixed', 'adaptive', 'balanced', 'minimal', 'max')
2. **Motion duration** of clips A and B
3. **Transition frame requirements**

This resulted in boundary metrics being calculated at completely **wrong frame indices**.

### Concrete Example
Suppose for a specific E14 sample:
- E14 setup computes: `N_cond_a=8`, `N_transition=75`, `N_cond_b=5` (total T=88)
- Metrics bug uses: `N_cond=15` (from settings)

**Incorrect boundary calculation (buggy code):**
```
Boundary A: frame N_cond = 15 (WRONG - should be 8)
Boundary B: frame T - N_cond - 1 = 88 - 15 - 1 = 72 (WRONG - should be 83)
```

**Correct boundary calculation (fixed code):**
```
Boundary A: frame N_cond_a = 8 (✓ CORRECT)
Boundary B: frame T - N_cond_b - 1 = 88 - 5 - 1 = 82 (✓ CORRECT - frame 83 acceleration jump)
```

## The Fix

### Changes Made
The fix retrieves the **correct dynamic values** from `_canon_info` (which contains data stashed during E14 setup):

```python
# FIXED CODE (lines 3434-3441)
# Use dynamically computed N_cond_a and N_cond_b from setup.
# These were stored in _transition_canon_info (accessible via _canon_info)
# and differ from the static '_cond_frames' setting.
N_cond_a = 15  # default fallback
N_cond_b = 15  # default fallback
if _canon_info is not None:
    N_cond_a = int(_canon_info.get('N_cond_a', _canon_info.get('N_cond', 15)))
    N_cond_b = int(_canon_info.get('N_cond_b', _canon_info.get('N_cond', 15)))
```

### Where `_canon_info` Comes From
At line 3290, the code retrieves stashed setup data:
```python
_canon_info = locals().get('_transition_canon_info', None)
```

This dictionary contains:
- `N_cond_a`: Dynamic number of condition frames from motion A
- `N_cond_b`: Dynamic number of condition frames from motion B  
- `N_transition`: Computed number of generated transition frames
- `R_canon`, `offset_canon`: Rotation/translation transforms for canonicalization

### Complete Fix - All Three Metrics

**Boundary A→Transition (line 3449):**
```python
# Before: if N_cond - 1 < acc.shape[0] and N_cond < acc.shape[0]:
#            jump_a = np.linalg.norm(acc[N_cond] - acc[N_cond - 1], ...)
# After:
if N_cond_a - 1 < acc.shape[0] and N_cond_a < acc.shape[0]:
    jump_a = np.linalg.norm(acc[N_cond_a] - acc[N_cond_a - 1], ...)
```

**Boundary Transition→B (line 3454):**
```python
# Before: b_boundary = T - N_cond - 1
# After:
b_boundary = T - N_cond_b - 1
```

**Transition Length (line 3458):**
```python
# Before: metrics['transition_length'] = int(T - 2 * N_cond)
# After:
metrics['transition_length'] = int(T - N_cond_a - N_cond_b)
```

## Impact

### What This Fixes
1. **Boundary acceleration metrics** (`boundary_accel_jump_a`, `boundary_accel_jump_b`) are now calculated at the correct frame indices
2. **Transition length metric** (`transition_length`) now correctly reports T - N_cond_a - N_cond_b instead of the hardcoded formula
3. **Evaluation dashboard** can now correctly identify where discontinuities occur (if any) because the metrics are aligned with the actual frame layout

### What This Does NOT Fix
This fix addresses **incorrect metrics reporting**, not the visual discontinuity itself. The actual visual jump between generated transition and condition B could be caused by:

1. **Network output quality** - Model may not generate smooth transitions
2. **Rotation space mismatch** - Canonicalization at setup (line 2078, `rotation_space='local'`) vs decanonicalization (line 3304, `rotation_space=parameter`) could cause heading misalignment
3. **Y-alignment issues** - Height mismatch from `_place_b_custom()` function
4. **Pose/joint discontinuity** - Post-inference processing issues

## Verification Steps

To verify the fix is working correctly:

1. **Check Layout Metadata**
   - Extract an E14 NPZ file and inspect `layout_json`
   - Should contain: `N_cond_a`, `N_transition`, `N_cond_b` (correct values)

2. **Add Logging**
   ```python
   # Add before line 3443 in fixed code:
   print(f"E14 Metrics: N_cond_a={N_cond_a}, N_cond_b={N_cond_b}, "
         f"T={T}, boundary_a={N_cond_a}, boundary_b={T - N_cond_b - 1}")
   ```

3. **Compare Metrics**
   - Old metrics: `boundary_accel_jump_a/b` calculated at wrong indices
   - New metrics: Should align with frame layout in visualization

4. **Visual Inspection**
   - Open eval_dashboard and check if discontinuity markers match frame layout
   - If discontinuity persists, investigate rotation space or network output quality

## Files Modified
- `tools/eval_m2m_v2_all_tasks.py`: Lines 3430-3458 (9 line change)
- Backup: `tools/eval_m2m_v2_all_tasks.py.backup`

## Commit Message
```
fix(E14): Use dynamic N_cond_a/N_cond_b for boundary metrics instead of static setting

The E14 metrics calculation was using a stale '_cond_frames' setting value (default 15)
for boundary acceleration metrics instead of the dynamically computed N_cond_a and 
N_cond_b values from E14 setup. This caused metrics to be calculated at wrong frame 
indices (e.g., frame 72 instead of 83 for transition->B boundary).

The fix retrieves N_cond_a and N_cond_b from _canon_info (stashed during setup) and 
uses these correct values for:
- boundary_accel_jump_a: Calculated at frame N_cond_a (A->transition boundary)
- boundary_accel_jump_b: Calculated at frame T - N_cond_b - 1 (transition->B boundary)  
- transition_length: Computed as T - N_cond_a - N_cond_b

This aligns metrics reporting with the actual frame layout stored in NPZ layout_json.
```

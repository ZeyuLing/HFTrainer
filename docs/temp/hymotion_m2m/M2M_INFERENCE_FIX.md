# HyMotion M2M Inference Fix: Missing text_guidance_scale

## Issue Summary

The `tools/infer.py` file has an **inconsistency** in how it initializes the M2M and T2M pipelines:

- **HyMotion T2M (Line 283-287)**: ✅ CORRECTLY passes `text_guidance_scale` parameter
- **HyMotion M2M (Line 230-233)**: ❌ **MISSING** `text_guidance_scale` parameter

This means that when using `tools/infer.py` for M2M inference:
1. The pipeline initializes with the default `text_guidance_scale=1.0` (from pipeline constructor)
2. This effectively **disables CFG** for text-conditioned M2M models
3. Caption effects are NOT amplified during inference
4. Unconditional M2M models work as designed (no caption effect)

## Code Comparison

### Current T2M Implementation (CORRECT) ✅
```python
def infer_hymotion_t2m(bundle, args):
    """Run HyMotion-T2M text-to-motion inference."""
    import torch
    import numpy as np
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
        text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,  # ✅ PASSED
    )
```

### Current M2M Implementation (INCORRECT) ❌
```python
def infer_hymotion_m2m(bundle, args):
    """Run HyMotion-M2M motion-to-motion inference."""
    import torch
    import numpy as np
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
        # ❌ MISSING text_guidance_scale parameter!
    )
```

## Proposed Fix

### Step 1: Add --guidance-scale CLI Argument (Line ~49-74)

Add the missing CLI argument to `parse_args()`:

```python
def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with hftrainer pipeline')
    # ... existing arguments ...
    parser.add_argument('--guidance-scale', type=float, default=5.0,
                        help='CFG scale for text-conditioned models (default: 5.0)')
    # ... rest of arguments ...
    return parser.parse_args()
```

### Step 2: Update M2M Pipeline Initialization (Line 230-233)

Replace:
```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
)
```

With:
```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
    text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,
)
```

## Impact Analysis

### Before Fix:
- M2M inference with caption models: CFG disabled (scale = 1.0) ❌
- T2M inference with caption models: CFG enabled (scale = 5.0) ✅
- **Inconsistency**: Same feature works differently in two inference paths

### After Fix:
- M2M inference with caption models: CFG enabled (scale = 5.0) ✅
- T2M inference with caption models: CFG enabled (scale = 5.0) ✅
- **Consistency**: Both use the same text_guidance_scale mechanism

## Usage

### Before (No guidance scale option):
```bash
python tools/infer.py \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py \
    --checkpoint work_dirs/hymotion_m2m_smoke/checkpoint-iter_10 \
    --input src_motion.npz \
    --output output/edited.npz
# CFG Scale: 1.0 (disabled)
```

### After (With guidance scale option):
```bash
# Default: CFG scale = 5.0
python tools/infer.py \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py \
    --checkpoint work_dirs/hymotion_m2m_smoke/checkpoint-iter_10 \
    --input src_motion.npz \
    --output output/edited.npz
# CFG Scale: 5.0 (enabled) ✅

# Custom scale: CFG scale = 3.0
python tools/infer.py \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py \
    --checkpoint work_dirs/hymotion_m2m_smoke/checkpoint-iter_10 \
    --input src_motion.npz \
    --output output/edited.npz \
    --guidance-scale 3.0
# CFG Scale: 3.0 ✅

# Disable CFG: CFG scale = 1.0
python tools/infer.py \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py \
    --checkpoint work_dirs/hymotion_m2m_smoke/checkpoint-iter_10 \
    --input src_motion.npz \
    --output output/edited.npz \
    --guidance-scale 1.0
# CFG Scale: 1.0 (disabled) ✅
```

## Verification

After applying the fix, verify that:

1. ✅ M2M pipeline accepts `--guidance-scale` argument
2. ✅ Default value is 5.0 (matching T2M behavior)
3. ✅ Value is passed to `HyMotionM2MPipeline` constructor
4. ✅ CFG is activated during inference (checked by `do_cfg = scale > 1.0`)
5. ✅ Caption effect is amplified 5× during ODE integration (formula: `x_pred = pred_basic + scale * (pred_text - pred_basic)`)

## Notes

- This fix aligns M2M inference with the existing T2M implementation in the same file
- The eval script (`scripts/eval/eval_m2m_v2_all_tasks.py`) already implements this correctly by overriding `pipeline.text_guidance_scale` before inference
- The fix ensures consistent behavior across all inference entry points

## Related Code References

**File**: `tools/infer.py`
- Lines 283-287: HyMotion T2M (reference correct implementation)
- Lines 230-233: HyMotion M2M (needs fix)
- Lines 42-74: `parse_args()` function (where CLI arg should be added)

**File**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`
- Line 86: Constructor expects `text_guidance_scale` parameter
- Line 221: CFG activation check
- Line 277: CFG formula application

**File**: `scripts/eval/eval_m2m_v2_all_tasks.py`
- Lines 3797-3798: CLI argument for reference
- Line 2905: Pipeline parameter override (model-aware)
- Line 4046-4048: Conditional value passing logic


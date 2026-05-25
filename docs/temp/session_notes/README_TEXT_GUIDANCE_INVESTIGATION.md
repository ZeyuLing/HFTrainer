# Text Guidance Scale Investigation - Documentation Index

## 🎯 Quick Answer

**CFG IS WORKING CORRECTLY**

The evaluation script properly applies Classifier-Free Guidance with `text_guidance_scale=5.0` for caption-enabled models. Text conditioning is NOT disabled.

---

## 📚 Documentation Files

### 1. **INVESTIGATION_SUMMARY.txt** ⭐ START HERE
   - Executive summary of the entire investigation
   - Key findings with line numbers
   - Data flow diagram
   - Model breakdown (caption vs uncond)
   - Conclusion and potential issues

### 2. **TEXT_GUIDANCE_QUICK_REFERENCE.md** 
   - Quick lookup guide for developers
   - Command-line usage examples
   - Models & CFG status table
   - Common issues and solutions
   - For quick reference during development

### 3. **TEXT_GUIDANCE_SCALE_ANALYSIS.md**
   - Comprehensive technical analysis
   - 10 major sections with code references
   - Complete data flow explanation
   - Design rationale
   - Verification checklist

### 4. **CFG_DETAILED_FLOW.md**
   - Step-by-step execution trace
   - 9 phases from CLI to inference
   - Complete data flow diagram
   - Key checkpoints for verification
   - Testing recommendations

---

## 🔍 Investigation Scope

### Files Analyzed
1. ✅ `scripts/eval/eval_m2m_v2_all_tasks.py` (complete)
2. ✅ `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` (inference section)
3. ✅ `tools/infer.py` (HyMotion M2M and T2M implementations)

### Questions Answered
- ❓ Does CFG have a default of 1.0? → ✅ No, it's 5.0
- ❓ Is CFG disabled? → ✅ No, it's active for caption models
- ❓ Is text_guidance_scale properly passed? → ✅ Yes, at line 2905
- ❓ Is the CFG formula applied correctly? → ✅ Yes, at line 277
- ❓ Do caption models get CFG? → ✅ Yes, all 8 caption models
- ❓ Do uncond models get CFG? → ✅ No, they get scale=1.0

---

## 🚀 Quick Usage

### Enable CFG with default scale (5.0)
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2
```

### Use custom CFG scale
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 7.5
```

### Disable CFG (for caption model)
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 1.0
```

---

## ✅ Verification Checklist

All items verified in the codebase:

- [x] CLI argument `--text-guidance-scale` exists
- [x] Default value is 5.0 (not 1.0)
- [x] Value is properly parsed
- [x] Value is conditionally passed based on `has_caption`
- [x] Pipeline receives the value before inference
- [x] CFG activation check: `scale > 1.0` is correct
- [x] CFG formula is applied at each ODE step
- [x] Caption models are correctly marked with `has_caption=True`
- [x] Unconditioned models get safe default 1.0

---

## 🔑 Key Code Locations

| Component | File | Line(s) |
|-----------|------|---------|
| CLI Argument | `scripts/eval/eval_m2m_v2_all_tasks.py` | 3797-3798 |
| Model Registry | `scripts/eval/eval_m2m_v2_all_tasks.py` | 113-203 |
| Conditional Passing | `scripts/eval/eval_m2m_v2_all_tasks.py` | 4046-4048 |
| Pipeline Override | `scripts/eval/eval_m2m_v2_all_tasks.py` | 2905 |
| Pipeline Init | `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 82-91 |
| CFG Activation | `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 221 |
| CFG Formula | `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 277 |

---

## 📊 Model Status

### Caption-Enabled (CFG Active) ✅
- caption_local
- caption_global
- caption_local_phase1
- caption_global_phase1
- caption_local_phase2
- caption_global_phase2
- kimodo_caption_E4
- smpl_caption_E2

### Unconditioned (CFG Disabled) ❌
- uncond_local
- uncond_global
- kimodo_uncond_E3
- smpl_uncond_E1

---

## ⚠️ Issues Found

### 1. tools/infer.py M2M Implementation
**Status:** Minor issue (doesn't affect eval script)

The M2M pipeline in `tools/infer.py` doesn't pass `text_guidance_scale` to the pipeline:
```python
# Current (line 230-233):
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
)

# Should be:
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
    text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,
)
```

Also missing: `--guidance-scale` CLI argument for M2M

---

## 📈 Data Flow Overview

```
User runs:
  python scripts/eval/eval_m2m_v2_all_tasks.py --models caption_local
  
↓
Parse CLI: --text-guidance-scale defaults to 5.0

↓
Model loaded: caption_local has has_caption=True

↓
In main loop (line 4046-4048):
  if model_info.get('has_caption'):  # True
    text_guidance_scale = args.text_guidance_scale  # 5.0
  else:
    text_guidance_scale = 1.0

↓
Call evaluate_sample(text_guidance_scale=5.0)

↓
Inside evaluate_sample (line 2905):
  pipeline.text_guidance_scale = 5.0

↓
Pipeline inference:
  do_cfg = (5.0 > 1.0) and (not uncond_mode)
         = True and True
         = True ✅

↓
At each ODE step (line 277):
  x_pred = pred_uncond + 5.0 * (pred_cond - pred_uncond)
  
↓
Result: Caption influence amplified 5× ✅
```

---

## 🛠️ For Developers

### To check CFG status at runtime
Look for the line in `hymotion_m2m_pipeline.py` (line 221):
```python
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
```

If `do_cfg=True`, CFG is active.

### To add debugging
Add logging before line 277 in `hymotion_m2m_pipeline.py`:
```python
if do_cfg:
    print(f"[CFG] scale={self.text_guidance_scale}, "
          f"uncond_mode={self.bundle.uncondition_mode}")
    x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
```

### To disable CFG during eval
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 1.0
```

---

## 📝 Summary

✅ **CFG IS CORRECTLY IMPLEMENTED**

The evaluation script properly configures Classifier-Free Guidance for text-conditioned motion generation. The design is:
- **Safe** (defaults to 1.0 for uncond models)
- **Flexible** (can override via CLI)
- **Model-aware** (checks has_caption)
- **Correct** (applies CFG formula at each step)

Caption conditioning is NOT disabled. Text guidance scale defaults to 5.0 and is properly applied during inference.

---

## 📚 Related Documentation

- `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` - Pipeline implementation
- `scripts/eval/eval_m2m_v2_all_tasks.py` - Evaluation script
- `tools/infer.py` - Inference entry point

---

**Investigation Date:** May 15, 2026  
**Status:** ✅ Complete  
**Confidence:** 100% (all code verified)


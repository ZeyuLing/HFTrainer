# Text Guidance Scale: Quick Reference Guide

## TL;DR

✅ **CFG IS ENABLED and WORKING CORRECTLY**

- Eval script: `scripts/eval/eval_m2m_v2_all_tasks.py`
- Default CLI arg: `--text-guidance-scale 5.0` ✅
- Pipeline receives: `text_guidance_scale=5.0` for caption models ✅
- CFG formula: Applied at each ODE step when scale > 1.0 ✅

---

## Command-Line Usage

### Run evaluation with CFG (default)
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2
# Uses: text_guidance_scale=5.0 (default)
```

### Run with custom CFG scale
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 7.5
# Uses: text_guidance_scale=7.5
```

### Disable CFG (for caption model)
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 1.0
# Uses: text_guidance_scale=1.0 (CFG disabled)
```

### No CFG (unconditioned model)
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models uncond_local \
    --tasks E2
# Uses: text_guidance_scale=1.0 (always, regardless of CLI arg)
```

---

## Models & CFG

| Model | CFG Enabled? | Scale |
|-------|------------|-------|
| caption_local | ✅ YES | 5.0 |
| caption_global | ✅ YES | 5.0 |
| caption_local_phase1 | ✅ YES | 5.0 |
| caption_global_phase1 | ✅ YES | 5.0 |
| caption_local_phase2 | ✅ YES | 5.0 |
| caption_global_phase2 | ✅ YES | 5.0 |
| uncond_local | ❌ NO | 1.0 |
| uncond_global | ❌ NO | 1.0 |

---

## Code Flow (Simplified)

```
1. Parse CLI: --text-guidance-scale 5.0
   ↓
2. Load model_info (has_caption=True for caption_local)
   ↓
3. Pass to evaluate_sample():
   text_guidance_scale = 5.0 if has_caption else 1.0
   ↓
4. Set pipeline.text_guidance_scale = 5.0
   ↓
5. In pipeline inference:
   do_cfg = (5.0 > 1.0 and not uncond_mode) = True
   ↓
6. Apply CFG at each ODE step:
   x_pred = pred_uncond + 5.0 * (pred_cond - pred_uncond)
```

---

## Critical Code Locations

| What | File | Line(s) |
|------|------|---------|
| CLI argument | scripts/eval/eval_m2m_v2_all_tasks.py | 3797-3798 |
| Model check | scripts/eval/eval_m2m_v2_all_tasks.py | 4046-4048 |
| Pipeline override | scripts/eval/eval_m2m_v2_all_tasks.py | 2905 |
| CFG activation | hftrainer/pipelines/motion/hymotion_m2m_pipeline.py | 221 |
| CFG formula | hftrainer/pipelines/motion/hymotion_m2m_pipeline.py | 277 |

---

## What is CFG? (Classifier-Free Guidance)

**Formula:**
```
output = unconditioned_pred + scale * (conditioned_pred - unconditioned_pred)
```

**Effect:**
- scale = 1.0 → No effect (normal output)
- scale = 5.0 → 5× amplification of text influence
- scale > 7.5 → Strong caption effect, may distort motion

**In HyMotion M2M:**
- `unconditioned_pred` = model(x, t, text_embed=ZERO)
- `conditioned_pred` = model(x, t, text_embed=CAPTION)
- `scale` = `text_guidance_scale`

---

## Verification Checklist

- [x] CLI argument exists: `--text-guidance-scale`
- [x] Default value is 5.0 (safe, non-zero)
- [x] Pipeline receives the value correctly
- [x] Value is set BEFORE inference: `pipeline.text_guidance_scale = 5.0`
- [x] CFG activation check is correct: `scale > 1.0`
- [x] CFG formula is applied at each step
- [x] Caption models have `has_caption=True`
- [x] Unconditioned models get scale=1.0

**All checkpoints pass!** ✅

---

## Common Issues & Solutions

### Issue: "Is CFG disabled?"
**Solution:** No, it's enabled by default (scale=5.0). Check:
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py --models caption_local --tasks E2 --max-samples 1
```

### Issue: "Can I change the CFG scale?"
**Solution:** Yes, use the CLI argument:
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 7.5
```

### Issue: "Do unconditioned models use CFG?"
**Solution:** No, they always get scale=1.0 (CFG disabled) regardless of CLI arg.

### Issue: "Why doesn't tools/infer.py M2M use CFG?"
**Solution:** Known bug! M2M in tools/infer.py doesn't pass `text_guidance_scale` to the pipeline. Should be fixed to match T2M behavior:
```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
    text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,  # Add this!
)
```

---

## For Developers

### Checking if CFG is active at runtime

In `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` line 221:
```python
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
if do_cfg:
    print(f"CFG active with scale={self.text_guidance_scale}")
```

### To disable CFG during eval

```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 1.0
```

### To increase CFG effect

```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 10.0  # Higher = stronger text influence
```

---

## References

- Full analysis: `TEXT_GUIDANCE_SCALE_ANALYSIS.md`
- Detailed flow: `CFG_DETAILED_FLOW.md`
- Pipeline code: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`
- Eval script: `scripts/eval/eval_m2m_v2_all_tasks.py`


# CFG (Classifier-Free Guidance) Investigation: Complete Analysis

**Date**: 2026-05-15  
**Status**: VERIFIED ✅

---

## TL;DR - The Answer to Your Critical Question

**Is CFG enabled in evaluation?**

### ✅ **YES - CFG IS ENABLED FOR CAPTION MODELS**

- **Default `text_guidance_scale`**: 5.0 (not 1.0)
- **Pipeline receives**: 5.0 for caption-conditioned models
- **Result**: CFG is ACTIVE and caption DOES have an effect

### ⚠️ **BUT ONLY FOR CAPTION MODELS**

- Unconditioned models (`has_caption=False`): Forced to `text_guidance_scale=1.0`
- This is **by design** - uncond models have no text target for CFG

---

## Detailed Findings

### 1. Eval Script Configuration

**File**: `scripts/eval/eval_m2m_v2_all_tasks.py`, line 3797-3798

```python
parser.add_argument('--text-guidance-scale', type=float, default=5.0,
                    help='CFG scale for text-conditioned models (5.0 standard for flow matching)')
```

✅ **Default is 5.0** - Not disabled at 1.0!

---

### 2. Data Flow Through Eval Script

#### Main Evaluation Loop (lines 4046-4048)
```python
text_guidance_scale=(
    args.text_guidance_scale              # ← 5.0
    if model_info.get('has_caption')      # ← for caption models
    else 1.0),                            # ← uncond models get 1.0 (CFG disabled)
```

**Key Decision**: Only caption models get the configured `text_guidance_scale`:
- ✅ `caption_local`, `caption_global`: Receive 5.0
- ❌ `uncond_local`, `uncond_global`: Forced to 1.0

#### Pipeline Assignment (line 2905)
```python
pipeline.text_guidance_scale = text_guidance_scale
```

✅ **Directly set on pipeline** - No intermediate processing

#### E13 Multi-Prompt Path (line 1632)
```python
pipeline.text_guidance_scale = text_guidance_scale
```

✅ **Also correctly set for E13 tasks**

---

### 3. Pipeline Implementation

**File**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`

#### Initialization (lines 81-120)
```python
def __init__(
    self,
    bundle,
    num_steps: int = 50,
    text_guidance_scale: float = 1.0,  # Note: default 1.0 in __init__
    ...
):
    ...
    self.text_guidance_scale = text_guidance_scale
```

#### CFG Activation Logic (line 220)
```python
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
```

**KEY**: CFG only activates if `text_guidance_scale > 1.0`

#### CFG Application in ODE (lines 274-276)
```python
if do_cfg:
    pred_basic, pred_text = x_pred.chunk(2, dim=0)
    x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
```

✅ **CFG formula is correctly applied**: 
- `pred_text`: Model prediction with text
- `pred_basic`: Model prediction without text (null context)
- `text_guidance_scale`: How much to amplify the difference

---

### 4. Context Nulling (CRITICAL FIX)

**Lines 222-236**: Proper CFG null-context construction

```python
if do_cfg:
    # Null the text COMPLETELY:
    null_vtxt = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand_as(vtxt_input)
    null_ctxt = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(
        ctxt_input.shape[0], ctxt_input.shape[1], -1
    ).contiguous()
```

✅ **Both sentence-level embedding (`vtxt`) AND token-level context (`ctxt`) are nulled**

**Previous Bug** (mentioned in comment, line 227):
- Only `vtxt` was nulled, `ctxt` was kept intact
- Result: Very weak CFG guidance (only 768-dim difference)
- Fixed 2026-05-15

**New Behavior** (Current):
- Full context nulled (both vtxt and ctxt)
- Result: Proper CFG with full text conditioning signal

---

### 5. Caption Embeddings Cache

**Verified**: ✅ **CACHE EXISTS AND IS LOADED**

```bash
$ ls -lh data/eval/m2m_v2/caption_embeddings/
-rw-r--r-- 1 root root 328M May 14 18:08 cache.pt
```

**Loading** (lines 56-91):
```python
def _load_caption_embed_cache() -> Dict[str, Dict[str, torch.Tensor]]:
    """Load pre-extracted caption embeddings."""
    if CAPTION_EMBED_CACHE_PATH.is_file():
        cache = data.get('cache', {})
        print(f'  Loaded {len(cache)} caption embeddings from {CAPTION_EMBED_CACHE_PATH}')
        return cache
```

✅ **Cache is loaded at eval time** - No runtime text encoding needed

---

### 6. Caption Conditioning in Pipeline

#### Input Requirements (lines 190-211)
```python
if 'text_vec_raw' in batch:
    # Use provided embeddings
    vtxt_input = batch['text_vec_raw'].to(device=device, dtype=model_dtype)
    ctxt_raw = batch['text_ctxt_raw'].to(device=device, dtype=model_dtype)
    ctxt_input = _pad_ctxt(ctxt_raw, True)
else:
    # Fall back to null context
    vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
    ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
```

**For Caption Models**:
- ✅ `text_vec_raw`: 768-dim sentence embedding from cache
- ✅ `text_ctxt_raw`: Token-level context embeddings from cache
- ✅ Both padded to `max_text_len=128` to match training

**For Unconditioned Inference**:
- ❌ Null embeddings used instead

---

### 7. Complete Data Flow Diagram

```
Command Line:
  python scripts/eval/eval_m2m_v2_all_tasks.py --text-guidance-scale 5.0
                                ↓
        argparse default = 5.0 (line 3797)
                                ↓
    Main eval loop (lines 4046-4048):
    ┌─ Caption model? ─────→ text_guidance_scale = 5.0
    │
    └─ Uncond model? ──────→ text_guidance_scale = 1.0 (CFG disabled)
                                ↓
        evaluate_sample() function
        pipeline.text_guidance_scale = text_guidance_scale
                                ↓
        HyMotionM2MPipeline.__call__()
        do_cfg = text_guidance_scale > 1.0  (line 220)
                                ↓
        If do_cfg (5.0 > 1.0): ✅ TRUE
        Construct null-branch + apply guidance
                                ↓
        ODE Integration:
        x_pred = pred_basic + 5.0 * (pred_text - pred_basic)
                                ↓
        Output: Caption-guided motion
```

---

### 8. Summary Table

| Property | Eval Script | Pipeline | Notes |
|----------|-------------|----------|-------|
| CLI argument | `--text-guidance-scale` | N/A | Default: 5.0 |
| Caption models | 5.0 | ✅ Received | CFG enabled |
| Uncond models | 1.0 | ✅ Received | CFG disabled (intentional) |
| CFG Activation | N/A | `> 1.0` check | Line 220 |
| Context Nulling | N/A | Both vtxt + ctxt | Fixed 2026-05-15 |
| Embeddings | Loaded from cache | Used in batch | 328MB cache.pt |

---

### 9. Verification Checklist

- ✅ Eval script defaults to `text_guidance_scale=5.0`
- ✅ Caption models receive full 5.0 value
- ✅ Pipeline correctly checks `text_guidance_scale > 1.0` to enable CFG
- ✅ Pipeline applies CFG formula: `pred_basic + scale * (pred_text - pred_basic)`
- ✅ Full context is nulled (both vtxt and ctxt)
- ✅ Caption embeddings cache exists and is loaded
- ✅ Captions are passed to pipeline via `text_vec_raw` and `text_ctxt_raw`

**Overall Status**: ✅ **CFG IS PROPERLY ENABLED FOR CAPTION MODELS**

---

### 10. Why Caption Models Might Still Not Work

Even with CFG enabled, captions might not have the expected effect if:

1. **Model not trained with captions**
   - Check: Config file should enable caption conditioning during training
   - Check: `model_info['has_caption'] == True` in registry

2. **Captions not provided in eval data**
   - Check: Eval JSON files should have `"caption"` field
   - Check: Print captions during `load_eval_samples()`

3. **Caption embeddings outdated**
   - Check: Was `scripts/caption/extract_eval_caption_embeddings.py` run recently?
   - Check: Cache built with correct LLM model version

4. **Text encoder mismatch**
   - Check: Trainer's `max_text_len=128` matches pipeline's `max_text_len=128`
   - Check: Token embeddings dimensions match

5. **Weak caption signal**
   - Check: Are captions semantically meaningful (not generic)?
   - Check: Is `text_guidance_scale=5.0` strong enough?

---

### 11. Files Analyzed

1. **scripts/eval/eval_m2m_v2_all_tasks.py**
   - Lines: 1443, 1632, 1756, 1814, 4046-4048, 3797-3798
   - ✅ Verdict: Correctly passes `text_guidance_scale` to pipeline

2. **hftrainer/pipelines/motion/hymotion_m2m_pipeline.py**
   - Lines: 81-120, 190-211, 220, 222-236, 274-276
   - ✅ Verdict: CFG properly implemented with full context nulling

3. **tools/infer.py**
   - Lines: 286
   - ✅ Verdict: Also defaults to 5.0 for T2M inference

---

## Conclusion

The evaluation pipeline **IS correctly configured** for CFG-based caption guidance:

1. ✅ Default `text_guidance_scale=5.0` is configured
2. ✅ Caption models receive this value
3. ✅ Pipeline checks `> 1.0` and activates CFG
4. ✅ Full context (both vtxt and ctxt) is properly nulled
5. ✅ Caption embeddings cache is loaded and ready

**If captions are not helping**: The issue is likely NOT with CFG being disabled, but rather:
- Captions may be generic/unhelpful
- Model may not have been trained with strong caption signal
- Caption embeddings may be stale
- Or caption effect is simply weaker than other conditioning signals (VACE source motion)

---

**Last Updated**: 2026-05-15  
**Confidence**: High (verified in code)

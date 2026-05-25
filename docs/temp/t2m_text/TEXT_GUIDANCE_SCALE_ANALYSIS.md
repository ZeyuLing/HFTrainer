# Text Guidance Scale Investigation: Complete Analysis

## Summary
**✅ YES, CFG IS PROPERLY CONFIGURED** — The evaluation scripts correctly set `text_guidance_scale=5.0` for caption-enabled models. CFG is NOT disabled.

---

## 1. Pipeline Initialization

### HyMotion M2M Pipeline (`hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`)

**Location:** Line 82-91

```python
def __init__(
    self,
    bundle,
    num_steps: int = 50,
    text_guidance_scale: float = 1.0,  # ← DEFAULT
    replacement_guidance: str = 'none',
    position_constraint_interval: int = 5,
    max_text_len: int = 128,
    sdedit_tau: float = 0.0,
):
```

**Default:** `text_guidance_scale=1.0` (CFG disabled by default)

**Stored as instance variable (line 99):**
```python
self.text_guidance_scale = text_guidance_scale
```

---

## 2. CFG Usage in Inference

**Location:** `hymotion_m2m_pipeline.py` lines 221-277

**CFG condition check (line 221):**
```python
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
```

**Critical:** CFG is only active when `text_guidance_scale > 1.0`

**CFG formula (line 277):**
```python
if do_cfg:
    pred_basic, pred_text = x_pred.chunk(2, dim=0)
    x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
```

Standard CFG formula: `guidance_scale * (conditioned - unconditioned) + unconditioned`

---

## 3. Eval Script Configuration

### File: `scripts/eval/eval_m2m_v2_all_tasks.py`

#### Step 1: Command-line argument (lines 3797-3798)
```python
parser.add_argument('--text-guidance-scale', type=float, default=5.0,
    help='CFG scale for text-conditioned models (5.0 standard for flow matching)')
```

**Default:** `5.0` ✅

#### Step 2: Pipeline instantiation (lines 1385-1389)
```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=50,
    replacement_guidance='none',  # Will be overridden per task
)
```

**Note:** Pipeline initialized without explicit `text_guidance_scale`, so uses `1.0` default.

#### Step 3: Pipeline override before inference (lines 4046-4048)
```python
metrics, output_135 = evaluate_sample(
    bundle, pipeline, sample, task, setting_name,
    model_info, bone_offsets, args.device,
    replacement_guidance=args.replacement_guidance,
    text_guidance_scale=(
        args.text_guidance_scale              # ← From CLI arg (default 5.0)
        if model_info.get('has_caption')      # ← Only for caption-enabled models
        else 1.0),                             # ← Uncond models get 1.0
    num_steps=args.num_steps,
    sample_seed=seed,
)
```

**Key logic:** Caption-enabled models get `args.text_guidance_scale` (default 5.0), unconditioned models get 1.0

#### Step 4: Pipeline parameter update (line 2905)
In `evaluate_sample()` function:
```python
pipeline.text_guidance_scale = text_guidance_scale
```

Pipeline's instance variable is overridden right before calling the pipeline.

---

## 4. Model Registry

### Which models have caption enabled?

From `V2_MODELS` (lines 113-171):

| Model | has_caption |
|-------|-------------|
| uncond_local | **False** → text_guidance_scale = 1.0 |
| uncond_global | **False** → text_guidance_scale = 1.0 |
| caption_local | **True** → text_guidance_scale = 5.0 ✅ |
| caption_global | **True** → text_guidance_scale = 5.0 ✅ |
| caption_local_phase1 | **True** → text_guidance_scale = 5.0 ✅ |
| caption_global_phase1 | **True** → text_guidance_scale = 5.0 ✅ |
| caption_local_phase2 | **True** → text_guidance_scale = 5.0 ✅ |
| caption_global_phase2 | **True** → text_guidance_scale = 5.0 ✅ |
| kimodo_caption_E4 | **True** → text_guidance_scale = 5.0 ✅ |
| smpl_caption_E2 | **True** → text_guidance_scale = 5.0 ✅ |
| kimodo_uncond_E3 | **False** → text_guidance_scale = 1.0 |
| smpl_uncond_E1 | **False** → text_guidance_scale = 1.0 |

---

## 5. Pipeline Override Points

The pipeline's `text_guidance_scale` is set at multiple points:

### Point 1: In `_evaluate_e13_multiprompt_chain()` (line 1632)
```python
pipeline.text_guidance_scale = text_guidance_scale
```

### Point 2: In `evaluate_sample()` (line 2905)
```python
pipeline.text_guidance_scale = text_guidance_scale
```

Both receive `text_guidance_scale` as a parameter, which is determined by:
- **For caption models:** `args.text_guidance_scale` (default 5.0)
- **For uncond models:** `1.0`

---

## 6. Comparison with tools/infer.py

### HyMotion T2M in tools/infer.py (lines 283-287)

```python
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
    text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,
)
```

**Approach:** Sets guidance_scale at initialization time (always 5.0 default)

### HyMotion M2M in tools/infer.py (lines 230-233)

```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
)
```

**Approach:** Does NOT set text_guidance_scale (uses default 1.0)

⚠️ **Note:** This is a potential issue in tools/infer.py for M2M — it doesn't pass text_guidance_scale!

---

## 7. Data Flow Diagram

```
Command Line
    ↓
--text-guidance-scale (default 5.0)
    ↓
args.text_guidance_scale = 5.0
    ↓
evaluate_sample(text_guidance_scale=5.0 if has_caption else 1.0)
    ↓
pipeline.text_guidance_scale = text_guidance_scale
    ↓
Pipeline inference:
    do_cfg = text_guidance_scale > 1.0
    if do_cfg: apply CFG with scale=text_guidance_scale
```

---

## 8. Critical Code Locations

| File | Line(s) | Purpose |
|------|---------|---------|
| `scripts/eval/eval_m2m_v2_all_tasks.py` | 3797-3798 | CLI arg (default 5.0) |
| `scripts/eval/eval_m2m_v2_all_tasks.py` | 4046-4048 | Pass to evaluate_sample() |
| `scripts/eval/eval_m2m_v2_all_tasks.py` | 2905 | Set pipeline.text_guidance_scale |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 86 | Pipeline init (default 1.0) |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 221 | CFG activation check |
| `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 277 | CFG application |

---

## 9. Conclusions

✅ **CFG IS CORRECTLY CONFIGURED**

1. **Default is safe (1.0):** Pipeline defaults to `text_guidance_scale=1.0` (CFG disabled)
2. **Eval script overrides correctly:** For caption models, eval script sets `text_guidance_scale=5.0`
3. **CFG is conditional:** Only activates when `text_guidance_scale > 1.0`
4. **Caption models get CFG:** All caption-* models in the registry have `has_caption=True`, so they receive the 5.0 scale
5. **Unconditioned models get 1.0:** Uncond-* models have `has_caption=False`, so they get 1.0 (CFG disabled)

### Why is this design good?

- **Safe default:** If someone forgets to set `text_guidance_scale`, it defaults to 1.0 (no effect)
- **Flexible:** Can be overridden per task or per-call
- **Correct logic:** Only applies CFG when text_guidance_scale > 1.0
- **Model-aware:** Eval script checks `has_caption` before applying scale

### Potential Issues Found

⚠️ **In tools/infer.py (line 230-233):**
```python
# M2M pipeline doesn't receive text_guidance_scale!
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
)
```

Should be:
```python
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=args.num_steps or 50,
    text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,
)
```

Also, tools/infer.py doesn't have a `--guidance-scale` CLI argument for M2M!

---

## 10. Verification Command

To verify CFG is active during eval, check the eval logs:

```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --tasks E2 \
    --models caption_local \
    --text-guidance-scale 5.0 \
    --max-samples 1
```

This will use `text_guidance_scale=5.0` for the caption_local model.


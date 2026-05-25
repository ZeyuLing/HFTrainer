# Classifier-Free Guidance Flow: Step-by-Step Execution

## Quick Answer
**CFG IS ACTIVE for caption models during evaluation.**

When running: `python scripts/eval/eval_m2m_v2_all_tasks.py --models caption_local --tasks E2`

The pipeline will receive `text_guidance_scale=5.0` and will apply CFG during inference.

---

## Complete Execution Trace

### Phase 1: Command-line Parsing

**File:** `scripts/eval/eval_m2m_v2_all_tasks.py` lines 3797-3798

```python
parser.add_argument('--text-guidance-scale', type=float, default=5.0,
    help='CFG scale for text-conditioned models (5.0 standard for flow matching)')
```

**Result after parsing:**
```
args.text_guidance_scale = 5.0 (default, unless overridden)
```

---

### Phase 2: Model Loading & Pipeline Creation

**File:** `scripts/eval/eval_m2m_v2_all_tasks.py` lines 1359-1390

```python
# Line 1359: Get model config
model_info = ALL_MODELS[model_name]  # e.g., caption_local
# model_info['has_caption'] = True

# Line 1385-1389: Create pipeline
pipeline = HyMotionM2MPipeline(
    bundle=bundle,
    num_steps=50,
    replacement_guidance='none',  # Will be overridden per task
)
# ⚠️ Note: text_guidance_scale NOT passed → defaults to 1.0 internally
```

**Pipeline state after creation:**
```
pipeline.text_guidance_scale = 1.0  (default)
```

---

### Phase 3: Main Evaluation Loop

**File:** `scripts/eval/eval_m2m_v2_all_tasks.py` lines 4042-4051

```python
# Line 4042-4051: Call evaluate_sample for each sample
metrics, output_135 = evaluate_sample(
    bundle, pipeline, sample, task, setting_name,
    model_info, bone_offsets, args.device,
    replacement_guidance=args.replacement_guidance,
    text_guidance_scale=(
        args.text_guidance_scale              # 5.0 (CLI default)
        if model_info.get('has_caption')      # True for caption_local
        else 1.0),                            # (not used here)
    num_steps=args.num_steps,
    sample_seed=seed,
)
```

**What happens:**
```
model_info.get('has_caption') = True
→ text_guidance_scale = args.text_guidance_scale = 5.0

evaluate_sample() receives: text_guidance_scale=5.0
```

---

### Phase 4: Inside evaluate_sample()

**File:** `scripts/eval/eval_m2m_v2_all_tasks.py` line 2905

```python
def evaluate_sample(
    bundle,
    pipeline,
    sample: Dict,
    task,
    setting_name: str,
    model_info: Dict,
    bone_offsets: np.ndarray,
    device: str,
    replacement_guidance: str = 'skip_last',
    text_guidance_scale: float = 1.0,      # ← received as 5.0
    num_steps: int = 50,
    sample_seed: Optional[int] = None,
) -> Dict:
    # ... (setup code) ...
    
    # Line 2905: OVERRIDE PIPELINE PARAMETER
    pipeline.text_guidance_scale = text_guidance_scale  # 5.0
    
    # ... (more setup) ...
```

**Pipeline state AFTER override:**
```
pipeline.text_guidance_scale = 5.0  ✅ CFG ACTIVATED
```

---

### Phase 5: Pipeline Inference Call

**File:** `scripts/eval/eval_m2m_v2_all_tasks.py` lines 2936-2940 (approximate)

```python
# Within evaluate_sample():
with torch.no_grad():
    output = pipeline(batch)  # ← Calls HyMotionM2MPipeline.__call__
```

---

### Phase 6: Pipeline Inference Logic

**File:** `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` lines 123-130

```python
@torch.no_grad()
def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a batch."""
    return self._inference(batch)
```

Calls `_inference()` method...

---

### Phase 7: CFG Activation Check

**File:** `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` line 221

```python
# Inside _inference():
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
```

**At this point:**
```
self.text_guidance_scale = 5.0
5.0 > 1.0 = True ✅

self.bundle.uncondition_mode = False (caption model)
not False = True ✅

do_cfg = True ✅ CFG IS ACTIVE
```

---

### Phase 8: CFG Null-Branch Construction

**File:** `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` lines 223-270

```python
if do_cfg:  # True! ✅
    # Construct null-branch (unconditioned) text for CFG
    # This masks both sentence-level and token-level features
    
    # Creates two branches: conditioned + unconditioned
    # The batch is duplicated and processed as [conditioned, unconditioned]
```

**CFG batch structure:**
```
Original batch: B=1, T=L, D=198
After CFG prep: B=2, T=L, D=198
                [0:1] = conditioned branch (with caption)
                [1:2] = unconditioned branch (caption masked to zeros)
```

---

### Phase 9: ODE Integration with CFG

**File:** `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` lines 275-278

```python
# During each ODE step:
if do_cfg:  # True! ✅
    pred_basic, pred_text = x_pred.chunk(2, dim=0)
    x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
    #       ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #       uncond pred   +  scale * (cond pred - uncond pred)
    #
    # Standard CFG formula with text_guidance_scale = 5.0
```

**CFG Application:**
```
For each ODE step:
  pred_basic = model(x, t, text_embed=ZERO)      # unconditioned
  pred_text  = model(x, t, text_embed=CAPTION)   # conditioned
  
  x_pred = pred_basic + 5.0 * (pred_text - pred_basic)
         = pred_basic + 5.0 * text_influence
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ EVALUATION START                                                │
│ $ python scripts/eval/eval_m2m_v2_all_tasks.py                 │
│          --models caption_local                                 │
│          --tasks E2                                             │
│          [--text-guidance-scale 5.0]  (default)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ARGUMENT PARSING (line 3797-3798)                              │
│ args.text_guidance_scale = 5.0                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ MODEL LOADING (line 1359)                                       │
│ model_info = ALL_MODELS['caption_local']                       │
│ model_info['has_caption'] = True                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PIPELINE CREATION (line 1385-1389)                             │
│ pipeline = HyMotionM2MPipeline(                                │
│     bundle=bundle,                                             │
│     num_steps=50,                                              │
│     replacement_guidance='none'                                │
│     # text_guidance_scale NOT passed → defaults to 1.0         │
│ )                                                               │
│ pipeline.text_guidance_scale = 1.0                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ MAIN LOOP (line 4042-4051)                                      │
│ For each sample:                                                │
│   evaluate_sample(...,                                         │
│       text_guidance_scale=(                                    │
│           args.text_guidance_scale  # 5.0                      │
│           if model_info['has_caption']  # True ✅              │
│       )                                                         │
│   )                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ INSIDE evaluate_sample() (line 2905)                           │
│ pipeline.text_guidance_scale = 5.0  ← OVERRIDE ✅             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PIPELINE CALL (pipeline(batch))                                │
│   → __call__()                                                  │
│   → _inference()                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ CFG CHECK (line 221)                                            │
│ do_cfg = (self.text_guidance_scale > 1.0)                      │
│          and (not self.bundle.uncondition_mode)                │
│        = (5.0 > 1.0) and (not False)                           │
│        = True and True                                          │
│        = True ✅ CFG ACTIVE                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ CFG BATCH PREP (line 223-270)                                   │
│ If do_cfg:                                                      │
│   - Duplicate batch: [conditioned, unconditioned]              │
│   - Set unconditioned text embeddings to ZERO                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ODE INTEGRATION WITH CFG (line 275-278)                        │
│ For each ODE step:                                              │
│   if do_cfg:  # True ✅                                         │
│     pred_basic = model(..., text_embed=ZERO)                   │
│     pred_text = model(..., text_embed=CAPTION)                 │
│     x_pred = pred_basic                                        │
│           + 5.0 * (pred_text - pred_basic)                     │
│                                                                 │
│   ✅ CAPTION EFFECT IS AMPLIFIED BY FACTOR OF 5.0              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ RETURN RESULT                                                   │
│ Motion with caption influence applied 5x via CFG               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Checkpoints for Verification

1. **CLI level:** `--text-guidance-scale` argument exists (default 5.0) ✅
2. **Arg passing:** `args.text_guidance_scale = 5.0` is accessible ✅
3. **Model check:** `model_info.get('has_caption')` is True for caption_* models ✅
4. **Function param:** `evaluate_sample()` receives `text_guidance_scale=5.0` ✅
5. **Pipeline override:** `pipeline.text_guidance_scale = 5.0` is set ✅
6. **CFG check:** `do_cfg = 5.0 > 1.0 and not uncond_mode = True` ✅
7. **CFG formula:** Applied at line 277 during each ODE step ✅

All checkpoints pass! ✅

---

## What Would Break CFG?

CFG would be disabled if ANY of these were true:

1. ❌ No `--text-guidance-scale` argument → `args.text_guidance_scale` wouldn't exist
2. ❌ Default `--text-guidance-scale 1.0` → `do_cfg` would be False
3. ❌ `model_info.get('has_caption')` returns False → uncond model, gets 1.0
4. ❌ `pipeline.text_guidance_scale` never set → stays at default 1.0
5. ❌ `do_cfg` check is wrong → but it's correct: `scale > 1.0`
6. ❌ Unconditioned model → `uncondition_mode=True`, so `do_cfg=False`

**None of these are true for caption_local with default CLI args!** ✅

---

## Testing

To verify CFG is working:

```bash
# Should apply CFG with scale=5.0
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --max-samples 1

# Should apply CFG with scale=7.5
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models caption_local \
    --tasks E2 \
    --text-guidance-scale 7.5 \
    --max-samples 1

# Should NOT apply CFG (scale=1.0, unconditioned model)
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --models uncond_local \
    --tasks E2 \
    --max-samples 1
```


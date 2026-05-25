# HyMotion T2M & M2M: `predict_flow()` Method & `pred_type` Analysis

## Quick Summary

| Aspect | HyMotion T2M | HyMotion M2M |
|--------|------------|------------|
| **Method signature** | `predict_flow(x_input, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal)` | `predict_flow(x_input, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal, mask_density, task_emb)` |
| **`pred_type` attribute** | `'velocity'` or `'x1'` | `'velocity'` or `'x1'` |
| **What it returns** | Tensor shape `(B, L, motion_dim)` — model prediction | Tensor shape `(B, L, motion_dim)` — model prediction |
| **Meaning of output** | Depends on `pred_type`: velocity if `'velocity'`, clean motion if `'x1'` | Depends on `pred_type`: velocity if `'velocity'`, clean motion if `'x1'` |
| **Motion dimension** | 135-dim (T2M specific: translation 3D + rot6d 22×6) | 135-dim (M2M same motion space) |

---

## 1. What Does `bundle.predict_flow()` Return?

### Method Location & Signature

**HyMotion T2M** (`hftrainer/models/motion/hymotion_t2m/bundle.py`, lines 225-255):
```python
def predict_flow(
    self,
    x_input: Tensor,
    ctxt_input: Tensor,
    vtxt_input: Tensor,
    timesteps: Tensor,
    x_mask_temporal: Optional[Tensor] = None,
    ctxt_mask_temporal: Optional[Tensor] = None,
) -> Tensor:
    """Single forward pass through the MMDiT transformer.
    
    Args:
        x_input: noisy motion x_t, shape (B, L, motion_dim).
                 Unlike M2M, this is NOT concatenated with VACE context.
        ctxt_input: token-level text embeddings, (B, Lc, Dc).
        vtxt_input: sentence-level text embeddings, (B, 1, Dv).
        timesteps: diffusion timesteps, (B,).
        x_mask_temporal: (B, L) boolean mask for motion sequence.
        ctxt_mask_temporal: (B, Lc) boolean mask for text tokens.
    
    Returns:
        Model prediction, shape (B, L, motion_dim).
    """
    return self.motion_transformer(
        x=x_input,
        ctxt_input=ctxt_input,
        vtxt_input=vtxt_input,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )
```

**HyMotion M2M** (`hftrainer/models/motion/hymotion_m2m/bundle.py`, lines 540-575):
```python
def predict_flow(
    self,
    x_input: Tensor,
    ctxt_input: Tensor,
    vtxt_input: Tensor,
    timesteps: Tensor,
    x_mask_temporal: Optional[Tensor] = None,
    ctxt_mask_temporal: Optional[Tensor] = None,
    mask_density: Optional[Tensor] = None,
    task_emb: Optional[Tensor] = None,
) -> Tensor:
    """Single forward pass through the MMDiT transformer.
    
    Args:
        x_input: concatenated [x_t, vace_context], shape (B, L, D + 3*D_motion).
        ctxt_input: token-level text embeddings, (B, Lc, Dc).
        vtxt_input: sentence-level text embeddings, (B, 1, Dv).
        timesteps: diffusion timesteps, (B,).
        x_mask_temporal: (B, L) boolean mask for motion sequence.
        ctxt_mask_temporal: (B, Lc) boolean mask for text tokens.
        mask_density: (B,) optional mask density for CDE (CRFM v3).
        task_emb: (B, 1, 1024) optional task instruction embeddings.
    
    Returns:
        Model prediction, shape (B, L, D_motion).
    """
    return self.motion_transformer(...)
```

### Return Value

**Both T2M and M2M return:**
- **Type**: `torch.Tensor`
- **Shape**: `(B, L, motion_dim)` where:
  - `B` = batch size
  - `L` = sequence length (max of all sequences in batch, padded)
  - `motion_dim` = 135 for both T2M and M2M (SMPL-22 motion representation)
  
**Content (Semantic Meaning)**:
- The **model's prediction** for the motion at diffusion timestep `t`
- The exact interpretation depends on `bundle.pred_type` (see section 2 below)
- It is **NOT the final clean motion** — it needs further ODE integration or post-processing to get the final output

**Example Return Shape**:
```python
batch_size = 2
seq_length = 360
motion_dim = 135
output = bundle.predict_flow(x_t, ctxt_input, vtxt_input, timesteps, ...)
# output.shape = torch.Size([2, 360, 135])
```

---

## 2. What is `bundle.pred_type`?

### Definition & Location

**T2M** (`hftrainer/models/motion/hymotion_t2m/bundle.py`, line 73, 91):
```python
def __init__(
    self,
    ...
    pred_type: str = 'velocity',
    ...
):
    ...
    self.pred_type = pred_type
```

**M2M** (`hftrainer/models/motion/hymotion_m2m/bundle.py`, line 163, 192):
```python
def __init__(
    self,
    ...
    pred_type: str = 'velocity',
    ...
):
    ...
    self.pred_type = pred_type
```

### Values It Can Take

| Value | Meaning | Formula |
|-------|---------|---------|
| `'velocity'` | Model predicts the **flow velocity** (continuous time diffusion) | `v = dx/dt`, where `x(t) = (1-t)*x0 + t*x1` → `v = x1 - x0` |
| `'x1'` | Model predicts the **clean motion directly** (also called epsilon/noise prediction in discrete DDPM, but here it's the clean motion) | Model outputs `x1` directly; velocity is computed as `v = (x1 - x_t) / (1 - t)` |

### Where It's Used

Both are used in the **ODE integration function** inside the pipeline to determine how to handle the model output:

**T2M Pipeline** (`hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`, lines 139-149):
```python
def fn(t_val: Tensor, x: Tensor) -> Tensor:
    # ... forward pass through model ...
    x_pred = self.bundle.predict_flow(
        x_input=x_double,
        ctxt_input=ctxt_cfg,
        vtxt_input=vtxt_cfg,
        timesteps=t_val.expand(2 * B),
        ...
    )
    
    # CRITICAL: Handle different prediction types
    if self.bundle.pred_type == 'x1':
        t_eps = 0.05
        if do_cfg:
            x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
        else:
            x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)
    
    # If 'velocity', x_pred is used directly as dx/dt
    
    if do_cfg:
        pred_uncond, pred_text = x_pred.chunk(2, dim=0)
        x_pred = pred_uncond + self.text_guidance_scale * (pred_text - pred_uncond)
    
    return x_pred  # This is the dx/dt for ODE integration
```

**M2M Pipeline** (`hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`, similar pattern):
- Also checks `self.bundle.pred_type == 'velocity'` vs `'x1'`
- Applies same transformation when `pred_type == 'x1'`

---

## 3. Output Semantic: x1 vs Velocity?

### When `pred_type == 'velocity'`

```
Model output x_pred = velocity = dx/dt = x1 - x0

ODE Integration:
  x_{t+1} = x_t + v * dt
           = x_t + x_pred * dt    # x_pred is already the velocity
```

**Usage**:
```python
v = bundle.predict_flow(x_t, ctxt_input, vtxt_input, timesteps, ...)
# v is already velocity, can use directly in ODE

x_next = x_t + v * dt  # Euler step or higher-order ODE solver
```

### When `pred_type == 'x1'`

```
Model output x_pred = x1 (predicted clean motion at this step)

But we need velocity for ODE integration:
  v = (x1 - x_t) / (1 - t) + epsilon  # epsilon prevents division near t=1
  
ODE Integration:
  x_{t+1} = x_t + v * dt
           = x_t + (x1 - x_t) / (1 - t) * dt
```

**Usage**:
```python
x1_pred = bundle.predict_flow(x_t, ctxt_input, vtxt_input, timesteps, ...)
# x1_pred is the predicted clean motion, NOT velocity

# Convert to velocity for ODE
t_eps = 0.05
v = (x1_pred - x_t) / (1.0 - t).clamp_min(t_eps)

x_next = x_t + v * dt
```

---

## 4. Key Differences: T2M vs M2M

### Input (`x_input`)

| Aspect | T2M | M2M |
|--------|-----|-----|
| **x_input composition** | Just `x_t` (noisy motion) | `[x_t, vace_context]` where `vace_context = [inactive, reactive, mask]` or `[reactive, mask]` |
| **x_input shape** | `(B, L, 135)` | `(B, L, 135 + vace_dims)` — typically `(B, L, 540)` or `(B, L, 405)` depending on `vace_condition_mode` |
| **Conditioning info source** | Only text + timestep | Text + timestep + VACE (known/edit regions) |

**T2M Docstring** (line 237-239):
> Unlike M2M, this is NOT concatenated with VACE context.
> The input to the transformer is just x_t (motion_dim), not [x_t, vace_context]

**M2M Docstring** (line 554):
> x_input: concatenated [x_t, vace_context], shape (B, L, D + 3*D_motion).

### Output

Both return the **same output shape and meaning**:
- Shape: `(B, L, motion_dim)` = `(B, L, 135)`
- Meaning: Depends on `pred_type`

### Model Architecture

Both use **HunyuanMotionMMDiT** (same architecture), but:
- **T2M**: Input dim adjusted for 135-dim motion only
- **M2M**: Input dim adjusted for 135-dim motion + VACE context (typically 540-dim total)

---

## 5. Configuration Examples

### T2M Config

From `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`:
```python
model = dict(
    type='HyMotionT2MBundle',
    pred_type='velocity',  # or 'x1'
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        input_dim=135,  # Just x_t
        output_dim=135,
        ...
    ),
    ...
)
```

### M2M Config

From `configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py`:
```python
model = dict(
    type='HyMotionM2MBundle',
    pred_type='velocity',  # or 'x1' (for JiT variant)
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        input_dim=540,  # x_t (135) + vace_context (405 for split_reactive mode)
        output_dim=135,
        ...
    ),
    ...
)
```

---

## 6. How `pred_type` Affects Pipeline Behavior

### Flow Matching Framework

Both bundles use **flow matching** (continuous diffusion) with ODE integration from `t=0` to `t=1`:

```python
trajectory = odeint(fn, y0, t=[0, 1], method=method)
# where fn uses bundle.predict_flow to compute dx/dt
```

### For `pred_type='velocity'`:
```
predict_flow output is directly used as dx/dt in the ODE

fn = lambda t, x: bundle.predict_flow(x, ctxt, vtxt, t, ...)
# No conversion needed, output is already velocity
```

### For `pred_type='x1'`:
```
predict_flow output is the predicted clean motion x1

fn = lambda t, x: (bundle.predict_flow(x, ctxt, vtxt, t, ...) - x) / (1 - t)
# Must convert x1 to velocity by (x1 - x) / (1 - t)
```

---

## 7. Motion Dimension: What's in the 135 dims?

Both T2M and M2M use the same 135-dim motion representation (SMPL-22):

```
dims [0:3]     — translation (3D absolute position)
dims [3:9]     — Pelvis (root, 6D rot6d)
dims [9:15]    — L_Hip (6D rot6d)
dims [15:21]   — R_Hip (6D rot6d)
... (each joint gets 6D rot6d)
dims [123:129] — L_Wrist (6D rot6d)
dims [129:135] — R_Wrist (6D rot6d)
```

Total: 3 (translation) + 22 joints × 6 (rot6d) = 3 + 132 = **135 dims**

---

## 8. Critical Implementation Details

### T2M Pipeline ODE Function

From `hymotion_t2m_pipeline.py` lines 118-149:
```python
def fn(t_val: Tensor, x: Tensor) -> Tensor:
    """ODE dx/dt function."""
    if do_cfg:
        x_double = torch.cat([x, x], dim=0)  # [uncond sample, cond sample]
        x_pred = self.bundle.predict_flow(
            x_input=x_double,           # Shape: (2B, L, 135)
            ctxt_input=ctxt_cfg,        # Stacked ctxt for both
            vtxt_input=vtxt_cfg,        # Stacked vtxt for both
            timesteps=t_val.expand(2 * B),
            x_mask_temporal=tgt_padding_mask.repeat(2, 1),
            ctxt_mask_temporal=ctxt_mask_cfg,
        )
    else:
        x_pred = self.bundle.predict_flow(
            x_input=x,                  # Shape: (B, L, 135)
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=t_val.expand(B),
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )
    
    # Handle x1 prediction type: convert to velocity
    if self.bundle.pred_type == 'x1':
        t_eps = 0.05
        if do_cfg:
            # For each branch: v = (x1_pred - x_current) / (1 - t)
            x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
        else:
            x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)
    
    # Apply classifier-free guidance
    if do_cfg:
        pred_uncond, pred_text = x_pred.chunk(2, dim=0)
        x_pred = pred_uncond + self.text_guidance_scale * (pred_text - pred_uncond)
    
    return x_pred  # Returns dx/dt for ODE solver
```

### M2M Pipeline (Similar Pattern)

From `hymotion_m2m_pipeline.py`, the `fn` function is identical in structure, just with VACE input handling.

---

## 9. Training vs Inference

### Training (HyMotionT2MTrainer)

During training, `predict_flow` is called on clean motion samples that have been noised at different timesteps. The `pred_type` determines the loss target:

- If `pred_type='velocity'`: Loss compares model output with actual velocity `v = x1 - x0`
- If `pred_type='x1'`: Loss compares model output with actual clean motion `x1`

### Inference (HyMotionT2MPipeline)

During inference, `predict_flow` is called iteratively during ODE integration. The model output is interpreted differently based on `pred_type`:

- If `pred_type='velocity'`: Output is directly used as `dx/dt`
- If `pred_type='x1'`: Output is converted to velocity using the formula above

---

## 10. Summary Table

| Property | Value | Notes |
|----------|-------|-------|
| **predict_flow() return type** | `torch.Tensor` | Always 4D tensor or higher (if batched) |
| **Return shape** | `(B, L, 135)` | Batch × Sequence × Motion-dim |
| **Return content** | Model's diffusion prediction | Interpretation depends on `pred_type` |
| **pred_type attribute** | `'velocity'` or `'x1'` | Set during bundle initialization |
| **When pred_type='velocity'** | Output = dx/dt = x1 - x0 | Can be used directly in ODE |
| **When pred_type='x1'** | Output = x1 (clean motion) | Must convert to velocity: `(x1 - x_t)/(1-t)` |
| **Default pred_type** | `'velocity'` | Both T2M and M2M default to velocity |
| **Is output the final motion?** | **No** | It's an intermediate prediction that ODE integration processes further |

---

## Key Takeaway

**`bundle.predict_flow(x_t, t, cond)` returns the model's best guess of what the motion should look like given:**
- Current noisy state `x_t`
- Timestep `t` in the diffusion process
- Conditions (text, VACE, etc.)

**The interpretation of this guess depends on `bundle.pred_type`:**
- If **velocity**: This IS the velocity vector to integrate the ODE forward
- If **x1**: This IS the predicted clean motion, from which we must compute velocity

Both interpretations are mathematically valid formulations of flow matching; the configuration determines which one your model was trained to produce.

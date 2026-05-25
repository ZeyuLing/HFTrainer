# FlowMatchEulerDiscreteScheduler - Quick Reference

## The Main Equation (All You Need)

```python
prev_sample = sample + (sigma_next - sigma) * model_output
```

**That's it.** This is the entire inference update rule.

---

## Variables Meaning

| Variable | Type | Shape | Meaning |
|----------|------|-------|---------|
| `sample` | tensor | same as model input | Current noisy latent (x_t) |
| `model_output` | tensor | same as sample | Network prediction (m_t) |
| `sigma` | float | scalar | Current noise level from `sigmas[step_index]` |
| `sigma_next` | float | scalar | Next noise level from `sigmas[step_index + 1]` |
| `prev_sample` | tensor | same as sample | Denoised output (x_{t-1}) |

---

## Sigma-Timestep Relationship

```
σ = timestep / num_train_timesteps
timestep = σ * num_train_timesteps
```

Example (T=1000):
- timestep=1000 → σ=1.0 (most noise)
- timestep=500 → σ=0.5 (half denoised)
- timestep=0 → σ=0.0 (clean)

---

## Setup (Called Once Before Inference)

```python
scheduler.set_timesteps(num_inference_steps=50, device=device)
timesteps = scheduler.timesteps
```

This creates:
- `scheduler.timesteps`: array of shape `[50]`
- `scheduler.sigmas`: array of shape `[51]` (has extra 0 at end for padding)

---

## Inference Loop Pattern

```python
for t in scheduler.timesteps:
    # Run model to get prediction
    model_pred = model(latents, timestep=t, ...)
    
    # Update latents
    latents = scheduler.step(model_pred, t, latents).prev_sample
```

Inside `scheduler.step()`:
1. Looks up sigma values at current step index
2. Applies: `latents = latents + (sigma_next - sigma) * model_pred`
3. Increments step index for next iteration

---

## Key Properties

| Property | Value |
|----------|-------|
| `sigma_max` | 1.0 (start, max noise) |
| `sigma_min` | 0.0 (end, no noise) |
| `len(sigmas)` | `num_inference_steps + 1` |
| `len(timesteps)` | `num_inference_steps` |
| Dtype during computation | float32 |
| Dtype of output | restored to model.dtype |

---

## Shift Transform (Optional)

If `use_dynamic_shifting=False` (default):

```
σ_shifted = shift * σ / (1 + (shift - 1) * σ)
```

- Default `shift=1.0`: σ_shifted = σ (no change)
- Higher shift (e.g., 1.15): stretches schedule toward high-noise region

If `use_dynamic_shifting=True`:

```
σ_shifted = time_shift(μ, 1.0, σ)
where time_shift(μ, σ, t) = e^μ / (e^μ + (1/t - 1)^σ)
```

---

## Important Implementation Details

1. **Array Padding**: `sigmas` has one extra zero appended
   - Allows `sigmas[step_index + 1]` to always be valid
   - Even on the last step: `sigmas[-1] = 0.0`

2. **Step Index**: Auto-increments after each `step()` call
   - First call: initializes via `_init_step_index(timestep)`
   - Subsequent calls: just increments

3. **Timestep Input**: Pass the actual timestep value, NOT the index
   ```python
   # CORRECT:
   scheduler.step(pred, t=timesteps[i], sample=latents)
   
   # WRONG:
   scheduler.step(pred, t=i, sample=latents)  # Don't pass index!
   ```

4. **Precision**: 
   - Input `sample` is upcasted to float32 during computation
   - Output is downcast back to original dtype

5. **Sigma Sign**: 
   - `(sigma_next - sigma)` is **always negative** during inference
   - Movement from σ=1.0 toward σ=0.0
   - Effectively: `latents = latents - |Δσ| * model_pred`

---

## Code Location

**File**: `/ref_repo/MotionLab/rfmotion/models/operator/scheduling_flow_match_euler_discrete.py`

**Core function**: `FlowMatchEulerDiscreteScheduler.step()` (lines 235-308)

**Setup function**: `FlowMatchEulerDiscreteScheduler.set_timesteps()` (lines 171-211)

**Usage**: `/hftrainer/pipelines/motion/prism_backend.py` (line 442)

---

## Mathematical Form

**Euler Forward Method in Sigma Space:**

```
x_{t-1} = x_t + (σ_{t-1} - σ_t) · m_t
```

Where:
- x_t: sample at step t
- x_{t-1}: sample at next step (moving backward/cleaner)
- σ_t: noise schedule value
- m_t: model output (flow/denoising direction)

This is a **linear interpolation** scaled by the sigma change.

---

## Common Mistakes to Avoid

❌ Passing integer index instead of timestep value  
❌ Forgetting to call `set_timesteps()` before inference loop  
❌ Assuming `sigmas` and `timesteps` have same length (sigmas has +1)  
❌ Trying to access timesteps by index instead of iterating directly  
❌ Modifying `step_index` manually (it auto-increments)  
❌ Using different device for timesteps and sample  

---

## Debugging Tips

```python
# Check scheduler state
print(f"step_index: {scheduler.step_index}")
print(f"sigma shape: {scheduler.sigmas.shape}")
print(f"timesteps shape: {scheduler.timesteps.shape}")
print(f"sigma_max: {scheduler.sigma_max}, sigma_min: {scheduler.sigma_min}")

# Check step computation
sigma = scheduler.sigmas[scheduler.step_index]
sigma_next = scheduler.sigmas[scheduler.step_index + 1]
print(f"Δσ = {sigma_next - sigma}")  # Should be negative
```

---

## Reference Paper

This implementation is based on **Flow Matching** by Yildirim et al., adapted for discrete Euler stepping.

Key paper: "Flow Matching for Generative Modeling" 
- Flow matching defines the vector field m_t that the model learns
- Euler stepping is a simple ODE solver to follow this vector field

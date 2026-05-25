# FlowMatchEulerDiscreteScheduler Documentation Index

This directory contains comprehensive documentation about the scheduler used during PRISM inference.

## 📄 Files Overview

### 1. **SCHEDULER_QUICK_REFERENCE.md** ⭐ START HERE
- **Best for**: Quick lookup, implementation reference
- **Contains**: The main equation, key properties, common mistakes
- **Length**: ~2 min read
- **Use case**: "I need to remember the formula" or "What's the correct usage?"

### 2. **SCHEDULER_VISUAL_SUMMARY.txt** 
- **Best for**: Understanding the complete workflow
- **Contains**: Visual diagrams, step-by-step execution examples, denoising loop walkthrough
- **Length**: ~5 min read
- **Use case**: "How does the scheduler actually work end-to-end?"

### 3. **FLOWMATCH_SCHEDULER_ANALYSIS.md**
- **Best for**: Deep technical understanding
- **Contains**: All methods (step, set_timesteps, _init_step_index, etc.), complete source code sections, detailed equations
- **Length**: ~10 min read
- **Use case**: "I need to understand every detail" or "I'm debugging scheduler issues"

## 🎯 Quick Navigation

### I need to know...

**"What's the main equation?"**
→ See `SCHEDULER_QUICK_REFERENCE.md` Section 1

**"How do sigmas relate to timesteps?"**
→ See `SCHEDULER_QUICK_REFERENCE.md` Section: "Sigma-Timestep Relationship"

**"What's the complete inference loop?"**
→ See `SCHEDULER_VISUAL_SUMMARY.txt` Section 4: "Denoising Loop Execution"

**"Show me a worked example"**
→ See `SCHEDULER_VISUAL_SUMMARY.txt` Section 7: "Complete Inference Sequence Example"

**"All the methods and their details"**
→ See `FLOWMATCH_SCHEDULER_ANALYSIS.md` Sections 1-5

**"I'm getting wrong results, what could be wrong?"**
→ See `SCHEDULER_QUICK_REFERENCE.md` Section: "Common Mistakes to Avoid"

**"What about the shift transform?"**
→ See `SCHEDULER_VISUAL_SUMMARY.txt` Section 5 or `FLOWMATCH_SCHEDULER_ANALYSIS.md` Section 6

## 📍 Source Code Locations

- **Implementation**: `/ref_repo/MotionLab/rfmotion/models/operator/scheduling_flow_match_euler_discrete.py`
- **Usage**: `/hftrainer/pipelines/motion/prism_backend.py` (line 442)

## 🔑 Key Takeaways

The entire scheduler behavior boils down to:

```python
# Before loop (once):
scheduler.set_timesteps(num_inference_steps=50, device=device)

# In loop (50 times):
for t in scheduler.timesteps:
    pred = model(latents, t)
    latents = scheduler.step(pred, t, latents).prev_sample

# Inside scheduler.step():
sigma = scheduler.sigmas[scheduler.step_index]
sigma_next = scheduler.sigmas[scheduler.step_index + 1]
prev_sample = sample + (sigma_next - sigma) * model_output
```

## 🧮 The Main Equation

```
x_{t-1} = x_t + (σ_{t-1} - σ_t) · m_t
```

**That's it.** This linear equation is the entire inference rule.

## 📊 Reference Table

| Concept | Formula | Range | Meaning |
|---------|---------|-------|---------|
| Sigma | σ = t / T | [0.0, 1.0] | Noise level (1.0=noisy, 0.0=clean) |
| Change | Δσ = σ_next - σ | Negative | Always moves toward clean (σ decreases) |
| Step | x_new = x_old + Δσ·m | Varies | Euler update rule |

## 🚀 Common Tasks

**Task: Add custom timestep schedule**
→ Modify `set_timesteps()` method to create custom `timesteps` and `sigmas`

**Task: Understand inference timing**
→ `num_inference_steps` controls total denoising steps
→ More steps = better quality but slower
→ Fewer steps = faster but lower quality

**Task: Debug step computation**
→ Print `sigma` and `sigma_next` at each step
→ Verify `sigma_next < sigma` (moving toward clean)
→ Check that `(sigma_next - sigma)` scales model output correctly

**Task: Implement custom scheduler**
→ Must implement: `__init__()`, `set_timesteps()`, `step()`, and `_init_step_index()`
→ Key requirement: maintain `sigmas` array with extra 0 padding

## 📚 Related Concepts

- **Flow Matching**: The underlying theory (model predicts flow/velocity field)
- **Euler Stepping**: Simple first-order ODE solver (our method uses this)
- **Noise Scheduling**: How noise level changes over diffusion steps
- **Timestep Conditioning**: How the model knows which noise level we're at

## ✅ Verification Checklist

Use this when implementing or debugging:

- [ ] `len(sigmas) == num_inference_steps + 1` (has padding)
- [ ] `len(timesteps) == num_inference_steps`
- [ ] `sigma_max == 1.0`, `sigma_min == 0.0`
- [ ] During inference: `sigma_next < sigma` always
- [ ] `(sigma_next - sigma)` is always negative
- [ ] `step_index` auto-increments after each step
- [ ] First `step()` call initializes step_index
- [ ] Timestep input is a value, not an index
- [ ] Sample is float32 during computation
- [ ] Output dtype matches model dtype

## 🔗 Connections to Other Components

- **Transformer**: Receives timestep `t` as conditioning input
- **Pipeline**: Calls `set_timesteps()` once, then `step()` in the denoising loop
- **Latent**: Gets updated at each step by the scheduler
- **Condition**: Combined with latents for first-frame masking

---

**Last Updated**: 2026-05-19
**Scheduler Version**: FlowMatchEulerDiscreteScheduler (from MotionLab)
**Status**: Complete documentation

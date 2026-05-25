# FlowMatchEulerDiscreteScheduler - Complete Analysis

## 📋 Quick Summary

You now have **complete documentation** of the FlowMatchEulerDiscreteScheduler used in PRISM inference.

**The core equation is simple:**
```python
prev_sample = sample + (sigma_next - sigma) * model_output
```

This is a **first-order Euler step** that moves samples through noise space from σ=1.0 (noisy) toward σ=0.0 (clean).

---

## 📄 Documentation Files

All files are in `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`:

### 1. **SCHEDULER_QUICK_REFERENCE.md** ⭐ **START HERE**
- **Purpose**: Quick lookup and implementation reference
- **Contains**: 
  - The main equation with all variables explained
  - Sigma-timestep relationships with examples
  - Setup code pattern for inference
  - Key properties (sigma_max, sigma_min, array shapes)
  - Common mistakes to avoid
  - Debugging tips
- **Read time**: ~2-3 minutes
- **Best for**: "I need the formula" or "How do I use this?"

### 2. **SCHEDULER_VISUAL_SUMMARY.txt**
- **Purpose**: Understand the complete workflow visually
- **Contains**:
  - ASCII diagrams of the complete system
  - Step-by-step execution walkthrough
  - Complete denoising loop with code
  - Full inference sequence example (with numbers)
  - Shift transformations explained
  - Key takeaways
- **Read time**: ~5-7 minutes
- **Best for**: "How does everything work together?"

### 3. **FLOWMATCH_SCHEDULER_ANALYSIS.md**
- **Purpose**: Deep technical reference
- **Contains**:
  - All 7 methods explained in detail
  - Complete source code sections
  - Mathematical interpretations
  - Sigma initialization and relationships
  - Summary table of all equations
  - Important implementation details
- **Read time**: ~10 minutes
- **Best for**: "I need to understand every detail" or "I'm debugging"

### 4. **SCHEDULER_DOCUMENTATION_INDEX.md**
- **Purpose**: Navigation guide
- **Contains**:
  - Which document to read for what question
  - Quick answer lookup table
  - Verification checklist
  - Common tasks and solutions
  - Related concepts
  - Connections to other components
- **Read time**: ~3 minutes
- **Best for**: "Where should I look?"

---

## 🎯 Quick Navigation by Question

**"What's the core equation?"**
→ See: SCHEDULER_QUICK_REFERENCE.md, Section 1

**"How do sigmas relate to timesteps?"**
→ See: SCHEDULER_QUICK_REFERENCE.md, "Sigma-Timestep Relationship"

**"Show me the full workflow"**
→ See: SCHEDULER_VISUAL_SUMMARY.txt, Sections 1-4

**"Give me a complete worked example"**
→ See: SCHEDULER_VISUAL_SUMMARY.txt, Section 7

**"All the methods and their details"**
→ See: FLOWMATCH_SCHEDULER_ANALYSIS.md, Sections 1-5

**"I'm getting wrong results, what's wrong?"**
→ See: SCHEDULER_QUICK_REFERENCE.md, "Common Mistakes to Avoid"

**"What about shift transformations?"**
→ See: SCHEDULER_VISUAL_SUMMARY.txt, Section 5

**"I need to find something specific"**
→ See: SCHEDULER_DOCUMENTATION_INDEX.md

---

## 🔬 The Exact Equations

### Core Step (The Main Computation)
```
prev_sample = sample + (sigma_next - sigma) * model_output

Mathematical form:
x_{t-1} = x_t + (σ_{t-1} - σ_t) · m_t

First-order Euler method in sigma space
(σ_{t-1} - σ_t) is ALWAYS NEGATIVE during inference
```

### Sigma-Timestep Conversion
```
σ = timestep / num_train_timesteps
timestep = σ * num_train_timesteps

Range: σ ∈ [0.0, 1.0]
  0.0 = clean (no noise)
  1.0 = noisy (max noise)
```

### Static Shift Transform (if not using dynamic shifting)
```
σ_shifted = shift · σ / (1 + (shift - 1) · σ)

Default shift=1.0: σ_shifted = σ (no change)
Higher shift: stretches toward high-noise region
```

### Dynamic Shift Transform
```
σ_shifted = time_shift(μ, 1.0, σ)
where: time_shift(μ, σ, t) = e^μ / (e^μ + (1/t - 1)^σ)
```

### Set Timesteps Schedule
```
timesteps = linspace(σ_max·T, σ_min·T, num_inference_steps)
          = linspace(1000, 0, 50)  [for num_train_timesteps=1000]

sigmas = torch.cat([sigmas_computed, torch.zeros(1)])
Result: len(sigmas) = num_inference_steps + 1
```

---

## 🏗️ Complete Inference Structure

```python
# Setup (once before inference)
scheduler.set_timesteps(num_inference_steps=50, device=device)

# Denoising loop
for t in scheduler.timesteps:
    # Run model to get prediction
    model_pred = transformer(latents, timestep=t, ...)
    
    # Update latents using scheduler
    latents = scheduler.step(model_pred, t, latents).prev_sample

# After loop: latents are clean samples
```

**Inside `scheduler.step()`:**
```python
sigma = self.sigmas[self.step_index]
sigma_next = self.sigmas[self.step_index + 1]
prev_sample = sample + (sigma_next - sigma) * model_output
self._step_index += 1
```

---

## 💾 Source Code

**Implementation:**
- File: `/ref_repo/MotionLab/rfmotion/models/operator/scheduling_flow_match_euler_discrete.py`
- `step()` method: Lines 235-308
- `set_timesteps()` method: Lines 171-211
- `_init_step_index()` method: Lines 227-233

**Usage:**
- File: `/hftrainer/pipelines/motion/prism_backend.py`
- `scheduler.set_timesteps()`: Line 378
- `scheduler.step()` in loop: Line 442

---

## ✅ Verification Checklist

When implementing or debugging the scheduler:

- [ ] Core equation: `prev_sample = sample + Δσ * model`
- [ ] Sigma range: [0.0 (clean) to 1.0 (noisy)]
- [ ] Sigma relationship: `σ = t / num_train_timesteps`
- [ ] Array shapes: `len(sigmas) = num_inference_steps + 1`
- [ ] Step change: `(σ_next - σ)` is always negative
- [ ] Step index: auto-increments, initialized on first call
- [ ] Precision: float32 during computation, restored to model.dtype
- [ ] Timestep input: value, not index (e.g., t=1000, not t=0)
- [ ] Shift transform: `σ_shifted = shift·σ/(1+(shift-1)·σ)`
- [ ] Direction: moves from σ=1.0 toward σ=0.0

---

## ⚠️ Common Mistakes

❌ **Passing integer index instead of timestep value**
- Wrong: `scheduler.step(pred, 0, latents)` 
- Right: `scheduler.step(pred, scheduler.timesteps[0], latents)`

❌ **Forgetting to call `set_timesteps()` before loop**
- Must be called once before inference

❌ **Assuming `sigmas` and `timesteps` have same length**
- `len(timesteps) = num_inference_steps`
- `len(sigmas) = num_inference_steps + 1` (has padding)

❌ **Trying to access timesteps by index in the loop**
- Wrong: `for i in range(len(timesteps)): t = i`
- Right: `for t in scheduler.timesteps:`

❌ **Modifying `step_index` manually**
- It auto-increments after each step
- Don't touch it

❌ **Using different device for timesteps and sample**
- Ensure timesteps are on the same device as latents

---

## 📊 Key Properties

| Property | Value | Meaning |
|----------|-------|---------|
| `sigma_max` | 1.0 | Start (maximum noise) |
| `sigma_min` | 0.0 | End (no noise/clean) |
| `len(sigmas)` | num_steps + 1 | With padding for final access |
| `len(timesteps)` | num_steps | Used in loop |
| Dtype in step() | float32 | Computation precision |
| Output dtype | model.dtype | Restored from float32 |
| `(sigma_next - sigma)` | Always < 0 | Moving toward clean |
| `step_index` | Auto-increments | No manual management |

---

## 🎓 Learning Path

**Level 1: Quick Understanding (5 min)**
1. Read SCHEDULER_QUICK_REFERENCE.md Section 1-2
2. Understand: core equation + sigma relationships

**Level 2: Practical Usage (15 min)**
1. Read SCHEDULER_QUICK_REFERENCE.md completely
2. Read SCHEDULER_VISUAL_SUMMARY.txt Section 4
3. Understand: how to set up and use in inference

**Level 3: Complete Mastery (30 min)**
1. Read all documents in order
2. Study FLOWMATCH_SCHEDULER_ANALYSIS.md
3. Trace through SCHEDULER_VISUAL_SUMMARY.txt Section 7 example

**Level 4: Implementation/Debugging (As needed)**
1. Use SCHEDULER_DOCUMENTATION_INDEX.md to find specific topics
2. Cross-reference with source code
3. Use verification checklist

---

## 🔗 Related Concepts

- **Flow Matching**: The underlying theory (model predicts velocity field)
- **Euler Stepping**: Simple first-order ODE solver (what we use)
- **Noise Scheduling**: How noise changes over diffusion steps
- **Timestep Conditioning**: How model knows which noise level
- **Diffusion Models**: General concept this is based on

---

## 📝 Notes

- This scheduler is DETERMINISTIC (no randomness unless s_churn parameters used)
- The step index MUST be incremented for loop continuation
- Sigmas are stored on CPU by default to reduce memory transfers
- The padding zero is crucial for valid array access on final step
- Precision is carefully managed to prevent loss of information

---

## 🚀 Next Steps

1. **If you're implementing**: Use SCHEDULER_QUICK_REFERENCE.md as template
2. **If you're debugging**: Check SCHEDULER_DOCUMENTATION_INDEX.md checklist
3. **If you're learning**: Follow the "Learning Path" section above
4. **If you need details**: Reference FLOWMATCH_SCHEDULER_ANALYSIS.md

---

**Last Updated**: 2026-05-19  
**Scheduler**: FlowMatchEulerDiscreteScheduler (from MotionLab)  
**Status**: Complete, thoroughly documented

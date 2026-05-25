# FlowMatchEulerDiscreteScheduler Analysis

## Overview
The `FlowMatchEulerDiscreteScheduler` is a custom scheduler from the MotionLab reference repository used during inference in the PRISM motion generation pipeline. It implements Euler-based discrete time stepping for flow matching diffusion models.

## Key File Location
- **Custom Implementation**: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/MotionLab/rfmotion/models/operator/scheduling_flow_match_euler_discrete.py`
- **Usage Location**: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/pipelines/motion/prism_backend.py` (line 442)

---

## 1. The `step()` Method - Core Inference Equation

### Location in Code
Lines 235-308 of the scheduler file

### Exact Formula for Computing `prev_sample`

```python
prev_sample = sample + (sigma_next - sigma) * model_output
```

**Where:**
- `sample`: Current latent tensor (the noisy sample at timestep t)
- `model_output`: The neural network's prediction (denoising output)
- `sigma`: Current sigma value (noise level) from `self.sigmas[self.step_index]`
- `sigma_next`: Next sigma value (noise level) from `self.sigmas[self.step_index + 1]`
- `prev_sample`: Output sample (should be used as next model input)

### Mathematical Interpretation

This is a **first-order Euler forward method**:

```
x_{t-1} = x_t + (σ_{t-1} - σ_t) * model_prediction
```

This implements:
- Linear interpolation in the sigma space
- Direct flow matching: the model predicts the direction of change, scaled by the sigma difference
- One-step update from sigma_t to sigma_{t-1}

### Code Section (Lines 288-303)
```python
if self.step_index is None:
    self._init_step_index(timestep)

# Upcast to avoid precision issues when computing prev_sample
sample = sample.to(torch.float32)

sigma = self.sigmas[self.step_index]              # Current sigma
sigma_next = self.sigmas[self.step_index + 1]    # Next sigma

prev_sample = sample + (sigma_next - sigma) * model_output  # ← CORE FORMULA

# Cast sample back to model compatible dtype
prev_sample = prev_sample.to(model_output.dtype)

# upon completion increase step index by one
self._step_index += 1
```

---

## 2. Sigma Initialization and Relationship to Timesteps

### During Initialization (`__init__`, Lines 65-90)

```python
# 1. Create initial timesteps from 0 to num_train_timesteps, reversed
timesteps = np.linspace(0, num_train_timesteps, num_train_timesteps+1, dtype=np.float32)[::-1].copy()
# Result: [1000, 999, 998, ..., 1, 0]

# 2. Convert timesteps to sigmas (normalized)
sigmas = timesteps / num_train_timesteps
# Result: σ ∈ [1.0, 0.999, 0.998, ..., 0.001, 0.0]

# 3. Apply shift transformation (if use_dynamic_shifting=False)
if not use_dynamic_shifting:
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    # With default shift=1.0: sigmas = 1.0 * sigmas / (1 + 0 * sigmas) = sigmas (no change)
    # With shift > 1.0: applies frequency shift to timeline

# 4. Convert back to timestep scale for reference
self.timesteps = sigmas * num_train_timesteps

# 5. Store sigmas on CPU
self.sigmas = sigmas.to("cpu")
```

**Key Relationships:**
```
σ = timestep / num_train_timesteps
timestep = σ * num_train_timesteps
```

**Sigma Range:**
- `sigma_max = 1.0` (start, most noise)
- `sigma_min = 0.0` (end, no noise)

### During Inference (`set_timesteps`, Lines 171-211)

Called once before the denoising loop to set up the inference schedule:

```python
def set_timesteps(
    self,
    num_inference_steps: int = None,
    device: Union[str, torch.device] = None,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
):
    """
    Sets up the discrete timesteps used for inference denoising.
    """
    
    if sigmas is None:
        # Create linearly-spaced timesteps from sigma_max to sigma_min
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max),      # Start: 1000
            self._sigma_to_t(self.sigma_min),      # End: 0
            num_inference_steps
        )
        # Convert to sigma scale
        sigmas = timesteps / self.config.num_train_timesteps
    
    # Apply shift transformation
    if self.config.use_dynamic_shifting:
        sigmas = self.time_shift(mu, 1.0, sigmas)
    else:
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
    
    # Convert to tensors and add a zero at the end (for sigma_next at final step)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    timesteps = sigmas * self.config.num_train_timesteps
    
    self.timesteps = timesteps.to(device=device)
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    # ↑ IMPORTANT: Appends a zero so sigmas has one more element than timesteps
    # This enables sigma_next = sigmas[step_index + 1] at every step
    
    self._step_index = None
    self._begin_index = None
```

**Example with num_inference_steps=50:**
- Input: linspace from timestep 1000 to 0 in 50 steps
- Output: `self.sigmas` has 51 elements (50 + 1 zero at end)
- During denoising: loop through timesteps[0] to timesteps[49], each can access sigmas[i] and sigmas[i+1]

---

## 3. Sigma-Timestep Conversion Helper

### `_sigma_to_t()` Method (Line 165-166)
```python
def _sigma_to_t(self, sigma):
    return sigma * self.config.num_train_timesteps
```

**Purpose:** Convert normalized sigma (0.0 to 1.0) back to timestep scale (0 to 1000)

---

## 4. Step Index Management

### `_init_step_index()` (Lines 227-233)
Initializes the step index on first call:

```python
def _init_step_index(self, timestep):
    if self.begin_index is None:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        self._step_index = self.index_for_timestep(timestep)
    else:
        self._step_index = self._begin_index
```

### `index_for_timestep()` (Lines 213-225)
Maps a given timestep to its index in the sigma schedule:

```python
def index_for_timestep(self, timestep, schedule_timesteps=None):
    if schedule_timesteps is None:
        schedule_timesteps = self.timesteps
    
    # Find indices where timestep matches (within tolerance 1e-4)
    indices = ((schedule_timesteps - timestep) < 1e-4).nonzero()
    
    # Take the second index if multiple found (to avoid skipping sigmas)
    # Used for image-to-image (starting mid-denoising)
    pos = 1 if len(indices) > 1 else 0
    
    return indices[pos].item()
```

---

## 5. Usage in PRISM Pipeline

### Location: `/hftrainer/pipelines/motion/prism_backend.py`

#### Setup Phase (Line 378)
```python
self.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = self.scheduler.timesteps  # Shape: [num_inference_steps]
```

#### Denoising Loop (Line 442)
```python
for i, t in enumerate(timesteps):
    # ... run transformer model to get noise_pred ...
    
    # Call scheduler step with:
    # - noise_pred: model output (predicted flow direction)
    # - t: current timestep from schedule
    # - latents: current sample
    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

---

## 6. Summary of Exact Equations

| Component | Formula |
|-----------|---------|
| **Core Step** | `x_{t-1} = x_t + (σ_{t-1} - σ_t) · m_t` |
| **Sigma-Timestep** | `σ = t / num_train_timesteps` |
| **Timestep-Sigma** | `t = σ · num_train_timesteps` |
| **Shift Transform** | `σ_shifted = shift · σ / (1 + (shift - 1) · σ)` |
| **Dynamic Shift** | `σ_shifted = time_shift(μ, 1.0, σ)` where `time_shift(μ, σ, t) = e^μ / (e^μ + (1/t - 1)^σ)` |
| **Sigma Range** | `σ ∈ [0.0, 1.0]` where 1.0 = max noise, 0.0 = min noise |
| **Schedule** | `timesteps = linspace(σ_max·T, σ_min·T, num_inference_steps)` |

**Notation:**
- `x_t`: Sample at current step
- `x_{t-1}`: Sample at next step (moving backward in time)
- `σ_t`, `σ_{t-1}`: Sigmas at steps t and t-1
- `m_t`: Model output (noise/flow prediction)
- `T`: `num_train_timesteps` (typically 1000)

---

## 7. Important Details

1. **Sigma Array Padding**: The `self.sigmas` array has `num_inference_steps + 1` elements (extra zero at end)
   - Allows `sigmas[step_index + 1]` to always be valid
   - Last element is 0.0 for final denoising step

2. **Precision Handling**: 
   - Sample upcast to float32 during computation (line 292)
   - Result cast back to model's dtype (line 300)

3. **Step Index Auto-increment**: After each step, `self._step_index += 1` (line 303)

4. **Linear Interpolation**: `(sigma_next - sigma)` is always negative during denoising
   - Moves from σ=1.0 toward σ=0.0
   - `sigma_next < sigma` throughout

5. **Timestep Format**: Input timestep `t` should be a value from `scheduler.timesteps`
   - NOT an integer index
   - Automatic index lookup via `_init_step_index()`


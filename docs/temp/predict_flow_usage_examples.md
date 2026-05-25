# predict_flow() Usage Examples — Practical Guide

## Example 1: Basic T2M Inference Call

### Setup
```python
from hftrainer.models.motion.hymotion_t2m.bundle import HyMotionT2MBundle
import torch

# Initialize bundle from config
bundle = HyMotionT2MBundle(
    motion_transformer=dict(...),
    pred_type='velocity',  # <-- CRITICAL
    ...
)
bundle.load_state_dict(checkpoint['model'])
bundle.eval()

# Prepare batch
batch_size = 2
seq_length = 360
motion_dim = 135

# Noisy motion at timestep t
x_t = torch.randn(batch_size, seq_length, motion_dim)  # Shape: (2, 360, 135)

# Text embeddings (pre-encoded)
ctxt_input = torch.randn(batch_size, 77, 4096)         # Context tokens
vtxt_input = torch.randn(batch_size, 1, 768)           # Sentence embedding

# Timesteps (t ∈ [0, 1])
timesteps = torch.tensor([0.5, 0.5])                   # Shape: (2,)

# Temporal masks
x_mask_temporal = torch.ones(batch_size, seq_length, dtype=torch.bool)
ctxt_mask_temporal = torch.ones(batch_size, 77, dtype=torch.bool)
```

### Call predict_flow()
```python
with torch.no_grad():
    prediction = bundle.predict_flow(
        x_input=x_t,                           # (2, 360, 135)
        ctxt_input=ctxt_input,                 # (2, 77, 4096)
        vtxt_input=vtxt_input,                 # (2, 1, 768)
        timesteps=timesteps,                   # (2,)
        x_mask_temporal=x_mask_temporal,       # (2, 360)
        ctxt_mask_temporal=ctxt_mask_temporal, # (2, 77)
    )

print(f"prediction.shape = {prediction.shape}")
# Output: prediction.shape = torch.Size([2, 360, 135])
```

### Interpret Based on pred_type
```python
if bundle.pred_type == 'velocity':
    # prediction is already velocity (dx/dt)
    # Can use directly in ODE
    print(f"Output is velocity: dx/dt")
    print(f"Typical range: [-1, 1]")
    velocity = prediction
    
elif bundle.pred_type == 'x1':
    # prediction is clean motion x1
    # Must convert to velocity
    print(f"Output is predicted clean motion x1")
    t_val = timesteps.view(-1, 1, 1)  # (2, 1, 1)
    t_eps = 0.05
    velocity = (prediction - x_t) / (1.0 - t_val).clamp_min(t_eps)
```

---

## Example 2: Using predict_flow in ODE Loop (T2M)

### Full inference pipeline with ODE integration
```python
from torchdiffeq import odeint

# Parameters
bundle = ...  # loaded HyMotionT2MBundle
batch_size = 1
seq_length = 360
motion_dim = 135
num_ode_steps = 50
text_guidance_scale = 5.0

# Prepare text
text = ["a person is walking"]
text_feats = bundle.encode_text(text)
vtxt_input = text_feats['text_vec_raw']       # (1, 1, 768)
ctxt_input = text_feats['text_ctxt_raw']      # (1, L_c, 4096)

# Setup masks
device = next(bundle.motion_transformer.parameters()).device
tgt_padding_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
ctxt_mask_temporal = torch.ones(batch_size, ctxt_input.shape[1], device=device, dtype=torch.bool)

# Initial noise
y0 = torch.randn(batch_size, seq_length, motion_dim, device=device)
t_schedule = torch.linspace(0, 1, num_ode_steps + 1, device=device)

# ODE function using predict_flow
@torch.no_grad()
def ode_fn(t_val, x):
    """Returns dx/dt at timestep t_val."""
    
    # For CFG: prepare double batch [uncond, cond]
    x_double = torch.cat([x, x], dim=0)
    
    # Prepare null embeddings
    null_vtxt = bundle.null_vtxt_feat.expand_as(vtxt_input)
    vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)  # (2, 1, 768)
    ctxt_cfg = torch.cat([ctxt_input, ctxt_input], dim=0)  # (2, L_c, 4096)
    
    # Call predict_flow
    x_pred = bundle.predict_flow(
        x_input=x_double,  # (2, 360, 135)
        ctxt_input=ctxt_cfg,
        vtxt_input=vtxt_cfg,
        timesteps=t_val.expand(2 * batch_size),
        x_mask_temporal=tgt_padding_mask.repeat(2, 1),
        ctxt_mask_temporal=ctxt_mask_temporal.repeat(2, 1),
    )
    
    # Handle pred_type conversion
    if bundle.pred_type == 'x1':
        t_eps = 0.05
        x_pred = (x_pred - x_double) / (1.0 - t_val).clamp_min(t_eps)
    
    # Apply CFG
    pred_uncond, pred_text = x_pred.chunk(2, dim=0)
    x_pred = pred_uncond + text_guidance_scale * (pred_text - pred_uncond)
    
    return x_pred  # Returns dx/dt

# Run ODE
trajectory = odeint(ode_fn, y0, t_schedule, method='midpoint')
final_motion_noisy = trajectory[-1]  # (1, 360, 135)

# Denormalize
final_motion = bundle.decode_motion_from_latent(final_motion_noisy)
rot6d = final_motion['rot6d']      # (1, 360, 22, 6)
transl = final_motion['transl']    # (1, 360, 3)
keypoints = final_motion['keypoints3d']  # (1, 360, 24, 3) or None
```

---

## Example 3: M2M Inference with VACE Conditioning

### Setup M2M bundle
```python
from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle

bundle = HyMotionM2MBundle(
    motion_transformer=dict(...),
    pred_type='velocity',
    vace_condition_mode='split_reactive',  # or 'split_inactive_reactive'
    ...
)
bundle.eval()
```

### Prepare M2M VACE input
```python
# Known motion region (to be preserved)
known_motion = torch.randn(batch_size, seq_length, 135)

# Mask: 0 = keep, 1 = generate
src_mask = torch.zeros(batch_size, seq_length, 135)
src_mask[:, 50:300, :] = 1.0  # Generate frames 50-300

# Normalize motion
motion_norm = bundle.normalize_motion(known_motion)

# Zero out mask=1 regions (for Completion mode)
src_motion = motion_norm * (1 - src_mask)

# Prepare VACE context
vace_context = bundle.prepare_vace_input(
    src_motion=src_motion,        # (B, L, 135)
    src_mask=src_mask,            # (B, L, 135)
    ref_pose=None,                # Optional reference pose
)
print(f"vace_context.shape = {vace_context.shape}")
# Output: (B, L, 405) for split_reactive mode
# = (B, L, 270) if split_inactive_reactive

# Final x_input = x_t + vace_context
x_t = torch.randn(batch_size, seq_length, 135)
x_input = torch.cat([x_t, vace_context], dim=-1)  # (B, L, 540)
```

### Call predict_flow for M2M
```python
with torch.no_grad():
    prediction = bundle.predict_flow(
        x_input=x_input,                      # (B, L, 540) for split_reactive
        ctxt_input=ctxt_input,                # (B, L_c, 4096)
        vtxt_input=vtxt_input,                # (B, 1, 768)
        timesteps=timesteps,                  # (B,)
        x_mask_temporal=x_mask_temporal,      # (B, L)
        ctxt_mask_temporal=ctxt_mask_temporal, # (B, L_c)
        mask_density=mask_density,            # Optional for CRFM
        task_emb=None,                        # Optional task instruction
    )

print(f"prediction.shape = {prediction.shape}")
# Output: prediction.shape = torch.Size([B, L, 135])
# Note: output is ALWAYS 135-dim (motion_dim), not including VACE dims
```

---

## Example 4: Checking pred_type at Runtime

### Quick diagnostic
```python
# Load checkpoint
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
bundle = HyMotionT2MBundle(...)
bundle.load_state_dict(checkpoint['model'])

# Check pred_type
print(f"bundle.pred_type = {bundle.pred_type!r}")

# This determines how to interpret predict_flow output
if bundle.pred_type == 'velocity':
    print("✅ Output will be velocity (dx/dt)")
    print("   → Can use directly in ODE integration")
elif bundle.pred_type == 'x1':
    print("✅ Output will be clean motion (x1)")
    print("   → Must convert to velocity: (x1 - x_t) / (1 - t)")
else:
    raise ValueError(f"Unknown pred_type: {bundle.pred_type}")
```

---

## Example 5: Debugging predict_flow Output

### Check output statistics
```python
with torch.no_grad():
    output = bundle.predict_flow(x_t, ctxt, vtxt, t, ...)

print("=== Output Statistics ===")
print(f"Shape: {output.shape}")
print(f"Mean: {output.mean().item():.6f}")
print(f"Std: {output.std().item():.6f}")
print(f"Min: {output.min().item():.6f}")
print(f"Max: {output.max().item():.6f}")

# Typical values by pred_type:
if bundle.pred_type == 'velocity':
    # Velocity should be roughly symmetric around 0
    # Range typically [-1, 1] or so
    print(f"\nExpected for 'velocity':")
    print(f"  Mean ≈ 0 (symmetric)")
    print(f"  Std ≈ 0.1-0.5")
    print(f"  Range ≈ [-1.5, 1.5]")
    
elif bundle.pred_type == 'x1':
    # x1 should match the motion space range
    # Similar to normalized motion
    print(f"\nExpected for 'x1':")
    print(f"  Mean ≈ 0")
    print(f"  Std ≈ 0.5-1.0")
    print(f"  Range ≈ [-3, 3]")
```

---

## Example 6: Multi-GPU CFG with predict_flow

### Handling classifier-free guidance across GPUs
```python
@torch.no_grad()
def forward_with_cfg(bundle, batch, guidance_scale=5.0):
    x_t = batch['x_t']  # (B, L, 135)
    B, L, D = x_t.shape
    
    # Prepare text (both conditional and unconditional)
    text = batch['text']
    text_feats = bundle.encode_text(text)
    
    # Stack for CFG: [null, real]
    null_vtxt = bundle.null_vtxt_feat.expand(B, -1, -1)  # (B, 1, 768)
    vtxt_cfg = torch.cat([null_vtxt, text_feats['text_vec_raw']], dim=0)  # (2B, 1, 768)
    
    null_ctxt = bundle.null_ctxt_input.expand(B, text_feats['text_ctxt_raw'].shape[1], -1)
    ctxt_cfg = torch.cat([null_ctxt, text_feats['text_ctxt_raw']], dim=0)
    
    # Replicate x_t for CFG
    x_t_double = torch.cat([x_t, x_t], dim=0)  # (2B, L, D)
    
    # Single forward pass for both
    pred = bundle.predict_flow(
        x_input=x_t_double,
        ctxt_input=ctxt_cfg,
        vtxt_input=vtxt_cfg,
        timesteps=torch.tensor([0.5] * (2*B), device=x_t.device),
        ...
    )
    
    # Split and apply CFG
    pred_null, pred_real = pred.chunk(2, dim=0)
    pred_cfg = pred_null + guidance_scale * (pred_real - pred_null)
    
    return pred_cfg
```

---

## Example 7: Trajectory Extraction for Visualization

### Getting motion from start to finish
```python
# ODE integration (as in Example 2)
trajectory = odeint(ode_fn, y0, t_schedule, method='midpoint')

# Extract final prediction
final_latent = trajectory[-1]  # (B, L, 135)

# Decode to motion space
result = bundle.decode_motion_from_latent(final_latent)

# Extract components
rot6d = result['rot6d']           # (B, L, 22, 6) — local rotations
transl = result['transl']         # (B, L, 3) — absolute translation
keypoints3d = result['keypoints3d']  # (B, L, 24, 3) — 3D keypoints via FK

# Optional: save intermediate trajectory
# (useful for debugging)
trajectories = trajectory.cpu().numpy()
np.save('ode_trajectory.npy', trajectories)
print(f"Full ODE trajectory shape: {trajectories.shape}")
# Shape: (num_steps+1, B, L, 135)
```

---

## Key Takeaways for predict_flow() Usage

1. **Check `bundle.pred_type` first** — this determines how to interpret the output
2. **Output is ALWAYS shape `(B, L, 135)`** — for both T2M and M2M
3. **T2M uses simpler input** — just `x_t`, no VACE context
4. **M2M concatenates VACE** — `x_input = [x_t, vace_context]`
5. **Interpretation depends on pred_type**:
   - `'velocity'` → use directly as `dx/dt`
   - `'x1'` → convert to velocity: `(x1 - x_t) / (1-t)`
6. **Always use in a `torch.no_grad()` context** during inference
7. **CFG requires stacking** — call once for both null and real, then split and blend


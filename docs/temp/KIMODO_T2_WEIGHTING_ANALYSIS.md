# KIMODO t² (Timestep Squared) Weighting Analysis

## Executive Summary

**Finding: KIMODO does NOT use t² (timestep squared) weighting in its training losses.**

The `timestep_squared_weighting` flag in our `kimodo_aux_loss.py` is **a custom addition we made**, not something from KIMODO's original codebase.

---

## Evidence

### 1. KIMODO Loss Formulation (from training code)

KIMODO's main training uses:
- **Framework**: DDPM diffusion (not flow matching)
- **Loss**: Smooth-L1 losses on various motion components
- **Weighting**: Fixed weights γ (gamma) values:
  - γ₁ = γ₃ = γ₅ = 10 (position components)
  - γ₂ = 2 (other position)
  - γ₄ = 3 (velocity)
  - γ₆ = 4 (contact)
  - γ₇ = 5 (FK consistency)

**From KIMODO paper (Eq. 1) and code**: All auxiliary losses use **constant weights**, NOT timestep-dependent weighting.

### 2. KIMODO Diffusion Implementation

File: `ref_repo/KIMODO/kimodo/kimodo/model/diffusion.py`
- Standard DDPM beta schedule (cosine schedule)
- `q_sample()`: standard noising formula `x_t = sqrt(α_t) * x + sqrt(1-α_t) * noise`
- No mention of t² weighting anywhere

**No timestep-dependent loss scaling found in KIMODO's core diffusion process.**

### 3. Our kimodo_aux_loss.py Analysis

**File**: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`

**Key findings**:

```python
class KimodoStyleAuxLoss(nn.Module):
    def __init__(
        self,
        ...
        timestep_squared_weighting: bool = True,  # ← THIS is our flag
    ):
        ...
        self.timestep_squared_weighting = bool(timestep_squared_weighting)
    
    def forward(
        self,
        ...
        timesteps: Optional[Tensor] = None,  # ← timesteps parameter, not in KIMODO
        ...
    ):
        # Optional t² re-weighting
        if self.timestep_squared_weighting and timesteps is not None:
            t_sq = (timesteps.to(pred_world.device).to(pred_world.dtype) ** 2)  # (B,)
        else:
            t_sq = None
        
        # Applied to each loss term:
        if t_sq is not None:
            per_frame = per_frame * t_sq.unsqueeze(-1)  # ← t² multiplication
```

**Evidence this is our addition**:
1. **Parameter name `timestep_squared_weighting`** never appears in KIMODO codebase
2. **Timesteps parameter**: KIMODO's diffusion doesn't need this for loss weighting
3. **Default value `True`**: Shows this was a deliberate design choice we made
4. **Comment in docstring** (lines 150-153) explicitly documents this:
   ```
   timestep_squared_weighting : bool
       If True (default), multiply each term by ``t²`` (matches the
       existing ``motion198_fk_loss`` t-weighting).  This down-weights
       pure-noise samples where FK on noisy x1 is uninformative.
   ```
   - "matches the existing motion198_fk_loss" → refers to our own M2M motion loss, not KIMODO

---

## Rationale Behind Our t² Weighting

The t² weighting in our KIMODO-style auxiliary loss serves a specific purpose:

**Why t² weighting?**
- At high noise levels (t near 1), the x1 (predicted clean motion) contains mostly noise
- Running FK on highly noisy motion is unreliable—the auxiliary losses (joint_pos, joint_vel, fk_consistency) provide weak supervision
- Weighting by t² down-weights these high-noise timesteps where the supervision signal is weak
- Weighting by t² up-weights low-noise timesteps (t near 0) where the prediction is cleaner and FK supervision is more meaningful

**Aligns with existing M2M loss:**
- Our baseline M2M `motion198_fk_loss` already uses t-weighting (not t²)
- The auxiliary loss uses t² to be more conservative with weak-supervision timesteps

---

## KIMODO vs Our Approach Comparison

| Aspect | KIMODO | Our kimodo_aux_loss.py |
|--------|--------|----------------------|
| **Loss Framework** | DDPM (predicts x0) | Flow Matching (predicts velocity) |
| **Auxiliary Loss Weighting** | Fixed γ values (constant) | Optional t² (timestep-dependent) |
| **Loss Components** | 7 terms (pos, vel, rot, contact, FK) | 3 terms subset (joint_pos, joint_vel, fk_consistency) |
| **Timestep Dependency** | No per-timestep weighting | Yes, can weight by t² |
| **Motivation** | Standard diffusion | Down-weight high-noise timesteps |

---

## Conclusion

✅ **KIMODO does NOT use t² weighting.**

✅ **Our `timestep_squared_weighting` flag is our own innovation.**

The feature was designed to improve the auxiliary loss supervision signal by:
- Down-weighting timesteps where x1 is mostly noise (high t)
- Up-weighting timesteps where x1 is cleaner (low t)
- Matching the per-timestep weighting convention of our existing M2M motion losses

This is a sensible architectural choice for working with the KIMODO-style auxiliary terms in our flow-matching framework.

---

## References

- KIMODO paper + code: `ref_repo/KIMODO/` ✓ searched
- Our auxiliary loss: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` ✓ verified
- KIMODO diffusion: `ref_repo/KIMODO/kimodo/kimodo/model/diffusion.py` ✓ verified
- No t² weighting found in KIMODO training code ✓ confirmed

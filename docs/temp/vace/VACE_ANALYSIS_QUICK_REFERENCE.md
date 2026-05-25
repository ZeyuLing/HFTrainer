# PRISM VACE Analysis - Quick Reference

## TL;DR

**VACE channel mismatch is NOT the problem.** PRISM doesn't use VACE channels at all. Both training and inference pass exactly 16 latent channels with no concatenation.

---

## Three Key Code Locations Verified

### 1. Training (prism_trainer.py:87-93)
```python
model_pred = self.bundle.transformer(
    hidden_states=noisy_latents,  # ← [B, 16, T_latent, J]
    encoder_hidden_states=text_states,
    timestep=timesteps,
    hidden_states_mask=padding_mask,
    encoder_hidden_states_mask=None,
).float()
```
✅ **16 channels only, no VACE**

### 2. Inference (prism_backend.py:420-427)
```python
noise_pred = current_model(
    hidden_states=latent_model_input,  # ← [B, 16, T_latent, J]
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    attention_kwargs=attention_kwargs,
    is_causal=self.config.is_causal,
    hidden_states_mask=motion_mask,
)
```
✅ **16 channels only, no VACE**

### 3. Config (prism_1b_tp2m_1frame.py:31)
```python
in_channels=16,  # ← Transformer expects 16
```
✅ **Config matches: 16 = 16**

---

## Training vs Inference Comparison

| Component | Training | Inference | Status |
|-----------|----------|-----------|--------|
| Input to transformer | `noisy_latents` [B,16,T,J] | `latent_model_input` [B,16,T,J] | ✓ Match |
| VACE channels | None | None | ✓ No mismatch |
| Config in_channels | 16 | 16 | ✓ Match |
| Concatenation | No | No | ✓ No mismatch |

---

## Where VACE Actually Exists

VACE is **exclusive to HyMotion M2M models**:

- `hymotion_m2m_trainer.py` (line 276): `x_input = torch.cat([x_t, vace_context])`
- `hymotion_m2m_soar_trainer.py` (line 228): `z_re_input = torch.cat([z_re, vace_context])`
- `hymotion_m2m_pipeline.py` (line 250): `x_input = torch.cat([x, vace_context])`

**PRISM has zero VACE references.**

---

## Real Issues to Debug Instead

Since VACE is ruled out, focus on:

1. **Timestep mismatch** → Train uses random [0,1000], inference uses sparse 10-step schedule
2. **Sigma lookup precision** → Float32/BF16 conversion errors
3. **Frame mask inconsistency** → `condition_frame_mask_vae` vs `first_frame_mask` shape differences
4. **Input distribution shift** → Model trained on 10% conditioning, inference uses 100% first-frame

---

## Debug Commands

### Verify no VACE in PRISM
```bash
grep -r "vace\|VACE" hftrainer/trainers/motion/prism_trainer.py
grep -r "vace\|VACE" hftrainer/pipelines/motion/prism_backend.py
```
Expected: No results

### Compare channel dimensions
```bash
# In training/inference logs:
print(f"hidden_states shape: {hidden_states.shape}")  # Should be [B, 16, T, J]
print(f"in_channels: {transformer.config.in_channels}")  # Should be 16
```

### Test without per-token expansion
```python
# Isolate per-token logic as culprit:
pipe = PrismARPipeline(..., expand_timesteps=False)
```

---

## Files to Check Next

**For timestep issues:**
- `hftrainer/models/motion/prism/bundle.py` (create_sequence_ts, add_flow_noise)
- `hftrainer/pipelines/motion/prism_backend.py` (generate_single_segment, timestep handling)

**For frame masking:**
- `hftrainer/models/motion/prism/bundle.py` (create_condition_mask)
- `hftrainer/pipelines/motion/prism_backend.py` (prepare_latents, first_frame_mask)

**For latent normalization:**
- `hftrainer/pipelines/motion/prism_backend.py` (latents_mean, latents_std)

---

## Key Evidence: Zero VACE Concatenation

```bash
$ grep -rn "torch.cat.*vace\|concat.*vace" hftrainer/trainers/motion/prism_trainer.py
[EMPTY - no results]

$ grep -rn "torch.cat.*vace\|concat.*vace" hftrainer/pipelines/motion/prism_backend.py
[EMPTY - no results]

$ grep -rn "torch.cat.*hidden\|concat.*hidden" hftrainer/trainers/motion/prism_trainer.py
[EMPTY - no results]

$ grep -rn "torch.cat.*hidden\|concat.*hidden" hftrainer/pipelines/motion/prism_backend.py
[EMPTY - no results]
```

---

## Conclusion

✅ VACE channel mismatch is **definitively ruled out**
→ Focus investigation on timestep handling and input distribution

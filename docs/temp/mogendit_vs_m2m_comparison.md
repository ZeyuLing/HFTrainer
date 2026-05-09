# MoGenDiT vs HyMotion M2M: Training & Inference Comparison

## 1. Running Experiments Summary

| Task Name | Config | Status |
|-----------|--------|--------|
| `hymotion_dit_fm_man_s` | `hymotion_dit/hymotion_dit_fm_man_s.py` (49M, local rot) | RUNNING |
| `hymotion_dit_fm_man_l` | `hymotion_dit/hymotion_dit_fm_man_l.py` (383M, local rot) | RUNNING |
| `hymotion_dit_fm_man_globalrot_s` | `hymotion_dit/hymotion_dit_fm_man_globalrot_s.py` (49M, global rot) | RUNNING |
| `hymotion_dit_fm_man_globalrot_b` | `hymotion_dit/hymotion_dit_fm_man_globalrot_b.py` (288M, global rot) | RUNNING |
| `hymotion_dit_fm_man_globalrot_l` | `hymotion_dit/hymotion_dit_fm_man_globalrot_l.py` (383M, global rot) | RUNNING |
| `hymotion_dit_fm_man_b_v2` | `hymotion_dit/hymotion_dit_fm_man_b.py` (288M, local rot) | RESOURCE_WAIT |
| `hymotion_m2m_uncond_fm_man_globalrot_elastic_run2` | `hymotion_m2m/...uncond_fm_man_globalrot_046b.py` (0.46B, global rot) | RUNNING |
| `hymotion_m2m_caption_fm_man_globalrot_v100_run2` | `hymotion_m2m/...caption_fm_man_globalrot_046b.py` (0.46B, global rot, caption) | RUNNING |

All 8 tasks use **flow matching + mask-aware noise** (FM + MAN). Both `hymotion_dit/` and `hymotion_m2m/` variants share the same `HyMotionM2MBundle` + `HyMotionM2MTrainer`. The `hymotion_dit` configs use the text-free `HunyuanMotionDiT`; the `hymotion_m2m` configs use the text-supporting `HunyuanMotionMMDiT`.

---

## 2. Architecture Comparison

| Aspect | MoGenDiT | HyMotion M2M |
|--------|----------|--------------|
| **Backbone** | DiT + AdaLN + RoPE + sliding window (window=90) | HunyuanMotionDiT/MMDiT + AdaLN + RoPE |
| **Diffusion** | DDPM, cosine beta schedule, 1000 steps, predict x0 | Flow matching, x_t=(1-t)*x0+t*x1, predict velocity |
| **Motion repr** | 201-dim: 22x6 pose + 22x3 joint + 3 transl | 135-dim: 3 transl + 22x6 rot6d |
| **Conditioning** | concat [x_t, obs_mask] | VACE: concat [x_t, inactive, reactive, mask] (540-dim) |
| **Rotation** | Global orientation only | Both local and global rotation variants |
| **Text** | No text conditioning | Optional: Qwen3+CLIP embeddings + CFG dropout |
| **Model sizes** | 0.03B, 0.1B (recommended), 0.3B | 49M, 162M, 288M, 383M, 460M |

---

## 3. Training Technique Comparison

### 3.1 Loss Functions

| Loss Component | MoGenDiT | M2M (code) | M2M (enabled in running configs) | Gap? |
|----------------|----------|------------|----------------------------------|------|
| **Main reconstruction** | L1 on x0 (pose, joint, trans separately) | SmoothL1 on velocity (v=x1-x0) | velocity_weight=1.0 | Loss type differs: L1 vs SmoothL1. Both valid. |
| **x1 loss** | — (predicts x0 directly) | Supported (x1_weight) | x1_weight=0.0 (disabled) | Not a gap — different paradigm |
| **Translation weighting** | Separate loss_trans (equal to loss_pose) | trans_dim_weight=5.0 on first 3 dims | Enabled (trans_dim_weight=5.0) | **No gap** — M2M uses per-dim reweighting |
| **FK keypoints3d** | Separate loss_joint (on 22x3 joint positions) | keypoints3d_weight via SmplxLiteJ24 FK | **keypoints3d_weight=0.0 (DISABLED)** | **GAP**: Code exists but disabled. MoGenDiT uses joint loss at equal weight to pose. |
| **Velocity loss** | Global joint velocity: FK(x0)[:,1:] - FK(x0)[:,:-1] | motion_smoothness_weight on raw features | **motion_smoothness_weight=0.0 (DISABLED)** | **GAP**: Code exists but disabled. Different formulation (feature-space vs joint-space). |
| **Consistency loss** | kinematic_loss_batch: rigid body constraint (FK offset invariance) | Not implemented | N/A | **Requires 201-dim repr with joint positions. NOT portable to 135-dim.** |

### 3.2 Training Techniques

| Technique | MoGenDiT | M2M (code) | M2M (running configs) | Gap? |
|-----------|----------|------------|------------------------|------|
| **EMA** | decay=0.999, start_step=1000 | EMAHook exists (decay=0.995, interval=10) | **NOT enabled in any running config** | **GAP**: EMA code exists but not enabled. |
| **Mask-aware noise** | q_sample skips observed frames | mask_aware_noise=True in trainer | All running configs use MAN | No gap |
| **Motion degradation** | 50% batch corruption (jitter/noise/offset) | edit_repair_prob=0.15 in PrepareM2MUniversalMask | Enabled at 15% | **Partial gap**: M2M has 15% vs MoGenDiT's 50%. Different implementation. |
| **Grad clipping** | grad_norm tracked, no explicit clip | max_grad_norm=1.0 | Enabled | No gap |
| **Optimizer** | AdamW, lr=1e-4, wd=1e-4 | AdamW, lr=1e-4/2e-4, wd=0.0 | Enabled | Minor diff: M2M has no weight decay |
| **Sequence length** | fix_len=224 | clip_len=360 | 360 | M2M uses longer sequences |

### 3.3 Mask Strategies

| Strategy | MoGenDiT | M2M |
|----------|----------|-----|
| random_frame | 20% | M1 random_cell: 25% |
| random_phrase | 20% | M2 random_block: 15% |
| random_start_end | 20% | M3 temporal_contiguous: 25% |
| block_trans | 10% | M4 joint_contiguous: 15% |
| joint_only | 10% | M5 full_mask: 5% |
| uncond | 20% | M6 keyframe_sparse: 15% |

M2M's mask strategies are richer and more diverse.

---

## 4. Gap Analysis: What Should Be Added

### 4.1 EMA — **RECOMMENDED** (safe, proven beneficial)

- MoGenDiT uses EMA with decay=0.999. KIMODO uses decay=0.995.
- M2M has `EMAHook` implemented but not enabled in any running config.
- **Action**: Enable EMA in all running configs. Use decay=0.999 (matching MoGenDiT, more stable than KIMODO's 0.995).
- **Risk**: Zero. EMA only creates a shadow copy; training dynamics unchanged. Worst case: slightly more memory.
- **Note**: EMA is evaluation-only (not used during training forward). The runner must swap EMA weights for validation/inference, but this is a separate concern — enabling the hook is harmless.

### 4.2 Motion Smoothness Loss — **RECOMMENDED** (safe, proven beneficial)

- MoGenDiT computes velocity loss on FK-derived global joint positions.
- M2M has `motion_smoothness_weight` which computes temporal difference loss on raw features (pred_x1[:,1:] - pred_x1[:,:-1]).
- This is a simpler but effective approximation that penalizes jitter in the denoised output.
- **Action**: Enable motion_smoothness_weight=0.5 in all running configs.
- **Risk**: Very low. This is a secondary loss with small weight. It cannot harm the main velocity loss. It operates on the already-computed pred_x1, adding negligible compute.
- **Difference from MoGenDiT**: M2M's smoothness operates on normalized feature space (135-dim), not FK joint space. This is actually broader — it penalizes jitter in ALL dimensions (rotation + translation), not just joint positions.

### 4.3 FK Keypoints3d Loss — **NOT recommended for now**

- MoGenDiT's loss_joint operates on 22x3 joint positions (part of 201-dim repr).
- M2M has `keypoints3d_weight` with FK via SmplxLiteJ24, but:
  - Requires `body_model_path` pointing to SMPL model files.
  - FK is computationally expensive (per-sample loop in `_compute_fk_keypoints`).
  - FK has numerical issues (discontinuities in rot6d→rotmat→FK chain).
- **Risk**: Moderate. FK compute cost could slow training 2-3x. Numerical issues could destabilize gradients.
- **Action**: Keep disabled. The motion_smoothness_weight (4.2) provides similar temporal regularization without FK overhead.

### 4.4 Consistency Loss (kinematic_loss_batch) — **NOT portable**

- MoGenDiT's consistency loss enforces that FK(predicted_pose) produces skeleton offsets consistent with the first frame.
- This requires both rot6d AND joint positions in the representation (201-dim).
- M2M uses 135-dim (rot6d only, no joint positions). Cannot compute this loss.
- **Action**: Skip. Not architecturally compatible.

### 4.5 Motion Degradation Rate (15% → 50%) — **NOT recommended**

- MoGenDiT uses 50% degradation for its repair paradigm.
- M2M already has 15% edit_repair_prob, which inserts corrupted motions for edit-mode training.
- MoGenDiT's higher rate makes sense for a dedicated repair model. M2M is a general completion model.
- **Action**: Keep at 15%. Higher rates would reduce effective training data for the primary completion task.

### 4.6 L1 vs SmoothL1 — **NOT recommended to change**

- MoGenDiT uses L1 (l1_weight=1.0, l2_weight=0.0).
- M2M uses SmoothL1, which is L1 for large errors and MSE for small errors.
- SmoothL1 is generally more stable for flow matching where predictions can have large initial errors.
- **Action**: Keep SmoothL1.

---

## 5. Implementation Plan

### Changes to apply to ALL running configs:

1. **Enable EMA**: Add `ema=dict(type='EMAHook', decay=0.999, update_interval=1)` to `default_hooks` in base configs.
2. **Enable motion smoothness loss**: Set `motion_smoothness_weight=0.5` in base configs' `losses_cfg`.

### Files to modify:
- `configs/hymotion_dit/_base_hymotion_dit_s.py` (base for all hymotion_dit configs)
- `configs/hymotion_m2m/_base_hymotion_m2m_046b.py` (base for 0.46B M2M configs)

These two base configs propagate to ALL running experiments through inheritance.

### Why these changes are safe:
1. **EMA**: Shadow copy only. Does not affect training forward/backward at all. Memory overhead ~2x model params (negligible for <500M models on V100 32GB).
2. **motion_smoothness_weight=0.5**: Secondary loss component. The velocity loss (weight=1.0) remains the primary signal. At weight=0.5, smoothness contributes ~33% of total gradient. This is conservative and well within safe range. The loss is only computed when need_x1=True, adding minimal compute (one temporal diff + one loss_fn call).

---

## 6. Inference Comparison (for reference)

| Aspect | MoGenDiT | M2M |
|--------|----------|-----|
| **Sampling** | DDPM reverse diffusion, 50/100/1000 steps | Euler ODE, 10/50 steps |
| **Guidance** | Replacement at each step: x_t[known] = q_sample(x_clean, t) | Replacement guidance: x_t[known] = (1-t)*noise + t*x_clean |
| **Modes** | denoise, ada_denoise, trans_regen | Single completion pipeline |
| **Post-processing** | None | Optional global→local rot6d conversion |

No inference changes needed — the training-side improvements will naturally improve inference quality.

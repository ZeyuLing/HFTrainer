# Physics Simulation + SOAR Training Integration

**Date:** 2026-05-18  
**Status:** Architecture Design Complete  
**Scope:** How physics-corrected motion enhances SOAR post-training  
**Synergy:** Exposure bias correction + Physical plausibility

---

## Executive Summary

SOAR's core innovation is **correcting for exposure bias** — training uses off-policy rollouts to match actual inference behavior. Physics simulation complements this by providing **physically plausible correction targets** that the model learns to predict.

### The Opportunity

**Standard SOAR:** Uses clean target motion as correction signal
- ✅ Reduces train/test mismatch
- ✅ Exposes bias in diffusion rollout
- ❌ May learn physically impossible corrections

**SOAR + Physics:** Uses physics-corrected motion as correction signal
- ✅ Reduces train/test mismatch (SOAR)
- ✅ Enforces physical plausibility (Physics)
- ✅ Model learns realistic embodied motion
- ✅ Naturally handles ground contact, gravity

---

## Part 1: SOAR Correction Flow (Baseline)

### Standard SOAR Algorithm

```python
# Training iteration (standard SOAR)
for batch in dataloader:
    x0_clean, caption = batch
    
    # [1] Base supervised loss (standard training)
    x1_noise = randn_like(x0_clean)
    t ~ U[0, 1]
    x_t = (1-t) * x0_clean + t * x1_noise
    v_pred = model(x_t, caption, t)
    v_gt = x1_noise - x0_clean
    L_base = ||v_pred - v_gt||²
    
    # [2] SOAR correction loss (off-policy)
    with torch.no_grad():
        # Rollout: one ODE step with current model
        v_rollout = model(x_t, caption, t)
        x_hat = x_t + dt * v_rollout  # Off-trajectory
        
        # Re-noise: sample intermediate timesteps
        for n_aux in range(N):
            t_prime = U[t_hat, 1]
            z_re = (1 - t_prime) * x_hat + t_prime * x1_noise
            
            # [CRITICAL] Correction target: clean motion
            v_corr_target = (x0_clean - z_re) / (1 - t_prime)
    
    # On-policy forward: predict correction
    v_pred_corr = model(z_re, caption, t_prime)
    L_soar = ||v_pred_corr - v_corr_target||²
    
    # Total loss
    L = L_base + λ * L_soar
```

### Problem: Correction Target Always "Clean"

The vector `v_corr_target = (x0_clean - z_re) / (1 - t_prime)` always points toward the clean reference motion, regardless of whether it's **physically realizable**.

Example: Humanoid motion
- **Clean motion:** High knee raise, foot slightly floating (motion capture)
- **Correction target:** Model learns to predict toward floating foot
- **Inference:** Model predicts floating foot, simulation falls or clip interpenetrates

---

## Part 2: Physics-Enhanced SOAR

### Key Insight: Replace Correction Target

Instead of always using `x0_clean`, use **physics-corrected reference** as the correction signal:

```python
# Physics-enhanced SOAR
for batch in dataloader:
    x0_clean, caption = batch
    x0_physics = physics_corrected_motion(x0_clean)  # <-- NEW
    
    # Base loss: unchanged
    ...
    L_base = ||v_pred - v_gt||²
    
    # SOAR correction: replace target
    with torch.no_grad():
        v_rollout = model(x_t, caption, t)
        x_hat = x_t + dt * v_rollout
        
        for n_aux in range(N):
            t_prime = U[t_hat, 1]
            z_re = (1 - t_prime) * x_hat + t_prime * x1_noise
            
            # [MODIFIED] Correction target: physics-corrected
            v_corr_target = (x0_physics - z_re) / (1 - t_prime)  # <-- PHYSICS
    
    v_pred_corr = model(z_re, caption, t_prime)
    L_soar = ||v_pred_corr - v_corr_target||²
    
    L = L_base + λ * L_soar
```

### Advantages

| Aspect | Standard SOAR | Physics SOAR |
|--------|---|---|
| **Correction target** | Clean (may be unphysical) | Physics-constrained |
| **Model learns** | Off-trajectory prediction | Off-trajectory + physics |
| **Inference output** | May violate ground contact | Naturally ground-contacting |
| **Foot penetration** | Possible | Prevented by simulation |
| **Falls** | Possible | Unlikely |
| **Compute cost** | Low (just model forward) | Higher (MuJoCo sim + smoothing) |

---

## Part 3: Data Preparation Pipeline

### End-to-End Workflow

```
HyMotion M2M v2 SFT Checkpoint (pretrained)
  ↓
[Inference] Generate motion_135 on training caption set
  ├─ 10K captions × 1 sample each
  └─ Output: motion_135 NPZ files
  
[Conversion] motion_135 → SMPL (via scripts/embodied/motion135_to_smplx.py)
  ├─ Decode 6D rotations → axis-angle
  ├─ Ensure valid joint limits
  └─ Output: SMPL NPZ
  
[Physics Simulation] SMPL → Physics-corrected SMPL
  (via scripts/embodied/run_smpl_physics_sim.py)
  ├─ Y-up → Z-up conversion
  ├─ MuJoCo PD tracking simulation
  ├─ Post-smoothing (Savitzky-Golay)
  ├─ Z-up → Y-up conversion
  └─ Output: Physics-corrected motion_135 NPZ
  
[Dataset Creation] motion_135 (clean) + motion_135 (physics) → Training dataset
  ├─ x0_clean: Original SFT output
  ├─ x0_physics: Physics-corrected
  ├─ caption: Same as inference
  └─ Store both in single NPZ per sample
  
[SOAR Training] Train HyMotion M2M v2 with physics targets
  ├─ Base loss: v_pred → original clean motion
  ├─ SOAR loss: v_pred_correction → physics-corrected motion
  ├─ Hyperparameters: soar_lambda, soar_num_aux, soar_K, soar_sigma_clamp
  └─ Output: Physics-SOAR checkpoint
```

### Dataset Format

Training dataset per sample:
```python
# Single NPZ file per motion
data = {
    'motion_135_clean': (T, 135),      # Original SFT output
    'motion_135_physics': (T, 135),    # Physics-corrected version
    'caption': str,                    # Text description
    'duration': float,                 # Seconds
    'fps': int,                        # Frame rate
    'sim_stats': {                     # Physics simulation statistics
        'total_frames': int,
        'simulated_frames': int,
        'completed': bool,
        'joint_tracking_error_rad': float,
        'root_position_drift_m': float,
        ...
    }
}
```

### Loading in Training Loop

```python
class PhysicsSoarDataset:
    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True)
        
        # Decode both versions
        motion_clean = data['motion_135_clean']   # (T, 135)
        motion_physics = data['motion_135_physics']  # (T, 135)
        caption = str(data['caption'])
        
        # Convert to latent space (if using VAE encoder)
        x0_clean = self.encode(motion_clean)       # (T, latent_dim)
        x0_physics = self.encode(motion_physics)   # (T, latent_dim)
        
        # Tokenize caption
        caption_emb = self.tokenizer(caption)
        
        return {
            'x0_clean': x0_clean,
            'x0_physics': x0_physics,
            'caption': caption_emb,
        }
```

---

## Part 4: SOAR Trainer Modifications

### Current Implementation (Motion Standard)

```python
class HyMotionM2MSoarTrainer(HyMotionM2MTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.soar_lambda = cfg.soar_lambda          # Loss weight
        self.soar_num_aux = cfg.soar_num_aux        # Auxiliary points
        self.soar_K = cfg.soar_K                    # ODE steps
        self.soar_cfg_scale = cfg.soar_cfg_scale    # Classifier-free guidance
        self.soar_sigma_clamp = cfg.soar_sigma_clamp
    
    def train_step(self, batch):
        # [1] Base forward + loss (standard SFT)
        x0 = batch['motion_135']
        ctx = self._prepare_and_forward(batch)
        loss_base = self._compute_base_loss(ctx)
        
        # [2] SOAR correction
        loss_soar = self._soar_correction_loss(x0, ctx)
        
        # [3] Total
        loss = loss_base + self.soar_lambda * loss_soar
        return loss
```

### Modified for Physics-Enhanced SOAR

```python
class PhysicsSoarTrainer(HyMotionM2MSoarTrainer):
    """SOAR trainer with physics-corrected correction targets."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_physics = cfg.get('use_physics', True)
        self.physics_weight = cfg.get('physics_weight', 1.0)
    
    def train_step(self, batch):
        # [1] Base loss: standard SFT on clean motion
        x0_clean = batch['motion_135']
        ctx = self._prepare_and_forward(batch)
        loss_base = self._compute_base_loss(ctx)
        
        # [2] SOAR correction: Physics-enhanced target
        if self.use_physics and 'motion_135_physics' in batch:
            x0_target = batch['motion_135_physics']
        else:
            x0_target = x0_clean  # Fallback to standard SOAR
        
        loss_soar = self._soar_correction_loss(x0_target, ctx)
        
        # [3] Total
        loss = loss_base + self.soar_lambda * loss_soar
        return loss
    
    def _soar_correction_loss(self, x0_target, ctx):
        """Override: use x0_target instead of hardcoded x0_clean."""
        # [Existing implementation, but replace x0_clean → x0_target]
        ...
        v_corr = (x0_target - z_re) / (1 - t_prime)
        ...
```

---

## Part 5: Configuration for Physics SOAR

### Config Structure

```python
# configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_physics_soar.py

# Inherit base SOAR config
_base_ = ['hymotion_m2m_v2_uncond_local_046b_soar.py']

# Physics-specific settings
trainer_cfg = dict(
    type='PhysicsSoarTrainer',
    
    # SOAR hyperparameters (from base)
    soar_lambda=0.1,
    soar_num_aux=1,
    soar_K=50,
    soar_cfg_scale=1.0,
    soar_sigma_clamp=0.05,
    
    # Physics enhancement
    use_physics=True,
    physics_weight=1.0,  # Scale contribution if needed
)

# Data settings
data_cfg = dict(
    type='PhysicsSoarDataset',  # Custom loader
    path='/path/to/physics_corrected_motions',
    split='train',
    batch_size=14,
    num_workers=8,
    physics_enabled=True,  # Load both clean and physics versions
)

# Training settings (tuned for physics-enhanced learning)
optimizer = dict(
    type='AdamW',
    lr=1e-5,  # Even smaller for physics-aware learning
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

max_iters = 10000  # More iterations to learn physics-constrained targets
val_interval = 500
save_interval = 500

checkpoint = dict(
    load_from='epoch_485',  # Latest SFT checkpoint
)
```

---

## Part 6: Hyperparameter Recommendations

### Physics SOAR Hyperparameter Tuning

| Hyperparameter | Recommended | Rationale |
|---|---|---|
| `soar_lambda` | 0.15-0.2 | Physics targets are stricter; higher weight needed |
| `soar_num_aux` | 2-3 | More auxiliary points to sample physics manifold |
| `soar_K` | 50-100 | Longer horizon to enforce physics constraints |
| `soar_cfg_scale` | 0.5-1.0 | Physics is implicit; lower CFG may help |
| `soar_sigma_clamp` | 0.03-0.1 | Tighter noise clamping for deterministic physics |
| `lr` | 1e-5 | Half of standard SOAR (more careful learning) |
| `batch_size` | 8-12 | Smaller due to 2x data loading (clean + physics) |
| `max_iters` | 10K-20K | More iterations to learn correction manifold |
| `physics_weight` | 1.0 | Equal importance to base and SOAR losses |

### Ablation Study Plan

1. **Baseline:** Standard SOAR (physics_weight = 0)
2. **Partial physics:** physics_weight = 0.5
3. **Full physics:** physics_weight = 1.0
4. **Higher physics weight:** physics_weight = 1.5 (emphasize physics)
5. **Longer SOAR horizon:** soar_K = 100 (vs. default 50)

Compare: Downstream embodied generation quality, physics compliance, visual quality.

---

## Part 7: Evaluation Metrics

### Physics Compliance Metrics

After physics-SOAR training, evaluate on:

#### 1. **Joint Tracking Error** (during simulation)
```python
# During physics sim
joint_error = mean(|q_sim - q_ref|) over all frames
# Good: < 0.05 rad (3° average)
```

#### 2. **Ground Contact Integrity**
```python
# Check foot-ground distance at simulation time
foot_z = [left_foot_height, right_foot_height]
# Good: all > -0.01 m (feet on/above ground, no penetration)
```

#### 3. **Foot Sliding Distance**
```python
# Track foot X/Y position while in contact
# Calculate distance traveled horizontally during vertical stasis
sliding_dist = sum(||foot_pos[t+1] - foot_pos[t]||_xy)
# where foot is in contact (vertical velocity low)
# Good: < 0.1 m per step
```

#### 4. **Energy Efficiency** (optional)
```python
# Integral of torque-velocity product
# High energy = unrealistic, jerky motion
work = sum(tau * omega) over all joints and time
# Good: < baseline (SFT output)
```

### Generation Quality Metrics

#### 1. **FID (Fréchet Inception Distance)**
- Compare features of physics-trained model vs. baseline
- Measure mode coverage

#### 2. **User Study**
- Realism ranking: Physics-SOAR vs. Standard SOAR vs. SFT
- Embodiment appeal: How likely to use for embodied applications

#### 3. **Diversity in Physics Compliance**
- Generate 100 motions per caption
- Measure variance in joint_tracking_error
- Good: Low variance (consistent physics quality)

---

## Part 8: Integration with Existing Pipeline

### File Modifications

**1. Add physics dataset loader:**
```
hftrainer/
  └─ data/
      └─ embodied/
          └─ physics_soar_dataset.py  (NEW)
```

**2. Add physics SOAR trainer:**
```
hftrainer/
  └─ trainers/
      └─ motion/
          └─ hymotion_m2m_physics_soar_trainer.py  (NEW or extend existing)
```

**3. Add config:**
```
configs/
  └─ hymotion_m2m_v2/
      └─ soar/
          └─ hymotion_m2m_v2_physics_soar.py  (NEW)
```

**4. Add data preprocessing script:**
```
scripts/
  └─ embodied/
      └─ prepare_physics_dataset.py  (NEW)
          • Orchestrates inference → sim → dataset creation
```

### Launch Commands

```bash
# [1] Generate motions on training set
python3 scripts/embodied/inference_batch.py \
    --checkpoint checkpoints/epoch_485 \
    --captions data/train_captions.txt \
    --output-dir data/generated_motions \
    --batch-size 32

# [2] Apply physics simulation
python3 scripts/embodied/run_smpl_physics_sim.py \
    --npz-dir data/generated_motions \
    --output-dir data/physics_corrected \
    --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --stats-dir data/sim_stats

# [3] Create training dataset
python3 scripts/embodied/prepare_physics_dataset.py \
    --clean-dir data/generated_motions \
    --physics-dir data/physics_corrected \
    --captions data/train_captions.txt \
    --output-dir data/physics_soar_dataset

# [4] Train physics SOAR
python3 -m torch.distributed.launch --nproc_per_node=8 \
    hftrainer/train.py \
    --config configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_physics_soar.py \
    --exp-name physics_soar_v1 \
    --output-dir checkpoints/
```

---

## Part 9: Computational Cost Analysis

### Training Data Preparation

| Step | Time per Motion | 1000 Motions | 10K Motions |
|---|---|---|---|
| Inference (30s @ 30fps) | 0.5s | 8 min | 1.4 hrs |
| Conversion (motion135→SMPL) | 0.01s | 10s | 100s |
| Physics simulation (PD tracking) | 30-60s | 8-16 hrs | 80-160 hrs |
| Post-smoothing | 0.1s | 100s | 1000s |
| **Total (bottleneck: physics sim)** | **30-60s** | **8-16 hrs** | **80-160 hrs** |

**Speedup strategy:**
- Parallelize physics sim: 8 GPUs × 4 processes = 32× speedup
- 10K motions: 80-160 hours → 2.5-5 hours wall-clock
- Feasible for post-training refinement

### Training Compute

| Resource | Quantity | Time |
|---|---|---|
| GPU | 8× A100 40GB | 10K iterations |
| Batch size | 14 motions | Per step |
| Iterations | 10K | ~24 hours |
| Memory | 32 GB per GPU | Sufficient |

**Total:** ~24 GPU-hours = 3 hours wall-clock on 8 A100s

---

## Part 10: Expected Improvements

### Quantitative Expectations

Based on related work (physics-guided diffusion models):

- **FID:** -5% to -15% (narrower mode coverage, but higher quality)
- **Joint tracking error:** 0.05 → 0.04 rad (better physics compliance)
- **User preference:** +20-30% (physics SOAR preferred over standard SOAR)
- **Embodied success rate:** +15-25% (fewer falls in downstream simulation)

### Qualitative Expectations

**Standard SOAR output:**
- Smooth, natural motion
- May have subtle foot penetration
- Occasional unnatural joint configurations

**Physics SOAR output:**
- Smooth, natural motion
- Zero foot penetration (physics-enforced)
- Joint configurations respect limits
- Better suited for embodied applications

---

## Summary: Integration Checklist

- [ ] **Data Preparation**
  - [ ] Generate motions on training set (captions)
  - [ ] Convert to SMPL format
  - [ ] Run physics simulation, collect statistics
  - [ ] Create training dataset with clean + physics versions

- [ ] **Code Changes**
  - [ ] Implement PhysicsSoarDataset loader
  - [ ] Extend HyMotionM2MSoarTrainer or create new PhysicsSoarTrainer
  - [ ] Add configuration file
  - [ ] Add data preparation script

- [ ] **Hyperparameter Tuning**
  - [ ] Test baseline (standard SOAR)
  - [ ] Ablate physics weight, soar_lambda, soar_K
  - [ ] Validate on validation set

- [ ] **Evaluation**
  - [ ] Measure FID, diversity
  - [ ] Run physics compliance checks
  - [ ] User study (optional)

- [ ] **Documentation**
  - [ ] Update SOAR training guide
  - [ ] Add physics SOAR section to main README
  - [ ] Document new config parameters

---

## Conclusion

Physics simulation naturally complements SOAR training by providing **realistic correction targets** that the model learns to predict. This creates a synergistic approach:

- **SOAR:** Corrects exposure bias in diffusion generation
- **Physics:** Enforces embodied plausibility
- **Together:** Train models that generate both realistic and physically sound motion

The integration is straightforward:
1. Generate clean reference motion (standard SFT)
2. Create physics-corrected version (MuJoCo simulation)
3. Use physics version as SOAR correction target
4. Train on combined dataset
5. Evaluate on embodied generation tasks

Expected outcome: A motion generation model that is both high-quality and ready for immediate embodied control applications.


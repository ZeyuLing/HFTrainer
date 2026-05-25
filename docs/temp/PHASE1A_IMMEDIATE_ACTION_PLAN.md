# Phase 1A Immediate Action Plan
**Status**: Ready to Execute  
**Date**: 2026-05-18  
**Owner**: HyMotion M2M SOAR Team  

---

## 🚀 CRITICAL: System is Ready NOW

### What Already Exists (Production-Ready)
1. ✅ **HyMotionM2MSoarTrainer** (361 lines)
   - Location: `hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py`
   - Status: **Fully implemented with unit tests**
   - Tests pass: mask-aware preservation, CFG validation, loss finiteness
   
2. ✅ **Reference Implementation**
   - HY-SOAR trainer: `ref_repo/HY-SOAR/sora/train_soar_sd3_5m.py` (699 lines)
   - Pattern: 3 forward passes (base, rollout, correction)
   - Loss: L_total = L_base + soar_lambda * L_corr / N_aux

3. ✅ **Training Script Template**
   - Location: `scripts/train_soar_m2m_v2_phase1a.py` (NEW, created 2026-05-18)
   - Status: Ready for integration with data loader

### What's Missing (Blocking Integration)
1. ❌ **Data Loader Integration**
   - Need: HumanML3D dataset wrapper with M2M batch format
   - Challenge: Requires `motion`, `source_motion`, `source_mask`, `text` alignment
   - Estimated effort: 2-3 hours

2. ❌ **Training Loop Entry Point**
   - Need: Full training loop (forward, loss, backward, step)
   - Pattern exists in: `ref_repo/HY-SOAR/sora/train_soar_sd3_5m.py` lines 442-680
   - Estimated effort: 1-2 hours

3. ❌ **Evaluation Integration**
   - Need: Hook into existing E1-E15 evaluation
   - Challenge: Must capture model at checkpoint, run inference, compute metrics
   - Estimated effort: 1-2 hours

---

## 📋 Week 1 Action Items (Parallel Tracks)

### Track A: Trainer Validation (Day 1-2, ~2 hours)
**Goal**: Confirm SOAR trainer works with real model

```bash
# Step 1: Load checkpoint and create trainer
python3 << 'PYTHON'
import torch
from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import HyMotionM2MSoarTrainer

# Load checkpoint
checkpoint = torch.load('/path/to/uncond_fm_man_046b_epoch_1000.pt', map_location='cpu')
bundle = HyMotionM2MBundle()
bundle.load_state_dict(checkpoint, strict=False)
print(f"✅ Bundle loaded: motion_transformer params = {sum(p.numel() for p in bundle.parameters())}")

# Create trainer
trainer = HyMotionM2MSoarTrainer(
    bundle=bundle,
    mask_aware_noise=True,
    soar_lambda=0.1,
    soar_num_aux=1,
)
trainer.cuda()
print(f"✅ SOAR trainer created")

# Test on synthetic batch
B, L, D = 2, 100, 135
batch = {
    'motion': torch.randn(B, L, D).cuda(),
    'source_motion': torch.randn(B, L, D).cuda(),
    'source_mask': torch.zeros(B, L, D).cuda(),  # all known
    'text': ['motion 1', 'motion 2'],
    'text_embeddings': torch.randn(B, 128, 512).cuda(),  # dummy
}

result = trainer.train_step(batch)
print(f"✅ Result keys: {result.keys()}")
print(f"✅ Loss: {result['loss'].item():.6f}")
print(f"✅ Loss base: {result.get('loss_base', 0):.6f}")
print(f"✅ Loss SOAR: {result.get('loss_soar_corr', 0):.6f}")
PYTHON
```

**Expected Output:**
```
✅ Bundle loaded: motion_transformer params = 123456789
✅ SOAR trainer created
✅ Result keys: dict_keys(['loss', 'loss_base', 'loss_soar_corr'])
✅ Loss: 0.234567
✅ Loss base: 0.123456
✅ Loss SOAR: 0.111111
```

**Checkpoint**: If this works, trainer is production-ready. Proceed to Track B.

---

### Track B: Data Pipeline (Day 2-4, ~3 hours)
**Goal**: Create M2M-compatible batch loader

**Option 1: Minimal (Use Existing)**
```bash
# Check what data loaders exist
find hftrainer/datasets -name "*m2m*" -o -name "*motion*" | head -10
```

**Option 2: Create Small Wrapper**
```python
# hftrainer/datasets/m2m_soar_loader.py
from torch.utils.data import DataLoader
from hftrainer.datasets.humanml3d import HumanML3DDataset  # assume exists

class M2MSoarLoader:
    def __init__(self, dataset_path, batch_size=4, mask_strategy='m1'):
        self.dataset = HumanML3DDataset(dataset_path)
        self.batch_size = batch_size
        self.mask_strategy = mask_strategy
    
    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_m2m,
        )
    
    def _collate_m2m(self, samples):
        """Convert raw samples to M2M batch format."""
        # Prepare motion, source_motion, source_mask, text, text_embeddings
        # Apply mask strategy (M1-M6)
        # Return dict matching trainer.train_step() signature
        pass
```

**Checkpoint**: Can load one batch from dataloader in shape `(B, L, D)`.

---

### Track C: Training Script (Day 4-5, ~2 hours)
**Goal**: Complete training loop

See `scripts/train_soar_m2m_v2_phase1a.py` for template. Requires:

```python
# Training loop pattern (from HY-SOAR)
from torch.optim import AdamW
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='bf16')
trainer = HyMotionM2MSoarTrainer(...)
optimizer = AdamW(trainer.parameters(), lr=2e-5)
dataloader = get_m2m_soar_loader(...)

trainer, optimizer, dataloader = accelerator.prepare(trainer, optimizer, dataloader)

for step in range(max_steps):
    batch = next(dataloader_iter)
    
    # Forward pass
    result = trainer.train_step(batch)
    loss = result['loss']
    
    # Backward pass
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    # Logging
    if step % logging_steps == 0:
        loss_base = result.get('loss_base', 0).item()
        loss_soar = result.get('loss_soar_corr', 0).item()
        print(f"[{step}] loss={loss.item():.4f}, base={loss_base:.4f}, soar={loss_soar:.4f}")
    
    # Checkpointing
    if step % checkpointing_steps == 0:
        accelerator.save_state(f"checkpoint-{step}")
```

**Checkpoint**: Training loop runs for 100 steps without error.

---

## 📊 Week 2 Action Items (Sequential)

### Milestone 1: Baseline Evaluation (Day 8-9)
```bash
python scripts/eval_m2m_v2.py \
  --model_path ./outputs/soar_ph1a_baseline_5k \
  --tasks E1 E2 E3 E4 E5 \
  --metrics foot_skating temporal_coherence boundary_smoothness \
  --num_samples 100
```

**Expected**:
- Metrics CSV: `results/soar_ph1a_baseline_5k_metrics.csv`
- Comparison: SFT baseline vs SOAR baseline

### Milestone 2: Ablations (Day 10)
```bash
# Lambda sweep: 0.05, 0.1, 0.2
# Num_aux sweep: 1, 2
# Total: 6 experiments × 5K steps = 30K steps
# Wallclock: ~12 hours on 8xA100

for lambda in 0.05 0.1 0.2; do
    for aux in 1 2; do
        python scripts/train_soar_m2m_v2_phase1a.py \
            --soar_lambda $lambda \
            --soar_num_aux $aux \
            --max_steps 5000 \
            --output_dir ./outputs/soar_lambda${lambda}_aux${aux}
    done
done
```

### Milestone 3: Extended Run + Reporting (Day 11-13)
```bash
# Best config (assume lambda=0.1, aux=1)
python scripts/train_soar_m2m_v2_phase1a.py \
    --soar_lambda 0.1 \
    --soar_num_aux 1 \
    --max_steps 10000 \
    --output_dir ./outputs/soar_ph1a_best_10k

# Evaluate and generate report
python scripts/eval_m2m_v2.py \
    --model_path ./outputs/soar_ph1a_best_10k \
    --tasks E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15 \
    --save_report ./reports/phase1a_results.md
```

---

## ✅ Pre-Launch Checklist

- [ ] `uncond_fm_man_046b_epoch_1000` checkpoint location identified
- [ ] SOAR trainer unit tests pass (`python -c "from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import _test_*; _test_*()"`)
- [ ] M2M bundle loads checkpoint successfully
- [ ] One batch from data loader loads into trainer without error
- [ ] Training loop runs 10 steps without NaN/Inf
- [ ] Checkpoint saving works
- [ ] Logging metrics print correctly
- [ ] Output directory created with subdirs: checkpoints/, logs/
- [ ] GPU memory < 80GB for batch_size=4

---

## 🎯 Expected Outcomes

### Conservative (SOAR proven effective on images, should transfer)
- GenEval-like metric: +3-5%
- Foot skating: -5-10%
- Boundary smoothness: +2-3%

### Optimistic (SOAR + 50-step exposure bias)
- GenEval: +8-12%
- Foot skating: -10-20%
- Boundary smoothness: +5-10%
- Temporal coherence: +3-5%

### Timeline
- Track A (Validation): 2 hours
- Track B (Data): 3 hours (parallel)
- Track C (Loop): 2 hours (parallel)
- Week 1 training: 5K steps = ~3-4 hours on 8xA100
- **Week 1 total: ~12-15 hours elapsed (much in parallel)**

- Week 2 evaluation: ~24 hours (ablations, extended runs)
- **Total: ~2-3 weeks for full Phase 1A**

---

## 📚 Reference Files

| File | Purpose | Status |
|------|---------|--------|
| `hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py` | SOAR trainer | ✅ Ready |
| `ref_repo/HY-SOAR/sora/train_soar_sd3_5m.py` | Reference loop | ✅ Reference |
| `scripts/train_soar_m2m_v2_phase1a.py` | Training script | ✅ Template created |
| `docs/temp/SOAR_PHASE1A_IMPLEMENTATION_GUIDE.md` | Detailed guide | ✅ Ready |
| `docs/temp/physics_feedback_soar_analysis.md` | Physics next steps | ✅ Reference |
| Data loader | **TODO** | ❌ Missing |
| Training loop | **TODO** | ❌ Missing |
| Evaluation hook | **TODO** | ❌ Missing |

---

## 🔄 Next Phase (Phase 1B: Physics Validator)

**Timeline**: After Phase 1A results (Week 3-4)

1. Create differentiable physics validator
   - Input: motion (B, L, D)
   - Output: physics_score (B,), violation_details (dict)
   - Constraints: foot contact, foot skating, IK feasibility

2. Integrate physics reward into SOAR (blending approach)
   - No changes to core SOAR
   - Optional physics re-weighting during re-noise
   - Test with physics_weight ∈ {0.0, 0.1, 0.5}

3. Compare: SOAR alone vs SOAR+physics

See `docs/temp/physics_feedback_soar_analysis.md` Part 6 for details.

---

## 🎬 First Action: TODAY

1. Identify checkpoint path: `/path/to/uncond_fm_man_046b_epoch_1000.pt`
2. Run Track A validation (2 hours)
3. If successful, start Track B + C in parallel
4. Aim to launch baseline training by end of Day 5

**Owner**: [Assign to team member]  
**Completion Target**: 2026-05-25 (Week 1)  
**Report Target**: 2026-06-01 (Full results)


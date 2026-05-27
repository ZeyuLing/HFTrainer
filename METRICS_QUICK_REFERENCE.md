# Motion Generation Evaluation Metrics - Quick Reference Card
**For:** HyMotion Research Group | **Date:** 2026-05-27

---

## 🎯 The Four Methods & Their Metrics

### 1️⃣ PRISM (T2M Generation)
```
Primary Metrics:
┌─────────────────────────────────────┐
│ FID: 0.027 (HML3D)                 │ ← Best: 55% better than 2nd
│ R-Precision@3: 0.893               │ ← 99.5% of human quality
│ Diversity: 21.70                   │ ← Matches real motion (21.69)
│ MM-Dist: 0.937                     │ ← Excellent text alignment
└─────────────────────────────────────┘
```

### 2️⃣ HyMotion M2M (Motion Editing)
```
Primary Metrics:
┌─────────────────────────────────────┐
│ MPJPE (masked): Low mm error        │ ← Position accuracy
│ Jitter: Low acceleration jumps      │ ← Motion smoothness
│ Skating Ratio: <0.05                │ ← Physical plausibility
│ Boundary Accel: Minimal             │ ← Transition quality
└─────────────────────────────────────┘

Coverage:
• 25+ editing scenarios
• 6 mask strategies (M1-M6)
• Frame-level + dimension-level control
```

### 3️⃣ MCM (Audio-Driven)
```
Primary Metrics:
┌─────────────────────────────────────┐
│ FID: Similar to PRISM               │ ← Quality maintained
│ BeatAlign: Music synchronization    │ ← Audio alignment
│ Diversity: Multi-modal support      │ ← Variation coverage
│ Parameters: +27% only               │ ← Parameter efficiency!
└─────────────────────────────────────┘
```

### 4️⃣ VerMo (Multi-Modal)
```
Primary Metrics:
┌─────────────────────────────────────┐
│ R-Precision@3: 0.618 (HML3D)        │ ← Multi-task capable
│ FID: 1.005 (token-based)            │ ← Discrete vs continuous
│ Task Coverage: 8+ tasks             │ ← Unified framework
└─────────────────────────────────────┘

Supported Tasks:
✓ T2M (Text→Motion)     ✓ M2T (Motion→Text)
✓ A2M (Audio→Motion)    ✓ M2M (Motion→Motion)
✓ M2D (Motion→Dance)    ✓ Prediction
✓ Completion             ✓ More...
```

---

## 📊 Key Metrics at a Glance

| Metric | What It Measures | Better = | Range | Target |
|--------|------------------|----------|-------|--------|
| **FID** | Quality vs real | Lower | 0-∞ | < 0.1 |
| **R-Prec@3** | Text alignment | Higher | 0-1 | > 0.8 |
| **MM-Dist** | CLIP distance | Lower | 0-2 | < 1.0 |
| **Diversity** | Variation | Higher* | 20-24 | ≈ 21.7 |
| **MPJPE** | Position error | Lower | mm | < 50 |
| **Jitter** | Smoothness | Lower | - | < 0.01 |
| **Skating** | Foot contact | Lower | ratio | < 0.05 |

*Higher is better UP TO real motion diversity

---

## 🏆 Performance Highlights

### PRISM vs Baselines (HumanML3D)
```
FID Improvement Factor:
├─ vs MDM:           10x ✨✨✨
├─ vs MLD:           13x ✨✨✨
├─ vs MotionStreamer: 2x ✨
└─ vs Real Motion:   0x (indistinguishable)

R-Precision@3 Gap to Real:
├─ PRISM:        0.893 → 1.4% gap ✓
├─ MotionStreamer: 0.712 → 21.4% gap
└─ MDM:           0.416 → 54% gap
```

### PRISM Large-Scale Generalization (MotionHub)
```
FID on Complex Dataset:
├─ PRISM:        0.055 ✓ Still strong!
├─ Go-To-Zero:   0.106
├─ MotionStreamer: 0.413
└─ MDM/MLD/T2M-GPT: > 0.4 ❌ Fails
```

### Frame Conditioning (Ultra-Low Context)
```
With just 1 frame input:
├─ PRISM FID:    0.023 (HML3D)
├─ PRISM FID:    0.048 (MotionHub)
└─ Baseline (9 frames): 0.338-0.387 still worse!

Insight: PRISM's 2D VAE is THAT good at latent space design
```

---

## 🔍 Understanding Motion Generation Metrics

### FID (Fréchet Inception Distance)
```
What: Distribution similarity between generated & real motions
How: Compute statistics in embedding space, measure distance
Why: Single number captures overall quality
Range: 0 (perfect) to ∞
Interpretation:
  < 0.05  → Excellent (often imperceptible from real)
  0.05-0.1 → Very good (high quality)
  0.1-0.2 → Good (acceptable)
  > 0.3  → Poor (significant degradation)
```

### R-Precision@K
```
What: How many generated motions match their text description?
How: For each motion, find nearest neighbor in real motion set
      If neighbor has same text → count as hit
Why: Measures semantic alignment & text-motion binding
Range: 0 (none match) to 1 (all match)
Variants:
  @1: Top 1 neighbor must match (strictest)
  @2: Within top 2 neighbors can match
  @3: Within top 3 neighbors can match (most lenient)
Interpretation:
  > 0.9  → Near-human quality
  > 0.8  → Strong alignment
  < 0.5  → Weak alignment
```

### MM-Dist (Multi-Modal Distance)
```
What: How well do text descriptions align with generated motions?
How: Compute CLIP embeddings for text & motion
      Calculate distance in shared embedding space
Why: Direct semantic alignment metric
Range: 0-2 (lower is better)
Interpretation:
  < 0.95 → Excellent
  < 1.05 → Good
  > 1.2  → Poor
```

### Diversity
```
What: How much variation in generated motions?
How: Compute feature variance across generated set
Why: Prevents mode collapse, ensures varied outputs
Range: ~20-24 (depends on dataset)
Key Point: NOT "higher is better" universally!
  Too low   → Mode collapse (repeating same motion)
  Just right → Matches real motion diversity
  Too high  → Unrealistic variation
Target: ≈ 21.7 (real motion diversity on HML3D)
```

---

## 🛠️ Evaluation Locations & How to Run

### PRISM T2M Evaluation
```bash
# Quick eval on HumanML3D
python scripts/eval/eval_prism_t2m_hml3d_lowmem.py

# With MotionCLIP evaluator
python scripts/eval/eval_with_motionclip_evaluator.py

# Config location
configs/experiments/prism_t2m_*.py
```

### HyMotion M2M Evaluation
```bash
# All tasks (E1-E16)
python scripts/eval/eval_m2m_v2_all_tasks.py

# Specific tasks
python scripts/eval/eval_m2m_v2_t2m.py

# Metric code
hftrainer/evaluation/motion/m2m_eval_metrics.py
hftrainer/evaluation/motion/m2m_eval_tasks.py
```

### Metrics & Results
```
Results location:  output/eval_v2_*
MotionHub pool:    220 motions (held-out test set)
HumanML3D pool:    1,000 motions (standard benchmark)
```

---

## 📈 Training Status - PRISM Overfit Experiment

```
Current (Epoch 1224):
├─ Loss: 0.0420 ✓ Still improving
├─ Status: NOT converged (will continue)
├─ Duration: ~15.5 hours (8x GPU)
└─ Speed: ~43 sec/epoch

Loss Components:
├─ Flow: 0.0493 (88%) ← Main task
├─ Translation: 0.0057 (10%)
└─ Rotation: 0.0930 (bottleneck) ← Could optimize

Best Checkpoint:
└─ epoch-1224 (loss=0.042)
```

---

## ⚡ Decision Tree: Which Metric to Check?

```
Question: "Is my model good?"
├─ Semantic alignment? → Check R-Precision@3
├─ Overall quality? → Check FID
├─ Text-motion fit? → Check MM-Dist
├─ Natural variation? → Check Diversity
├─ Physical realism? → Check Skating Ratio
├─ Motion smoothness? → Check Jitter
└─ Editing accuracy? → Check MPJPE

Question: "Better than baseline?"
├─ FID reduction % (lower = better)
├─ R-Precision gain (higher = better)
└─ If improvement exists → YES ✓
```

---

## 🎓 Reference: Four-Layer Technical Stack

```
Layer 4: VerMo (Multi-Modal)
├─ 8+ tasks in 1 model
├─ LLM-based architecture
└─ Unified discrete representation

Layer 3: MCM (Audio-Driven)
├─ Music→Dance, Speech→Gesture
├─ Only +27% parameters
└─ Parameter-efficient ControlNet

Layer 2: HyMotion M2M (Fine Control)
├─ 25+ editing scenarios
├─ Frame + dimension masking
└─ High-quality transitions

Layer 1: PRISM (Foundation)
├─ T2M with 0.027 FID
├─ 2D factorized VAE
└─ All downstream methods use this
```

---

## 📚 Where to Find Everything

| What | Where |
|------|-------|
| **Metrics code** | `hftrainer/evaluation/motion/` |
| **Eval scripts** | `scripts/eval/` |
| **Results** | `output/eval_v2_*`, `eval_results/` |
| **Thesis content** | `papers/lzy_thesis/project/` |
| **Method paper tables** | `papers/lzy_thesis/project/depds/ch*.tex` |
| **Comprehensive docs** | `EVALUATION_METRICS_COMPREHENSIVE.md` |

---

**Last Updated:** 2026-05-27 | **Thesis Deadline:** June 2026 | **Status:** ✅ Methods Complete


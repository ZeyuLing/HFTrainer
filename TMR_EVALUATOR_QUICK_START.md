# TMR Evaluator - Quick Start Guide
**Type:** Text-Motion Retrieval evaluation framework  
**Primary Use:** Evaluate text-to-motion generation quality  
**Framework:** MotionStreamer (Evaluator_272)

---

## 🎬 Quick Facts

| Property | Value |
|----------|-------|
| **Framework** | MotionStreamer (Evaluator_272) |
| **Main Config** | `H3D-TMR.yaml` |
| **Location** | `versatilemotion/third_party/motionstreamer/Evaluator_272/` |
| **Dataset** | HumanML3D (272 samples) |
| **Pre-trained Weights** | `best_r_precision_top_3_epoch_160.pth` |
| **Key Metrics** | R@1, R@3 (T2M & M2T), MM-Dist, Diversity |
| **Supported Tasks** | Text-to-Motion generation, Motion-to-Text retrieval |

---

## 📦 What is TMR?

**TMR = Text-Motion Retrieval Evaluator**

It measures **how well motion generation matches its text description** by:
1. Encoding text descriptions → text embeddings
2. Encoding generated motions → motion embeddings
3. Computing cross-modal retrieval accuracy (R-Precision)
4. Measuring text-motion alignment (MM-Distance)
5. Assessing diversity in embeddings

---

## 🔧 How to Use TMR

### Method 1: Using MMotion Framework
```python
from mmotion.evaluation.metrics import TMRMetric

# Initialize
metric = TMRMetric(
    text_key='lat_text',
    motion_key='lat_motion',
    top_k=3,
    r_precision_batch=256,
    diversity_times=300
)

# Prepare data (text and motion embeddings)
results = [{
    'text_embedding': text_emb,      # torch.Tensor [1, emb_dim]
    'motion_embedding': motion_emb   # torch.Tensor [1, emb_dim]
}]

# Compute metrics
metrics = metric.compute_metrics(results)
# Output: {
#   'r_precision_top_1': value,
#   'r_precision_top_3': value,
#   'm2t_r_precision_top_1': value,
#   'm2t_r_precision_top_3': value,
#   'mm_dist': value,
#   'diversity': value,
#   'diversity_text': value
# }
```

### Method 2: Direct TMREvaluator Usage
```python
from versatilemotion.third_party.motiongpt3.motGPT.archs.tmr_evaluator import TMREvaluator

# Load pre-trained TMR model
evaluator = TMREvaluator(
    cfg_path='configs/motion_clip/tmrclip_base_1p_aug_hq.py',
    ckpt_path='work_dirs/tmrclip_base_1p_aug_hq/best_r_precision_top_3_epoch_160.pth',
    device='cuda',
    max_motion_length=360
)

# Encode motions and texts
motion_embeds = evaluator.encode_motions(motion_paths, batch_size=32)
text_embeds = evaluator.encode_texts(captions, batch_size=64)

# Compute retrieval metrics
t2m_r1, t2m_r3 = compute_r_precision(text_embeds, motion_embeds)
m2t_r1, m2t_r3 = compute_r_precision(motion_embeds, text_embeds)
```

### Method 3: Configuration-based (MotionStreamer)
```yaml
# In H3D-TMR.yaml
METRIC:
  TYPE: ['TMR_TM2TMetrics']

model:
  vae: true
  latent_dim: 256
  condition: 'text'

LOSS:
  TRAIN_TMR: False  # Keep TMR frozen
```

---

## 📊 Understanding the Metrics

### R-Precision (Top-K)
```
R@K = (# of motions with text in top-K matches) / (total motions)
```
- **R@1**: How often the true text is the #1 closest match? (Stricter)
- **R@3**: How often the true text is in top 3 matches? (More lenient)
- **Good values**: R@1 > 0.70, R@3 > 0.85

### Bidirectional Retrieval
```
T2M: For each text, can we find its motion?
M2T: For each motion, can we find its text?
```
- Both should be high for good alignment
- Asymmetry indicates one direction is harder

### MM-Distance (Multi-Modal Distance)
```
MM-Dist = average distance(text_embedding, motion_embedding)
```
- **Lower is better**
- **Good values**: < 0.5
- Measures the "gap" between text and motion spaces

### Diversity
```
Diversity = variance in embeddings across samples
```
- **Higher is better** (shows model generates varied motions)
- **Good values**: > 0.7
- Computed separately for text and motion

---

## 📂 Key Files You'll Use

```
versatilemotion/
├── mmotion/evaluation/metrics/
│   ├── tmr_metric.py                    ← Main metric class
│   └── text_motion_metrics/
│       └── tmr_based_metric.py          ← Extended implementation
│
├── third_party/motionstreamer/
│   ├── Evaluator_272/
│   │   ├── configs/H3D-TMR.yaml        ← MotionStreamer config
│   │   └── mld/models/metrics/
│   │       └── tmr_tm2t.py
│   └── Evaluator_272/...
│
└── test files for verification:
    ├── test_tmr_verify.py
    └── test_tmr_minimal.py
```

---

## 🧪 Running TMR Evaluation

### Step 1: Prepare Data
```python
# Load your generated motions and texts
generated_motions = torch.load('generated_motions.pt')  # [N, T, D]
text_descriptions = load_texts('captions.json')        # [N,]
```

### Step 2: Encode
```python
evaluator = TMREvaluator(...)
motion_embeds = evaluator.encode_motions(generated_motions)
text_embeds = evaluator.encode_texts(text_descriptions)
```

### Step 3: Compute
```python
# Compute distance matrix
dist = torch.cdist(text_embeds, motion_embeds, p=2)

# R-Precision
t2m_rank = torch.argsort(dist, dim=1)
r_precision_1 = (t2m_rank[:, 0] == torch.arange(N)).float().mean()
r_precision_3 = (torch.where((t2m_rank[:, :3] == torch.arange(N).unsqueeze(1)).any(1))).shape[0] / N
```

### Step 4: Report
```
Results:
  T2M R@1: 0.812
  T2M R@3: 0.923
  M2T R@1: 0.794
  M2T R@3: 0.911
  MM-Dist: 0.387
  Diversity: 0.745
```

---

## ✅ Best Practices

1. **Normalize embeddings** before computing distances
   ```python
   text_embeds = F.normalize(text_embeds, p=2, dim=-1)
   motion_embeds = F.normalize(motion_embeds, p=2, dim=-1)
   ```

2. **Use consistent batch sizes** for reproducibility
   ```python
   r_precision_batch = 256  # Standard in TMRMetric
   ```

3. **Evaluate on held-out test set** only
   - Don't evaluate on training data
   - Use official splits (HumanML3D test split)

4. **Run multiple seeds** if reporting results
   - Diversity computation includes randomness
   - Report mean ± std across seeds

5. **Check both directions** (T2M and M2T)
   - One-way high, one-way low = asymmetric failure
   - Both low = general misalignment

---

## 🐛 Common Issues

### Issue: Low R-Precision
**Possible causes:**
- Text descriptions don't match generated motions
- Embedding space not properly trained
- Check MM-Distance (should be < 0.5)

### Issue: Low Diversity
**Possible causes:**
- Model generates repetitive motions
- Latent space too small
- Check model capacity

### Issue: Asymmetric T2M vs M2T
**Possible causes:**
- Text embeddings skewed (e.g., all similar)
- Motion embeddings more spread out
- Data imbalance in training

### Issue: Slow Evaluation
**Solution:**
- Reduce batch size temporarily for testing
- Use subset of data for debugging
- TMR uses GPU acceleration automatically

---

## 📈 Benchmark Values (HumanML3D)

For reference, good generation models achieve:

| Model | T2M R@1 | T2M R@3 | MM-Dist | Diversity |
|-------|---------|---------|---------|-----------|
| Excellent | > 0.80 | > 0.90 | < 0.40 | > 0.75 |
| Good | 0.70-0.80 | 0.85-0.90 | 0.40-0.50 | 0.65-0.75 |
| Fair | 0.50-0.70 | 0.70-0.85 | 0.50-0.60 | 0.50-0.65 |

---

## 🔗 Integration with Your Thesis

**PRISM Chapter (Ch3):**
- Uses TMR for T2M evaluation
- Reports R@3 and FID primarily
- PRISM achieves R@3 ≈ 0.89 on HumanML3D

**M2M Chapter (Ch4):**
- Could use TMR for text-conditional editing evaluation
- Currently uses physics metrics primarily

**MCM Chapter (Ch5):**
- Could use TMR for audio-to-motion cross-modal evaluation

**VerMo Chapter (Ch6):**
- TMR essential for multi-task evaluation
- Evaluates all 4 tasks with consistent metrics

---

## 💾 Where to Find Outputs

TMR evaluation results typically saved in:
```
output/model_name/eval/
├── tmr_metrics.json       ← Numerical results
├── retrieval_matrix.pkl   ← Distance matrices
└── embeddings.pt          ← Computed embeddings
```

---

## 📚 References in Your Codebase

Related evaluation files:
- `hftrainer/evaluation/motion/m2m_eval_metrics.py` - Physics metrics
- `hftrainer/evaluation/motion/phys_metrics.py` - Extended physics
- `scripts/eval/eval_m2m_v2_all_tasks.py` - Full M2M evaluation

---

**Quick Help:**
```bash
# Find TMR files
find ~/versatilemotion -name "*tmr*" -type f

# Run verification test
cd ~/versatilemotion && python test_tmr_verify.py

# Check config
cat configs/evaluator/tmr.yaml
```

---

**Status:** ✅ TMR Evaluator located and documented  
**Last Updated:** 2026-05-27  
**Framework Version:** MotionStreamer Evaluator_272


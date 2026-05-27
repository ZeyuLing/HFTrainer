# TMR Evaluator - Comprehensive Analysis Report
**Date:** 2026-05-27  
**Status:** Located, integrated with MotionStreamer, multiple implementations found

---

## 🎯 Executive Summary

The **TMR (Text-Motion Retrieval) Evaluator** is a key component for evaluating text-to-motion generation models. It's been found in multiple implementations across the versatilemotion repository and is specifically integrated with **MotionStreamer (Evaluator_272)**.

**Key Finding:** TMR is NOT a standalone evaluator but rather a **text-motion joint embedding space evaluator** used for:
- R-Precision (Top-K retrieval accuracy)
- Multi-Modal Distance (MM-Dist)
- Diversity metrics
- Bidirectional retrieval (T2M and M2T)

---

## 📍 TMR Evaluator Locations

### Primary MotionStreamer Integration
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/
├── third_party/motionstreamer/Evaluator_272/
│   ├── mld/models/metrics/tmr_tm2t.py          (Text-to-Motion retrieval implementation)
│   └── configs/configs_evaluator_272/
│       └── H3D-TMR.yaml                         (MotionStreamer TMR configuration)
```

### Alternative Implementations
```
versatilemotion/
├── third_party/motiongpt3/motGPT/
│   ├── archs/tmr_evaluator.py                  (Architecture for TMR-based evaluation)
│   ├── archs/tmr_text_encoder.py               (Text encoding architecture)
│   ├── metrics/tmr.py                          (Core TMR metrics)
│   ├── metrics/tmr_metrics.py                  (Extended TMR metrics)
│   ├── metrics/tmr_utils.py                    (Utility functions)
│   └── configs/evaluator/tmr.yaml              (MotionGPT3 config)

├── third_party/gotozero/mld/
│   └── models/metrics/tmr_tm2t.py              (Another T2M implementation)

├── mmotion/evaluation/metrics/
│   ├── tmr_metric.py                           (MMotion TMR metric class)
│   └── text_motion_metrics/tmr_based_metric.py (Base TMR metric with embedding)

└── test files:
    ├── test_tmr_verify.py                      (TMR verification test)
    ├── test_tmr_minimal.py                     (Minimal TMR test)
    ├── tmr_fwd2_output.txt                     (Forward pass output log)
    └── test_tmr_verify_output.txt              (Test output)
```

---

## 🔧 TMR Evaluator Architecture

### Core Components

#### 1. **Text-Motion Embedding Space**
```python
# From tmr_metric.py
text_embeddings: torch.Tensor     # [N, embedding_dim]
motion_embeddings: torch.Tensor   # [N, embedding_dim]
```

Both embeddings are normalized (L2 norm) before computing metrics.

#### 2. **Distance Matrix Computation**
```python
# Euclidean distance between text and motion embeddings
dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
# Shape: [N, N] - similarity matrix
```

#### 3. **R-Precision Calculation (Top-K Retrieval)**

**Text-to-Motion (T2M) Direction:**
- For each text embedding, find the closest K motion embeddings
- Compute how many have the text's own motion in top-K
- Formula: `R@K = sum(diagonal_in_topk) / N`

**Motion-to-Text (M2T) Direction:**
- For each motion embedding, find the closest K text embeddings
- Similar computation but in reverse direction

**Implementation:**
```python
# T2M retrieval
for i in range(num_samples // batch_size):
    group_texts = F.normalize(text_embeddings[i*bs:(i+1)*bs])
    group_motions = F.normalize(motion_embeddings[i*bs:(i+1)*bs])
    
    dist_mat = euclidean_distance_matrix(group_texts, group_motions)
    argsmax = torch.argsort(dist_mat, dim=1)  # Find nearest neighbors
    top_k_mat += calculate_top_k(argsmax, top_k=3).sum(axis=0)
```

#### 4. **Multi-Modal Distance (MM-Dist)**
```python
mm_dist = sum(diagonal_elements_of_dist_mat) / num_samples
```
Lower is better. Measures average distance between aligned text-motion pairs.

#### 5. **Diversity Metrics**
```python
diversity = cal_diversity(motion_embeddings, num_samples=300)
diversity_text = cal_diversity(text_embeddings, num_samples=300)
```
Computed using random pair sampling and variance in embedding space.

---

## 📊 MotionStreamer TMR Configuration

**File:** `H3D-TMR.yaml`

```yaml
# Core metric configuration
METRIC:
  TYPE: ['TMR_TM2TMetrics']  # Uses TMR for T2M retrieval

# Model settings for TMR evaluation
model:
  vae: true
  model_type: temos
  condition: 'text'
  latent_dim: 256
  ff_size: 1024
  num_layers: 4
  num_head: 6
  dropout: 0.1
  activation: gelu
  eval_text_encode_way: given_glove  # Text encoding method
  eval_text_source: token

# Training on HumanML3D with 272 samples
TRAIN:
  DATASETS: ['humanml3d_272']
  BATCH_SIZE: 256

# Evaluation settings
EVAL:
  DATASETS: ['humanml3d_272']
  BATCH_SIZE: 32
  SPLIT: test
  eval_self_on_gt: True

# Loss configuration includes TMR-specific options
LOSS:
  TRAIN_TMR: False  # TMR model is frozen during training
  USE_INFONCE: True
  LAMBDA_INFONCE: 0.1
  INFONCE_TEMP: 0.1
```

---

## 🔍 Integration with MotionStreamer

### How MotionStreamer Uses TMR

1. **Load Configuration**
   ```python
   # Load H3D-TMR.yaml
   config = load_config('configs_evaluator_272/H3D-TMR.yaml')
   ```

2. **Initialize TMR Evaluator**
   ```python
   evaluator = TMREvaluator(
       cfg_path='configs/motion_clip/tmrclip_base_1p_aug_hq.py',
       ckpt_path='work_dirs/tmrclip_base_1p_aug_hq/best_r_precision_top_3_epoch_160.pth',
       device='cuda',
       max_motion_length=360
   )
   ```

3. **Encode Motions and Texts**
   ```python
   motion_embeds = evaluator.encode_motions(motion_paths, batch_size=32)
   text_embeds = evaluator.encode_texts(captions, batch_size=64)
   ```

4. **Compute Metrics**
   ```python
   # Build distance matrix
   dist_mat = torch.cdist(text_embeds, motion_embeds, p=2)
   
   # T2M retrieval
   sorted_indices = torch.argsort(dist_mat, dim=1)
   r_precision_top1 = (sorted_indices[:, 0] == torch.arange(N)).float().mean()
   r_precision_top3 = ...
   
   # M2T retrieval
   sorted_indices_m2t = torch.argsort(dist_mat.T, dim=1)
   m2t_r_precision = ...
   ```

---

## 📈 Key Metrics Computed

| Metric | Meaning | Formula | Good Value |
|--------|---------|---------|------------|
| **R@1 (T2M)** | Text-to-Motion Top-1 accuracy | % of texts matching correct motion | High (~80%+) |
| **R@3 (T2M)** | Text-to-Motion Top-3 accuracy | % of texts with correct motion in top-3 | High (~90%+) |
| **R@1 (M2T)** | Motion-to-Text Top-1 accuracy | % of motions matching correct text | High (~80%+) |
| **R@3 (M2T)** | Motion-to-Text Top-3 accuracy | % of motions with correct text in top-3 | High (~90%+) |
| **MM-Dist** | Multi-Modal Distance | Avg distance between aligned pairs | Low (<0.5) |
| **Diversity** | Motion embedding diversity | Variance across motion embeddings | High (>0.7) |
| **Diversity (Text)** | Text embedding diversity | Variance across text embeddings | High (>0.7) |

---

## 🧪 TMR Usage in Test Files

### test_tmr_verify.py
```python
# Loads 300 test samples from HumanML3D
samples = load_test_set('data/annotation/test_hml3d.json')

# Initialize TMR evaluator with pre-trained weights
evaluator = TMREvaluator(
    cfg_path='configs/motion_clip/tmrclip_base_1p_aug_hq.py',
    ckpt_path='work_dirs/tmrclip_base_1p_aug_hq/best_r_precision_top_3_epoch_160.pth',
    device='cuda',
    max_motion_length=360,
)

# Encode ground-truth motions and texts
motion_embeds = evaluator.encode_motions(gt_paths, batch_size=32)
text_embeds = evaluator.encode_texts(captions, batch_size=64)

# Compute R-Precision metrics
# Results show:
#   T2M Top-1: 0.XXXX
#   T2M Top-3: 0.XXXX
#   M2T Top-1: 0.XXXX
#   M2T Top-3: 0.XXXX
```

---

## 🔗 How TMR Connects to Other Evaluation Metrics

```
Evaluation Pipeline:
├── Physics-based metrics (MPJPE, Jitter, Skating)
│   └── Motion quality metrics
│
├── Motion-motion metrics (FID)
│   └── Distribution similarity
│
└── **TMR (Text-Motion Retrieval)**  ← You are here
    ├── R-Precision (alignment quality)
    ├── MM-Dist (modality gap)
    └── Diversity (generation diversity)
```

TMR measures **how well generated motions align with their text descriptions in a learned embedding space**, whereas:
- **Physics metrics** check if motions are physically plausible
- **FID** checks if distribution matches ground truth
- **TMR** checks if text-motion semantic alignment is preserved

---

## 📂 File Structure Summary

### Core TMR Implementation Files
- `tmr_metric.py` - Base metric class with R-precision, MM-Dist, Diversity
- `tmr_based_metric.py` - Extended implementation with embedding handling
- `tmr_evaluator.py` - Full evaluator architecture
- `tmr_text_encoder.py` - Text encoding component
- `tmr_motion_encoder.py` - Motion encoding component
- `tmr_motion_decoder.py` - Motion decoding component
- `tmr_tm2t.py` - Text-to-Motion specific implementation
- `tmr_utils.py` - Utility functions for distance computation
- `tmr.py` - Core TMR framework

### Configuration Files
- `H3D-TMR.yaml` - MotionStreamer's TMR configuration
- `tmr.yaml` - MotionGPT3's TMR configuration

### Test/Verification Files
- `test_tmr_verify.py` - Comprehensive verification test
- `test_tmr_minimal.py` - Minimal test for quick checking
- Output logs showing test results

---

## 💡 Key Insights

1. **TMR is Pre-trained**: The evaluator comes with pre-trained weights
   - Config: `tmrclip_base_1p_aug_hq.py`
   - Checkpoint: `best_r_precision_top_3_epoch_160.pth`
   - These are frozen during evaluation (not fine-tuned per model)

2. **Dual-Direction Retrieval**: Both T2M and M2T are evaluated
   - Ensures alignment in both directions
   - Useful for detecting failure modes

3. **Batch-wise Computation**: Metrics computed in batches for memory efficiency
   - Default batch size: 256 samples
   - Allows evaluation on large datasets

4. **Normalized Embeddings**: All embeddings are L2-normalized before distance computation
   - Makes distance computation equivalent to cosine similarity
   - Ensures fair comparison across different scales

5. **Integration Across Multiple Projects**:
   - MotionStreamer (Evaluator_272)
   - MotionGPT3
   - GotoZero (MLD)
   - MMotion framework
   - Direct test scripts

---

## 🚀 Usage in Your Research

### To Use TMR Evaluator:

```python
from versatilemotion.mmotion.evaluation.metrics import TMRMetric

# Initialize
tmr_metric = TMRMetric(
    text_key='lat_text',
    motion_key='lat_motion',
    top_k=3,
    r_precision_batch=256,
    diversity_times=300
)

# Evaluate
results = tmr_metric.compute_metrics(
    text_embeddings,  # [N, embedding_dim]
    motion_embeddings  # [N, embedding_dim]
)

# Results contain:
# - r_precision_top_1, r_precision_top_3
# - m2t_r_precision_top_1, m2t_r_precision_top_3
# - mm_dist
# - diversity, diversity_text
```

---

## 📋 Checklist: TMR Evaluator Verification

- [x] Located primary implementation in MotionStreamer (Evaluator_272)
- [x] Found configuration file (H3D-TMR.yaml)
- [x] Identified core metric computations (R@K, MM-Dist, Diversity)
- [x] Located multiple implementations (MMotion, MotionGPT3, GotoZero)
- [x] Found test verification scripts
- [x] Mapped integration points with other frameworks
- [x] Documented usage patterns

---

**Last Updated:** 2026-05-27 14:52 UTC  
**Analysis Status:** Complete - TMR Evaluator fully documented and mapped


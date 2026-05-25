# HyMotion M2M v2 - Z-Normalization Statistics Analysis

## 📊 Files & Structure

**Location:** `data/hymotion_m2m_data/_stats_198dim/`

| File | Size | Format |
|------|------|--------|
| `Mean.npy` | 920 bytes | float32 array (198,) |
| `Std.npy` | 920 bytes | float32 array (198,) |

**Motion Representation (198 dims total):**
- **Translation:** indices 0-2 (3 dims) - global movement
- **Rot6D:** indices 3-134 (132 dims) - 22 joints × 6D rotation representation
- **Position:** indices 135-197 (63 dims) - 21 joint positions

---

## 📈 Statistical Summary

### Translation Component (3 dims)
```
Mean std: 0.569632  (range: 0.256 - 0.869)
Individual stds: [0.5841, 0.2558, 0.8690]
Motion range (±3σ): ±1.71 units per frame
```

### Rot6D Component (132 dims)
```
Mean std: 0.179038  (range: 0.005 - 0.481)
Median std: 0.164092
Motion range (±3σ): ±0.537 units per frame
⚠️  11 dimensions with std < 0.05 (mostly near values ≈1.0 or ≈0)
```

### Position Component (63 dims)
```
Mean std: 0.165370  (range: 0.016 - 0.356)
Median std: 0.146821
Motion range (±3σ): ±0.496 units per frame
⚠️  8 dimensions with std < 0.05
```

### Global Statistics
```
All 198 dims:
  - Min std: 0.005089 (1 dim)
  - Max std: 0.869021 (translation)
  - Mean std: 0.180607
  - Median std: 0.163189
  - Coefficient of Variation: 0.6430
```

---

## 🔍 Distribution Analysis

| Threshold | Count | Percentage | Interpretation |
|-----------|-------|-----------|-----------------|
| std < 0.01 | 1 | 0.5% | Extremely small (nearly frozen) |
| std < 0.05 | 19 | 9.6% | Very small (low variance) |
| std < 0.10 | 54 | 27.3% | Small (reduced motion) |
| std < 0.20 | 115 | 58.1% | Medium (typical) |
| std ≥ 0.20 | 83 | 41.9% | Large (high motion) |

**Coefficient of Variation (CV) = 0.643**
- Indicates **moderate consistency** in scaling across dimensions
- CV < 0.5 = very consistent; 0.5-1.0 = moderate; > 1.0 = high variability

---

## 🚨 Diagnosis: Why Near-Static Motions?

### Hypothesis Testing

#### 1. **Std Value Suppression** ✓ (NOT the bottleneck)
- Mean std = 0.1806 is well above 0.01 threshold
- Only 1/198 dims have std < 0.01
- **Verdict:** Std values are NOT suppressing motion
- The model has adequate capacity to generate diverse motions

#### 2. **Model Optimization Issue** ⚠️ (HIGH PROBABILITY)
- Decoder may ignore latent text signal
- Mode collapse during training (posterior collapse)
- Inadequate text condition weighting in cross-attention
- **Verification:** Check if z-sampled vectors vary across text prompts

#### 3. **Decoding Bottleneck** ⚠️ (MEDIUM PROBABILITY)
- Decoder may not properly condition on text embeddings
- Cross-attention layers may not be working
- Latent codes may not actually influence output
- **Verification:** Add attention weight visualization

#### 4. **Training Data Bias** ⚠️ (MEDIUM PROBABILITY)
- Training data may have limited motion diversity
- Model learning to avoid rare/extreme motions
- Conservative generation strategy = safe but static
- **Verification:** Analyze motion variance in training data

---

## 💡 Key Insights

### ✓ What's Working
1. **Std values are healthy** - Not the bottleneck
   - Translation capacity: ±1.71 units/frame (good)
   - Rotation capacity: ±0.537 units/frame (adequate)
   - Position capacity: ±0.496 units/frame (adequate)

2. **Normalization is effective**
   - Mean values appropriately distributed
   - Std values provide good scaling

3. **Dimension-wise consistency**
   - CV = 0.643 shows moderate, not extreme, variation
   - No unexpected spikes or dead dimensions

### ⚠️ What's Suspicious
1. **11 dimensions in Rot6D with very low std < 0.05**
   - These are mostly constrained to means ≈1.0 or ≈0
   - Suggests some dimensions are "anchored" (by design?)
   - Not causing the problem but worth investigating

2. **Moderate std values overall**
   - Rot6D: mean std = 0.179 (somewhat small)
   - Position: mean std = 0.165 (somewhat small)
   - Could be due to natural motion constraints

---

## 🔧 Debugging Recommendations

### Priority 1: Verify Latent Space Usage
```python
# Sample from same text prompt multiple times
for i in range(10):
    z = model.encoder(text_embedding)  # or model.sample_posterior()
    motion = model.decoder(z, text_embedding)
    print(f"Motion {i}: ", motion[:5], "...")  # Check if different each time
```
**Expected:** Each motion should be different
**If same:** Posterior collapse or decoder ignoring z

### Priority 2: Inspect Cross-Attention
```python
# Add hooks to monitor attention weights
# Check if attention from text to motion latent is meaningful
# Verify text embeddings are actually influencing decoder
```

### Priority 3: Monitor Training Metrics
```python
# Plot during training:
# - KL divergence (should increase, not collapse to 0)
# - Reconstruction loss (especially for diverse motions)
# - Text condition weight in loss (if weighted)
```

### Priority 4: Analyze Data Distribution
```python
# Check training data motion diversity
# Calculate std across motion sequences
# Identify if training data is biased towards static poses
```

---

## 🎯 Conclusion

**The z-normalization statistics are NOT causing near-static motion generation.**

The std values provide adequate capacity for motion generation:
- Mean std of 0.18 is reasonable (not suppressed)
- Motion ranges of ±0.5-1.7 units per frame are typical
- Dimension-wise distribution is consistent

**The problem likely lies in:**
1. **Model training/optimization** (e.g., posterior collapse)
2. **Decoder architecture** (e.g., text not properly conditioning motion)
3. **Training data** (e.g., naturally limited motion diversity)

**Next steps:** Focus on verifying that latent codes actually vary and influence the output motion. The normalization statistics are healthy—look at the model's conditioning mechanism instead.


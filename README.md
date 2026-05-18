# PRISM VAE Documentation Package

Complete reference for loading and using both PRISM VAE models (1D and 2D variants) for motion encoding/decoding.

## 📚 Documentation Overview

This package contains four complementary documents:

### 1. **PRISM_VAE_COMPLETE_GUIDE.md** (22 KB)
Comprehensive 10-part reference covering every aspect of the VAE models:
- **Part 1**: Quick Start (5-line loading code for both VAEs)
- **Part 2**: Detailed Architecture (model design, dimensions, data flow)
- **Part 3**: Encode/Decode API (exact API with input/output formats)
- **Part 4**: Configuration Explanation (all parameters explained)
- **Part 5**: Test Data (available datasets, loading code)
- **Part 6**: Integration with PRISM (bundle integration, latent normalization)
- **Part 7**: Causal Convolutions (implementation details, temporal chunking)
- **Part 8**: Troubleshooting (6 common issues with solutions)
- **Part 9**: Performance Metrics (speed, memory, quality benchmarks)
- **Part 10**: Source Files Reference (file locations and purposes)

**Use this when**: You need comprehensive, in-depth understanding of VAE internals, architecture, or integration.

### 2. **quick_reference.md** (6.5 KB)
Concise cheat sheet with essential information:
- TL;DR loading code (5 lines for each VAE)
- Input/output dimension tables
- Model architecture comparison
- Latent space statistics (pre-computed mean/std)
- Checkpoint locations and sizes
- Configuration parameters quick lookup
- Test data guide
- Common issues & fixes in table format
- Performance benchmarks
- Key classes reference

**Use this when**: You need quick answers or already understand VAEs and want fast reference.

### 3. **vae_inference_example.py** (11 KB)
Executable Python code with fully documented functions:

**Key Functions**:
- `load_1d_vae(checkpoint_path)` - Load 1D VAE from MMEngine checkpoint
- `load_2d_vae(checkpoint_path)` - Load 2D VAE from HuggingFace checkpoint
- `infer_1d_vae(vae_1d, motion, ...)` - Encode/decode with 1D VAE
- `infer_2d_vae(vae_2d, motion, ...)` - Encode/decode with 2D VAE
- `calculate_reconstruction_error(original, reconstructed)` - Compute MSE/MAE
- `test_vae_roundtrip(vae, motion)` - Full test with error metrics
- `main()` - Comprehensive test suite for both VAEs

**Features**:
- Handles various input tensor shapes automatically
- Includes latent normalization for 2D VAE
- Error handling and helpful error messages
- Shape verification and dimension checks
- Reconstruction quality metrics

**Use this when**: You want to start coding immediately with working examples.

### 4. **README.md** (This File)
Navigation guide and quick overview of the entire package.

**Use this when**: You're new to the documentation or need to navigate between documents.

---

## ⚡ Quick Start (Choose Your Path)

### Path A: I'm New, Give Me Everything
1. Start with: **quick_reference.md** (5 min read)
2. Then read: **PRISM_VAE_COMPLETE_GUIDE.md** Part 1-3 (15 min)
3. Run: **vae_inference_example.py** to see it work (5 min)
4. Deep dive: **PRISM_VAE_COMPLETE_GUIDE.md** Parts 4-10 as needed

### Path B: I Know VAEs, Just Show Me Code
1. Read: **quick_reference.md** (skim the TL;DR sections)
2. Copy: Loading code from **PRISM_VAE_COMPLETE_GUIDE.md** Part 1
3. Run: **vae_inference_example.py** main() function
4. Refer: **quick_reference.md** for quick lookup

### Path C: I Need Specific Information
Use this index:

| Need | See |
|------|-----|
| 5-line loading code | PRISM_VAE_COMPLETE_GUIDE.md Part 1 or quick_reference.md |
| Input/output shapes | quick_reference.md table or PRISM_VAE_COMPLETE_GUIDE.md Part 2 |
| Encode/decode API | PRISM_VAE_COMPLETE_GUIDE.md Part 3 |
| Configuration details | PRISM_VAE_COMPLETE_GUIDE.md Part 4 |
| Test data location | PRISM_VAE_COMPLETE_GUIDE.md Part 5 or quick_reference.md |
| Integration with diffusion | PRISM_VAE_COMPLETE_GUIDE.md Part 6 |
| Causal conv details | PRISM_VAE_COMPLETE_GUIDE.md Part 7 |
| Error solving | PRISM_VAE_COMPLETE_GUIDE.md Part 8 or quick_reference.md |
| Speed/memory/quality | PRISM_VAE_COMPLETE_GUIDE.md Part 9 or quick_reference.md |
| Source file locations | PRISM_VAE_COMPLETE_GUIDE.md Part 10 |
| Executable examples | vae_inference_example.py |

---

## 🎯 Key Information at a Glance

### Model Architectures

**1D VAE (AutoencoderKLPrism1D)**
- **Type**: Joint-agnostic, flattened representation
- **Input**: `[B, T, 138]` (T=121 frames, 138 = 23 joints × 6D)
- **Latent**: `[B, 16, 30]` (z_dim=16, temporal 4x downsample)
- **Output**: `[B, T, 138]`
- **Checkpoint**: `../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth` (189 MB)

**2D VAE (AutoencoderKLPrism2DTK)**
- **Type**: Joint-aware, per-joint representation
- **Input**: `[B, T, 22, 6]` (T=121, 22 joints, 6D per joint)
- **Latent**: `[B, 16, 30, 22]` (z_dim=16, temporal 4x downsample, joint dimension preserved)
- **Output**: `[B, T, 22, 6]`
- **Checkpoint**: `checkpoints/vermo_vae/` (69.6 MB)

### Loading Code (Copy & Paste)

**1D VAE**:
```python
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from hftrainer.models.motion.prism.autoencoder_kl_1d import AutoencoderKLPrism1D

config = Config.fromfile('../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/smpl_vae1d_nostatic_aug_hq.py')
vae_1d = AutoencoderKLPrism1D(**config.model.vae)
load_checkpoint(vae_1d, '../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth')
vae_1d.eval()
```

**2D VAE**:
```python
from diffusers import AutoencoderKL
import torch

vae_2d = AutoencoderKL.from_pretrained('checkpoints/vermo_vae/', torch_dtype=torch.float32)
vae_2d.eval()
```

### Encode/Decode Pattern

```python
# For both VAEs (same pattern)
with torch.no_grad():
    latent_dist = vae.encode(motion)        # motion → distribution
    latent = latent_dist.mode()              # deterministic encoding
    reconstruction = vae.decode(latent)      # latent → motion
```

### Latent Space Normalization (2D VAE only)

```python
import json

config = json.load(open('checkpoints/vermo_vae/config.json'))
latents_mean = torch.tensor(config['latents_mean']).view(1, 16, 1, 1)
latents_std = torch.tensor(config['latents_std']).view(1, 16, 1, 1)

# For diffusion model input
z_normalized = (z - latents_mean) / latents_std
```

---

## 📊 Key Statistics

| Metric | 1D VAE | 2D VAE |
|--------|--------|--------|
| **Model Size** | 189 MB | 69.6 MB |
| **Checkpoint Format** | MMEngine .pth | HuggingFace safetensors |
| **Input Channels** | 138 | 6 (per-joint) |
| **Latent Dim** | 16 | 16 |
| **Temporal Downsample** | 4x | 4x |
| **Encode Speed (V100, B=32)** | ~5 ms | ~8 ms |
| **Decode Speed (V100, B=32)** | ~3 ms | ~5 ms |
| **Reconstruction MSE** | ~0.001 | ~0.0005-0.001 |

---

## 📁 File Locations

### Checkpoints
- **1D VAE**: `../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth`
- **2D VAE**: `checkpoints/vermo_vae/` (config.json + diffusion_pytorch_model.safetensors)

### Configuration Files
- **1D Config**: `../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/smpl_vae1d_nostatic_aug_hq.py`
- **2D Config**: `checkpoints/vermo_vae/config.json`

### Test Data
- **Reconstruction**: `data/annotation/test_motionhub_recon.json`
- **Single-person**: `data/annotation/test_motionhub_1p.json`
- **Multi-person**: `data/annotation/test_motionhub_2p.json`
- **Text-to-motion**: `data/annotation/test_motionhub_t2m.json`
- **Training**: `data/annotation/train_hymotion_400h.json` (400 hours)

### Source Code
- **1D VAE**: `hftrainer/models/motion/prism/autoencoder_kl_1d.py`
- **2D VAE**: `hftrainer/models/motion/prism/autoencoder_kl_2d.py`
- **Integration**: `hftrainer/models/motion/prism/bundle.py`

---

## 🔍 Verification Checklist

Before using these models, verify:

- [ ] Both checkpoint files exist at specified paths
- [ ] Config files are accessible and readable
- [ ] Required packages installed: `mmengine`, `diffusers`, `torch`
- [ ] GPU available (optional but recommended for inference)
- [ ] Test motion data available in `data/motionhub/`

Run this to verify:
```bash
ls ../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth
ls checkpoints/vermo_vae/config.json
python vae_inference_example.py  # Run test
```

---

## 🚀 Usage Scenarios

### Scenario 1: Reconstruct Motion
Goal: Load motion → encode → decode → get reconstruction
1. See: PRISM_VAE_COMPLETE_GUIDE.md Part 3
2. Use: `vae_inference_example.py` function `infer_1d_vae()` or `infer_2d_vae()`

### Scenario 2: Extract Latent Codes (for Diffusion Model)
Goal: Motion → latent codes for diffusion training
1. See: PRISM_VAE_COMPLETE_GUIDE.md Part 6 (Integration)
2. Remember: Normalize latents using mean/std from config
3. Use: `infer_2d_vae()` returns both `latent` and `latent_normalized`

### Scenario 3: Integrate with PRISM Bundle
Goal: Use VAEs as part of full PRISM system
1. See: PRISM_VAE_COMPLETE_GUIDE.md Part 6
2. Use: `PrismBundle.encode_motion()` handles everything

### Scenario 4: Evaluate Reconstruction Quality
Goal: Measure how well VAE reconstructs motion
1. Use: `vae_inference_example.py` function `test_vae_roundtrip()`
2. Compare: MSE/MAE to benchmarks in quick_reference.md

### Scenario 5: Debug Issues
Goal: Something's not working
1. Check: quick_reference.md "Common Issues & Solutions"
2. Read: PRISM_VAE_COMPLETE_GUIDE.md Part 8 (Troubleshooting)
3. If stuck: Verify shapes match expected dimensions in Part 2

---

## 🛠️ Troubleshooting Quick Links

**Model won't load?**
- See: PRISM_VAE_COMPLETE_GUIDE.md Part 8, Issue 3

**Shape mismatch error?**
- See: PRISM_VAE_COMPLETE_GUIDE.md Part 8, Issue 1

**Reconstruction quality bad?**
- See: PRISM_VAE_COMPLETE_GUIDE.md Part 8, Issue 5 & 6

**NaN losses during training?**
- See: PRISM_VAE_COMPLETE_GUIDE.md Part 8, Issue 4

**Don't know which VAE to use?**
- 1D: Better for simple applications, faster
- 2D: Better reconstruction, preserves joint structure

---

## 📖 Learning Resources

1. **Understand Basics** (10 min)
   - Read: quick_reference.md "Model Architecture Overview"

2. **Understand Deep** (30 min)
   - Read: PRISM_VAE_COMPLETE_GUIDE.md Part 2-3

3. **Understand Implementation** (60 min)
   - Read: PRISM_VAE_COMPLETE_GUIDE.md Part 7 (Causal Convolutions)
   - Reference: Source files in Part 10

4. **Practice** (30 min)
   - Run: vae_inference_example.py
   - Modify: Load your own motion data

5. **Master** (2 hours)
   - Read: All of PRISM_VAE_COMPLETE_GUIDE.md
   - Integrate: VAEs into your pipeline

---

## 📝 Document Statistics

| Document | Size | Topics | Code Examples |
|----------|------|--------|----------------|
| PRISM_VAE_COMPLETE_GUIDE.md | 22 KB | 10 parts | 15+ |
| quick_reference.md | 6.5 KB | Quick lookup | 5+ |
| vae_inference_example.py | 11 KB | 7 functions | Full examples |
| README.md | This file | Navigation | Quick reference |

**Total**: ~49.5 KB comprehensive documentation with working code

---

## 🎓 Key Concepts Summary

### VAE Architecture
- **Encoder**: Maps motion to latent distribution (mean + logvar)
- **Latent Space**: 16-dimensional bottleneck
- **Decoder**: Maps latent back to motion
- **Temporal Downsampling**: 4x compression in time dimension

### Causality
- Convolutions only use past/current frames (no future)
- Enables streaming inference
- Feature caching for smooth temporal coherence

### Representation
- **1D VAE**: 23 joints × 6D per joint = 138-D flattened
- **2D VAE**: 22 joints × 6D per joint = treated as 2D (time, joint) grid

### Normalization
- **1D**: No pre-computed normalization
- **2D**: Pre-computed mean/std for ~N(0,1) latent space (optimized for diffusion)

---

## 📞 Quick Reference Commands

```bash
# View complete guide
cat PRISM_VAE_COMPLETE_GUIDE.md | less

# View quick reference
cat quick_reference.md | less

# Run inference example
python vae_inference_example.py

# Check checkpoint exists
ls -lh ../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth
ls -lh checkpoints/vermo_vae/

# Search for specific info (example)
grep -n "Causality\|latent\|normalize" PRISM_VAE_COMPLETE_GUIDE.md
```

---

## 📦 Package Contents

```
.
├── README.md                           # This navigation guide
├── PRISM_VAE_COMPLETE_GUIDE.md        # 10-part comprehensive reference (22 KB)
├── quick_reference.md                  # Cheat sheet and quick lookup (6.5 KB)
└── vae_inference_example.py           # Executable Python examples (11 KB)
```

---

**Last Updated**: 2026-05-15  
**Status**: Complete and verified  
**Framework**: PyTorch + MMEngine (1D) + Diffusers (2D)  
**Tested On**: CUDA / CPU

---

**Next Steps**: Pick your learning path above and start with the first document!

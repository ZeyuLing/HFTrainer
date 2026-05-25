# KAFS Quick Reference

## 🎯 KAFS Location & Usage

### Where KAFS is Implemented
```
hftrainer/pipelines/motion/prism_backend.py
├── _kafs_alpha_map       [Lines 75-78] - Per-joint scaling factors
├── _kafs_mode            [Lines 75-78] - Current mode ('none', 'depth_driven', etc)
├── set_kafs_alpha()      [Lines 134-221] - Configure KAFS
└── generate_single_segment() [Lines 383-384] - Apply KAFS in denoising
```

## 🚀 How to Use KAFS

### Option 1: Python API (Recommended)
```python
from hftrainer.tools.infer import load_bundle_from_checkpoint
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

# Load model
bundle = load_bundle_from_checkpoint(cfg, checkpoint_path, device='cuda')
pipeline = PrismPipeline(bundle=bundle)

# Enable KAFS
pipeline.backend.set_kafs_alpha(mode="depth_driven")

# Generate with KAFS
output = pipeline(prompts="a person walks forward")
```

### Option 2: Modify tools/infer.py (for CLI)
**Add to parse_args():**
```python
parser.add_argument('--kafs-mode', default='none',
                    choices=['none', 'depth_driven', 'uniform', 'random', 'custom'],
                    help='KAFS mode for per-joint timestep scaling')
```

**Add to infer_prism():**
```python
pipeline.backend.set_kafs_alpha(mode=args.kafs_mode, device=args.device)
```

## 📋 KAFS Modes Explained

| Mode | Effect | Use Case |
|------|--------|----------|
| **none** | Standard baseline (no scaling) | Baseline comparison |
| **depth_driven** | Kinematic-based alphas (0.85-1.15) | Main mode - improves joint hierarchy |
| **uniform** | All alphas = 1.0 | Ablation control |
| **random** | Random alphas in [0.85, 1.15] | Ablation/robustness testing |
| **custom** | User-provided alpha values | Fine-tuning experiments |

## 📐 Depth-Driven Alpha Values (23 SMPL Joints)

```
Root Motion (0.85): Translation, Pelvis
Legs (0.90-1.10):   Hips → Knees → Ankles → Feet
Spine (1.00):       All spine joints
Arms (1.10-1.15):   Shoulders → Elbows → Wrists
```

**Key Insight:** Lower alpha for proximal joints (root), higher for distal joints (wrists)

## 🔧 Technical Details

### KAFS Timestep Scaling Formula
```
Without KAFS:   t = [t, t, t, ..., t]  (shared timestep)
With KAFS:      t_j = t × α_j          (per-joint scaling)

Effect:
- Proximal joints (α=0.85): Slower diffusion, more stable
- Distal joints (α=1.15):   Faster diffusion, more flexible
```

### Activation Requirement
KAFS only works when `config.expand_timesteps = True` in the pipeline config.

## 📊 Code Snippets

### Check KAFS Status
```python
print(f"KAFS Mode: {pipeline.backend._kafs_mode}")
print(f"Alpha Map: {pipeline.backend._kafs_alpha_map}")
```

### Custom Alpha Values
```python
import torch

# Custom alphas for each joint
custom_alphas = torch.tensor([
    0.85,  # trans
    0.85,  # pelvis
    0.90, 0.90,  # hips
    # ... (23 total values)
])

pipeline.backend.set_kafs_alpha(
    mode="custom",
    alpha_vals=custom_alphas,
    device='cuda'
)
```

## 🎬 Example: T2M Generation with KAFS

```bash
# Standard (no KAFS)
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_1frame.py \
    --checkpoint work_dirs/prism_1b_tp2m_1frame/checkpoint-iter_11000 \
    --prompt "a person walks forward" \
    --output output_baseline.npz

# With KAFS (after modifying infer.py)
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_1frame.py \
    --checkpoint work_dirs/prism_1b_tp2m_1frame/checkpoint-iter_11000 \
    --prompt "a person walks forward" \
    --kafs-mode depth_driven \
    --output output_kafs.npz
```

## 📁 Related Files

### Core Implementation
- `hftrainer/pipelines/motion/prism_backend.py` (854 lines)
  - `PrismARPipeline` class with KAFS support

### Inference Entry Points
- `tools/infer.py` - Main CLI tool (needs modification)
- `hftrainer/pipelines/motion/prism_pipeline.py` - Wrapper (KAFS-ready)

### PRISM Configurations
- `configs/prism/prism_1b_tp2m_1frame.py` (Main T2M config)
- `configs/prism/prism_mcm_motionhub.py` (Motion-conditioned variant)

### Evaluation
- `scripts/eval/eval_m2m_v2_t2m.py` (HyMotion T2M eval, not PRISM)

## ❓ FAQ

**Q: Is KAFS currently exposed in CLI?**
A: No. You must either:
1. Modify `tools/infer.py` to add KAFS arguments, OR
2. Use Python API directly

**Q: What's the best KAFS mode for inference?**
A: Start with `depth_driven` - it's based on kinematic principles and should improve motion quality.

**Q: Does KAFS require retraining?**
A: No. KAFS is an inference-time technique. Just set it after loading a trained model.

**Q: Which models support KAFS?**
A: PRISM models using `PrismARPipeline`. HyMotion doesn't have KAFS support.

**Q: How much does KAFS slow down inference?**
A: Negligible - it's just element-wise multiplication of timesteps.

---

**For detailed information, see KAFS_SEARCH_REPORT.md**

# PRISM Checkpoint and Inference Configuration Report

## Summary
This report documents all PRISM-related checkpoint paths, inference scripts, and configuration files found in `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

---

## 1. LATEST CHECKPOINT PATHS

### PRISM T2M Single-Frame Model
- **Base Model (1-frame conditioning):**
  - Path: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/prism_1b_tp2m_1frame/checkpoint-iter_11000`
  - Size: 27GB (model.pt)
  - Config: `configs/prism/prism_1b_tp2m_1frame.py`
  - Status: Pre-migrated checkpoint (from iteration 11000)

### PRISM T2M Multi-Frame Model (Latest - MAIN MODEL)
- **Multi-Frame Text+Pose-to-Motion:**
  - Path: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`
  - Size: Likely 27GB+ (parent checkpoint for MCM models)
  - Config: `configs/prism/prism_1b_tp2m_multiframe.py`
  - Training Stages: 
    1. Starts from 1-frame model (checkpoint-iter_11000)
    2. Fine-tunes with multi-frame conditioning (1/5/9 frames)
    3. Final iteration: 15000

### PRISM MCM (Music-Conditioned Motion) Model
- **Music-Conditioned Motion Generator:**
  - Pretrained Parent: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`
  - Config: `configs/prism/prism_mcm_motionhub.py`
  - Status: Requires parent checkpoint to load control transformer
  - Training Config: `load_from = dict(path="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000", load_scope="model")`

### Other Model Variants
- **Spectral KT-RoPE Variant:**
  - Checkpoint: `work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0`
  - Config: `configs/prism/prism_1b_tp2m_1frame_kt_spectral.py`, `prism_1b_tp2m_multiframe_kt_spectral.py`
  
- **DFS KT-RoPE Variant:**
  - Config: `configs/prism/prism_1b_tp2m_1frame_kt_dfs.py`, `prism_1b_tp2m_multiframe_kt_dfs.py`

- **Debug/Loss Split Model:**
  - Config: `configs/prism/prism_debug_loss_split.py`
  - Loads from: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`

---

## 2. MAIN INFERENCE SCRIPT

### Primary Inference Entry Point
- **File:** `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/tools/infer.py`
- **Size:** 14.3 KB
- **Supported Models:**
  - PRISM (text-to-motion)
  - PRISM-MCM (music-conditioned motion)
  - HyMotion T2M
  - HyMotion M2M
  - VerMo (multi-task)

### PRISM-Specific Inference Functions

#### Function: `infer_prism(bundle, args)`
```python
def infer_prism(bundle, args):
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
    
    pipeline = PrismPipeline(bundle=bundle)
    output = pipeline(
        prompts=args.prompt or 'a person walks forward',
        negative_prompt=args.negative_prompt,
        first_frame_motion_path=args.first_frame_motion,
        num_frames_per_segment=args.num_frames or 33,
        num_inference_steps=args.num_steps or 4,
        guidance_scale=5.0,
        use_static=False,
        use_smooth=False,
        normalize=False,
    )
```

#### Function: `infer_prism_mcm(bundle, args)`
```python
def infer_prism_mcm(bundle, args):
    from hftrainer.pipelines.motion.prism_mcm_pipeline import PrismMCMPipeline
    
    pipeline = PrismMCMPipeline(bundle=bundle)
    # Supports audio/music loading via librosa
    output = pipeline(
        prompts=args.prompt or 'a person dances to music',
        audio=audio_tensor,
        num_frames_per_segment=args.num_frames or 33,
        num_inference_steps=args.num_steps or 4,
        guidance_scale=5.0,
    )
```

---

## 3. INFERENCE COMMAND EXAMPLES

### PRISM Text-to-Motion Inference
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person walks forward" \
    --output output/motion.npz \
    --num-frames 33 \
    --num-steps 4 \
    --guidance-scale 5.0
```

### PRISM with First-Frame Conditioning
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person walks forward" \
    --first-frame-motion condition_motion.npz \
    --output output/motion.npz \
    --num-frames 33 \
    --num-steps 4
```

### PRISM-MCM Music-Conditioned Motion
```bash
python tools/infer.py \
    --config configs/prism/prism_mcm_motionhub.py \
    --checkpoint work_dirs/prism_mcm_motionhub/checkpoint-latest \
    --prompt "a person dances to music" \
    --music music.wav \
    --output output/motion.npz \
    --num-frames 33 \
    --num-steps 4
```

---

## 4. CONFIGURATION FILES

### Main PRISM Configs (Located in `configs/prism/`)

| Config File | Model Type | Training Stage | Base Model |
|-------------|-----------|-----------------|-----------|
| `prism_1b_tp2m_1frame.py` | T2M | Stage 1 (1-frame) | Initial training |
| `prism_1b_tp2m_multiframe.py` | T2M | Stage 2 (multiframe) | Extends 1-frame |
| `prism_1b_tp2m_1frame_kt_spectral.py` | T2M + Spectral KT-RoPE | 1-frame variant | - |
| `prism_1b_tp2m_multiframe_kt_spectral.py` | T2M + Spectral KT-RoPE | Multi-frame variant | - |
| `prism_1b_tp2m_1frame_kt_dfs.py` | T2M + DFS KT-RoPE | 1-frame variant | - |
| `prism_1b_tp2m_multiframe_kt_dfs.py` | T2M + DFS KT-RoPE | Multi-frame variant | - |
| `prism_mcm_motionhub.py` | MCM (Music-conditioned) | Control Transformer | `prism_1b_tp2m_multiframe` |
| `prism_mcm_motionhub_16v100.py` | MCM (16 V100 variant) | Control Transformer | `prism_1b_tp2m_multiframe` |
| `prism_mcm_motionhub_64v100.py` | MCM (64 V100 variant) | Control Transformer | `prism_1b_tp2m_multiframe` |
| `prism_debug_loss_split.py` | T2M Debug variant | Loss analysis | `prism_1b_tp2m_multiframe` |

### Key Config Parameters (from `prism_1b_tp2m_multiframe.py`)

```python
# Model Architecture
model = dict(
    type="PrismBundle",
    transformer=dict(
        type="PrismTransformerMotionModel",
        trainable=True,
        gradient_checkpointing=True,
        module_dtype="bf16",
        num_layers=30,
        num_attention_heads=12,
        ffn_dim=8960,
        in_channels=16,
        out_channels=16,
        rope_max_seq_len=1024,
        # KT-RoPE: Kinematic-Topology Rotary Position Embedding
        joint_pos_mode="sequential",  # Options: "sequential", "spectral", "dfs"
    ),
    # Frozen components
    vae=dict(from_pretrained="checkpoints/vermo_vae"),
    text_encoder=dict(from_pretrained="checkpoints/Wan2.1-VACE-1.3B-diffusers"),
)

# Training Configuration
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1, 5, 9],  # Multi-frame conditioning
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
)

# Dataset
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        type="MotionHubSingleAgentTextDataset",
        anno_file="data/annotation/train_hq_motionhub_hymotion.json",
        data_dir="data/motionhub",
    ),
)
```

---

## 5. EVALUATION/INFERENCE SCRIPTS

### Evaluation Script for T2M
- **File:** `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval/eval_prism_t2m_hml3d.py`
- **Purpose:** Parallel multi-GPU evaluation on HumanML3D test set
- **Features:**
  - Multi-GPU parallel inference (shards test set across GPUs)
  - Per-sample NPZ output for metric computation
  - Generates motions from text captions
  
#### Usage:
```bash
python scripts/eval/eval_prism_t2m_hml3d.py \
    --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
    --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --gpus 0 1 2 3 4 5 6 7
```

### Evaluation Script Wrapper
- **File:** `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval/run_prism_eval_fixed.sh`
- **Purpose:** Bash wrapper for evaluation with preset parameters

### Other PRISM Scripts
- **Jitter Diagnostics:** `scripts/debug/diagnose_prism_jitter.py`
- **Jitter Test:** `scripts/debug/quick_prism_jitter_test.py`
- **KAFS Ablation Eval:** `scripts/eval/eval_prism_kafs_ablation.py`

---

## 6. DEPENDENCY MODELS & CHECKPOINTS

### Required Checkpoints (Referenced in Configs)

| Checkpoint | Purpose | Location |
|-----------|---------|----------|
| `checkpoints/vermo_vae` | VAE for motion encoding/decoding | `checkpoints/vermo_vae` |
| `checkpoints/Wan2.1-VACE-1.3B-diffusers` | Text encoder (UMT5) | `checkpoints/Wan2.1-VACE-1.3B-diffusers` |
| `checkpoints/BEATs_iter3_plus_AS2M.pt` | Audio encoder (for MCM) | `checkpoints/BEATs_iter3_plus_AS2M.pt` |
| `checkpoints/smpl_models/smplx` | SMPL-X model | `checkpoints/smpl_models/` |

---

## 7. INFERENCE PARAMETERS

### Key Command-Line Arguments (from `tools/infer.py`)

```python
parser.add_argument('--config', required=True, help='Path to config file (.py)')
parser.add_argument('--checkpoint', required=True, help='Path to checkpoint directory')
parser.add_argument('--prompt', help='Text prompt for generation tasks')
parser.add_argument('--first-frame-motion', help='Path to first-frame condition motion (.npz) for PRISM')
parser.add_argument('--num-frames', type=int, default=None, help='Number of output frames (default: 33)')
parser.add_argument('--num-steps', type=int, default=None, help='Number of denoising steps (default: 4)')
parser.add_argument('--guidance-scale', type=float, default=5.0, help='CFG scale (default: 5.0)')
parser.add_argument('--music', help='Music/audio wav path for dance tasks (for MCM)')
parser.add_argument('--output', help='Output file path (e.g., output.npz)')
parser.add_argument('--device', default='cuda', help='Device (cuda, cpu)')
```

### Default Values for PRISM Inference
- `num_frames_per_segment`: 33 frames
- `num_inference_steps`: 4 steps
- `guidance_scale`: 5.0
- `use_static`: False
- `use_smooth`: False
- `normalize`: False

---

## 8. WORKING DIRECTORY STRUCTURE

### PRISM Work Directories
```
work_dirs/
├── prism_1b_tp2m_1frame/
│   └── checkpoint-iter_11000/          # Base 1-frame model (27GB)
│
├── prism_1b_tp2m_multiframe/
│   └── checkpoint-iter_15000/          # Main multi-frame model (27GB+)
│
├── prism_1b_tp2m_multiframe_kt_spectral/
│   └── checkpoint-epoch_0/             # Spectral KT-RoPE variant
│
├── prism_mcm_motionhub/                # MCM training runs
│   └── (various timestamped dirs)
│
└── (other variants and ablations)
```

---

## 9. RECOMMENDED INFERENCE SETUP

### Quick Start (T2M)
```bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Single prompt inference
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person walks forward" \
    --output output/test_motion.npz \
    --device cuda:0

# Multi-step evaluation on test set
python scripts/eval/eval_prism_t2m_hml3d.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --output-dir eval_output/ \
    --gpus 0 1 2 3
```

### For First-Frame Conditioning
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person waves" \
    --first-frame-motion path/to/condition.npz \
    --output output/conditioned_motion.npz \
    --num-frames 100
```

---

## 10. KEY ARCHITECTURAL FEATURES

### Model Architecture
- **Type:** Diffusion-based motion transformer (PRISM = Pose-Relative Image-driven Sequence Motion)
- **Size:** ~1B parameters
- **Conditioning:** Text (T5-based) + Pose (multi-frame SMPL-X)
- **Architecture:** PrismTransformerMotionModel (30 layers, 12 attention heads)

### Key Features
1. **KT-RoPE (Kinematic-Topology RoPE):** Joint-aware position encoding
   - Options: sequential, spectral, dfs
2. **Multi-Frame Conditioning:** 1, 5, 9 frame options
3. **Flow Matching:** Continuous diffusion with Euler discrete scheduler
4. **FSDP Training:** Distributed training across multiple GPUs
5. **Music Conditioning (MCM):** Additional audio encoder via BEATs

---

## 11. FILES REFERENCE TABLE

| File | Type | Size | Purpose |
|------|------|------|---------|
| `tools/infer.py` | Python Script | 14.3 KB | Main inference entry point |
| `configs/prism/prism_1b_tp2m_multiframe.py` | Config | ~15 KB | Multi-frame T2M config |
| `configs/prism/prism_mcm_motionhub.py` | Config | ~12 KB | Music-conditioned variant |
| `scripts/eval/eval_prism_t2m_hml3d.py` | Python Script | ~15 KB | HML3D evaluation |
| `scripts/eval/run_prism_eval_fixed.sh` | Bash Script | ~500 B | Evaluation wrapper |
| `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000/model.pt` | Checkpoint | ~27 GB | Latest pretrained weights |

---

## Summary of Key Paths

**Latest PRISM Checkpoint (RECOMMENDED):**
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
```

**Main Inference Script:**
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/tools/infer.py
```

**Main Config (recommended):**
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/configs/prism/prism_1b_tp2m_multiframe.py
```

**Evaluation Script:**
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval/eval_prism_t2m_hml3d.py
```

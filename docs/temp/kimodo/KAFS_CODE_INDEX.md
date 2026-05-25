# KAFS Code Index

## File Tree

```
hftrainer/
├── pipelines/
│   └── motion/
│       ├── prism_backend.py          [✅ KAFS CORE]
│       ├── prism_pipeline.py         [🔗 KAFS WRAPPER]
│       ├── prism_mcm_pipeline.py
│       └── hymotion_t2m_pipeline.py
├── tools/
│   └── infer.py                      [⚠️ NEEDS KAFS]
└── configs/
    └── prism/
        ├── prism_1b_tp2m_1frame.py   [📝 T2M CONFIG]
        ├── prism_mcm_motionhub.py    [📝 MCM CONFIG]
        └── ...

scripts/
└── eval/
    └── eval_m2m_v2_t2m.py            [❌ HyMotion, not PRISM]
```

---

## 1. KAFS Core Implementation

### File: `hftrainer/pipelines/motion/prism_backend.py`
**Lines: 854 total**

#### Class: `PrismARPipeline(DiffusionPipeline)`

##### 1.1 Initialization (Lines 75-78)
```python
# KAFS-Inference: Per-joint adaptive timestep scaling
# Shape: [num_joints] with values in range [0.85, 1.15] based on kinematic depth
self._kafs_alpha_map = None
self._kafs_mode = "none"  # Tracks which KAFS mode is active
```

**Members:**
- `_kafs_alpha_map`: Tensor shape [1, 1, 1, 23] or None
- `_kafs_mode`: String tracking active mode

---

##### 1.2 Method: `set_kafs_alpha()` (Lines 134-221)

**Signature:**
```python
def set_kafs_alpha(
    self,
    mode: str = "none",
    alpha_vals: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> None:
```

**Implementation Details:**

| Mode | Lines | Implementation |
|------|-------|-----------------|
| none | 156-159 | Sets `_kafs_alpha_map = None` |
| depth_driven | 161-186 | Hardcoded 23-joint kinematic alphas |
| uniform | 188-194 | All alphas = 1.0 |
| random | 196-202 | Random [0.85, 1.15] with seed=42 |
| custom | 204-218 | User-provided tensor validation |

**Key Code Blocks:**

**Mode: depth_driven (Lines 166-182)**
```python
alpha_vals = torch.tensor([
    0.85,        # Translation (root motion, depth 0)
    0.85,        # Pelvis (depth 0)
    0.90, 0.90,  # L_Hip, R_Hip (depth 1)
    1.00,        # Spine1 (depth 1)
    1.00, 1.00,  # L_Knee, R_Knee (depth 2)
    1.00,        # Spine2 (depth 2)
    1.05, 1.05,  # L_Ankle, R_Ankle (depth 3)
    1.00,        # Spine3 (depth 3)
    1.10, 1.10,  # L_Foot, R_Foot (depth 4)
    1.00,        # Neck (depth 2)
    1.05, 1.05,  # L_Collar, R_Collar (depth 3)
    1.00,        # Head (depth 3)
    1.10, 1.10,  # L_Shoulder, R_Shoulder (depth 4)
    1.12, 1.12,  # L_Elbow, R_Elbow (depth 5)
    1.15, 1.15,  # L_Wrist, R_Wrist (depth 6)
], dtype=self.vae.dtype, device=device)

self._kafs_alpha_map = alpha_vals.view(1, 1, 1, -1)  # [1, 1, 1, 23]
```

**Output Messages:**
- Line 159: `"KAFS: Disabled (standard baseline)"`
- Line 186: `f"KAFS: Depth-driven mode enabled. Alpha range: [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]"`
- Line 194: `"KAFS: Uniform mode enabled. All alphas = 1.0"`
- Line 202: `f"KAFS: Random mode enabled. Alpha range: [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]"`
- Line 218: `f"KAFS: Custom mode enabled. Alpha range: [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]"`

---

##### 1.3 Application: `generate_single_segment()` (Lines 375-390)

**Location in denoising loop:**
```python
for i, t in enumerate(timesteps):
    # ... [setup code]
    
    if self.config.expand_timesteps:
        latent_model_input = (
            (1 - first_frame_mask) * condition + first_frame_mask * latents
        ).to(transformer_dtype)
        
        # ✅ KAFS APPLIED HERE (Lines 383-384)
        if self._kafs_alpha_map is not None:
            temp_ts = (first_frame_mask[0][0] * t * self._kafs_alpha_map).flatten()
        else:
            temp_ts = (first_frame_mask[0][0] * t).flatten()
        
        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
    else:
        latent_model_input = latents.to(transformer_dtype)
        timestep = t.expand(latents.shape[0])
```

**KAFS Effect:**
- Line 384: `temp_ts = (first_frame_mask[0][0] * t * self._kafs_alpha_map).flatten()`
- Element-wise multiplication: `t_j = t × α_j` for each joint j
- Only affects non-condition frames (where first_frame_mask == 1)
- Creates per-joint adaptive timestep in the denoising loop

**Tensor Shapes:**
- `t`: Scalar timestep
- `self._kafs_alpha_map`: [1, 1, 1, 23]
- `first_frame_mask[0][0]`: [T, 23] (T frames, 23 joints)
- `temp_ts`: [T×23] after flatten
- `timestep`: [B, T×23] after expansion (B batch size)

---

##### 1.4 Method: `prepare_latents()` (Lines 80-131)

**Prepares latents for denoising:**
- Creates random noise tensor [B, C, T_latent, J]
- Sets up condition tensor and first_frame_mask
- Used by generate_single_segment()

---

##### 1.5 Method: `__call__()` (Lines 429-558)

**Main autoregressive generation entry point:**
- Takes prompts (list of strings)
- Generates segments sequentially
- Uses extract_last_frame_motion() for autoregressive conditioning
- Returns full motion as smplx_dict

**Note:** Doesn't directly call set_kafs_alpha() - user must call it first

---

##### 1.6 Supporting Methods

| Method | Lines | Purpose |
|--------|-------|---------|
| `load_condition_pose()` | 224-258 | Load first frame condition from npz |
| `extract_last_frame_motion()` | 260-271 | Extract last frame for next segment |
| `encode_motion()` | 273-301 | Encode motion to VAE latents |
| `decode_motion()` | 560-574 | Decode latents to motion |
| `post_process_motion()` | 576-628 | Convert to SMPL-X format |
| `encode_prompt()` | 630-685 | Text encoding (T5) |

---

#### Standalone main() Function (Lines 734-848)

**Purpose:** Direct entry point for PRISM AR generation
**Usage:**
```bash
python -m hftrainer.pipelines.motion.prism_backend \
    --trainer_cfg configs/prism/prism_1b_tp2m_1frame.py \
    --trainer_ckpt work_dirs/.../checkpoint.pth \
    --prompts "A person walks;A person runs"
```

**KAFS Status:** ❌ NOT integrated into main()
**Recommendation:** Add `--kafs-mode` parameter to main()

---

## 2. KAFS Pipeline Wrapper

### File: `hftrainer/pipelines/motion/prism_pipeline.py`
**Lines: 49 total**

#### Class: `PrismPipeline(BasePipeline)`

```python
class PrismPipeline(BasePipeline):
    """HFTrainer wrapper around the vendored PRISM AR pipeline."""

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle)
        from hftrainer.pipelines.motion.prism_backend import PrismARPipeline

        self.backend = PrismARPipeline(
            tokenizer=bundle.tokenizer,
            text_encoder=bundle.text_encoder,
            vae=bundle.vae,
            scheduler=bundle.scheduler,
            smpl_processor=bundle.smpl_pose_processor,
            transformer=bundle.transformer,
        )

    def __call__(self, prompts, negative_prompt=None, ...):
        return self.backend(...)
```

**KAFS Access:** ✅ Available via `self.backend`

**Usage:**
```python
pipeline = PrismPipeline(bundle)
pipeline.backend.set_kafs_alpha(mode="depth_driven")
output = pipeline(prompts="a person walks")
```

---

## 3. Inference CLI Tool

### File: `tools/infer.py`
**Lines: 359 total**

#### Function: `infer_prism(bundle, args)` (Lines 110-129)

**Current Implementation:**
```python
def infer_prism(bundle, args):
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    pipeline = PrismPipeline(bundle=bundle)
    prompts = args.prompt or 'a person walks forward'
    output = pipeline(
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        first_frame_motion_path=args.first_frame_motion,
        num_frames_per_segment=args.num_frames or 33,
        num_inference_steps=args.num_steps or 4,
        guidance_scale=5.0,
        use_static=False,
        use_smooth=False,
        normalize=False,
    )
    # ... save output
```

**KAFS Status:** ❌ NOT integrated

**Required Changes:**
1. Add CLI argument in `parse_args()` (Line 42-74):
   ```python
   parser.add_argument('--kafs-mode', default='none',
       choices=['none', 'depth_driven', 'uniform', 'random', 'custom'])
   ```

2. Apply KAFS after pipeline creation:
   ```python
   pipeline.backend.set_kafs_alpha(mode=args.kafs_mode, device=args.device)
   ```

---

#### Function: `parse_args()` (Lines 42-74)

**Current Arguments:**
- `--config`: Config file path
- `--checkpoint`: Checkpoint path
- `--prompt`: Text prompt
- `--output`: Output path
- `--num-steps`: Inference steps
- `--num-samples`: Number of samples
- Plus 15+ other arguments

**Missing:** `--kafs-mode` parameter

---

#### Function: `main()` (Lines 315-358)

**Orchestrates the inference pipeline:**
1. Parse arguments
2. Load config
3. Load bundle from checkpoint
4. Auto-detect trainer type
5. Call appropriate inference function

**Note:** Uses trainer_type detection (line 331) to select inference function

---

## 4. PRISM Configuration Files

### File: `configs/prism/prism_1b_tp2m_1frame.py`
**Lines: 179 total** (Main T2M config)

**Structure:**
- Model config (transformer, VAE, text_encoder, scheduler)
- Trainer config (Lines 95-101)
- Data pipeline config
- Optimizer and scheduler
- Accelerator config (FSDP)

**Trainer Config (Lines 95-101):**
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
)
```

**KAFS Status:** ❌ NO KAFS settings
- No `expand_timesteps` config
- Trainer doesn't mention KAFS

**Recommendation:** Document `expand_timesteps` as optional parameter

---

### File: `configs/prism/prism_mcm_motionhub.py`
**Lines: 232 total** (Motion-conditioned music variant)

**Additional Components:**
- `control_transformer`: VACE-based architecture
- `audio_encoder`: BEATs for music conditioning
- Multi-task training setup

**KAFS Status:** ❌ NO KAFS support yet

---

## 5. Evaluation Scripts

### File: `scripts/eval/eval_m2m_v2_t2m.py`
**Lines: 400+ (not PRISM)**

**Purpose:** Multi-GPU T2M evaluation for HyMotion M2M
- Evaluates on Yiran subset (240 prompts)
- Supports CFG sweep ablations
- Multi-worker architecture

**KAFS Status:** ❌ NO KAFS
- Evaluates HyMotion, not PRISM
- HyMotion doesn't have KAFS implementation

---

## 6. Imports & Dependencies

### KAFS Dependencies in prism_backend.py
```python
import torch                          # Tensor operations
from transformers import UMT5EncoderModel  # Text encoding
from diffusers import DiffusionPipeline    # Base class
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # Scheduler
```

### KAFS-Related Types
```python
Optional[torch.Tensor]   # For alpha_vals
Optional[torch.device]   # For device placement
torch.dtype              # Data type (fp32, bf16)
```

---

## 7. Summary Table

| Component | File | Lines | Status | KAFS Support |
|-----------|------|-------|--------|--------------|
| **Core KAFS** | prism_backend.py | 75-390 | ✅ Complete | ✅ Full |
| **Pipeline Wrapper** | prism_pipeline.py | 12-48 | ✅ Complete | ✅ Via backend |
| **CLI Inference** | tools/infer.py | 110-129 | ⚠️ Partial | ❌ NO |
| **T2M Config** | prism_1b_tp2m_1frame.py | 95-101 | ✅ Complete | ❌ NO |
| **T2M Eval** | eval_m2m_v2_t2m.py | Full | ✅ Complete | ❌ HyMotion |

---

## 8. Integration Checklist

- [ ] Add `--kafs-mode` argument to `tools/infer.py`
- [ ] Call `set_kafs_alpha()` in `infer_prism()` function
- [ ] Document KAFS in config templates
- [ ] Add KAFS support to `prism_backend.py` main() if standalone usage needed
- [ ] Create PRISM-specific T2M evaluation script with KAFS
- [ ] Test all KAFS modes (none, depth_driven, uniform, random, custom)
- [ ] Benchmark performance impact
- [ ] Document in README


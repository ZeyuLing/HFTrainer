# PhysFlow Demo Data Generation - Quick Reference

**⭐ Primary Script for Demo Data:** `scripts/embodied/physflow_eval_and_export.py`

**Output Location:** `output/physflow/eval_demo/`

---

## TL;DR - Generate Demo Data in 2 Steps

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# STEP 1: Pre-compute text embeddings (one-time, ~5 min)
python3 scripts/embodied/physflow_precompute_text.py \
    --output output/physflow/text_embeddings.pt

# STEP 2: Generate eval demo (original + trained model comparison, ~20 min for quick mode)
python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/run_500iter/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick                                    # (2 prompts per level, 3 levels = 6 motions)
```

---

## What Gets Generated

### Directory Structure
```
output/physflow/eval_demo/
├── data/
│   ├── npz/                          # motion_135 NPZ files (motion data)
│   ├── smpl_mesh/                    # SMPL mesh JSON for web viewer (kinematic)
│   ├── smpl_mesh_physics/            # SMPL mesh JSON (physics-corrected)
│   ├── meta/                         # Per-motion metadata (text, stats)
│   └── sim_stats/                    # Physics simulation success rates
├── metrics.json                      # Summary table: completion rate, tracking error, etc.
└── batch_report.json
```

### Key Files in output/physflow/eval_demo/
- `metrics.json` - Comparison metrics (original vs trained model)
- `data/npz/original_*.npz` - Original model motions (motion_135 format)
- `data/npz/physflow_*.npz` - PhysFlow-trained model motions (motion_135 format)
- `data/smpl_mesh/original_*.json` - Kinematic motion for 3D viewer
- `data/smpl_mesh/physflow_*.json` - Trained model for 3D viewer
- `data/smpl_mesh_physics/original_*.json` - Physics-corrected reference
- `data/smpl_mesh_physics/physflow_*.json` - Physics-corrected trained

---

## Motion Data Format (motion_135)

**Shape:** (T, 135) where T = number of frames @ 30fps

**Layout:**
- `[0:3]` → Translation (x, y, z in meters, Y-up)
- `[3:135]` → 22 joint rotations in 6D format (row-major):
  - Each joint: 6D representation [R00, R01, R10, R11, R20, R21]
  - 22 joints total (SMPL skeleton, no hands)

**Loading in Python:**
```python
import numpy as np
data = np.load('output/physflow/eval_demo/data/npz/original_000_a_person_stands_still.npz')
motion_135 = data['motion_135']  # (T, 135) float32

transl = motion_135[:, :3]           # (T, 3) translation
rot6d = motion_135[:, 3:].reshape(motion_135.shape[0], 22, 6)  # (T, 22, 6) rotations
```

---

## Evaluation Metrics in metrics.json

### Per-Prompt Metrics
```json
{
  "prompt": "a person walks forward slowly",
  "completion_rate": 0.95,              # Physics sim success (0-1)
  "tracking_error": 0.0342,             # Joint angle error (radians)
  "correction_magnitude": 0.0125,       # How much physics changed motion
  "jerk": 0.0052,                       # Motion smoothness proxy (lower = smoother)
  "mean_speed": 0.85                    # Translation velocity (m/s)
}
```

### Comparison (original vs physflow-trained)
- `correction_reduction_pct` - How much less physics needs to correct
- `tracking_reduction_pct` - How much tracking error improved
- `jerk_reduction_pct` - How much smoother the motion became

---

## Curriculum Levels (5 difficulty levels, 27 total prompts)

| Level | Name | Frames | Prompts | Examples |
|-------|------|--------|---------|----------|
| 0 | Standing | 90 | 5 | "stands still", "shifts weight" |
| 1 | Walking | 120 | 5 | "walks forward slowly", "walks in straight line" |
| 2 | Upper Body | 90 | 6 | "waves hand", "raises both arms", "claps" |
| 3 | Transitions | 150 | 5 | "walks then turns", "walks in circle" |
| 4 | Dynamic | 120 | 6 | "kicks", "squats", "lunges", "balances on one foot" |

---

## Scripts Used in Pipeline

### 1. physflow_precompute_text.py
- **Input:** Curriculum prompts from PHYSFLOW_LEVELS
- **Output:** `text_embeddings.pt` (~226MB)
- **Time:** ~5 minutes
- **Purpose:** Pre-compute Qwen3 + CLIP-L text encodings to avoid loading 8B model during evaluation

### 2. physflow_eval_and_export.py ⭐
- **Input:** T2M model configs + checkpoint, physics sim config, text cache
- **Output:** eval_demo/ directory with motion data + metrics
- **Time:** ~20 min (quick mode), ~2 hours (full mode)
- **Purpose:** Generate motions, physics-correct, export for web visualization

**Key Options:**
- `--quick` - 2 prompts per level, 3 levels = 6 motions (FAST)
- `--no-original` - Skip original model (only eval trained)
- `--num-ode-steps` - Diffusion denoising steps (default 20)

---

## Viewing Demo Data

### 1. Programmatic Access (Python)
```python
import numpy as np
import json

# Load motion data
motion_npz = np.load('output/physflow/eval_demo/data/npz/physflow_000_a_person_stands_still.npz')
motion_135 = motion_npz['motion_135']  # (T, 135)

# Load metrics
with open('output/physflow/eval_demo/metrics.json') as f:
    metrics = json.load(f)

# Load SMPL mesh JSON for Three.js viewer
with open('output/physflow/eval_demo/data/smpl_mesh/physflow_000_a_person_stands_still.json') as f:
    mesh_data = json.load(f)
    print(f"Frames: {len(mesh_data['frames'])}")
    print(f"FPS: {mesh_data['fps']}")
```

### 2. Web Viewer
```bash
python3 motion_annot_web/embodied_viz/app.py \
    --data-dir output/physflow/eval_demo \
    --port 8095
# Then open: http://localhost:8095
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM when loading text encoder | Use `--text-cache` pointing to pre-computed embeddings |
| "No 'model_state_dict' in checkpoint" | Check checkpoint format: `torch.load(path)` and inspect keys |
| NaN in generated motions | Reduce `--num-ode-steps` (20→10) or use float32 dtype |
| Physics sim fails (low completion_rate) | Check MJCF XML validity, tune PD controller gains |
| Slow generation | Use `--quick` mode or reduce number of ODE steps |

---

## Performance Expectations

### Quick Mode (`--quick`)
- Prompts: 2 × 3 levels = 6 motions total
- Time per motion: ~2-3 minutes (gen + physics correction)
- Total time: ~20-30 minutes
- Output size: ~100MB (6 motions × data/)

### Full Mode (all 27 curriculum prompts)
- Prompts: 27 unique curriculum prompts
- Per model: 27 motions
- With original + trained: 54 motions total
- Time: ~1.5-2 hours per model
- Output size: ~2-3GB

---

## Metrics Interpretation Guide

### completion_rate (0.0 - 1.0)
- **High (>0.9):** Physics sim ran successfully, motion is physically plausible
- **Low (<0.6):** Motion has extreme values or contact geometry issues

### tracking_error_rad (radians)
- **Low (<0.05):** Motion was kinematically smooth, little deviation
- **High (>0.1):** Motion had artifacts, physics sim had to correct it

### correction_magnitude (0.0 - ??)
- **Low (<0.01):** Generated motion was already physics-friendly
- **High (>0.05):** Physics correction had to make significant changes

### jerk (smoothness proxy)
- **Low (<0.01):** Smooth, natural-looking motion
- **High (>0.05):** Jerky, unnatural motion

---

## File Naming Convention

**Original model outputs:**
```
original_NNN_prompt_slug.npz
original_NNN_prompt_slug.json
```

**PhysFlow-trained model outputs:**
```
physflow_NNN_prompt_slug.npz
physflow_NNN_prompt_slug.json
```

Example:
```
original_000_a_person_stands_still.npz
physflow_000_a_person_stands_still.npz
```

---

## Next Steps After Generation

1. **Analyze metrics** - Check metrics.json for improvements
2. **Visualize motions** - Use web viewer to see 3D comparisons
3. **Extract specific motions** - Load NPZ files for custom processing
4. **Retarget for robot** - Use `batch_t2m_to_embodied.py` to convert to G1 robot format
5. **Compare with baselines** - Run on other models for comparison

---

## Document References

- **Full guide:** See `PHYSFLOW_SCRIPTS_GUIDE.md`
- **All scripts:** See `PHYSFLOW_ALL_SCRIPTS.md`
- **This file:** `PHYSFLOW_QUICK_REFERENCE.md`

---

## Support & Debugging

For detailed information on:
- **Physics correction pipeline:** See "physflow_physics_oracle.py" section in PHYSFLOW_SCRIPTS_GUIDE.md
- **Curriculum definition:** See "physflow_curriculum.py" section
- **Motion format details:** See "Data Format Summary" section
- **Script dependencies:** See "Script Dependencies Graph" section
- **All 51 scripts:** See PHYSFLOW_ALL_SCRIPTS.md

---

**Last Updated:** 2026-05-20


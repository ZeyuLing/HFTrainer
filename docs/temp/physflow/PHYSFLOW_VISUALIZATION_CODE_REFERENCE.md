# PhysFlow 4-Panel Visualization - Code Reference Guide

## Quick Navigation

- **Main visualization script:** `scripts/embodied/physflow_visualize_compare.py`
- **Latest output:** `output/physflow_v2_compare/`
- **Related tools:** `physflow_eval_demo.py`, `physflow_rl_oracle.py`

---

## Main Comparison Pipeline (physflow_visualize_compare.py)

### Phase 1-2: Generate with Both Models

```python
# PHASE 1: Pretrained model (lines 469-488)
pretrained_motions = load_bundle_and_generate(
    config_path=args.t2m_config,
    checkpoint_path=args.pretrained_ckpt,
    text_cache_path=args.text_cache,
    prompts=prompts,
    num_frames=args.num_frames,
    device=args.device,
    finetuned_ckpt=None,  # No fine-tuning
)

# PHASE 2: Fine-tuned model (lines 495-513)
finetuned_motions = load_bundle_and_generate(
    config_path=args.t2m_config,
    checkpoint_path=args.pretrained_ckpt,
    text_cache_path=args.text_cache,
    prompts=prompts,
    num_frames=args.num_frames,
    device=args.device,
    finetuned_ckpt=args.finetuned_ckpt,  # Fine-tuned weights
)
```

### Phase 3: RL Physics Correction (Creates 4-Panel)

```python
# For EACH motion set, run RL simulation (lines 521-536)
pretrained_stats = run_rl_physics_evaluation(
    pretrained_motions, prompts, npz_dir, label="pretrained")

finetuned_stats = run_rl_physics_evaluation(
    finetuned_motions, prompts, npz_dir, label="finetuned")

# Result: 4 files per prompt
# - pretrained_00_prompt_raw.npz
# - pretrained_00_prompt_rl.npz
# - finetuned_00_prompt_raw.npz
# - finetuned_00_prompt_rl.npz
```

### Phase 4: Generate Comparison Report

```python
# Generate human-readable + JSON reports (lines 541-548)
print_comparison_report(
    pretrained_stats, finetuned_stats, prompts, args.output_dir)

# Outputs:
# - output/physflow_v2_compare/comparison_report.txt
# - output/physflow_v2_compare/comparison_results.json
```

---

## Core Function: run_rl_physics_evaluation()

**Location:** Lines 194-257

This function creates the NPZ files for the 4-panel visualization:

```python
def run_rl_physics_evaluation(
    motions_135: List[np.ndarray],
    prompts: List[str],
    output_dir: str,
    label: str,  # "pretrained" or "finetuned"
) -> List[dict]:
    """Run RL physics simulation on generated motions.
    
    Produces 2 NPZ files per motion (raw + RL corrected).
    """
    from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle
    
    oracle = RLPhysicsOracle()
    results = []
    
    for i, (motion_135, prompt) in enumerate(zip(motions_135, prompts)):
        if motion_135 is None:
            results.append({'status': 'skipped', 'prompt': prompt})
            continue
        
        print(f"  RL sim [{i+1}/{len(motions_135)}] ({label}): \"{prompt}\"")
        
        # Run RL correction
        motion_135_rl, stats = oracle.correct(motion_135)
        
        # Save 2 NPZ files (PANEL 1 & 2 or PANEL 3 & 4)
        safe_name = prompt.replace(' ', '_')[:40]
        
        # Panel 1/3: Raw generation
        npz_raw = os.path.join(
            output_dir, 
            f"{label}_{i:02d}_{safe_name}_raw.npz"
        )
        np.savez(npz_raw, motion_135=motion_135, fps=30, prompt=prompt)
        
        # Panel 2/4: RL-corrected
        npz_rl = os.path.join(
            output_dir, 
            f"{label}_{i:02d}_{safe_name}_rl.npz"
        )
        np.savez(npz_rl, motion_135=motion_135_rl, fps=30, prompt=prompt)
        
        # Extract metrics
        status = stats.get('status', 'unknown')
        completion = stats.get('completion_ratio', 0)
        root_h = stats.get('root_height_min', 0)
        
        print(f"    Status={status}, completion={completion:.2f}, "
              f"root_h_min={root_h:.3f}")
        
        results.append({
            'prompt': prompt,
            'label': label,
            'status': status,
            'completion_ratio': completion,
            'root_height_min': root_h,
            'npz_raw': npz_raw,
            'npz_rl': npz_rl,
            # ... + many more metrics
        })
    
    return results
```

---

## 4-Panel Structure Visualization

```
OUTPUT DIRECTORY: output/physflow_v2_compare/npz/

For prompt i="a person stands still":

PANEL 1                          PANEL 2
┌──────────────────────┐        ┌──────────────────────┐
│ Pretrained Model     │        │ Pretrained + RL      │
│ (Raw Generation)     │   →    │ (Physics Corrected)  │
│ motion_135 (T, 135)  │        │ motion_135_rl        │
└──────────────────────┘        └──────────────────────┘
pretrained_00_*.npz            pretrained_00_*_rl.npz
    ↓                               ↓
    Generate with T2M             Run RL Oracle.correct()
    
PANEL 3                          PANEL 4
┌──────────────────────┐        ┌──────────────────────┐
│ Fine-tuned Model     │        │ Fine-tuned + RL      │
│ (Raw Generation)     │   →    │ (Physics Corrected)  │
│ motion_135 (T, 135)  │        │ motion_135_rl        │
└──────────────────────┘        └──────────────────────┘
finetuned_00_*.npz             finetuned_00_*_rl.npz
    ↓                               ↓
    Generate with T2M             Run RL Oracle.correct()
    (+ fine-tuned weights)

COMPARISON METRICS:
- completion_ratio: (actual_sim_steps / total_sim_steps)
- root_height_min: minimum height during physics sim
- status: 'success' (reached end) or 'fell' (fell over)
- delta: finetuned_completion - pretrained_completion
```

---

## Comparison Report Output

**File:** `output/physflow_v2_compare/comparison_report.txt`

```
================================================================================
PhysFlow Visualization Comparison Report
================================================================================

Prompt                                   |      Pretrained      |      Fine-tuned      | Δ         
-----------------------------------------------------------------------------------------------
a person stands still                    |  fell c=0.41 h=0.26  |  fell c=0.88 h=0.27  | +0.47     
a person stands in a relaxed pose        |  fell c=0.21 h=0.29  |  fell c=0.20 h=0.29  | -0.01     
a person shifts weight from left to ri.. |  fell c=0.42 h=0.27  |  fell c=0.50 h=0.26  | +0.08     
a person walks forward at a normal pace  |  fell c=0.38 h=0.26  |  fell c=0.88 h=0.30  | +0.50     
...

SUMMARY STATISTICS
  Total prompts evaluated: 19
  Pretrained:
    Avg completion:   0.438
    Success rate:     0/19 (0.0%)
  Fine-tuned (PhysFlow blend50, iter 500):
    Avg completion:   0.500
    Success rate:     2/19 (10.5%)
  Improvement:
    Avg completion:   +0.062
    Success rate:     +2 (+10.5%)

PER-CATEGORY BREAKDOWN
  standing (n=3): pretrained=0.282 → finetuned=0.493 (Δ=+0.212)
  walking (n=7): pretrained=0.446 → finetuned=0.549 (Δ=+0.103)
  upper_body (n=3): pretrained=0.448 → finetuned=0.398 (Δ=-0.050)
  dynamic (n=6): pretrained=0.502 → finetuned=0.498 (Δ=-0.004)

================================================================================
NPZ files saved in output directory for 3D visualization.
Use motion_annot_web/embodied_viz to view side-by-side.
================================================================================
```

---

## JSON Output Structure

**File:** `output/physflow_v2_compare/comparison_results.json`

```json
{
  "prompts": ["a person stands still", ...],
  
  "pretrained_stats": [
    {
      "status": "fell",
      "total_ref_frames": 120,
      "total_sim_steps": 200,
      "actual_sim_steps": 81,
      "fall_frame": 80,
      "root_height_min": 0.2622689575820713,
      "completion_ratio": 0.405,
      "control_dt": 0.02,
      "oracle_time_s": 0.8899613451212645,
      "npz_raw": "output/physflow_v2_compare/npz/pretrained_00_..._raw.npz",
      "npz_rl": "output/physflow_v2_compare/npz/pretrained_00_..._rl.npz",
      "prompt": "a person stands still",
      "label": "pretrained"
    },
    ...
  ],
  
  "finetuned_stats": [...],
  
  "summary": {
    "pretrained_avg_completion": 0.438,
    "finetuned_avg_completion": 0.500,
    "pretrained_success_rate": 0.0,
    "finetuned_success_rate": 0.105,
    "improvement_completion": 0.062,
    "improvement_success_rate": 0.105263
  }
}
```

---

## Test Prompts (19 Curriculum Levels)

**Location:** Lines 406-431 in `physflow_visualize_compare.py`

```python
TEST_PROMPTS = [
    # Standing (easy - level 0)
    "a person stands still",
    "a person stands in a relaxed pose",
    "a person shifts weight from left to right foot",
    
    # Walking (medium - level 1)
    "a person walks forward at a normal pace",
    "a person walks in a small circle",
    "a person walks forward slowly",
    "a person walks with long strides",
    
    # Upper body (medium - level 2)
    "a person waves with their right hand",
    "a person raises both arms above their head",
    "a person claps their hands together",
    "a person stretches arms to the sides",
    
    # Transitions (hard - level 3)
    "a person walks and then stops",
    "a person walks forward then turns around",
    "a person jogs slowly then walks",
    
    # Dynamic (hardest - level 4)
    "a person kicks with their right leg",
    "a person squats down and stands back up",
    "a person jumps in place",
    "a person does a jumping jack",
    "a person does a high kick",
]
```

---

## Related Visualization Code

### Alternative: physflow_eval_demo.py (3-way comparison)

```python
# Evaluation demo with baseline comparison
# Outputs: baseline_tag.npz, v5_tag.npz, v5_rl_tag.npz
# + SMPL mesh JSON files for web viewer

def motion_135_to_mesh_json(motion_135: np.ndarray, fps: int = 30) -> dict:
    """Convert motion_135 array to SMPL mesh JSON for web viewer."""
    T = motion_135.shape[0]
    transl = motion_135[:, :3]
    rot6d = motion_135[:, 3:].reshape(T, 22, 6)
    aa = rot6d_to_axis_angle_np(rot6d)
    
    # Build SMPL-X format (55 joints)
    root_orient = aa[:, 0, :]
    body_pose = aa[:, 1:22, :]
    
    poses_per_frame = np.zeros((T, 165), dtype=np.float32)
    poses_per_frame[:, :3] = root_orient
    poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)
    
    # Export JSON frames
    frames = []
    for t in range(T):
        frame = [{
            "id": 0,
            "gender": "neutral",
            "smpl_type": "smplx",
            "Rh": [root_orient[t].tolist()],
            "Th": [transl[t].tolist()],
            "poses": [poses_per_frame[t].tolist()],
            "shapes": [[0.0] * 16],
            "mocap_framerate": fps,
        }]
        frames.append(frame)
    
    return {"type": "frames", "fps": fps, "frames": frames}
```

---

## Command-line Usage

```bash
# Standard 4-panel comparison
python3 scripts/embodied/physflow_visualize_compare.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --pretrained-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --finetuned-ckpt output/physflow_v2_train_blend50/model_iter500.pt \
    --text-cache output/physflow_v2_test/text_embeddings.pt \
    --output-dir output/physflow_v2_compare \
    --device cuda:0 \
    --num-frames 120 \
    --seed 42

# Output:
# output/physflow_v2_compare/
# ├── comparison_report.txt
# ├── comparison_results.json
# └── npz/ (76 NPZ files = 19 prompts × 4 panels)
```

---

## Viewing the Results

**From comparison_report.txt (line 366-367):**
```
NPZ files saved in output directory for 3D visualization.
Use motion_annot_web/embodied_viz to view side-by-side.
```

**External viewers:**
- `motion_annot_web` - Motion annotation + visualization web tool
- `embodied_viz` - 3D embodied motion viewer
- Supports SMPL mesh rendering via three.js/Babylon.js

---

## Key Data Structures

### motion_135 Format
```
Shape: (T, 135) where T = number of frames

[0:3]        → Translation (X, Y, Z)
[3:9]        → Root rotation (6D representation)
[9:135]      → Body pose (21 joints × 6D)

Total: 3 + 6 + (21 × 6) = 135 dimensions
```

### NPZ File Contents
```python
{
  'motion_135': np.ndarray (T, 135) - motion kinematics
  'fps': int - 30 fps
  'prompt': str - text prompt used
  [optional] 'rl_status': str - 'success' or 'fell'
}
```

### Stats Dictionary (from RLPhysicsOracle)
```python
{
  'status': 'fell' | 'success',
  'completion_ratio': float (0-1),
  'root_height_min': float (meters),
  'fall_frame': int (frame number when fell, or -1),
  'total_sim_steps': int (expected steps),
  'actual_sim_steps': int (steps before falling),
  'oracle_time_s': float (seconds),
  'npz_raw': str (path),
  'npz_rl': str (path),
}
```

---

## Integration with RLPhysicsOracle

**File:** `scripts/embodied/physflow_rl_oracle.py`

```python
from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle

oracle = RLPhysicsOracle()

# Correct a single motion
motion_135_rl, stats = oracle.correct(motion_135)

# Returns:
# - motion_135_rl: corrected motion (typically shorter)
# - stats: physics metrics
```

---

## Summary of Files

| File | Size | Purpose |
|------|------|---------|
| `physflow_visualize_compare.py` | 21.3 KB | Main 4-panel comparison |
| `physflow_rl_oracle.py` | 39 KB | RL physics correction |
| `physflow_eval_demo.py` | 19.7 KB | Demo with optional baseline |
| `comparison_report.txt` | ~2 KB | Text report |
| `comparison_results.json` | ~50 KB | Structured metrics |
| `npz/*.npz` | 76 files | Motion data (4 per prompt) |


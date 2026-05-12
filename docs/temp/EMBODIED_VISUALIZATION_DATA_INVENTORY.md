# Embodied Pipeline: Complete Data Inventory & Visualization Opportunities

**Date**: 2026-05-12  
**Project**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## Executive Summary

The embodied pipeline (HyMotion → SMPL-X → G1 Robot) produces **rich multi-modal data** suitable for interactive comparison visualization:

✅ **Videos**: Reference (reference motion) vs. Tracked (ONNX policy simulation)  
✅ **Per-frame data**: Joint positions/velocities, body positions, tracking errors  
✅ **Metrics**: Stability (root height), tracking fidelity, motion quality  
✅ **Multiple test cases**: 10 HyMotion motions with varying success rates (80% stable)  

---

## 1. Directory Structure & Data Assets

### `data/embodied_debug/` (30 MB total)

```
data/embodied_debug/
├── Motion Cache (.pt files) - 33-body state at 50Hz
│   ├── g1_motion_cache.pt           (446 KB, 159 frames, 3.2s)
│   ├── g1_standing_cache*.pt        (3 variants with different heights)
│   ├── g1_walking_cache_fixed.pt    (446 KB, ~200 frames)
│   ├── hymotion_real_cache.pt       (1.4 MB, 487 frames, 9.7s)  ★ Real data
│   ├── hymotion_real_cache_*.pt     (Height variants: h1.66/h1.8/h2.0)
│   └── pipeline_test_*.pt           (3 test motions: 00001/00002/00005)
│
├── Intermediate Formats
│   ├── *_smplx.npz                 (SMPL-X output: pose_body, root_orient, trans)
│   ├── *_gmr.pkl                   (GMR retarget PKL: root_pos, root_rot, dof_pos)
│   └── *_retarget.pkl              (GMR output before FK conversion)
│
├── Renders/ (13 MB)                ★ Main visualization asset
│   ├── ref_motion_00000.mp4        (2.5 MB, reference motion video)
│   ├── ref_motion_00001.mp4        (1.9 MB)
│   ├── ref_motion_00003_FELL.mp4   (1.8 MB, tracking failure case)
│   ├── tracked_motion_00000.mp4    (1.4 MB, ONNX policy result)
│   ├── ref_*_frame_*.png           (Key frames for reference motions)
│   └── tracked_00000/              (100 individual PNG frames, ~51KB each)
│
└── Analysis Scripts (8 .py files)
    ├── analyze_motion_quality.py    (Joint ranges, height, velocities)
    ├── compare_motions.py           (Multi-motion comparison)
    ├── compare_standing_vs_ref.py   (Tracking quality analysis)
    └── [others: debug, test utilities]
```

### `output/` (26 GB - mostly eval results)
Contains many historical evaluation runs. Relevant for embodied:
- Directories named `eval_v2_e9_*` contain multi-motion evaluation results
- Would need investigation to extract embodied-specific outputs

---

## 2. Motion Cache Format (Detailed)

### File: `*.pt` (PyTorch tensors)

**Keys in cache dict:**
```python
{
    'dof_pos':      (T, 29)      # Joint angles [rad] — 29 DOFs for G1
    'dof_vel':      (T, 29)      # Joint velocities [rad/s] — finite diff
    'body_pos':     (T, 33, 3)   # Body positions [m] — all 33 rigid bodies
    'body_rot':     (T, 33, 4)   # Body rotations (xyzw) — quaternions
    'body_vel':     (T, 33, 3)   # Body linear velocities [m/s]
    'body_ang_vel': (T, 33, 3)   # Body angular velocities [rad/s]
    'control_dt':   float        # Time step = 0.02s (50 Hz)
    'num_frames':   int          # Total frames T
}
```

**Example: `hymotion_real_cache.pt`**
- **487 frames** @ 50Hz = **9.7 seconds** of motion
- 29 DOF values per frame × 487 frames = 14,123 data points
- 33 body positions per frame × 487 frames = 16,071 positions

**Key insight:** Each cache contains **complete kinematic state** needed to reconstruct motion, compare tracking errors, and compute metrics.

---

## 3. Video Assets (13 MB Total)

### Reference Motions (Ground Truth)
| File | Size | Frames/Duration | Motion Type | Status |
|------|------|-----------------|-------------|--------|
| `ref_motion_00000.mp4` | 2.5 MB | 1280×720 | Standing motion | ✅ Stable tracking |
| `ref_motion_00001.mp4` | 1.9 MB | 1280×720 | Walking motion | ✅ Stable tracking |
| `ref_motion_00003_FELL.mp4` | 1.8 MB | 1280×720 | Crouching (aggressive) | ❌ Tracking failed |

**What they show:**
- Direct rendering of motion cache (qpos set directly, no simulation)
- G1 robot in 1280×720 MuJoCo visualization
- Shows what the **reference motion looks like** before ONNX policy

### Tracked Motion (ONNX Simulation Result)
| File | Size | Frames/Duration | Status |
|------|------|-----------------|--------|
| `tracked_motion_00000.mp4` | 1.4 MB | 1280×720 | ✅ Successful tracking |

**What it shows:**
- **ONNX tracker policy running in closed-loop** (not direct playback)
- Same motion (00000) with policy trying to track reference
- Smaller file indicates **tracking divergence** or shorter duration
- Can compare frame-by-frame against `ref_motion_00000.mp4`

### Frame Sequences
- `tracked_00000/`: 100 PNG frames (~51KB each = 5.1 MB) for motion 00000
- Can be reconstructed into video or used for frame-level analysis
- Pixel-level comparison possible between reference and tracked

---

## 4. Intermediate Data (Debugging & Reprocessing)

### SMPL-X Format (NPZ files)
- `hymotion_real_smplx.npz`: SMPL-X body pose (23 joints) + root position/orientation
- `pipeline_test_00001_smplx.npz`: Intermediate for pipeline test

**Size**: ~28-81 KB per motion
**Content**: 3D joint positions in human skeleton space
**Use case**: Debug motion conversion step

### GMR IK Output (PKL files)
- `hymotion_real_retarget.pkl`: GMR's inverse kinematics solution
  - `root_pos`: (T, 3) pelvis translation
  - `root_rot`: (T, 4) pelvis rotation (xyzw quaternion)
  - `dof_pos`: (T, 29) robot joint angles
  - `fps`: Source frame rate (e.g., 30 Hz)

**Size**: ~28-83 KB per motion
**Use case**: Understand GMR IK accuracy, debug coordinate frame conversions

---

## 5. Rendering Pipeline & Output Modes

### File: `scripts/embodied/render_tracker_headless.py`

**Two rendering modes:**

#### Mode 1: `--mode reference` (Fast, no ONNX needed)
```bash
python scripts/embodied/render_tracker_headless.py \
    --motion data/embodied_debug/g1_motion_cache.pt \
    --output-dir /tmp/render_ref \
    --mode reference \
    --video
```
- Sets `qpos` directly from motion cache (no physics simulation)
- Renders all frames with zero velocity
- Fast: 1000 render fps
- **Output**: PNG frames + optional MP4
- **Use case**: Visualize reference motion, generate ground truth videos

#### Mode 2: `--mode tracked` (Slower, requires ONNX model)
```bash
python scripts/embodied/render_tracker_headless.py \
    --motion data/embodied_debug/g1_motion_cache.pt \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --output-dir /tmp/render_tracked \
    --mode tracked \
    --video
```
- Runs ONNX policy in **closed-loop simulation**
- Uses MuJoCo physics with PD actuators to track reference
- Slower but shows **real tracking performance**
- **Output**: PNG frames showing policy result

**Configurable parameters:**
- `--skip-frames N`: Render every N-th frame (default: 2)
- `--width, --height`: Resolution (default: 1280×720)
- `--max-frames N`: Limit frames processed
- `--camera-distance, --elevation, --azimuth`: Camera position
- `--video-fps`: Video playback speed

**Output formats:**
- Individual PNG frames with frame number overlay
- MP4 video (tries libx264 → h264_nvenc → mpeg4)
- ~50KB per PNG frame at 1280×720

---

## 6. Available Metrics & Quality Data

### From `analyze_motion_quality.py`

**Per-joint analysis:**
- Min/max/mean DOF positions per joint (29 DOFs)
- DOF velocity ranges (identifies abrupt/explosive motions)
- Joint ranges vs. reference dataset

**Motion-level analysis:**
- Pelvis height (Z) over time: Identifies standing stability
  - Stable: 0.77–0.79m throughout
  - Failed: Drops to 0.27m (robot fell)
- Pelvis rotation: Should be near-identity for standing, only yaw for turning

**Tracking quality:**
- Root height drift: Initial vs. final Z position
- Max reference tracking error (normalized distance to reference pose)
- Failure indicators: Sudden height drops, high DOF velocities

### From debug report (`docs/temp/embodied_pipeline_debug_report.md`)

**10-motion test results:**
```
Motion | Init root_h | Final root_h | Max ref err | Status
00000  | 0.776       | 0.787        | 0.6492     | ✅ STABLE
00001  | 0.770       | 0.780        | 0.5836     | ✅ STABLE
00002  | 0.778       | 0.784        | 0.5948     | ✅ STABLE
00003  | 0.780       | 0.273        | 1.1986     | ❌ FELL
00004  | 0.774       | 0.786        | 0.5953     | ✅ STABLE
00005  | 0.550       | 0.079        | 2.4172     | ❌ FELL
00006  | 0.778       | 0.783        | 0.6991     | ✅ STABLE
00007  | 0.780       | 0.790        | 0.6525     | ✅ STABLE
00008  | 0.546       | 0.784        | 1.4872     | ✅ STABLE (recovered)
00009  | 0.777       | 0.783        | 0.7634     | ✅ STABLE

Success Rate: 80% (8/10)
```

**Metrics captured:**
- Pelvis height stability (root_h)
- Tracking error (max reference error)
- Motion difficulty classification
- ONNX policy performance under different conditions

---

## 7. What Data Is Available for Comparison Website

### ✅ Currently Available

1. **Side-by-side video comparison**
   - Reference motion (direct playback)
   - Tracked motion (ONNX policy result)
   - Both already rendered as MP4

2. **Frame-by-frame inspection**
   - 100 PNG frames for tracked motion 00000
   - Can generate PNG sequences for all reference motions
   - Overlay frame numbers, metadata

3. **Performance metrics per motion**
   - Root height over time (stability curve)
   - Tracking error over time (from motion cache)
   - DOF angles per joint
   - Joint velocity profiles

4. **Motion classification**
   - Standing vs. walking vs. complex
   - Stability outcome (pass/fail)
   - Difficulty assessment

5. **Kinematic data for visualization**
   - 33 body positions per frame (skeleton visualization)
   - Joint angles (29 DOF plots)
   - Velocity profiles (for trend analysis)

### 🔧 Easily Derivable (from motion cache)

- **Tracking error**: Euclidean distance between reference and tracked body positions
- **Joint tracking error**: Per-DOF error plot
- **Foot skating detection**: Foot velocity when contact expected
- **Ground penetration**: Z-position of ankle/foot below ground
- **Energy metrics**: Kinetic + potential energy over time
- **Smoothness**: Velocity variance per joint
- **Height stability**: Z variance over motion duration

### 📊 Suggested Visualization Components

```
┌─────────────────────────────────────────────────┐
│  Embodied Pipeline Comparison Dashboard         │
├──────────────────────┬──────────────────────────┤
│  Motion Selector     │  Metrics Summary         │
│  (dropdown: 00-09)   │  • Success: 80% (8/10)  │
│                      │  • Root H: 0.776→0.787  │
├──────────────────────┼──────────────────────────┤
│  [▶] Reference Video │  [▶] Tracked Video      │
│  1280×720, 2.5 MB    │  1280×720, 1.4 MB       │
│  Time: 0:00 / 3:00   │  Time: 0:00 / 2:50      │
├──────────────────────┴──────────────────────────┤
│  Per-Frame Analysis                             │
│  ┌─────────────────────────────────────────┐   │
│  │ Root Height Over Time (m)               │   │
│  │ 0.8 ├─────┐                             │   │
│  │     │ Ref ├─ Tracked                    │   │
│  │ 0.7 ├─────┘                             │   │
│  │     0    60   120  180  240  300        │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │ DOF Tracking Error (rad)                │   │
│  │ 1.0 ├─ Max error over all joints        │   │
│  │     ├─                                  │   │
│  │ 0.5 │                                   │   │
│  │     ├─                                  │   │
│  │ 0.0 └───────────────────────────────    │   │
│  │     0    60   120  180  240  300        │   │
│  └─────────────────────────────────────────┘   │
├────────────────────────────────────────────────┤
│  Joint Inspector (29 DOF)                      │
│  Left Hip Pitch: ┌──────────────────┐          │
│                  │ Min: -0.5 rad    │          │
│                  │ Max: +1.2 rad    │          │
│                  │ Mean: 0.3 rad    │          │
│                  └──────────────────┘          │
│  [prev] [next] Filter by: ○ All ○ Active      │
└────────────────────────────────────────────────┘
```

---

## 8. Current Test Dataset

**10 HyMotion motions tested** (from eval split E2_B):
- **8 successful** (80%): Motions with normal standing poses
- **2 failed** (20%):
  - Motion 00003: Extreme crouching (pelvis drops to 0.44m) → robot can't track
  - Motion 00005: Starts in squat (0.55m) → non-standing pose → falling

**Data location**: Results in `data/embodied_debug/` with cached `.pt` files and renders

**Test coverage:**
- Walking (00001)
- Standing with motion (00000, 00004, 00006, 00007, 00009)
- Recovery scenarios (00008: starts at 0.546m but recovers to 0.784m)

---

## 9. Pipeline Data Flow (For Context)

```
Input: HyMotion eval NPZ (motion_135 format)
  ↓ motion135_to_smplx.py
Output: SMPL-X NPZ (pose_body, root_orient, trans)
  ↓ gmr_retarget_headless.py
Output: GMR PKL (root_pos, root_rot, dof_pos)
  ↓ gmr_to_protomotions.py [FK + resampling + velocities]
Output: ProtoMotions cache .pt (✨ Core visualization asset)
  ├─ render_tracker_headless.py --mode reference
  │  → ref_motion_00000.mp4 + ref_*_frame_*.png
  └─ render_tracker_headless.py --mode tracked
     → tracked_motion_00000.mp4 + tracked_00000/*.png
```

**Each .pt file contains:**
- 33 body positions/rotations (per frame)
- 29 DOF angles/velocities (per frame)
- Control timestep and duration

**Data size:** ~1-1.5 MB per motion (487-500 frames @ 50Hz)

---

## 10. Recommendations for Comparison Website

### Data to Display

| Component | Source | Format | Update Freq |
|-----------|--------|--------|-------------|
| Reference video | MP4 | Video stream | Static |
| Tracked video | MP4 | Video stream | Static |
| Root height curve | Motion cache .pt | Time series | Per motion |
| DOF positions | Motion cache .pt | 29 × T table | Per motion |
| Tracking error | Derived from .pt | Time series | Per motion |
| Motion summary | debug_report.md | Text/metrics | Static |
| Frame inspector | PNG sequences | Image carousel | Per motion |

### Technical Stack Suggestions

**Backend:**
- Load `.pt` files with PyTorch (or NumPy via pickle)
- Compute metrics on-the-fly or cache as JSON
- Serve MP4s and PNG sequences via HTTP

**Frontend:**
- Video.js or HTML5 `<video>` for MP4 playback
- Chart.js or Plotly for time series (height, error, DOF)
- Interactive frame slider (scrub through 100 PNGs)
- Motion selector (dropdown: 00-09)
- Responsive design for tablet/desktop

**Deployment:**
- Static files: MP4s + PNGs (pre-generated, ~20 MB total)
- Dynamic data: Motion cache metrics (regenerated per session or cached as JSON)
- Lightweight server: Flask/FastAPI (just need to load `.pt` and compute derivatives)

---

## 11. Files You Can Access Today

All data is available at:
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
```

**Start here:**
1. `data/embodied_debug/renders/` — Watch the videos
2. `data/embodied_debug/hymotion_real_cache.pt` — Load and inspect motion data
3. `docs/temp/embodied_pipeline_debug_report.md` — Read performance summary
4. `scripts/embodied/render_tracker_headless.py` — Understand rendering pipeline
5. `scripts/embodied/gmr_to_protomotions.py` — Understand motion cache format

---

## Summary Table

| Aspect | What You Have | Ready for Web? |
|--------|---------------|----------------|
| **Videos** | 5 MP4 files (2-3 MB each) | ✅ Yes |
| **Frame sequences** | 100 PNGs for motion 00000 | ✅ Yes |
| **Metrics** | Root height, DOF angles, tracking error | ✅ Yes (derive from .pt) |
| **Multi-motion data** | 10 motions tested | ✅ Yes |
| **Success/failure cases** | 8 pass, 2 fail (80% rate) | ✅ Yes |
| **Per-frame kinematic data** | 33 body positions × 487 frames | ✅ Yes (in .pt) |
| **Velocity profiles** | DOF & body velocities | ✅ Yes (in .pt) |
| **Ground truth labels** | Stability outcome per motion | ✅ Yes |

**Total size of visual assets:** ~13 MB (fits in any web deployment)


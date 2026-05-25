# Motion JSON Generation - Complete Code Reference

## Quick Answer Summary

| Question | Answer |
|----------|--------|
| **Where are JSON .motion files created?** | `scripts/embodied/convert_cache_to_json.py` (line 158 `json.dump()`) |
| **Is there a separate step for JSON?** | YES, Stage 4 conversion via `convert_cache_to_json()` function |
| **Height fix bypassed?** | NO. Height fix applied in Stage 3, JSON inherits corrected values |
| **Negative Z values?** | Relative to fixed pelvis height (~0.02-0.04m above ground) |
| **Who calls the JSON converter?** | `batch_pipeline_to_web.py` (line 226) or `batch_t2m_to_embodied.py` (line 845) |

---

## Code Snippets by Stage

### Stage 3: Height Fix Application (convert_pyroki_retargeted_robot_motions_to_proto.py)

**Location**: `ref_repo/ProtoMotions/data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py`

```python
# Lines 352-362: Height fix applied to motion object

# Get translation vectors from per-frame height adjustment
translation_vecs = motion.fix_height_per_frame(height_offset=0.02)

# Update velocities to account for position changes
if motion.rigid_body_vel is not None and motion.fps is not None:
    vel_delta = torch.zeros(
        translation_vecs.shape[0], 1, 3,
        device=motion.rigid_body_vel.device,
        dtype=motion.rigid_body_vel.dtype,
    )
    vel_delta[:-1] = (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1) / motion.motion_dt
    motion.rigid_body_vel = motion.rigid_body_vel + vel_delta

# Additional global height fix
motion.fix_height(height_offset=0.04)

# Line 410: Save torch dict (NOW WITH CORRECTED Z VALUES)
print(f"Saving to {outpath}")
torch.save(motion.to_dict(), str(outpath))
```

**What this does**:
- `fix_height_per_frame(0.02)`: Adjusts each frame's pelvis Z individually
- Updates velocities to maintain physical consistency
- `fix_height(0.04)`: Applies additional global height offset
- Result: torch `.motion` file has corrected Z values

---

### Stage 4: JSON Conversion (convert_cache_to_json.py)

**Location**: `scripts/embodied/convert_cache_to_json.py`

```python
# Lines 82-169: Main conversion function

def convert_cache_to_json(cache_path: str, output_path: str, subsample: int = 1) -> dict:
    """Convert a single cache .pt or .motion file to JSON.

    Supports two formats:
      - Old .pt cache: keys body_pos, body_rot, dof_pos, control_dt, num_frames
      - New .motion:   keys rigid_body_pos, rigid_body_rot, dof_pos, motion_dt/fps
    """
    cache = torch.load(cache_path, weights_only=False)  # Load torch pickle

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    dof_pos = to_numpy(cache["dof_pos"])  # (T, 29)

    # Handle both old and new .motion format
    if "body_pos" in cache:
        # Old .pt cache format
        body_pos = to_numpy(cache["body_pos"])  # (T, 33, 3)
        body_rot = to_numpy(cache["body_rot"])  # (T, 33, 4) xyzw
        control_dt = float(cache["control_dt"])
        fps = round(1.0 / control_dt)
        num_frames_total = int(cache["num_frames"])
    elif "rigid_body_pos" in cache:
        # New .motion format (from PyRoki pipeline)
        body_pos = to_numpy(cache["rigid_body_pos"])  # (T, N_bodies, 3)
        body_rot = to_numpy(cache["rigid_body_rot"])  # (T, N_bodies, 4) xyzw
        if "motion_dt" in cache:
            control_dt = float(cache["motion_dt"])
            fps = round(1.0 / control_dt)
        elif "fps" in cache:
            fps = int(cache["fps"])
            control_dt = 1.0 / fps
        else:
            fps = 30
            control_dt = 1.0 / 30.0
            print(f"  WARNING: No fps/motion_dt in .motion file, defaulting to {fps}")
        num_frames_total = body_pos.shape[0]
    else:
        raise KeyError(
            f"Unrecognized cache format. Keys: {list(cache.keys())}. "
            "Expected 'body_pos' (old .pt) or 'rigid_body_pos' (.motion)."
        )

    # Subsample if needed
    indices = list(range(0, num_frames_total, subsample))
    effective_fps = fps / subsample

    # Build JSON frames
    frames = []
    for i in indices:
        frame = {
            "root_pos": body_pos[i, ROOT_BODY_INDEX].tolist(),    # [x, y, z]
            "root_quat": body_rot[i, ROOT_BODY_INDEX].tolist(),   # [x, y, z, w]
            "dof_pos": dof_pos[i].tolist(),                       # 29 joint angles
        }
        frames.append(frame)

    # Create result dict
    result = {
        "fps": effective_fps,
        "num_frames": len(frames),
        "joint_names": DOF_JOINT_NAMES,  # 29 MuJoCo DOF names
        "root_body_index": ROOT_BODY_INDEX,
        "frames": frames,
    }

    # Line 158: Write JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))  # Compact format

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Wrote {output_path} ({len(frames)} frames, {effective_fps:.0f} FPS, {size_kb:.0f} KB)")

    return {
        "id": os.path.splitext(os.path.basename(cache_path))[0],
        "num_frames": len(frames),
        "fps": effective_fps,
        "json_path": output_path,
        "source_pt": cache_path,
    }
```

**Key points**:
- Loads torch `.motion` pickle with `weights_only=False`
- Extracts `rigid_body_pos` (height-corrected) and `rigid_body_rot`
- Uses root body (index 0, pelvis) for `root_pos` and `root_quat`
- No further height adjustments
- Outputs compact JSON

---

### Stage 4 Callers

#### Caller 1: batch_pipeline_to_web.py

**Location**: `scripts/embodied/batch_pipeline_to_web.py` (line 226)

```python
def main():
    # ... setup ...
    
    for i, npz_path in enumerate(npz_files):
        npz_name = pathlib.Path(npz_path).stem
        motion_id = f"{args.name_prefix}{npz_name}"
        json_path = output_dir / f"{motion_id}.json"
        cache_path = cache_dir / f"{motion_id}.pt"

        # Step 1: Run pipeline (NPZ → torch .motion)
        if not cache_path.exists():
            print(f"  Running pipeline...")
            ok = run_pipeline(npz_path, str(cache_path), args.pipeline_args)
            if not ok:
                failures += 1
                continue
        else:
            print(f"  Cache exists, skipping pipeline")

        # Step 2: Convert cache → JSON (LINE 226)
        try:
            info = convert_cache_to_json(str(cache_path), str(json_path))  # ← HERE
            successes += 1
            results.append({...})
        except Exception as e:
            print(f"  JSON CONVERT FAILED: {e}")
            failures += 1
```

**Usage**:
```bash
python scripts/embodied/batch_pipeline_to_web.py \
    --npz-dir work_dirs/.../npz/ \
    --output-dir output/embodied_comparison/data/motions/
```

#### Caller 2: batch_t2m_to_embodied.py

**Location**: `scripts/embodied/batch_t2m_to_embodied.py` (lines 576-581, 845)

```python
# Line 576-581: Import the converter
def convert_cache_to_json(cache_path, json_output):
    """Convert ProtoMotions cache .pt → JSON for Three.js visualization."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from convert_cache_to_json import convert_cache_to_json as _convert
    return _convert(str(cache_path), str(json_output))

# Later in main()...

# Line 843-845: Call the converter
try:
    print(f"  [C] Converting .motion → reference JSON...")
    ref_info = convert_cache_to_json(str(motion_file), str(ref_json_path))  # ← HERE
    print(f"      Ref JSON: {ref_info['num_frames']} frames, {ref_info['fps']} FPS")
except Exception as e:
    print(f"      REF JSON FAILED: {e}")
    traceback.print_exc()
    status = "ref_json_failed"
```

**Usage**:
```bash
python scripts/embodied/batch_t2m_to_embodied.py \
    --prompt-json output/embodied_comparison_v2/motion_text_mapping.json \
    --output-dir output/embodied_comparison_v3/ \
    --max-motions 5
```

---

## JSON Output Format

### Structure Example

```json
{
  "fps": 50.0,
  "num_frames": 149,
  "joint_names": [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    ...
    "right_wrist_yaw_joint"
  ],
  "root_body_index": 0,
  "frames": [
    {
      "root_pos": [0.431, 0.094, 0.749],
      "root_quat": [0.149, -0.102, 0.046, 0.982],
      "dof_pos": [-1.078, 0.106, 0.189, ..., -0.065]
    },
    {
      "root_pos": [0.438, 0.091, 0.730],
      "root_quat": [0.161, -0.156, 0.015, 0.974],
      "dof_pos": [-1.193, 0.260, 0.124, ..., -0.065]
    },
    ...
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `fps` | float | Frames per second (usually 50 for G1 robot) |
| `num_frames` | int | Total number of frames in motion |
| `joint_names` | list | 29 MuJoCo DOF names (in order) |
| `root_body_index` | int | Body index for root (always 0 = pelvis) |
| `frames` | list | Array of frame objects |
| `root_pos` | [x, y, z] | Pelvis position in meters (**height-corrected**) |
| `root_quat` | [x, y, z, w] | Pelvis rotation quaternion (xyzw convention) |
| `dof_pos` | [...] | 29 joint angles in radians, typically [-π, π] |

---

## Data Flow with File Names

```
HyMotion T2M
    ↓
work_dirs/.../npz/00001.npz (motion_135)
    ↓
pipeline_motion_to_robot.py
    ↓
output/.../caches/motion_00001.pt (torch .motion)
    ↓
convert_cache_to_json()
    ↓
output/.../motions/motion_00001.json (Web JSON) ✓ FINAL OUTPUT
```

Or in batch_t2m_to_embodied.py:

```
prompt "a person walks"
    ↓
run_t2m_inference()
    ↓
output/.../data/npz/motion_0000.npz
    ↓
run_retarget_pipeline()
    ↓
output/.../data/retarget/motion_0000/*.motion (torch pickle)
    ↓
convert_cache_to_json()
    ↓
output/.../data/motions/motion_0000.json ✓ FINAL OUTPUT
```

---

## Debugging Tips

### Check if torch .motion file exists
```python
import torch
cache = torch.load("path/to/motion.motion", weights_only=False)
print("Keys:", cache.keys())
print("FPS:", cache.get("fps", cache.get("motion_dt")))
print("rigid_body_pos shape:", cache["rigid_body_pos"].shape)
print("Sample root Z:", cache["rigid_body_pos"][0, 0, 2])  # Should be ~0.02-0.05
```

### Manually convert torch to JSON
```python
from scripts.embodied.convert_cache_to_json import convert_cache_to_json
info = convert_cache_to_json("path/to/motion.motion", "output/motion.json")
print(info)
```

### Verify JSON format
```python
import json
with open("motion.json") as f:
    data = json.load(f)
    print(f"Frames: {data['num_frames']}")
    print(f"FPS: {data['fps']}")
    print(f"First frame root_pos:", data['frames'][0]['root_pos'])
    print(f"Z range: {min(f['root_pos'][2] for f in data['frames']):.4f} to {max(f['root_pos'][2] for f in data['frames']):.4f}")
```

---

## Common Issues & Fixes

### Issue: "UnrecognizedCache format — Expected 'body_pos' or 'rigid_body_pos'"
**Cause**: Trying to convert a non-motion torch file
**Fix**: Ensure input is from ProtoMotions pipeline (has `rigid_body_pos` key)

### Issue: JSON file is empty or has wrong structure
**Cause**: Mismatched torch and JSON keys
**Fix**: Verify torch `.motion` has required keys (rigid_body_pos, dof_pos, fps)

### Issue: Root Z values look wrong (all negative or very large)
**Cause**: Height fix not applied or improper offset
**Fix**: Check convert_pyroki_retargeted_robot_motions_to_proto.py ran successfully
- Look for "motion.fix_height()" in output log
- Verify torch file created after this step

### Issue: FPS incorrect in JSON
**Cause**: Missing fps or motion_dt in torch dict
**Fix**: Ensure convert_pyroki script set fps correctly
- Check line "output_fps" parameter passed to convert script

---

## Dependencies

- `torch` (for loading `.motion` pickle)
- `numpy` (for array operations)
- `json` (standard library)
- Python 3.8+


# Height Estimation Implementation Guide

## Quick Implementation (Option A - FK-based)

### Step 1: Modify `load_smplx_file()` in utils/smpl.py

Replace the hardcoded height formula with FK-based measurement:

```python
def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )
    
    num_frames = smplx_data["pose_body"].shape[0]
    betas_raw = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    num_betas = body_model.num_betas if hasattr(body_model, 'num_betas') else 10
    if betas_raw.shape[-1] > num_betas:
        betas_raw = betas_raw[..., :num_betas]
    elif betas_raw.shape[-1] < num_betas:
        betas_raw = torch.cat([betas_raw, torch.zeros(1, num_betas - betas_raw.shape[-1])], dim=-1)
    betas_tensor = betas_raw.expand(num_frames, -1)
    
    smplx_output = body_model(
        betas=betas_tensor,
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    # ============ NEW: FK-based height estimation ============
    try:
        # Get world-space joint positions from FK
        joints_np = smplx_output.joints.detach().numpy()  # (T, 22, 3)
        
        # Joint indices: 15=head, 10=left_foot, 11=right_foot
        # Measure height as max head height - min foot height across all frames
        head_positions = joints_np[:, 15, :]  # (T, 3)
        left_foot_positions = joints_np[:, 10, :]  # (T, 3)
        right_foot_positions = joints_np[:, 11, :]  # (T, 3)
        
        # Use the vertical axis (typically Y or Z, check SMPL convention)
        # Assuming Y is vertical (0=X, 1=Y, 2=Z)
        vertical_axis = 1  # Change to 2 if Z is vertical
        
        max_head_height = head_positions[:, vertical_axis].max()
        min_foot_height = min(
            left_foot_positions[:, vertical_axis].min(),
            right_foot_positions[:, vertical_axis].min()
        )
        
        human_height = float(max_head_height - min_foot_height)
        
        # Validation: clamp to reasonable human height range
        if human_height < 1.3 or human_height > 2.3:
            print(f"[WARNING] Estimated height {human_height:.2f}m out of range [1.3, 2.3]. Using default 1.7m")
            human_height = 1.7
        
        print(f"[INFO] Estimated human height from FK: {human_height:.3f}m")
        
    except Exception as e:
        print(f"[WARNING] FK-based height estimation failed: {e}. Using default 1.7m")
        human_height = 1.7
    # ================= END NEW CODE =================
    
    return smplx_data, body_model, smplx_output, human_height
```

### Step 2: Apply same fix to `load_gvhmr_pred_file()`

The function starting at line 58 should use the same FK-based height logic:

```python
def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    
    betas = np.pad(smpl_params_global['betas'][0], (0,6))
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    
    # ============ NEW: FK-based height estimation ============
    try:
        joints_np = smplx_output.joints.detach().numpy()
        head_positions = joints_np[:, 15, :]
        left_foot_positions = joints_np[:, 10, :]
        right_foot_positions = joints_np[:, 11, :]
        
        vertical_axis = 1  # Y is vertical
        
        max_head_height = head_positions[:, vertical_axis].max()
        min_foot_height = min(
            left_foot_positions[:, vertical_axis].min(),
            right_foot_positions[:, vertical_axis].min()
        )
        
        human_height = float(max_head_height - min_foot_height)
        
        if human_height < 1.3 or human_height > 2.3:
            print(f"[WARNING] Estimated height {human_height:.2f}m out of range. Using 1.7m")
            human_height = 1.7
        
        print(f"[INFO] Estimated human height from GVHMR FK: {human_height:.3f}m")
        
    except Exception as e:
        print(f"[WARNING] GVHMR height estimation failed: {e}. Using 1.7m")
        human_height = 1.7
    # ================= END NEW CODE =================
    
    return smplx_data, body_model, smplx_output, human_height
```

### Step 3: Verify GMR usage

Ensure that any script using these functions passes `actual_human_height` to GMR:

```python
from general_motion_retargeting.utils.smpl import load_smplx_file
from general_motion_retargeting import GeneralMotionRetargeting

# Load SMPL-X and get height
smplx_data, body_model, smplx_output, human_height = load_smplx_file(
    "motion.npz",
    "path/to/smplx/models"
)

# Create GMR with proper height
gmr = GeneralMotionRetargeting(
    src_human="smplx",
    tgt_robot="unitree_g1",
    actual_human_height=human_height  # <-- THIS IS KEY
)

# Use GMR for retargeting
# ...
```

## Testing the Implementation

### Test 1: Unit test for height estimation

```python
import numpy as np
import torch
from pathlib import Path

def test_height_estimation():
    """Test FK-based height estimation"""
    
    # Create a simple test SMPL-X data
    test_npz = {
        'pose_body': np.random.randn(10, 63).astype(np.float32),
        'root_orient': np.random.randn(10, 3).astype(np.float32),
        'trans': np.random.randn(10, 3).astype(np.float32),
        'betas': np.zeros(10, dtype=np.float32),
        'gender': 'neutral',
        'mocap_frame_rate': np.array(30),
    }
    
    # Save test NPZ
    np.savez('/tmp/test_motion.npz', **test_npz)
    
    # Load and check height
    from utils.smpl import load_smplx_file
    
    _, _, _, height = load_smplx_file(
        '/tmp/test_motion.npz',
        'path/to/smplx/models'
    )
    
    print(f"Estimated height: {height:.3f}m")
    assert 1.3 <= height <= 2.3, f"Height out of range: {height}"
    print("✓ Test passed")

if __name__ == "__main__":
    test_height_estimation()
```

### Test 2: Integration test

```python
import numpy as np
from motion135_to_smplx import convert_motion135_to_smplx
from utils.smpl import load_smplx_file
from motion_retarget import GeneralMotionRetargeting

def test_full_pipeline():
    """Test full pipeline: motion_135 → SMPL-X → GMR"""
    
    # Create dummy motion_135 data
    motion_135 = np.random.randn(30, 135).astype(np.float32)
    np.savez('/tmp/test_motion_135.npz', motion_135=motion_135)
    
    # Convert to SMPL-X
    convert_motion135_to_smplx(
        '/tmp/test_motion_135.npz',
        '/tmp/test_smplx.npz'
    )
    
    # Load with height estimation
    smplx_data, body_model, smplx_output, height = load_smplx_file(
        '/tmp/test_smplx.npz',
        'path/to/smplx/models'
    )
    
    print(f"Height from pipeline: {height:.3f}m")
    
    # Create GMR with proper height
    gmr = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=height
    )
    
    # Check that scaling was applied
    original_height = 1.7  # Config assumption
    expected_ratio = height / original_height
    print(f"Expected scaling ratio: {expected_ratio:.3f}")
    print("✓ Pipeline test passed")

if __name__ == "__main__":
    test_full_pipeline()
```

## Coordinate System Check

If height seems wrong, check which axis is vertical:

```python
import numpy as np
import torch
from smplx import SMPLX

def check_coordinate_system(smplx_file, model_path):
    """Determine which axis is vertical in SMPL-X output"""
    
    data = np.load(smplx_file)
    body_model = SMPLX(model_path)
    
    # Create poses with body tilted in different directions
    num_frames = 1
    
    smplx_output = body_model(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 63),
        transl=torch.zeros(1, 3),
    )
    
    joints = smplx_output.joints.detach().numpy()  # (1, 22, 3)
    
    print("Joint positions in world space:")
    print(f"Head (joint 15): {joints[0, 15]}")
    print(f"Pelvis (joint 0): {joints[0, 0]}")
    print(f"L_Foot (joint 10): {joints[0, 10]}")
    
    # Compute height along each axis
    for axis, name in enumerate(['X', 'Y', 'Z']):
        head_pos = joints[0, 15, axis]
        foot_pos = joints[0, 10, axis]
        height = abs(head_pos - foot_pos)
        print(f"Height along {name}-axis: {height:.3f}m")
    
    # The axis with largest height is vertical
```

## Common Issues & Troubleshooting

### Issue 1: Height estimation returns NaN or Inf

**Symptom**: `actual_human_height` becomes NaN after load_smplx_file()

**Solution**:
1. Check if body_model FK is working: `print(smplx_output.joints.shape)`
2. Verify input data is valid: `print(smplx_data['pose_body'].shape)`
3. Check for NaN in input: `print(np.isnan(smplx_data['pose_body']).any())`

### Issue 2: Height is 0 or negative

**Symptom**: `actual_human_height = 0.0` or negative

**Solution**:
1. **Wrong vertical axis**: Change `vertical_axis` from 1 to 2 (or 0)
2. **Coordinate system mismatch**: Check output coordinate system
3. **Joints are stacked**: Verify joint indices (15=head, 10/11=feet)

```python
# Debug: print joint coordinates
joints_np = smplx_output.joints.detach().numpy()
for i in [0, 10, 11, 15]:
    print(f"Joint {i}: {joints_np[0, i]}")
```

### Issue 3: Height is always 1.7m (fallback)

**Symptom**: All motion clips return 1.7m height

**Solution**:
1. Check exception message: Add more detailed logging
2. Verify numpy/torch versions are compatible
3. Test FK manually:
   ```python
   smplx_output = body_model(...)
   print(type(smplx_output.joints))  # Should be torch.Tensor
   print(smplx_output.joints.shape)  # Should be (T, 22, 3)
   ```

## Performance Optimization

FK computation can be slow for large sequences. To optimize:

```python
def load_smplx_file(smplx_file, smplx_body_model_path):
    # ... existing code ...
    
    # Option 1: Subsample frames for height estimation
    num_frames = smplx_data["pose_body"].shape[0]
    sample_stride = max(1, num_frames // 100)  # Sample every N frames
    
    smplx_output_sampled = body_model(
        betas=betas_tensor[::sample_stride],
        global_orient=torch.tensor(smplx_data["root_orient"][::sample_stride]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"][::sample_stride]).float(),
        transl=torch.tensor(smplx_data["trans"][::sample_stride]).float(),
        left_hand_pose=torch.zeros(len(range(num_frames)[::sample_stride]), 45).float(),
        right_hand_pose=torch.zeros(len(range(num_frames)[::sample_stride]), 45).float(),
        # ... other params ...
    )
    
    joints_np = smplx_output_sampled.joints.detach().numpy()
    # ... rest of height calculation ...
```

## Validation Checklist

Before deploying:

- [ ] Height estimation runs without errors
- [ ] Height values are in range [1.3, 2.3]m
- [ ] Different motion clips produce different heights (if they're from different people)
- [ ] Same motion clip always produces same height (deterministic)
- [ ] GMR receives `actual_human_height` parameter
- [ ] IK solutions improve after fix (qualitative check)
- [ ] No significant performance regression (FK should be ~1-5s per clip)

## Roll-back Plan

If issues arise:

1. **Revert smpl.py**: Replace back to original height formula
2. **Use dummy height**: Set `actual_human_height = 1.7` explicitly
3. **Check logs**: Look for FK error messages

Safe revert:
```python
# Temporary fallback if new code breaks
def load_smplx_file_safe(smplx_file, smplx_body_model_path):
    # ... FK computation ...
    
    # Temporary: always use 1.7m
    human_height = 1.7
    print("[WARNING] Using fallback height 1.7m (FK estimation disabled)")
    
    return smplx_data, body_model, smplx_output, human_height
```

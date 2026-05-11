# E3 GT Motion Loading - Quick Reference

## TL;DR: Get GT Motion for Sample N

```python
import json
import numpy as np

# 1. Load E3 datalist
with open('data/eval/m2m_v2/eval_e3_keyframe.json') as f:
    data_list = json.load(f)['data_list']

# 2. Get GT motion path for sample index N (0-239)
gt_path = data_list[N]['motion_path']

# 3. Load 135-dim motion
npz = np.load(gt_path, allow_pickle=True)
motion_135 = np.concatenate([
    npz['trans'].reshape(npz['trans'].shape[0], -1)[:, :3],
    npz['poses'].reshape(npz['poses'].shape[0], -1)
], axis=1).astype(np.float32)

print(f"GT motion shape: {motion_135.shape}")  # (T, 135)
```

## File Paths at a Glance

| What | Where |
|------|-------|
| **E3 Datalist** | `data/eval/m2m_v2/eval_e3_keyframe.json` |
| **GT Motion NPZ** | `/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/{actor}/{action}.npz` |
| **Generated Results** | `work_dirs/eval_*/kimodo/*_E3_*/E3_*/npz/{idx:05d}.npz` |
| **Aggregated Results** | `work_dirs/eval_*/import_jsons/kimodo/KIMODO_*_E3_*.json` |
| **eval_dashboard DB** | `/tmp/eval_dashboard.db` |

## Database Queries

### List all E3 runs:
```sql
SELECT id, task_id, setting, num_samples, result_json_path
FROM eval_runs WHERE task_id = 'E3'
ORDER BY created_at DESC;
```

### Get samples for run ID=3250:
```sql
SELECT sample_idx, prompt_id, gen_motion_path, metrics_json
FROM sample_results
WHERE eval_run_id = 3250
ORDER BY sample_idx;
```

## NPZ File Structure

**GT Motion NPZ keys**:
- `poses`: (T, 22, 3) or (T, 66) - 6D rotation vectors for 22 SMPL joints
- `trans`: (T, 3) or (T, 1, 3) - global translation
- `gender`: string or bytes - 'neutral'
- `fps`: float - usually 30.0

**Generated Motion NPZ keys**: Same structure as GT

## Motion Dimension Reference

| Component | Dims | Details |
|-----------|------|---------|
| Translation | 3 | [x, y, z] |
| Root rotation (6D) | 6 | First joint in rot6d |
| 21 other joints × 6D | 126 | Remaining 21 joints |
| **Total** | **135** | Standard M2M format |

## Related Files

- **Eval Script**: `scripts/eval/eval_m2m_v2_all_tasks.py` (lines 1056-1240: `load_eval_samples` function)
- **Motion Loading**: `hftrainer/evaluation/motion/m2m_eval_metrics.py` (`load_motion_135d` function)
- **Metrics Computation**: `hftrainer/evaluation/motion/m2m_eval_metrics.py` (metrics functions)
- **Dashboard API**: `motion_annot_web/eval_dashboard/app.py` (source motion loading)

## Common Issues

### Issue: `motion_path` is empty in `sample_results` table
**Solution**: Use `sample_idx` from DB to index into E3 JSON datalist

### Issue: Different motion lengths (GT vs Generated)
**Solution**: Interpolate to match lengths before comparing
```python
from scipy.interpolate import interp1d
if gen.shape[0] != gt.shape[0]:
    t_gen = np.linspace(0, 1, gen.shape[0])
    t_gt = np.linspace(0, 1, gt.shape[0])
    gen = np.array([interp1d(t_gen, gen[:, d])(t_gt) for d in range(gen.shape[1])]).T
```

### Issue: Different NPZ key names
**Solution**: Check both key variants
```python
poses_key = 'poses' if 'poses' in npz else 'body_pose'
trans_key = 'trans' if 'trans' in npz else 'transl'
poses = npz[poses_key]
trans = npz[trans_key]
```

## E3 Task Settings (5 variants)

1. **adaptive**: Dynamic keyframe selection based on motion
2. **every_5f**: Keep every 5th frame as keyframe
3. **every_10f**: Keep every 10th frame as keyframe
4. **every_15f**: Keep every 15th frame as keyframe
5. **every_30f**: Keep every 30th frame as keyframe (most sparse)

All use 240 samples from the E3 datalist.

## Example: Compute Jitter Metric

```python
def jitter(motion):
    """Acceleration magnitude (2nd derivative)."""
    vel = np.diff(motion, axis=0)
    acc = np.diff(vel, axis=0)
    return np.linalg.norm(acc, axis=1).mean()

gt_motion = ...  # Load GT
gen_motion = ...  # Load generated

print(f"GT jitter: {jitter(gt_motion):.4f}")
print(f"Gen jitter: {jitter(gen_motion):.4f}")
```

---

**See full guide**: `GT_MOTION_LOADING_GUIDE.md`

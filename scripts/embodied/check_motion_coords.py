#!/usr/bin/env python3
"""Quick diagnostic: check raw motion_135 translation values."""
import numpy as np
from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

data = np.load(NPZ, allow_pickle=True)
motion = data['motion_135']
fps = int(data.get('fps', 30))
T = motion.shape[0]

transl = motion[:, :3]
rot6d = motion[:, 3:].reshape(T, 22, 6)

print(f"Motion: T={T}, fps={fps}")
print(f"\nRaw translation (first 5 frames):")
for t in range(min(5, T)):
    print(f"  frame {t}: x={transl[t,0]:.4f}, y={transl[t,1]:.4f}, z={transl[t,2]:.4f}")

print(f"\nTranslation stats:")
print(f"  x: min={transl[:,0].min():.4f}, max={transl[:,0].max():.4f}")
print(f"  y: min={transl[:,1].min():.4f}, max={transl[:,1].max():.4f}")
print(f"  z: min={transl[:,2].min():.4f}, max={transl[:,2].max():.4f}")

# Decode root orientation
def rot6d_to_rotmat(r6d):
    shape = r6d.shape[:-1]
    r6d = r6d.reshape(-1, 6)
    r6d = r6d[:, [0, 2, 4, 1, 3, 5]]
    a1 = r6d[:, :3]
    a2 = r6d[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).reshape(*shape, 3, 3)

root_r6d = rot6d[:, 0]
root_rotmat = rot6d_to_rotmat(root_r6d)
root_euler = sRot.from_matrix(root_rotmat.reshape(-1,3,3)).as_euler('xyz', degrees=True).reshape(T, 3)

print(f"\nRoot orientation (euler xyz, degrees) first 5 frames:")
for t in range(min(5, T)):
    print(f"  frame {t}: rx={root_euler[t,0]:.1f}, ry={root_euler[t,1]:.1f}, rz={root_euler[t,2]:.1f}")

# Now check another motion (a walking one)
NPZ2 = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_walk_001.npz"
import os
if os.path.exists(NPZ2):
    d2 = np.load(NPZ2, allow_pickle=True)
    m2 = d2['motion_135']
    t2 = m2[:, :3]
    print(f"\nv4_walk_001:")
    print(f"  T={m2.shape[0]}, fps={int(d2.get('fps', 30))}")
    print(f"  y: min={t2[:,1].min():.4f}, max={t2[:,1].max():.4f}")
    print(f"  First frame: x={t2[0,0]:.4f}, y={t2[0,1]:.4f}, z={t2[0,2]:.4f}")

# Also check: what coordinate is UP in this motion?
# In Y-up: y should be ~0.9 for standing
# In Z-up: z should be ~0.9 for standing
# If neither: check which component has the standing-height range
print(f"\nHeight analysis for v4_crouch_002:")
print(f"  If Y-up: standing height (y) avg = {transl[:,1].mean():.4f}")
print(f"  If Z-up: standing height (z) avg = {transl[:,2].mean():.4f}")
print(f"  If X-up: standing height (x) avg = {transl[:,0].mean():.4f}")

# Check: which axis has the ~0.9m range typical for pelvis height?
for ax, name in [(0,'x'), (1,'y'), (2,'z')]:
    vals = transl[:, ax]
    print(f"  {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.4f}, {vals.max():.4f}]")

# List all available npz files
import glob
npz_files = sorted(glob.glob(f"{CEPH}/output/embodied_t2m_v4/data/npz/*.npz"))
print(f"\nTotal NPZ files: {len(npz_files)}")
# Check first 5 files' y values
for f in npz_files[:5]:
    d = np.load(f, allow_pickle=True)
    m = d['motion_135']
    t = m[:, :3]
    name = os.path.basename(f)
    print(f"  {name}: T={m.shape[0]}, y_avg={t[:,1].mean():.3f}, z_avg={t[:,2].mean():.3f}")

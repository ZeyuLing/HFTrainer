#!/usr/bin/env python3
"""Identify which joints have FK roundtrip errors."""
import sys, os, numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "motion_annot_web", "score_m2m"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "motion_annot_web", "score_m2m", "soma_model"))
from utils_soma import SOMASKEL30_IN_NVSKEL77, NVSKEL77_JOINT_NAMES
from soma_forward import _load_skin_model, _grm30_to_lrm30, _fk_77_numpy
import glob

os.chdir(PROJECT_ROOT)
f = sorted(glob.glob("work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_uncond_E3_adaptive_000_025/E3_adaptive/npz/*.npz"))[0]
data = np.load(f, allow_pickle=True)
gr = data["global_rot_mats"].astype(np.float32)[:10]
pj = data["posed_joints"].astype(np.float32)[:10]
T = gr.shape[0]

S30 = np.array(SOMASKEL30_IN_NVSKEL77, dtype=np.int64)
model = _load_skin_model()
lrm_30 = _grm30_to_lrm30(gr[:, S30])
lrm_77 = np.tile(np.eye(3, dtype=np.float32), (T, 77, 1, 1))
lrm_77[:, model["s30_in_77"]] = lrm_30
fk_gr, fk_pj = _fk_77_numpy(lrm_77, pj[:, 0], model)

err = np.linalg.norm(fk_pj - pj, axis=-1).mean(axis=0)  # (77,)
s30_set = set(S30.tolist())

print(f"Joint errors (top 20 by error):")
print(f"{'Idx':>4s} {'Name':>25s} {'Error(m)':>10s} {'In S30?':>8s}")
for idx in np.argsort(err)[::-1][:20]:
    name = NVSKEL77_JOINT_NAMES[idx] if idx < len(NVSKEL77_JOINT_NAMES) else f"j{idx}"
    in_s30 = "YES" if idx in s30_set else "no"
    print(f"{idx:4d} {name:>25s} {err[idx]:10.6f} {in_s30:>8s}")

print(f"\nS30 joints (all should be ~0):")
for i, idx in enumerate(S30):
    name = NVSKEL77_JOINT_NAMES[idx]
    print(f"  S30[{i:2d}] = nv77[{idx:2d}] {name:>25s}: error={err[idx]:.8f}m")

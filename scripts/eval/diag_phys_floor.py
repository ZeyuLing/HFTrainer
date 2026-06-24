#!/usr/bin/env python3
"""Diagnostic: why is GT not the best on Ground_Penetration?

For a few methods, dump foot-height statistics and the penetration metric under
several floor conventions, to expose the per-clip-percentile artifact.
"""
import os, sys, numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
import torch
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
from hftrainer.datasets.motion.representation.humanml_repr import recover_272_stored_positions

_FOOT = [10, 11]; _H = 1
_bone = torch.load(os.path.join(_ROOT, "data/hymotion_m2m_data/bone_offsets_22.pt"), map_location="cpu").float()

def wp_m135(p):
    t = torch.from_numpy(p[:, :135].astype(np.float32)); wp,_,_,_ = motion135_to_fk(t, _bone, rotation_space="local"); return wp.numpy()

def load(path, mode):
    if mode == "m135":
        d = np.load(path, allow_pickle=True)
        if "motion_135" not in d: return None
        return wp_m135(np.asarray(d["motion_135"], np.float32))
    m = np.load(path)
    if m.ndim != 2 or m.shape[1] != 272: return None
    return np.asarray(recover_272_stored_positions(m), np.float32)

def pen(foot_y, floor, tol=0.005):
    h = foot_y - floor; b = np.abs(h[h < -tol]); return float(b.mean()) if b.size else 0.0

def files(d, mode, n=40):
    pat = ".npz" if mode=="m135" else ".npy"
    fs = sorted(e.path for e in os.scandir(d) if e.name.endswith(pat))
    return fs[:n]

METHODS = [
    ("real_smpl","gt272","ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data"),
    ("ours","m135","outputs/evaluation/prism_kt_spectral_epoch17_gtlen/prep/ours_e17_gtlen"),
    ("kimodo","m135","outputs/evaluation/ms272_t1_fill_0609/prep/kimodo"),
    ("mld","m135","outputs/evaluation/ms272_t1_fill_0609/prep/mld"),
]

for tag, mode, d in METHODS:
    if not os.path.isdir(d):
        print(f"[skip] {tag}"); continue
    p_pct10=[]; p_min=[]; p_pct2=[]; p_contact=[]; fy_min=[]; fy_mean=[]; fy_std=[]; absmin=[]
    for fp in files(d, mode):
        pos = load(fp, mode)
        if pos is None or pos.shape[0] < 4: continue
        fy = pos[:, _FOOT, _H]          # (T,2)
        flat = fy.reshape(-1)
        absmin.append(float(flat.min()))
        fy_min.append(float(flat.min())); fy_mean.append(float(flat.mean())); fy_std.append(float(flat.std()))
        p_pct10.append(pen(fy, np.percentile(flat,10)))
        p_pct2.append(pen(fy, np.percentile(flat,2)))
        p_min.append(pen(fy, flat.min()))
        # contact floor: median height of the lowest-velocity (contact) foot frames
        vel = np.linalg.norm(np.diff(pos[:, _FOOT], axis=0), axis=-1)  # (T-1,2)
        vel = np.concatenate([vel, vel[-1:]],0)
        contact_h = fy[vel < 0.01]
        cf = np.median(contact_h) if contact_h.size else np.percentile(flat,10)
        p_contact.append(pen(fy, cf))
    print(f"{tag:12s} absMinFootY={np.mean(absmin):+.3f}  meanFootY={np.mean(fy_mean):.3f}  stdFootY={np.mean(fy_std):.3f}")
    print(f"{'':12s}  Penet_mm: pct10={np.mean(p_pct10)*1000:6.2f}  pct2={np.mean(p_pct2)*1000:6.2f}  min={np.mean(p_min)*1000:6.2f}  contactMed={np.mean(p_contact)*1000:6.2f}")

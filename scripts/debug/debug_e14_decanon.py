#!/usr/bin/env python3
"""Debug E14 decanon bug: run 1 sample of E14_M uncond_local and trace
the output_135 at each stage to find where the corruption happens.
"""
import numpy as np
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.pipelines.motion.transition_utils import (
    canonicalize_segment, decanonicalize_segment,
)
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_transl, process_smplx_pose,
)
from tools.eval_m2m_v2_all_tasks import (
    _place_b_custom, motion_135_to_198,
    MOTION_DIM_V2,
)
from hftrainer.evaluation.motion.m2m_eval_tasks import build_transition_mask
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

# ─── Setup ───
bone_offsets = torch.load(
    'data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu'
).float().numpy()

# Load model
from mmengine import Config
cfg = Config.fromfile('configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py')

from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bundle = HyMotionM2MBundle.from_config(cfg.model)
ckpt_path = 'work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2740/model.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
bundle.load_state_dict_selective(ckpt)
bundle = bundle.to(device).eval()
pipeline = HyMotionM2MPipeline(bundle)

print(f"Bundle mean shape: {bundle.mean.shape}")  # Should be (1,1,198)
motion_dim = MOTION_DIM_V2  # 198
print(f"motion_dim = {motion_dim}")

# ─── Load E14 sample 72 ───
with open('data/eval/m2m_v2/eval_e14_hq400h_static100.json') as f:
    items = json.load(f)['data_list']
item = items[72]

data_a = np.load(item['motion_a_path'], allow_pickle=True)
data_b = np.load(item['motion_b_path'], allow_pickle=True)
tk_a = 'trans' if 'trans' in data_a else 'transl'
tk_b = 'trans' if 'trans' in data_b else 'transl'
pk_a = 'poses' if 'poses' in data_a else 'body_pose'
pk_b = 'poses' if 'poses' in data_b else 'body_pose'

mA = np.concatenate([
    process_transl(data_a[tk_a].astype(np.float32), 'abs'),
    process_smplx_pose(data_a[pk_a].astype(np.float32), 'rotation_6d', 'smpl_22'),
], axis=-1).astype(np.float32)
mB = np.concatenate([
    process_transl(data_b[tk_b].astype(np.float32), 'abs'),
    process_smplx_pose(data_b[pk_b].astype(np.float32), 'rotation_6d', 'smpl_22'),
], axis=-1).astype(np.float32)

N_cond_a, N_cond_b, N_transition = 45, 45, 82

# ─── Place B (velocity) ───
motion_b_world_np = _place_b_custom(
    mA, mB, placement='velocity', N_transition=N_transition,
    bone_offsets=bone_offsets)

a_tail = mA[-N_cond_a:]
b_head = motion_b_world_np[:N_cond_b]
transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
world_segment = np.concatenate([a_tail, transition_pad, b_head], axis=0)
T = world_segment.shape[0]
print(f"\nT={T}, world_segment[0,:3] (a_tail[0]): {world_segment[0,:3]}")

# ─── Canonicalize ───
world_segment_t = torch.from_numpy(world_segment).float()
canon_t, R_canon, offset_canon = canonicalize_segment(
    world_segment_t, anchor_frame=0, rotation_space='local')
motion_135 = canon_t.numpy()
print(f"canonical[0,:3]: {motion_135[0,:3]}")  # Should be [0, Y, 0]

# ─── Convert to 198 ───
motion_raw = motion_135_to_198(motion_135, bone_offsets)
print(f"motion_raw shape: {motion_raw.shape}, [0,:3]: {motion_raw[0,:3]}")

# ─── Build mask ───
mask = build_transition_mask(T, 135, N_cond_a=N_cond_a,
                             N_transition=N_transition, N_cond_b=N_cond_b)
# Expand mask to 198 dims
mask_198 = np.ones((T, 198), dtype=np.float32)
mask_198[:, :135] = mask
mask_198[:N_cond_a, 135:] = 0.0
mask_198[N_cond_a + N_transition:, 135:] = 0.0
print(f"mask shape: {mask_198.shape}, cond_a sum: {mask_198[:N_cond_a].sum():.0f}, cond_b sum: {mask_198[N_cond_a+N_transition:].sum():.0f}")

# ─── Normalize + pad + inference ───
motion_norm = bundle.normalize_motion(
    torch.from_numpy(motion_raw).float().unsqueeze(0).to(device))
src_mask = torch.from_numpy(mask_198).float().unsqueeze(0).to(device)
T_PAD = 360
if T < T_PAD:
    pad_len = T_PAD - T
    motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), mode='constant', value=0.0)
    src_mask = torch.nn.functional.pad(src_mask, (0, 0, 0, pad_len), mode='constant', value=0.0)

src_motion_norm = motion_norm * (1 - src_mask)
clean_motion = motion_norm.clone()

batch = {
    'src_motion': src_motion_norm,
    'src_mask': src_mask,
    'src_length': [T],
    'tgt_length': [T],
    'clean_motion': clean_motion,
}
print(f"\nRunning inference (T_PAD={T_PAD})...")
pipeline.replacement_guidance = 'skip_last'
with torch.no_grad():
    output = pipeline(batch)
sampled_norm = output['latent']  # (1, T_PAD, 198)

# ─── Denormalize ───
output_denorm = bundle.denormalize_motion(sampled_norm)[0].cpu().numpy()
output_denorm = output_denorm[:T]
output_135 = output_denorm[:, :135]

print(f"\n=== BEFORE condition replacement ===")
print(f"output_135[0,:3] (should be near canonical[0,:3]=[0,Y,0]): {output_135[0,:3]}")
print(f"output_135[0,3:9] (pelvis rot6d): {output_135[0,3:9]}")
print(f"motion_135[0,:3] (canonical GT): {motion_135[0,:3]}")
print(f"motion_135[0,3:9] (canonical GT rot6d): {motion_135[0,3:9]}")
diff_before = np.abs(output_135[:N_cond_a, :3] - motion_135[:N_cond_a, :3]).max()
print(f"max trans diff in cond_a BEFORE replacement: {diff_before:.6f}")

# ─── Condition replacement ───
mask_135_only = mask_198[:T, :135]
cond_mask = (mask_135_only < 0.5)
print(f"\ncond_mask sum: {cond_mask.sum()} (should be {N_cond_a*135 + N_cond_b*135}={N_cond_a*135+N_cond_b*135})")
output_135_replaced = output_135.copy()
output_135_replaced[cond_mask] = motion_135[cond_mask]
print(f"\n=== AFTER condition replacement ===")
print(f"output_135[0,:3]: {output_135_replaced[0,:3]}")
diff_after = np.abs(output_135_replaced[:N_cond_a, :3] - motion_135[:N_cond_a, :3]).max()
print(f"max trans diff in cond_a AFTER replacement: {diff_after:.8f}")

# ─── Decanonicalize ───
out_t = torch.from_numpy(output_135_replaced).float()
out_world_t = decanonicalize_segment(out_t, R_canon, offset_canon, rotation_space='local')
output_world = out_world_t.numpy()

print(f"\n=== AFTER decanonicalize ===")
print(f"output_world[0,:3] (should match world_segment[0,:3]={world_segment[0,:3]}): {output_world[0,:3]}")
diff_world = np.abs(output_world[0,:3] - world_segment[0,:3])
print(f"  diff: {diff_world}")
print(f"  total: {np.linalg.norm(diff_world):.6f}m")

# ─── Compare with saved NPZ ───
npz_path = 'work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz/00072.npz'
if os.path.exists(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    npz_135 = d['motion_135'].astype(np.float32)
    print(f"\n=== Compare with existing NPZ ===")
    print(f"NPZ[0,:3]: {npz_135[0,:3]}")
    print(f"Our decanon[0,:3]: {output_world[0,:3]}")
    print(f"Diff: {npz_135[0,:3] - output_world[0,:3]}")
    print(f"NPZ cond_a vs our decanon cond_a max diff: {np.abs(npz_135[:N_cond_a] - output_world[:N_cond_a]).max():.6f}")
else:
    print(f"\nNPZ not found at {npz_path}")

print("\n=== DONE ===")

#!/usr/bin/env python3
"""A/B test: run E14 inference with and without ground-anchored canonicalization
on the worst-sliding cases, then compare foot_skating_ratio.

Runs on lzy_debug_machine GPU.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
from hftrainer.pipelines.motion.transition_utils import (
    canonicalize_segment, decanonicalize_segment,
)
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_transl, process_smplx_pose,
)
from tools.eval_m2m_v2_all_tasks import (
    _place_b_custom, motion_135_to_198, MOTION_DIM_V2,
)
from hftrainer.evaluation.motion.m2m_eval_tasks import build_transition_mask

WORST_PIDS = [0, 33, 20, 40, 48, 73, 45, 55, 34, 80, 16, 86, 53, 69]

def load_pair(items, pid, bone_offsets, N_cond_a=45, N_cond_b=45):
    item = items[pid]
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

    # Get N_transition from existing NPZ
    npz_path = f'work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz/{pid:05d}.npz'
    if os.path.exists(npz_path):
        layout = json.loads(bytes(np.load(npz_path, allow_pickle=True)['layout_json']).decode())
        N_transition = layout['N_transition']
    else:
        N_transition = 82

    motion_b_world = _place_b_custom(
        mA, mB, placement='velocity', N_transition=N_transition,
        bone_offsets=bone_offsets)

    return mA, mB, motion_b_world, N_cond_a, N_cond_b, N_transition


def build_input(mA, motion_b_world, N_cond_a, N_cond_b, N_transition,
                bone_offsets, use_ground_anchor=False):
    """Build canonical input segment for E14."""
    a_tail = mA[-N_cond_a:]
    b_head = motion_b_world[:N_cond_b]
    transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
    world_segment = np.concatenate([a_tail, transition_pad, b_head], axis=0)
    T = world_segment.shape[0]

    world_segment_t = torch.from_numpy(world_segment).float()

    if use_ground_anchor:
        bo_t = torch.from_numpy(bone_offsets).float()
        canon_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0, rotation_space='local',
            bone_offsets=bo_t, ground_anchor='y_min')
    else:
        canon_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0, rotation_space='local')

    motion_135 = canon_t.numpy()
    motion_raw = motion_135_to_198(motion_135, bone_offsets)

    mask = build_transition_mask(T, 135, N_cond_a=N_cond_a,
                                 N_transition=N_transition, N_cond_b=N_cond_b)
    # Expand to 198
    pos_mask = np.zeros((T, 63), dtype=np.float32)
    for j in range(21):
        rot_mask_val = mask[:, 3 + (j + 1) * 6]
        pos_mask[:, j * 3:(j + 1) * 3] = rot_mask_val[:, None]
    mask_198 = np.concatenate([mask, pos_mask], axis=-1)

    return motion_raw, mask_198, motion_135, R_canon, offset_canon, T


def run_inference(bundle, pipeline, motion_raw, mask_198, T, device):
    """Run a single inference pass."""
    motion_norm = bundle.normalize_motion(
        torch.from_numpy(motion_raw).float().unsqueeze(0).to(device))
    src_mask = torch.from_numpy(mask_198).float().unsqueeze(0).to(device)

    T_PAD = 360
    if T < T_PAD:
        pad_len = T_PAD - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0.0)
        src_mask = torch.nn.functional.pad(src_mask, (0, 0, 0, pad_len), value=0.0)

    src_motion_norm = motion_norm * (1 - src_mask)
    clean_motion = motion_norm.clone()

    batch = {
        'src_motion': src_motion_norm,
        'src_mask': src_mask,
        'src_length': [T],
        'tgt_length': [T],
        'clean_motion': clean_motion,
    }

    pipeline.replacement_guidance = 'skip_last'
    with torch.no_grad():
        output = pipeline(batch)

    sampled_norm = output['latent']
    output_denorm = bundle.denormalize_motion(sampled_norm)[0].cpu().numpy()[:T]
    output_135 = output_denorm[:, :135]

    return output_135


def compute_slide_ratio(motion_135, bone_offsets, gen_start, gen_end):
    positions = motion135_to_positions_np(motion_135, bone_offsets)
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]
    bad = 0; total = 0
    FOOT_JOINTS = [7, 8, 10, 11]
    for fi in range(1, N):
        for j in FOOT_JOINTS:
            if gen_pos[fi, j, 1] < 0.08:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > 0.015:
                    bad += 1
    return bad / max(1, total)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bone_offsets = torch.load(
        'data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu').float().numpy()

    # Load model
    from mmengine import Config
    cfg = Config.fromfile('configs/hymotion_m2m/hymotion_m2m_uncond_local_046b.py')
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    bundle = HyMotionM2MBundle.from_config(cfg.model)
    ckpt = torch.load(
        'work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2740/model.pt',
        map_location='cpu')
    bundle.load_state_dict_selective(ckpt)
    bundle = bundle.to(device).eval()
    pipeline = HyMotionM2MPipeline(bundle)

    with open('data/eval/m2m_v2/eval_e14_hq400h_static100.json') as f:
        items = json.load(f)['data_list']

    print(f"{'pid':>5}  {'baseline':>10}  {'grounded':>10}  {'delta':>8}")
    print('-' * 42)

    for pid in WORST_PIDS:
        # Use same seed for fair comparison
        seed = 0xE4A10000 + pid

        mA, mB, motion_b_world, nc_a, nc_b, n_trans = load_pair(
            items, pid, bone_offsets)

        # Baseline (no ground anchor)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        mr_base, mask_base, m135_base, R_base, off_base, T = build_input(
            mA, motion_b_world, nc_a, nc_b, n_trans, bone_offsets,
            use_ground_anchor=False)
        out_base = run_inference(bundle, pipeline, mr_base, mask_base, T, device)

        # Replace condition
        mask_135 = mask_base[:T, :135]
        cond_mask = mask_135 < 0.5
        out_base[cond_mask] = m135_base[cond_mask]

        out_base_t = torch.from_numpy(out_base).float()
        out_base_world = decanonicalize_segment(
            out_base_t, R_base, off_base, rotation_space='local').numpy()

        slide_base = compute_slide_ratio(out_base_world, bone_offsets, nc_a, nc_a + n_trans)

        # Grounded (with bone_offsets ground anchor)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        mr_ga, mask_ga, m135_ga, R_ga, off_ga, T2 = build_input(
            mA, motion_b_world, nc_a, nc_b, n_trans, bone_offsets,
            use_ground_anchor=True)
        out_ga = run_inference(bundle, pipeline, mr_ga, mask_ga, T, device)

        mask_135_ga = mask_ga[:T, :135]
        cond_mask_ga = mask_135_ga < 0.5
        out_ga[cond_mask_ga] = m135_ga[cond_mask_ga]

        out_ga_t = torch.from_numpy(out_ga).float()
        out_ga_world = decanonicalize_segment(
            out_ga_t, R_ga, off_ga, rotation_space='local').numpy()

        slide_ga = compute_slide_ratio(out_ga_world, bone_offsets, nc_a, nc_a + n_trans)

        delta = slide_ga - slide_base
        marker = '✓' if delta < -0.02 else ('✗' if delta > 0.02 else '~')
        print(f'{pid:05d}  {slide_base:10.1%}  {slide_ga:10.1%}  {delta:+8.1%} {marker}')

    print('\nDone.')


if __name__ == '__main__':
    main()

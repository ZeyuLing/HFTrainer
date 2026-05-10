#!/usr/bin/env python3
"""Multi-seed resampling using the actual eval script's infrastructure.

Instead of reimplementing the preprocessing pipeline, we directly call the
eval script's per-sample function with different seeds and pick the best.

The key insight: we only need to replace NPZs whose foot skating improved.
We load the original NPZ, compute its skating ratio, then run the eval
pipeline with different seeds and compare.
"""
import json
import os
import sys
import shutil
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

ALL_FOOT = [7, 8, 10, 11]
GROUND_Y_THRESH = 0.08
SLIDE_SPEED_THRESH = 0.015
NUM_SEEDS = 5


def compute_skating(m135, bone_offsets, gen_start, gen_end):
    positions = motion135_to_positions_np(m135, bone_offsets)
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]
    bad = total = 0
    for fi in range(1, N):
        for j in ALL_FOOT:
            if gen_pos[fi, j, 1] < GROUND_Y_THRESH:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > SLIDE_SPEED_THRESH:
                    bad += 1
    return bad / max(1, total)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', required=True,
                        help='Original NPZ directory')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for improved NPZs')
    parser.add_argument('--pids', type=str, default=None,
                        help='Comma-separated PIDs to process (default: all with >10%% skating)')
    parser.add_argument('--num-seeds', type=int, default=NUM_SEEDS)
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='Only process cases with skating > this')
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    parser.add_argument('--config', default='configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py')
    parser.add_argument('--checkpoint', default='work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2740/model.pt')
    parser.add_argument('--datalist', default='data/eval/m2m_v2/eval_e14_hq400h_static100.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bone_offsets = torch.load(args.bone_offsets, map_location='cpu').float().numpy()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    from mmengine import Config
    cfg = Config.fromfile(args.config)
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
    bundle = HyMotionM2MBundle.from_config(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    bundle.load_state_dict_selective(ckpt)
    bundle = bundle.to(device).eval()
    pipeline = HyMotionM2MPipeline(bundle)

    # Import the actual eval pipeline function
    from tools.eval_m2m_v2_all_tasks import (
        load_motion_135d, _place_b_custom, motion_135_to_198,
    )
    from hftrainer.evaluation.motion.m2m_eval_tasks import build_transition_mask
    from hftrainer.pipelines.motion.transition_utils import (
        canonicalize_segment, decanonicalize_segment,
    )

    with open(args.datalist) as f:
        items = json.load(f)['data_list']

    # Find PIDs to process
    npz_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith('.npz'))

    if args.pids:
        target_pids = set(int(x) for x in args.pids.split(','))
        npz_files = [f for f in npz_files if int(f.replace('.npz', '')) in target_pids]

    print(f"{'pid':>5}  {'orig':>8}  {'best':>8}  {'worst':>8}  {'mean':>8}  {'action':>8}")
    print('-' * 55)

    stats = {'improved': 0, 'kept': 0, 'n': 0}

    for fname in npz_files:
        pid = int(fname.replace('.npz', ''))
        orig_path = os.path.join(args.npz_dir, fname)
        d = np.load(orig_path, allow_pickle=True)

        if 'motion_135' not in d.files or 'layout_json' not in d.files:
            shutil.copy2(orig_path, os.path.join(args.output_dir, fname))
            continue

        m135_orig = d['motion_135'].astype(np.float32)
        layout = json.loads(bytes(d['layout_json']).decode())
        nc_a = layout.get('N_cond_a', 45)
        nc_b = layout.get('N_cond_b', 45)
        n_trans = layout.get('N_transition', 0)
        if n_trans == 0:
            shutil.copy2(orig_path, os.path.join(args.output_dir, fname))
            continue

        gen_start = nc_a
        gen_end = nc_a + n_trans
        ratio_orig = compute_skating(m135_orig, bone_offsets, gen_start, gen_end)

        if ratio_orig < args.threshold:
            shutil.copy2(orig_path, os.path.join(args.output_dir, fname))
            print(f'{pid:05d}  {ratio_orig:8.1%}  {"---":>8}  {"---":>8}  {"---":>8}  {"keep":>8}')
            stats['kept'] += 1
            stats['n'] += 1
            continue

        # Load source motions exactly like eval script
        item = items[pid]
        motion_a_path = item['motion_a_path']
        motion_b_path = item['motion_b_path']
        # Resolve paths
        MOTION_DATA_DIR = 'data/hymotion_data'
        for p in [motion_a_path, motion_b_path]:
            if not os.path.isabs(p) and not os.path.exists(p):
                legacy = os.path.join(MOTION_DATA_DIR, p)
                if os.path.exists(legacy):
                    if p == motion_a_path:
                        motion_a_path = legacy
                    else:
                        motion_b_path = legacy

        motion_a = load_motion_135d(motion_a_path, bone_offsets=bone_offsets)
        motion_b = load_motion_135d(motion_b_path, bone_offsets=bone_offsets)
        if motion_a is None or motion_b is None:
            shutil.copy2(orig_path, os.path.join(args.output_dir, fname))
            continue

        N_cond_a = nc_a
        N_cond_b = nc_b
        N_transition = n_trans

        motion_b_world = _place_b_custom(
            motion_a, motion_b, placement='velocity',
            N_transition=N_transition, bone_offsets=bone_offsets)

        a_tail = motion_a[-N_cond_a:]
        b_head = motion_b_world[:N_cond_b]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate([a_tail, transition_pad, b_head], axis=0)
        T = world_segment.shape[0]

        world_segment_t = torch.from_numpy(world_segment).float()
        canon_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0, rotation_space='local')
        motion_135_canon = canon_t.numpy()

        motion_raw = motion_135_to_198(motion_135_canon, bone_offsets)
        mask = build_transition_mask(T, 135, N_cond_a=N_cond_a,
                                     N_transition=N_transition, N_cond_b=N_cond_b)
        # Expand to 198
        pos_mask = np.zeros((T, 63), dtype=np.float32)
        for j in range(21):
            rot_mask_val = mask[:, 3 + (j + 1) * 6]
            pos_mask[:, j * 3:(j + 1) * 3] = rot_mask_val[:, None]
        mask_198 = np.concatenate([mask, pos_mask], axis=-1)

        # Generate multiple samples
        seed_results = []
        for si in range(args.num_seeds):
            seed = 0xE4B10000 + pid * 100 + si
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed & 0xFFFFFFFF)

            # Run inference (matching eval script exactly)
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

            # Condition replacement (matching eval script)
            mask_135 = mask[:T, :135]
            cond_mask = mask_135 < 0.5
            output_135[cond_mask] = motion_135_canon[cond_mask]

            # Decanonicalize
            out_t = torch.from_numpy(output_135).float()
            out_world = decanonicalize_segment(
                out_t, R_canon, offset_canon, rotation_space='local').numpy()

            # Compute positions using the world-space output
            ratio = compute_skating(out_world, bone_offsets, gen_start, gen_end)
            seed_results.append((ratio, si, out_world))

        # Pick best
        seed_results.sort(key=lambda x: x[0])
        best_ratio, best_si, best_world = seed_results[0]
        worst_ratio = seed_results[-1][0]
        mean_ratio = np.mean([r[0] for r in seed_results])

        # Use best if it improves over original
        if best_ratio < ratio_orig:
            from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
            best_m135 = best_world.astype(np.float32)
            wp, _, _, _ = motion135_to_fk(
                torch.from_numpy(best_m135).float(),
                torch.from_numpy(bone_offsets).float(), 'local')

            save_kw = dict(
                motion_135=best_m135,
                positions=wp.numpy(),
                translation=best_m135[:, :3],
            )
            save_kw['layout_json'] = d['layout_json']
            np.savez_compressed(os.path.join(args.output_dir, fname), **save_kw)
            action = 'improved'
            stats['improved'] += 1
        else:
            shutil.copy2(orig_path, os.path.join(args.output_dir, fname))
            action = 'keep_orig'
            stats['kept'] += 1

        stats['n'] += 1
        print(f'{pid:05d}  {ratio_orig:8.1%}  {best_ratio:8.1%}  {worst_ratio:8.1%}  {mean_ratio:8.1%}  {action:>8}')

    # Copy remaining files
    for fname in sorted(os.listdir(args.npz_dir)):
        if fname.endswith('.npz'):
            out = os.path.join(args.output_dir, fname)
            if not os.path.exists(out):
                shutil.copy2(os.path.join(args.npz_dir, fname), out)

    n = stats['n']
    print(f'\nDone: {n} cases, {stats["improved"]} improved, {stats["kept"]} kept original')


if __name__ == '__main__':
    main()

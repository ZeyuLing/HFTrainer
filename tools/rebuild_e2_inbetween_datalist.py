#!/usr/bin/env python3
"""Re-rank the E2 in-betweening datalist by pose-delta between head/tail.

E2 settings A/B/C use keep_start ∈ {5, 30} and keep_end ∈ {5}. The test is
only meaningful when the KEPT head and tail frames differ — otherwise the
model just has to output constant T-pose-ish filler. Many motions in the
existing datalist start AND end in near T-pose (the mocap recording idle
padding), so the in-between task is trivial.

This script loads each item, FKs to SMPL-22 positions, and computes:
  - pose_delta       : mean ‖rot6d(tail) − rot6d(head)‖ over 21 non-root
                       joints, averaged over keep_end × keep_start frames.
                       Measures how much the body POSE differs.
  - pelvis_delta_xz  : ‖pelvis_tail_xz − pelvis_head_xz‖ in meters.
                       Measures global TRANSLATION difference.
  - pelvis_y_delta   : |pelvis_tail_y − pelvis_head_y|.
  - first_is_tpose   : how close the first frame is to a neutral T-pose
                       (small shoulder-outstretched pose). Penalty if
                       head IS T-pose (boring start).

Score = pose_delta + 0.3·pelvis_delta_xz + 0.2·pelvis_y_delta − 0.5·first_is_tpose

Top-scored items get re-ranked to the top so `--max-samples 50` always
hits the hardest/most-diverse cases.

Reference: the existing rerank pattern lives in rebuild_e5_trajectory_datalist.py.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(ROOT))
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_transl, process_smplx_pose,
)
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

DATA_DIR = ROOT / 'data' / 'eval' / 'm2m_v2'
ORIG = DATA_DIR / 'eval_e2_inbetween.json'
REWRITTEN = DATA_DIR / 'eval_e2_inbetween_rewritten.json'
bone_offsets = torch.load(
    str(ROOT / 'data' / 'hymotion_m2m_data' / 'bone_offsets_22.pt'),
    map_location='cpu').float()

# keep_start/keep_end for E2 settings (max values we need to score over)
KEEP_START = 30
KEEP_END = 5


def load_motion_135(p):
    try:
        d = np.load(p, allow_pickle=True)
        tk = 'trans' if 'trans' in d.files else 'transl'
        pk = 'poses' if 'poses' in d.files else 'body_pose'
        return np.concatenate([
            process_transl(d[tk].astype(np.float32), 'abs'),
            process_smplx_pose(d[pk].astype(np.float32),
                               'rotation_6d', 'smpl_22'),
        ], axis=-1).astype(np.float32)
    except Exception as e:
        print(f'  [skip] {os.path.basename(p)}: {e}')
        return None


def fk_positions(m135):
    with torch.no_grad():
        wp, _, _, _ = motion135_to_fk(
            torch.from_numpy(m135).float().unsqueeze(0), bone_offsets, 'local')
    return wp.squeeze(0).numpy()


def score_item(m135):
    """Return (pose_delta, pelvis_dxz, pelvis_dy, tpose_score) — or None."""
    T = m135.shape[0]
    if T < max(KEEP_START, KEEP_END) * 2 + 5:
        return None

    rot6d = m135[:, 3:135].reshape(T, 22, 6)
    # Use the 21 NON-ROOT joints to score "body pose" independent of global orientation
    body6d = rot6d[:, 1:]          # (T, 21, 6)

    head = body6d[:KEEP_START]     # (ks, 21, 6)
    tail = body6d[-KEEP_END:]      # (ke, 21, 6)

    # Pose delta: compare mean head pose vs mean tail pose
    head_mean = head.mean(axis=0)  # (21, 6)
    tail_mean = tail.mean(axis=0)  # (21, 6)
    pose_delta = float(np.linalg.norm(head_mean - tail_mean, axis=-1).mean())

    # Pelvis delta from FK positions
    pos = fk_positions(m135)
    pelvis = pos[:, 0]              # (T, 3)
    pelvis_head = pelvis[:KEEP_START].mean(axis=0)
    pelvis_tail = pelvis[-KEEP_END:].mean(axis=0)
    pelvis_dxz = float(np.linalg.norm(pelvis_head[[0, 2]] - pelvis_tail[[0, 2]]))
    pelvis_dy = float(abs(pelvis_head[1] - pelvis_tail[1]))

    # T-pose proxy: in rot6d, identity rotation is [1, 0, 0, 1, 0, 0]
    # (row-major 6D = first two rows of identity matrix). Average over
    # non-root joints; lower = more T-pose-ish.
    identity_6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    head_dev_from_id = np.linalg.norm(head - identity_6d, axis=-1).mean()
    # tpose_score: lower = more T-pose-ish. We penalize LOW values.
    # Values typically ≥ 0 (magnitude of first 6 entries of rotation matrix).
    tpose_score = float(head_dev_from_id)

    return pose_delta, pelvis_dxz, pelvis_dy, tpose_score


def rerank(data):
    items = data.get('data_list', data) if isinstance(data, dict) else data
    scored = []
    total = len(items)
    for i, it in enumerate(items):
        mp = it.get('motion_path', '')
        if not mp:
            continue
        m135 = load_motion_135(mp)
        if m135 is None:
            continue
        s = score_item(m135)
        if s is None:
            continue
        pose_delta, pelvis_dxz, pelvis_dy, tpose_dev = s
        # Higher tpose_dev = less T-pose start = good
        score = (pose_delta
                 + 0.3 * pelvis_dxz
                 + 0.2 * pelvis_dy
                 + 0.15 * tpose_dev)
        it['_e2_stats'] = {
            'pose_delta': round(pose_delta, 4),
            'pelvis_dxz': round(pelvis_dxz, 4),
            'pelvis_dy': round(pelvis_dy, 4),
            'tpose_dev': round(tpose_dev, 4),
        }
        scored.append((score, i, it))
        if (i + 1) % 20 == 0:
            print(f'  scored {i+1}/{total}: pose_delta={pose_delta:.3f}, '
                  f'pelvis_dxz={pelvis_dxz:.2f}, tpose_dev={tpose_dev:.2f}, '
                  f'score={score:.3f}')

    if not scored:
        raise RuntimeError('no scorable items')

    # Drop obvious T-pose-start trivial cases
    non_trivial = [(sc, i, it) for sc, i, it in scored
                   if it['_e2_stats']['pose_delta'] >= 0.08  # at least some body motion difference
                   or it['_e2_stats']['pelvis_dxz'] >= 0.5]  # or some root movement
    trivial = [(sc, i, it) for sc, i, it in scored
               if (sc, i, it) not in non_trivial]
    non_trivial.sort(key=lambda x: -x[0])
    trivial.sort(key=lambda x: -x[0])

    print(f'  {len(non_trivial)} non-trivial, {len(trivial)} trivial (head≈tail)')

    n_keep = len(items)
    ordered = [it for _, _, it in non_trivial][:n_keep]
    if len(ordered) < n_keep:
        needed = n_keep - len(ordered)
        ordered.extend(it for _, _, it in trivial[:needed])
        print(f'  topped up with {needed} trivial (not enough non-trivial)')

    for it in ordered:
        it.pop('_e2_stats', None)

    if isinstance(data, dict) and 'data_list' in data:
        data['data_list'] = ordered
    else:
        data = ordered
    return data


def main():
    if not ORIG.exists():
        raise SystemExit(f'missing {ORIG}')
    print(f'Loading {ORIG.name}...')
    with open(ORIG) as f:
        data = json.load(f)
    n = len(data.get('data_list', data)) if isinstance(data, dict) else len(data)
    print(f'  {n} items')

    print('Scoring pose/position deltas...')
    new_data = rerank(data)

    backup = ORIG.with_suffix('.json.bak_before_pose_rerank')
    if not backup.exists():
        print(f'Backing up -> {backup.name}')
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    print(f'Writing {ORIG.name}...')
    with open(ORIG, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    if REWRITTEN.exists():
        print(f'Mirroring order into {REWRITTEN.name}...')
        with open(REWRITTEN) as f:
            rdata = json.load(f)
        ritems = rdata.get('data_list', rdata)
        rmap = {it.get('motion_path'): it for it in ritems}
        new_order = new_data.get('data_list', new_data)
        reordered = [rmap[it['motion_path']]
                     for it in new_order if it['motion_path'] in rmap]
        if isinstance(rdata, dict) and 'data_list' in rdata:
            rdata['data_list'] = reordered
        else:
            rdata = reordered
        rbackup = REWRITTEN.with_suffix('.json.bak_before_pose_rerank')
        if not rbackup.exists():
            with open(REWRITTEN) as f:
                rbackup.write_text(f.read())
        with open(REWRITTEN, 'w') as f:
            json.dump(rdata, f, ensure_ascii=False, indent=2)

    print('done.')


if __name__ == '__main__':
    main()

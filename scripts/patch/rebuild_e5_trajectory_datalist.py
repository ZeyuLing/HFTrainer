#!/usr/bin/env python3
"""Re-rank the E5 trajectory datalist by XZ displacement + curvature.

E5 tests root-trajectory following. The useful cases are ones where the
subject actually moves around, and ideally on curved paths (arcs / turns)
rather than straight walks. The old datalist was filtered by category +
keyword which mixed many in-place actions into the set.

This script:
  1. Loads the existing datalist (eval_e5_trajectory.json).
  2. For each item, FKs the motion to pelvis XZ positions and computes:
       - path_len       : sum of per-frame step lengths (meters)
       - chord_len      : straight-line |last - first| distance (meters)
       - curvature      : path_len / max(chord_len, 1e-3)  (>=1, 1=straight)
       - std_xz         : radial RMS of the XZ trajectory around centroid
  3. Scores each case:  score = path_len + 0.5 * std_xz + 0.3 * (curvature - 1)
     (path_len dominates; curvature breaks ties between equally-long walks).
  4. Drops items with path_len < 1.0 m (essentially in-place) unless needed
     to fill the quota.
  5. Re-ranks the data_list in-place. Item count is preserved.

Also emits an eval_e5_trajectory_rewritten.json mirror if present in the
original directory (only the re-ranked order changes; captions untouched).

Usage:
    python3 tools/rebuild_e5_trajectory_datalist.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(ROOT))
import torch
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    process_transl, process_smplx_pose,
)
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

DATA_DIR = ROOT / 'data' / 'eval' / 'm2m_v2'
ORIG = DATA_DIR / 'eval_e5_trajectory.json'
REWRITTEN = DATA_DIR / 'eval_e5_trajectory_rewritten.json'

bone_offsets_path = ROOT / 'data' / 'hymotion_m2m_data' / 'bone_offsets_22.pt'
bone_offsets = torch.load(str(bone_offsets_path), map_location='cpu').float()


def pelvis_xz_trajectory(motion_path: str) -> np.ndarray | None:
    """Return (T, 2) pelvis XZ world positions, or None on failure."""
    try:
        d = np.load(motion_path, allow_pickle=True)
        tk = 'trans' if 'trans' in d.files else 'transl'
        pk = 'poses' if 'poses' in d.files else 'body_pose'
        m135 = np.concatenate([
            process_transl(d[tk].astype(np.float32), 'abs'),
            process_smplx_pose(d[pk].astype(np.float32),
                               'rotation_6d', 'smpl_22'),
        ], axis=-1).astype(np.float32)
        with torch.no_grad():
            wp, _, _, _ = motion135_to_fk(
                torch.from_numpy(m135).float().unsqueeze(0),
                bone_offsets, 'local',
            )
        pos = wp.squeeze(0).numpy()  # (T, 22, 3)
        return pos[:, 0, [0, 2]]  # pelvis XZ
    except Exception as exc:
        print(f'  [skip] {motion_path[-60:]}: {exc}')
        return None


def score_trajectory(xz: np.ndarray) -> dict:
    """Compute path_len, chord_len, curvature, std_xz, total_turn.

    Updated 2026-04-21 to emphasize complex paths (S-curves, circles):
      - total_turn = sum of absolute per-step heading-angle changes
        (= total turning in radians). A straight walk has ~0 total_turn,
        a full circle has 2π. Much more discriminative than curvature
        (path/chord) for detecting curved trajectories vs straight walks.
      - score reweighted to strongly reward long AND curved motions.
    """
    steps_vec = np.diff(xz, axis=0)                         # (T-1, 2)
    step_lens = np.linalg.norm(steps_vec, axis=1)          # (T-1,)
    path_len = float(step_lens.sum())
    chord_len = float(np.linalg.norm(xz[-1] - xz[0]))
    curvature = path_len / max(chord_len, 1e-3) if path_len > 0 else 1.0
    centroid = xz.mean(axis=0)
    std_xz = float(np.sqrt(((xz - centroid) ** 2).sum(axis=1).mean()))

    # Total turning angle: sum of absolute heading deltas, only counting
    # steps with nonzero displacement (avoid ~0/0 noise during stationary frames).
    moving = step_lens > 0.02  # >2 cm/frame ≈ real motion
    headings = np.arctan2(steps_vec[:, 1], steps_vec[:, 0])  # (T-1,)
    dh = np.diff(headings)                                    # (T-2,)
    # Wrap to [-π, π]
    dh = np.mod(dh + np.pi, 2 * np.pi) - np.pi
    if moving[:-1].any():
        total_turn = float(np.abs(dh[moving[:-1]]).sum())
    else:
        total_turn = 0.0

    # New score: path_len dominates, plus large bonus for curved paths.
    # - path_len (meters)
    # - (curvature-1) bonus up to ~0.7 for curvatures 1-2 (S-curve territory)
    # - total_turn (radians) gives strong signal for circles (2π ≈ 6.28)
    score = (path_len
             + 0.7 * max(0.0, curvature - 1.0)
             + 0.4 * total_turn
             + 0.5 * std_xz)
    return {
        'path_len': path_len,
        'chord_len': chord_len,
        'curvature': curvature,
        'std_xz': std_xz,
        'total_turn': total_turn,
        'score': score,
    }


def rerank(data: dict) -> dict:
    items = data.get('data_list', data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise RuntimeError(f'unexpected datalist structure: {type(items)}')

    scored = []
    for i, it in enumerate(items):
        mp = it.get('motion_path', '')
        if not mp:
            continue
        xz = pelvis_xz_trajectory(mp)
        if xz is None or len(xz) < 2:
            continue
        s = score_trajectory(xz)
        it['_e5_traj_stats'] = s
        scored.append((s['score'], i, it))
        if (i + 1) % 10 == 0:
            print(f'  scored {i+1}/{len(items)}: path_len={s["path_len"]:.2f}m, '
                  f'curv={s["curvature"]:.2f}')

    if not scored:
        raise RuntimeError('no scorable items')

    # Split: movers (path_len >= 2m) vs stationary, keep all movers first.
    # Raised from 1m (2026-04-21): 1m includes walking one or two steps in
    # place, which is not an interesting trajectory test.
    movers = [(sc, i, it) for sc, i, it in scored
              if it['_e5_traj_stats']['path_len'] >= 2.0]
    statics = [(sc, i, it) for sc, i, it in scored
               if it['_e5_traj_stats']['path_len'] < 1.0]
    movers.sort(key=lambda x: -x[0])
    statics.sort(key=lambda x: -x[0])

    print(f'  {len(movers)} motions with >=1m path, {len(statics)} near-static')

    # Keep original item count. Movers first, then (if needed) best statics.
    n_keep = len(items)
    ordered = [it for _, _, it in movers][:n_keep]
    if len(ordered) < n_keep:
        needed = n_keep - len(ordered)
        ordered.extend(it for _, _, it in statics[:needed])
        print(f'  topped up with {needed} near-static (insufficient movers)')

    # Strip the helper stats before saving to keep the JSON clean.
    for it in ordered:
        it.pop('_e5_traj_stats', None)

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

    print('Scoring trajectories...')
    new_data = rerank(data)

    backup = ORIG.with_suffix('.json.bak_before_traj_rerank')
    if not backup.exists():
        print(f'Backing up original -> {backup.name}')
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f'Writing {ORIG.name}...')
    with open(ORIG, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    # Mirror the reordering into the _rewritten sister file by re-indexing.
    if REWRITTEN.exists():
        print(f'Mirroring order into {REWRITTEN.name}...')
        with open(REWRITTEN) as f:
            rdata = json.load(f)
        ritems = rdata.get('data_list', rdata)
        # Build motion_path -> item map for the rewritten set.
        rmap = {it.get('motion_path'): it for it in ritems}
        new_order = new_data.get('data_list', new_data)
        reordered = [rmap[it['motion_path']]
                     for it in new_order if it['motion_path'] in rmap]
        if isinstance(rdata, dict) and 'data_list' in rdata:
            rdata['data_list'] = reordered
        else:
            rdata = reordered
        rbackup = REWRITTEN.with_suffix('.json.bak_before_traj_rerank')
        if not rbackup.exists():
            with open(REWRITTEN) as f:
                rbackup.write_text(f.read())
        with open(REWRITTEN, 'w') as f:
            json.dump(rdata, f, ensure_ascii=False, indent=2)

    print('done.')


if __name__ == '__main__':
    main()

"""Patch existing KIMODO output NPZs by applying frame-0 Y-anchor.

KIMODO model output exhibits ~10-30 cm of upward Y drift over unconstrained
spans. Frame-0 Y-anchor shifts the entire motion vertically so the lowest
joint at frame 0 lines up with the GT lowest joint at frame 0. This removes
the visually-obvious "floating at the start" artifact without distorting
horizontal motion. Applied uniformly to:
  * positions (T, 22, 3) — used for metrics + skeleton viz
  * translation (T, 3) — pelvis trajectory
  * posed_joints (T, 77, 3) — used for SOMA mesh viz
The NPZ is rewritten in place; a sentinel key 'y_anchor_applied' marks files
that already have the patch so reruns are idempotent.

Usage (PROJECT_ROOT must be on sys.path so we can reuse FK helpers):
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python tools/patch_kimodo_y_anchor.py \
        --root work_dirs/kimodo_e2_v2_priv220_20260425_031208 \
        --datalist data/eval/m2m_v2/eval_e2_inbetween_v2_rewritten.json \
        --motion-data-dir /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch  # noqa: F401  -- ensures CUDA driver init quietly


def _patch_one_npz(npz_path: str, gt_floor0: float) -> str:
    """Patch one NPZ in place. Returns short status string."""
    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return f'load-error: {e}'
    files = list(d.files)
    if 'y_anchor_applied' in files:
        return 'skipped (already patched)'
    if 'positions' not in files:
        return 'skipped (no positions)'

    pos = d['positions'].astype(np.float32)
    n0 = min(5, pos.shape[0])
    pred_floor0 = float(pos[:n0, :, 1].min())
    delta = pred_floor0 - gt_floor0

    save_fields = {}
    for k in files:
        v = d[k]
        if k in ('positions', 'translation', 'posed_joints') and abs(delta) > 1e-4:
            v = v.astype(np.float32).copy()
            v[..., 1] -= delta
        save_fields[k] = v
    save_fields['y_anchor_applied'] = np.array(delta, dtype=np.float32)
    np.savez_compressed(npz_path, **save_fields)
    return f'patched (delta={delta:+.3f})' if abs(delta) > 1e-4 else 'noop'


def _gt_floors_from_eval_pipeline(datalist_path: str, motion_data_dir: str,
                                  max_samples: int = 0):
    """Use the same load_eval_samples + motion135 -> positions FK that the
    KIMODO runner uses. Returns dict: sample_idx -> frame-0 min joint Y."""
    sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
    from tools.eval_m2m_v2_all_tasks import load_eval_samples
    from hftrainer.evaluation.motion.m2m_eval_metrics import (
        motion135_to_positions_np,
    )

    bone_offsets = torch.load(
        '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/'
        'data/hymotion_m2m_data/bone_offsets_22.pt',
        map_location='cpu',
    ).numpy()

    # max_samples=0 means "no cap" semantically here, but load_eval_samples
    # treats it as len(samples) >= 0 and immediately breaks. Use a large
    # ceiling instead.
    samples = load_eval_samples(
        datalist_path, motion_data_dir,
        max_samples if max_samples > 0 else 100000,
        require_caption=False, bone_offsets=bone_offsets,
    )
    print(f'[gt-cache] loaded {len(samples)} eval samples')

    floors = {}
    for i, s in enumerate(samples):
        motion_135 = s['motion']
        gt_pos = motion135_to_positions_np(motion_135, bone_offsets)  # (T,22,3)
        n0 = min(5, gt_pos.shape[0])
        floors[i] = float(gt_pos[:n0, :, 1].min())
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(samples)}] gt_floor0 cached')
    return floors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='KIMODO work_dirs root (cond/ uncond/ subdirs)')
    ap.add_argument('--datalist', required=True,
                    help='Path to eval datalist JSON for GT FK lookup')
    ap.add_argument('--motion-data-dir', required=True,
                    help='Root of mocap NPZ pool used by load_eval_samples')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    gt_floors = _gt_floors_from_eval_pipeline(
        args.datalist, args.motion_data_dir)

    n_patched = n_skipped = n_noop = n_err = 0
    for sub in ('cond', 'uncond'):
        sub_root = os.path.join(args.root, sub)
        if not os.path.isdir(sub_root):
            continue
        for setting_dir in sorted(os.listdir(sub_root)):
            npz_glob = os.path.join(sub_root, setting_dir, 'npz', '*.npz')
            files = sorted(glob.glob(npz_glob))
            if not files:
                continue
            print(f'\n=== {sub}/{setting_dir} ({len(files)} npz) ===')
            for fp in files:
                fname = os.path.basename(fp)
                try:
                    sidx = int(fname.split('.')[0])
                except ValueError:
                    n_err += 1
                    continue
                gt_floor0 = gt_floors.get(sidx)
                if gt_floor0 is None:
                    n_skipped += 1
                    continue
                if args.dry_run:
                    print(f'  {fname}: would patch with gt_floor={gt_floor0:.3f}')
                    continue
                status = _patch_one_npz(fp, gt_floor0)
                if status.startswith('patched'):
                    n_patched += 1
                elif status.startswith('noop'):
                    n_noop += 1
                elif status.startswith('skipped'):
                    n_skipped += 1
                else:
                    n_err += 1
                    print(f'  {fname}: {status}')

    print(f'\nDone. patched={n_patched} noop={n_noop} '
          f'skipped={n_skipped} err={n_err}')


if __name__ == '__main__':
    main()

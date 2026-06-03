#!/usr/bin/env python3
"""Repackage our 135-dim predictions into the MotionStreamer-272 evaluator format.

Our generation writes ``<pred_id>.npz`` (SMPLX) and ``compute_kafs_metrics`` produced
``<mode>_135d/<pred_id>.npy`` (135-dim trans+rot6d). The 272 evaluator
(``eval_motionstreamer_272.py``) expects ``<canonical_id>.npz`` files keyed by
``motion_135``, where ``<canonical_id>`` is the HumanML3D id from
``humanml3d_272/split/test.txt`` (e.g. ``010541`` / ``M003137``).

The mapping ``<pred_id> -> <canonical_id>`` is the basename of ``smplx_path`` in the
annotation ``data_list``. This script writes the renamed npz files (parallel).
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def _init_worker():
    # conversion is CPU-only; hide GPU so per-worker torch/deepspeed import
    # does not contend a CUDA context (avoids fork+CUDA deadlock).
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _smplx_npz_to_row135(npz_path):
    """SMPLX npz -> ROW-major 135 (trans3 + 22x6D), matching motion135_to_fk.

    The pre-existing ``*_135d/*.npy`` use COLUMN-major rot6d (rotation_convert),
    which motion135_to_fk (row-major) mis-decodes -> garbage FK. The 272 path
    therefore needs row-major 135 built directly from the SMPLX rotations.
    """
    import torch
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
    )
    from hftrainer.pipelines.motion.differentiable_fk import (
        rotmat_to_rot6d_row_major,
    )
    npz = np.load(str(npz_path), allow_pickle=True)
    transl = np.asarray(npz['transl'], dtype=np.float32)
    T = transl.shape[0]
    go = torch.from_numpy(np.asarray(npz['global_orient'], dtype=np.float32)).reshape(T, 3)
    bp = torch.from_numpy(np.asarray(npz['body_pose'], dtype=np.float32)).reshape(T, 21, 3)
    aa = torch.cat([go[:, None], bp], dim=1)            # (T,22,3)
    R = axis_angle_to_matrix(aa)
    r6 = rotmat_to_rot6d_row_major(R).reshape(T, 132)
    m135 = torch.cat([torch.from_numpy(transl), r6], dim=1).numpy()
    return m135.astype(np.float32)


def _worker(task):
    src_path, out_npz, mode = task
    if os.path.exists(out_npz):
        return 'skip'
    try:
        if mode == 'row':
            m135 = _smplx_npz_to_row135(src_path)
        else:
            m135 = np.load(src_path)
        np.savez(out_npz, motion_135=m135.astype(np.float32))
        return 'ok'
    except Exception as e:  # noqa: BLE001
        return f'fail:{e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npy-dir', default=None,
                    help='dir of <pred_id>.npy (COLUMN-major 135), e.g. .../none_135d')
    ap.add_argument('--npz-dir', default=None,
                    help='dir of <pred_id>.npz (SMPLX); converts to ROW-major 135 '
                         'for the 272/FK path. Mutually exclusive with --npy-dir.')
    ap.add_argument('--anno-file', default='data/annotation/test_hml3d.json')
    ap.add_argument('--out-dir', required=True,
                    help='output dir for <canonical_id>.npz')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    assert bool(args.npy_dir) ^ bool(args.npz_dir), 'give exactly one of --npy-dir/--npz-dir'

    anno = json.load(open(args.anno_file))['data_list']
    pred2can = {
        pid: os.path.splitext(os.path.basename(e['smplx_path']))[0]
        for pid, e in anno.items()
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.npz_dir:
        src_dir, pat, mode = Path(args.npz_dir), '*.npz', 'row'
    else:
        src_dir, pat, mode = Path(args.npy_dir), '*.npy', 'col'

    tasks = []
    missing_map = 0
    for f in src_dir.glob(pat):
        pid = f.stem
        can = pred2can.get(pid)
        if can is None:
            missing_map += 1
            continue
        tasks.append((str(f), str(out_dir / f'{can}.npz'), mode))

    print(f'src({mode}): {len(tasks)+missing_map}, mapped: {len(tasks)}, '
          f'unmapped: {missing_map}')

    pool = mp.Pool(max(1, args.workers), initializer=_init_worker)
    ok = skip = fail = 0
    for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=16), 1):
        if r == 'ok':
            ok += 1
        elif r == 'skip':
            skip += 1
        else:
            fail += 1
            if fail <= 5:
                print('  ', r)
        if i % 1000 == 0:
            print(f'  {i}/{len(tasks)} (ok={ok} skip={skip} fail={fail})')
    pool.close()
    pool.join()
    print(f'DONE ok={ok} skip={skip} fail={fail} -> {out_dir}')


if __name__ == '__main__':
    main()

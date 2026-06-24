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
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _init_worker():
    # conversion is CPU-only; hide GPU so per-worker torch/deepspeed import
    # does not contend a CUDA context (avoids fork+CUDA deadlock).
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _smplx_npz_to_row135(npz_path):
    """SMPLX npz -> ROW-major 135 (trans3 + 22x6D), matching motion135_to_fk.

    The pre-existing ``*_135d/*.npy`` use COLUMN-major rot6d (rotation_convert),
    which motion135_to_fk (row-major) mis-decodes -> garbage FK. The 272 path
    therefore needs row-major 135 built directly from the SMPLX rotations.
    """
    import torch
    from hftrainer.motion.representation.rotation import (
        axis_angle_to_matrix,
    )
    from hftrainer.motion.skeleton.fk import (
        rotmat_to_rot6d_row_major,
    )
    npz = np.load(str(npz_path), allow_pickle=True)
    transl = np.asarray(npz['transl'], dtype=np.float32)
    T = transl.shape[0]
    go = torch.from_numpy(np.asarray(npz['global_orient'], dtype=np.float32)).reshape(T, 3)
    bp = torch.from_numpy(np.asarray(npz['body_pose'], dtype=np.float32)).reshape(T, 21, 3)
    aa = torch.cat([go[:, None], bp], dim=1)            # (T,22,3)
    R = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(T, 22, 3, 3)
    r6 = rotmat_to_rot6d_row_major(R).reshape(T, 132)
    m135 = torch.cat([torch.from_numpy(transl), r6], dim=1).numpy()
    return m135.astype(np.float32)


def _col135_to_row135(npy_path):
    """COLUMN-major 135 (trans3 + 22x6D column) -> ROW-major 135.

    motionclip135 ``.npy`` files store rot6d in COLUMN-major convention
    (``matrix_to_rotation_6d(convention='column')``). ``motion135_to_fk`` decodes
    ROW-major, so feeding column-major directly produces garbage FK. We round-trip
    column-6D -> rotation matrices -> row-major 6D (lossless on the actual
    rotation), keeping the translation block untouched.
    """
    import torch
    from hftrainer.motion.representation.rotation import (
        rotation_6d_to_matrix,
    )
    from hftrainer.motion.skeleton.fk import (
        rotmat_to_rot6d_row_major,
    )
    arr = np.load(str(npy_path)).astype(np.float32)
    T = arr.shape[0]
    trans = torch.from_numpy(arr[:, :3])
    col6d = torch.from_numpy(arr[:, 3:135]).reshape(T, 22, 6)
    mat = rotation_6d_to_matrix(col6d, convention='column')      # (T,22,3,3)
    row6d = rotmat_to_rot6d_row_major(mat).reshape(T, 132)
    m135 = torch.cat([trans, row6d], dim=1).numpy()
    return m135.astype(np.float32)


def _gt272_to_row135(npy_path):
    """Native GT 272 (.npy / .npz motion_272) -> ROW-major 135 (trans + 22x6D).

    Decodes the 272 local rotations + root translation and re-expresses them as a
    row-major 135 so the GT passes through the *same* canon272 FK->272 path as the
    predictions (a conversion-penalty Real control)."""
    import torch
    from hftrainer.datasets.motion.representation.humanml_repr import (
        recover_local_rotations_and_root,
    )
    from hftrainer.motion.skeleton.fk import (
        rotmat_to_rot6d_row_major,
    )
    p = str(npy_path)
    if p.endswith('.npz'):
        z = np.load(p, allow_pickle=True)
        m272 = np.asarray(z['motion_272'], dtype=np.float32)
    else:
        m272 = np.asarray(np.load(p), dtype=np.float32)
    rot, root = recover_local_rotations_and_root(m272)        # (T,22,3,3),(T,3)
    row6d = rotmat_to_rot6d_row_major(torch.from_numpy(np.asarray(rot, np.float32)))
    m135 = np.concatenate(
        [np.asarray(root, np.float32), row6d.reshape(row6d.shape[0], 132).numpy()],
        axis=-1)
    return m135.astype(np.float32)


def _load_motion272(npy_path):
    p = str(npy_path)
    if p.endswith('.npz'):
        z = np.load(p, allow_pickle=True)
        m272 = np.asarray(z['motion_272'], dtype=np.float32)
    else:
        m272 = np.asarray(np.load(p), dtype=np.float32)
    return m272.astype(np.float32)


def _worker(task):
    src_path, out_npz, mode = task
    if os.path.exists(out_npz):
        return 'skip'
    try:
        if mode == 'row':
            m135 = _smplx_npz_to_row135(src_path)
        elif mode == 'col':
            m135 = _col135_to_row135(src_path)
        elif mode == 'gt272':
            m135 = _gt272_to_row135(src_path)
        elif mode == 'motion272':
            m272 = _load_motion272(src_path)
            np.savez(out_npz, motion_272=m272)
            return 'ok'
        else:
            m135 = np.load(src_path)
        np.savez(out_npz, motion_135=m135.astype(np.float32))
        return 'ok'
    except Exception as e:  # noqa: BLE001
        return f'fail:{src_path}:{e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npy-dir', default=None,
                    help='dir of <pred_id>.npy (raw COLUMN-major 135 stored as-is, '
                         'legacy; usually wrong for FK -- prefer --col-npy-dir)')
    ap.add_argument('--col-npy-dir', default=None,
                    help='dir of <pred_id>.npy COLUMN-major motionclip135 -> converted '
                         'to ROW-major 135 for the 272/FK path.')
    ap.add_argument('--gt272-dir', default=None,
                    help='dir of native GT 272 files (<id>.npy or <id>.npz motion_272) '
                         '-> ROW-major 135 (conversion-penalty Real control).')
    ap.add_argument('--motion272-dir', default=None,
                    help='dir of native 272 files (<id>.npy or <id>.npz motion_272); '
                         'preserves motion_272 directly for evaluator input.')
    ap.add_argument('--npz-dir', default=None,
                    help='dir of <pred_id>.npz (SMPLX); converts to ROW-major 135 '
                         'for the 272/FK path. Mutually exclusive with --npy-dir.')
    ap.add_argument('--anno-file', default='data/annotation/test_hml3d.json')
    ap.add_argument('--id-passthrough', action='store_true',
                    help='if a source id is not found in the annotation map, use the '
                         'source stem itself as the canonical id (for dirs already '
                         'named by canonical HumanML3D ids, e.g. 000000.npz).')
    ap.add_argument('--out-dir', required=True,
                    help='output dir for <canonical_id>.npz')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    n_src = sum(bool(x) for x in (
        args.npy_dir, args.col_npy_dir, args.npz_dir, args.gt272_dir, args.motion272_dir))
    assert n_src == 1, (
        'give exactly one of --npy-dir/--col-npy-dir/--npz-dir/--gt272-dir/--motion272-dir')

    anno = json.load(open(args.anno_file))['data_list']
    pred2can = {
        pid: os.path.splitext(os.path.basename(e['smplx_path']))[0]
        for pid, e in anno.items()
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.npz_dir:
        src_dir, pats, mode = Path(args.npz_dir), ['*.npz'], 'row'
    elif args.col_npy_dir:
        src_dir, pats, mode = Path(args.col_npy_dir), ['*.npy'], 'col'
    elif args.gt272_dir:
        src_dir, pats, mode = Path(args.gt272_dir), ['*.npy', '*.npz'], 'gt272'
    elif args.motion272_dir:
        src_dir, pats, mode = Path(args.motion272_dir), ['*.npy', '*.npz'], 'motion272'
    else:
        src_dir, pats, mode = Path(args.npy_dir), ['*.npy'], 'raw'

    tasks = []
    missing_map = 0
    files = []
    for pat in pats:
        files.extend(src_dir.glob(pat))
    for f in files:
        pid = f.stem
        can = pred2can.get(pid)
        if can is None and (args.id_passthrough or mode == 'gt272'):
            can = pid
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
    if fail:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

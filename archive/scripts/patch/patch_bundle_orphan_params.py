"""Patch existing checkpoints' ``__bundle_params__`` from a source ckpt.

Why this exists
---------------
Until the fix in `accelerate_runner._restore_bundle_params_from_model_pt`,
``Accelerator.save_state`` / ``load_state`` silently dropped bundle-level
orphan ``nn.Parameter`` / buffers (e.g. ``null_vtxt_feat`` for
classifier-free guidance, ``null_source_feat`` for source-CFG).  The first
auto-resume cycle reset them to constructor-time zeros, and every
subsequent save persisted those zeros.

This tool retroactively repairs already-trained checkpoints by copying the
desired tensor values from a *source* checkpoint (typically the pretrained
T2M 1.0 lite checkpoint) into the target checkpoint's ``model.pt``.

Typical usage
-------------
    python3 tools/patch_bundle_orphan_params.py \\
        --source checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \\
        --target work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_2790 \\
        --keys null_vtxt_feat null_ctxt_input \\
        --backup

    # Bulk patch every checkpoint dir under a work_dir:
    python3 tools/patch_bundle_orphan_params.py \\
        --source checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \\
        --work-dir work_dirs/hymotion_m2m_v2_caption_local_phase2 \\
        --keys null_vtxt_feat null_ctxt_input \\
        --backup

Each target's ``model.pt`` is rewritten so that
``__bundle_params__[key] == source_sd[key]``.  Missing keys are reported
but not fatal.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Iterable, List

import torch


def _load_source_sd(path: str) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    return load_checkpoint(path, map_location='cpu')


def _list_ckpt_dirs(work_dir: str) -> List[str]:
    out = []
    for entry in sorted(os.listdir(work_dir)):
        full = os.path.join(work_dir, entry)
        if entry.startswith('checkpoint-') and os.path.isdir(full):
            out.append(full)
    return out


def _patch_one(target_dir: str, source_sd: dict, keys: Iterable[str], backup: bool) -> bool:
    mpt = os.path.join(target_dir, 'model.pt')
    if not os.path.exists(mpt):
        print(f'[skip] {target_dir}: no model.pt')
        return False
    try:
        blob = torch.load(mpt, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f'[err]  {target_dir}: failed to read model.pt ({exc})')
        return False

    if not isinstance(blob, dict):
        print(f'[skip] {target_dir}: model.pt is not a dict')
        return False

    bp = blob.get('__bundle_params__')
    if not isinstance(bp, dict):
        bp = {}
        blob['__bundle_params__'] = bp

    changed = []
    for k in keys:
        if k not in source_sd:
            print(f'[warn] {target_dir}: source has no key {k}')
            continue
        src = source_sd[k]
        if not torch.is_tensor(src):
            print(f'[warn] {target_dir}: source key {k} is not a tensor')
            continue
        prev = bp.get(k)
        if torch.is_tensor(prev) and prev.shape == src.shape:
            same = torch.equal(prev, src)
            if same:
                print(f'[ok]   {target_dir}: {k} already matches source (norm={src.float().norm():.4f})')
                continue
        bp[k] = src.detach().clone()
        changed.append(
            f'{k} (was '
            f'{f"norm={prev.float().norm():.4f}" if torch.is_tensor(prev) else "missing"} '
            f'-> norm={src.float().norm():.4f})'
        )

    if not changed:
        return False

    if backup:
        shutil.copy2(mpt, mpt + '.preorphanfix')

    torch.save(blob, mpt)
    print(f'[fix]  {target_dir}: patched {len(changed)} key(s) -> {changed}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='Path to source checkpoint to copy values from (e.g. T2M 1.0 lite).')
    parser.add_argument('--target', help='Single checkpoint directory to patch.')
    parser.add_argument('--work-dir', help='Work dir; every checkpoint-* subdirectory will be patched.')
    parser.add_argument('--keys', nargs='+', default=['null_vtxt_feat', 'null_ctxt_input'],
                        help='Bundle-level orphan parameter / buffer names to copy from source.')
    parser.add_argument('--backup', action='store_true', help='Save model.pt.preorphanfix before overwriting.')
    parser.add_argument('--dry-run', action='store_true', help='Report what would be patched, but do not write.')
    args = parser.parse_args()

    if args.target and args.work_dir:
        parser.error('Use --target OR --work-dir, not both.')
    if not args.target and not args.work_dir:
        parser.error('Need --target or --work-dir.')

    print(f'Loading source state_dict from {args.source} ...')
    source_sd = _load_source_sd(args.source)
    available = [k for k in args.keys if k in source_sd]
    missing = [k for k in args.keys if k not in source_sd]
    if missing:
        print(f'[warn] Source missing keys: {missing}')
    print(f'Patch keys (with source norms):')
    for k in available:
        print(f'  {k}: norm={source_sd[k].float().norm():.4f}, shape={tuple(source_sd[k].shape)}')

    if args.dry_run:
        print('DRY RUN: no files will be modified.')

    targets = [args.target] if args.target else _list_ckpt_dirs(args.work_dir)
    print(f'Patching {len(targets)} checkpoint dir(s)')

    n_changed = 0
    for t in targets:
        if args.dry_run:
            mpt = os.path.join(t, 'model.pt')
            if not os.path.exists(mpt):
                print(f'[dry-skip] {t}: no model.pt')
                continue
            try:
                blob = torch.load(mpt, map_location='cpu', weights_only=False)
            except Exception as exc:
                print(f'[dry-err] {t}: {exc}')
                continue
            bp = blob.get('__bundle_params__', {}) if isinstance(blob, dict) else {}
            for k in available:
                cur = bp.get(k)
                cur_str = f'norm={cur.float().norm():.4f}' if torch.is_tensor(cur) else 'missing'
                print(f'  [dry] {t} :: {k} = {cur_str}  (would set norm={source_sd[k].float().norm():.4f})')
        else:
            if _patch_one(t, source_sd, available, args.backup):
                n_changed += 1
    print(f'Done. Patched {n_changed} checkpoints.')


if __name__ == '__main__':
    main()

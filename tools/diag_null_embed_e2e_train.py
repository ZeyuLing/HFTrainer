"""End-to-end training validation for the orphan-Parameter resume fix.

Runs through the FULL AccelerateRunner flow (build → load_from →
prepare → train 1 step → save_checkpoint → quit) twice in the same
process, simulating a real auto_resume cycle, and checks that
``null_vtxt_feat`` / ``null_ctxt_input`` keep their pretrained values
across the cycle.

Stage A (fresh launch):
  - cfg.load_from points to T2M 1.0 lite (norm=10/45)
  - runner.train() for 1 step
  - save_checkpoint()
  - assert null_vtxt_feat / null_ctxt_input on bundle ≈ 10 / 45
  - assert checkpoint-iter_*/model.pt's __bundle_params__ ≈ 10 / 45

Stage B (simulated auto_resume):
  - re-build a SECOND runner from same cfg (load_from cleared,
    auto_resume=True)
  - this triggers _load(scope='full') → accelerator.load_state(path)
  - **without the fix** this would silently zero null_vtxt_feat
  - assert post-resume bundle still has null_vtxt_feat ≈ 10 / 45
  - run train() for 1 more step, save again, recheck.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys

import torch


def _setup_path():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))


def _norm(t):
    return float(t.detach().float().norm().item()) if t is not None else None


def _load_pt_norms(model_pt):
    if not os.path.exists(model_pt):
        return None
    blob = torch.load(model_pt, map_location='cpu', weights_only=False)
    bp = blob.get('__bundle_params__') if isinstance(blob, dict) else None
    if not isinstance(bp, dict):
        return None
    return {k: _norm(v) for k, v in bp.items() if torch.is_tensor(v)}


def _build_short_run_cfg(base_cfg_path: str, work_dir: str, max_iters: int, save_every: int):
    """Edit cfg in-memory: shorten train, single GPU, no eval."""
    from mmengine import Config
    cfg = Config.fromfile(base_cfg_path)
    # Strip image-style cfg objects we don't actually need here.
    cfg.work_dir = work_dir
    if hasattr(cfg, 'train_cfg'):
        tc = cfg.train_cfg
        if hasattr(tc, 'to_dict'):
            tc = tc.to_dict()
        else:
            tc = dict(tc)
        tc['by_epoch'] = False
        tc['max_iters'] = max_iters
        tc['save_interval'] = save_every
        tc['log_interval'] = 1
        tc['val_interval'] = max_iters * 100  # never
        cfg.train_cfg = tc
    # Make sure dataloader is small.
    for k in ('train_dataloader',):
        if hasattr(cfg, k):
            dl = getattr(cfg, k)
            if hasattr(dl, 'to_dict'):
                dl = dl.to_dict()
            else:
                dl = dict(dl)
            dl['batch_size'] = 1
            dl['num_workers'] = 0
            setattr(cfg, k, dl)
    return cfg


def _check(label, want, got_dict, atol=0.05):
    print(f'\n[{label}]')
    bad = []
    for k, want_v in want.items():
        got = got_dict.get(k)
        if got is None:
            print(f'  {k}: MISSING')
            bad.append(k)
            continue
        ok = abs(got - want_v) < atol
        print(f"  {k}: {got:.4f} (want ≈ {want_v:.4f})  {'✓' if ok else '✗'}")
        if not ok:
            bad.append(k)
    if bad:
        print(f'  FAIL: {bad}')
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--load-from', required=True)
    parser.add_argument('--max-iters', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=1)
    args = parser.parse_args()

    _setup_path()
    from hftrainer.runner.accelerate_runner import AccelerateRunner

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)

    target_norms = {'null_vtxt_feat': 10.1251, 'null_ctxt_input': 44.7500}

    # ============== STAGE A: fresh launch with load_from ==============
    print('=' * 70)
    print('STAGE A: fresh launch with load_from = T2M 1.0')
    print('=' * 70)
    cfg = _build_short_run_cfg(args.cfg, args.work_dir, args.max_iters, args.save_every)
    cfg.load_from = args.load_from
    cfg.auto_resume = False
    runner = AccelerateRunner.from_cfg(cfg)
    bundle = runner.bundle
    a_after_load = {
        'null_vtxt_feat': _norm(bundle.null_vtxt_feat),
        'null_ctxt_input': _norm(bundle.null_ctxt_input),
    }
    if not _check('A.1 after load_from (in-memory bundle)', target_norms, a_after_load):
        sys.exit(1)

    runner.train()  # runs max_iters
    a_after_train = {
        'null_vtxt_feat': _norm(bundle.null_vtxt_feat),
        'null_ctxt_input': _norm(bundle.null_ctxt_input),
    }
    if not _check('A.2 after train() (frozen orphan params should be unchanged)',
                  target_norms, a_after_train):
        sys.exit(1)

    ck_dirs = sorted(d for d in os.listdir(args.work_dir) if d.startswith('checkpoint-'))
    assert ck_dirs, 'No checkpoint dir was saved!'
    last_ck = os.path.join(args.work_dir, ck_dirs[-1])
    pt_norms = _load_pt_norms(os.path.join(last_ck, 'model.pt')) or {}
    if not _check(f'A.3 saved {ck_dirs[-1]}/model.pt __bundle_params__',
                  target_norms, pt_norms):
        sys.exit(1)

    # Free runner to release GPU and Accelerator state before second build.
    del runner
    del bundle
    torch.cuda.empty_cache()

    # ============== STAGE B: second launch with auto_resume ==============
    print()
    print('=' * 70)
    print('STAGE B: simulated auto_resume (no load_from; auto_resume=True)')
    print('=' * 70)
    cfg2 = _build_short_run_cfg(args.cfg, args.work_dir, args.max_iters * 2, args.save_every)
    cfg2.load_from = None
    cfg2.auto_resume = True
    runner2 = AccelerateRunner.from_cfg(cfg2)
    bundle2 = runner2.bundle
    b_after_resume = {
        'null_vtxt_feat': _norm(bundle2.null_vtxt_feat),
        'null_ctxt_input': _norm(bundle2.null_ctxt_input),
    }
    if not _check('B.1 after auto_resume (THE BUG: pre-fix this would be 0)',
                  target_norms, b_after_resume):
        print('  → fix did NOT take effect — orphan params dropped during resume')
        sys.exit(1)

    runner2.train()
    ck_dirs2 = sorted(d for d in os.listdir(args.work_dir) if d.startswith('checkpoint-'))
    last_ck2 = os.path.join(args.work_dir, ck_dirs2[-1])
    pt_norms2 = _load_pt_norms(os.path.join(last_ck2, 'model.pt')) or {}
    if not _check(f'B.2 post-resume save {ck_dirs2[-1]}/model.pt __bundle_params__',
                  target_norms, pt_norms2):
        sys.exit(1)

    print()
    print('=' * 70)
    print('✓ ALL STAGES PASSED — orphan params survive across resume cycle')
    print('=' * 70)


if __name__ == '__main__':
    main()

"""Verify the null-embed fix end-to-end.

Stages:
  1. Build a fresh bundle (norm=0).
  2. Inject T2M 1.0 weights into bundle.null_vtxt_feat (norm=10), null_ctxt_input (norm=44).
  3. accelerator.save_state(dir)         -> model.safetensors
     torch.save(_state_dict_to_save())   -> model.pt
  4. ZERO out the bundle's orphan params (simulates next-startup constructor zeros).
  5. accelerator.load_state(dir)         -> only restores prepared modules
     _restore_bundle_params_from_model_pt -> NEW: restores null embeds
  6. Assert null_vtxt_feat and null_ctxt_input recovered to non-zero.
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile

import torch


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))


def _norm(t):
    return float(t.detach().float().norm().item()) if t is not None else None


def main():
    _setup()
    from mmengine import Config
    from accelerate import Accelerator
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.runner.accelerate_runner import AccelerateRunner

    cfg = Config.fromfile('configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py')
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model_cfg['text_encoder'] = None
    bundle = HyMotionM2MBundle(**model_cfg)

    accelerator = Accelerator(mixed_precision='no')
    bundle.motion_transformer = bundle.motion_transformer.to(accelerator.device)
    bundle.motion_transformer = accelerator.prepare(bundle.motion_transformer)
    for p in bundle.parameters():
        if p.is_meta:
            continue
        p.data = p.data.to(accelerator.device)

    sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt')
    bundle.load_state_dict_selective(sd, strict=False)
    print(f'After T2M-1.0 load: null_vtxt_feat={_norm(bundle.null_vtxt_feat):.4f} '
          f'null_ctxt_input={_norm(bundle.null_ctxt_input):.4f}')

    tmpd = tempfile.mkdtemp(prefix='diag_fix_')
    try:
        accelerator.save_state(tmpd)
        # Write model.pt with __bundle_params__ — emulate _state_dict_to_save logic
        bp = {}
        for n, p in bundle.named_parameters(recurse=False):
            bp[n] = p.data.clone()
        for n, b in bundle.named_buffers(recurse=False):
            bp[n] = b.clone()
        torch.save({'__bundle_params__': bp}, os.path.join(tmpd, 'model.pt'))

        # Simulate next-launch: orphan params reset to constructor zeros
        bundle.null_vtxt_feat.data.zero_()
        bundle.null_ctxt_input.data.zero_()
        print(f'Pre  load_state: null_vtxt_feat={_norm(bundle.null_vtxt_feat):.4f} '
              f'null_ctxt_input={_norm(bundle.null_ctxt_input):.4f}')

        accelerator.load_state(tmpd)
        print(f'Post load_state (no fix): null_vtxt_feat={_norm(bundle.null_vtxt_feat):.4f} '
              f'null_ctxt_input={_norm(bundle.null_ctxt_input):.4f}')

        # Now invoke the fix (must be a runner method).
        # We instantiate a minimal runner-like object with the necessary attrs.
        class _R:
            pass
        r = _R()
        r.bundle = bundle
        r.accelerator = accelerator
        AccelerateRunner._restore_bundle_params_from_model_pt(r, tmpd)
        print(f'Post fix         : null_vtxt_feat={_norm(bundle.null_vtxt_feat):.4f} '
              f'null_ctxt_input={_norm(bundle.null_ctxt_input):.4f}')

        ok_v = abs(_norm(bundle.null_vtxt_feat) - 10.1251) < 0.01
        ok_c = abs(_norm(bundle.null_ctxt_input) - 44.7500) < 0.01
        print(f'\nFIX VERIFICATION: null_vtxt_feat={ok_v} null_ctxt_input={ok_c}')
        if ok_v and ok_c:
            print('✓ PASS: orphan params correctly restored after full-resume')
        else:
            print('✗ FAIL: fix did not work')
            sys.exit(1)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == '__main__':
    main()

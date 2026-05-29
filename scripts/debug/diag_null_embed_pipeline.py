"""Reproduce the EXACT from_cfg → prepare → load_from → save pipeline,
checkpoint-by-checkpoint, to identify where null_vtxt_feat becomes zero.

Steps tracked:
  S0  bundle just constructed (expected: zeros)
  S1  after `for p in bundle.parameters(): p.data = p.data.to(device)` (move-only)
  S2  after `accelerator.prepare(*trainable_modules)` (children become DDP)
  S3  after `bundle.load_state_dict_selective(T2M_1.0_state_dict)` (expected: norm > 0)
  S4  after `accelerator.save_state(dir)` then read back model.safetensors
  S5  after `_state_dict_to_save()` and torch.save → reload model.pt
  S6  after `accelerator.load_state(dir)` (full resume) — does it overwrite back to zero?

Everything runs single-GPU / no-DDP because we only need a logical trace,
not actual distributed semantics. The bug, if it's in save/load logic, will
show up identically.
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile

import torch


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path.insert(0, root)


def _norm(p):
    if p is None:
        return None
    return float(p.detach().float().norm().item())


def _state_str(bundle, label):
    nv = _norm(getattr(bundle, 'null_vtxt_feat', None))
    nc = _norm(getattr(bundle, 'null_ctxt_input', None))
    mean = _norm(getattr(bundle, 'mean', None))
    std = _norm(getattr(bundle, 'std', None))
    print(f'[{label}] null_vtxt_feat={nv:.4f} | null_ctxt_input={nc:.4f} | mean={mean:.4f} | std={std:.4f}')


def main():
    _setup()
    from mmengine import Config
    from accelerate import Accelerator
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile('configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py')
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model_cfg['text_encoder'] = None  # avoid loading Qwen3

    print('=' * 70)
    print('Stage S0: just constructed')
    bundle = HyMotionM2MBundle(**model_cfg)
    _state_str(bundle, 'S0')

    print('=' * 70)
    print('Stage S1: orphan params moved to device (mimics from_cfg L297-299)')
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device

    _child_params = set()
    for child in bundle.children():
        for p in child.parameters():
            _child_params.add(p.data_ptr())
        for b in child.buffers():
            _child_params.add(b.data_ptr())
    for p in bundle.parameters():
        if p.data_ptr() not in _child_params:
            p.data = p.data.to(device)
    for name, buf in bundle.named_buffers():
        if buf.data_ptr() not in _child_params:
            parts = name.rsplit('.', 1)
            if len(parts) == 1:
                bundle.register_buffer(parts[0], buf.to(device))
    _state_str(bundle, 'S1')

    print('=' * 70)
    print('Stage S2: children moved to device + prepare (skipping DDP since 1 GPU)')
    bundle.motion_transformer = bundle.motion_transformer.to(device)
    prepared = accelerator.prepare(bundle.motion_transformer)
    bundle.motion_transformer = prepared
    _state_str(bundle, 'S2')

    print('=' * 70)
    print('Stage S3: load T2M 1.0 via load_state_dict_selective (mimics _handle_load → _load)')
    sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt')
    print(f'  T2M 1.0 sd has null_vtxt_feat: {"null_vtxt_feat" in sd} '
          f'(norm={_norm(sd.get("null_vtxt_feat")):.4f}); '
          f'null_ctxt_input: norm={_norm(sd.get("null_ctxt_input")):.4f}')
    bundle.load_state_dict_selective(sd, strict=False)
    _state_str(bundle, 'S3')

    print('=' * 70)
    print('Stage S4: accelerator.save_state(dir) then read back model.safetensors')
    tmpd = tempfile.mkdtemp(prefix='diag_null_')
    try:
        accelerator.save_state(tmpd)
        st_path = os.path.join(tmpd, 'model.safetensors')
        if os.path.exists(st_path):
            from safetensors.torch import load_file
            saved_sd = load_file(st_path)
            keys_with_null = [k for k in saved_sd.keys() if 'null' in k.lower()]
            print(f'  model.safetensors total_keys: {len(saved_sd)}')
            print(f'  null-related keys in safetensors: {keys_with_null}')
        else:
            print(f'  no model.safetensors found in {tmpd}; files: {sorted(os.listdir(tmpd))}')

        print('=' * 70)
        print('Stage S5: torch.save(_state_dict_to_save()-equivalent) → reload model.pt')
        # Emulate _state_dict_to_save
        sd_to_save = {}
        for name in bundle._save_ckpt_modules:
            mod = getattr(bundle, name, None)
            if isinstance(mod, torch.nn.Module):
                # Unwrap DDP if any
                target = mod
                while hasattr(target, 'module'):
                    target = target.module
                sd_to_save[name] = target.state_dict()
        bundle_params = {}
        for pn, p in bundle.named_parameters(recurse=False):
            bundle_params[pn] = p.data.clone()
        for bn, b in bundle.named_buffers(recurse=False):
            bundle_params[bn] = b.clone()
        if bundle_params:
            sd_to_save['__bundle_params__'] = bundle_params
        mpt = os.path.join(tmpd, 'model.pt')
        torch.save(sd_to_save, mpt)
        reloaded = torch.load(mpt, map_location='cpu', weights_only=False)
        bp = reloaded.get('__bundle_params__', {})
        nv = bp.get('null_vtxt_feat')
        nc = bp.get('null_ctxt_input')
        print(f'  reloaded model.pt __bundle_params__.null_vtxt_feat norm: {_norm(nv):.4f}')
        print(f'  reloaded model.pt __bundle_params__.null_ctxt_input norm: {_norm(nc):.4f}')

        print('=' * 70)
        print('Stage S6: simulate auto_resume by accelerator.load_state(dir); does null_vtxt_feat survive?')
        # zero out the bundle params then call accelerator.load_state to see if it restores them
        bundle.null_vtxt_feat.data.zero_()
        bundle.null_ctxt_input.data.zero_()
        _state_str(bundle, 'S6.pre  (before load_state)')
        accelerator.load_state(tmpd)
        _state_str(bundle, 'S6.post (after load_state)')

    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == '__main__':
    main()

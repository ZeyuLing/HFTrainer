#!/usr/bin/env python3
"""Numerical parity check: hftrainer-port MotionCLIP vs versatilemotion original.

Loads:
  - Original mmengine .pth checkpoint into versatilemotion's MotionCLIPTrainer.
  - Converted safetensors into hftrainer's MotionCLIPBundle.

Then runs identical (motion, text) batches through both and reports max abs diff
of text embeddings, motion embeddings, logits.

Usage:
    python3 tools/test_motionclip_parity.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import torch

HF_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
VM_ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion')
SRC_PTH = VM_ROOT / 'work_dirs/motionclip_base_1p_aug_hq/best_r_precision_top_3_epoch_840.pth'
ST_PATH = HF_ROOT / 'checkpoints/motion_clip/motionclip_base_1p_aug_hq/motionclip_model.safetensors'
BUNDLE_CFG_PATH = HF_ROOT / 'checkpoints/motion_clip/motionclip_base_1p_aug_hq/bundle_config.json'
CLIP_PATH = HF_ROOT / 'checkpoints/clip-vit-base-patch32'


def build_hftrainer_bundle():
    sys.path.insert(0, str(HF_ROOT))
    from safetensors.torch import load_file
    from hftrainer.models.motion.motion_clip import MotionCLIPBundle

    with open(BUNDLE_CFG_PATH) as f:
        bcfg = json.load(f)

    bundle = MotionCLIPBundle(
        text_config=bcfg['text_config'],
        motion_config=bcfg['motion_config'],
        projection_dim=bcfg['projection_dim'],
        logit_scale_init_value=bcfg['logit_scale_init_value'],
        tokenizer={
            'type': 'CLIPTokenizer',
            'from_pretrained': {
                'pretrained_model_name_or_path': str(CLIP_PATH),
            },
        },
        smpl_pose_processor={
            'type': 'SMPLPoseProcessor',
            'do_normalize': True,
            'stats_file': str(HF_ROOT / 'data/statistic/smplx55_stats_hymotion_aug.json'),
            'rot_type': 'rotation_6d',
            'transl_type': 'abs',
            'smpl_type': 'smpl_22',
            'smpl_model': None,
            'smooth_model': None,
        },
        clip_pretrained=None,
        freeze_text_encoder=False,
    )
    sd = load_file(str(ST_PATH))
    missing, unexpected = bundle.motionclip_model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return bundle.eval()


def _import_module_from_path(mod_name: str, path: Path):
    """Import a Python module from a file path, registering it in sys.modules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_versatilemotion_model():
    """Bypass mmotion.trainers chain (broken on py3.9) and load MotionCLIPModel directly.

    Strategy:
      1. importlib-load the 5 motion_clip model files into a fake `vm_motionclip` package.
      2. Build MotionCLIPModel from the original config dicts.
      3. Strip 'motionclip_model.' prefix from the .pth state_dict and load.
    """
    vm_pkg = VM_ROOT / 'mmotion/models/transformers/motion_clip'

    # Create a minimal fake package
    import types
    pkg = types.ModuleType('vm_motionclip')
    pkg.__path__ = [str(vm_pkg)]
    sys.modules['vm_motionclip'] = pkg

    base = _import_module_from_path('vm_motionclip.modeling_motionclip_base', vm_pkg / 'modeling_motionclip_base.py')
    cfg_mod = _import_module_from_path('vm_motionclip.configuration_motionclip', vm_pkg / 'configuration_motionclip.py')
    text_mod = _import_module_from_path('vm_motionclip.modeling_motionclip_text', vm_pkg / 'modeling_motionclip_text.py')
    motion_mod = _import_module_from_path('vm_motionclip.modeling_motionclip_motion', vm_pkg / 'modeling_motionclip_motion.py')
    main_mod = _import_module_from_path('vm_motionclip.modeling_motionclip', vm_pkg / 'modeling_motionclip.py')

    MotionCLIPConfig = cfg_mod.MotionCLIPConfig
    MotionCLIPTextConfig = cfg_mod.MotionCLIPTextConfig
    MotionCLIPMotionConfig = cfg_mod.MotionCLIPMotionConfig
    MotionCLIPModel = main_mod.MotionCLIPModel

    # Build config matching motionclip_base_1p_aug_hq
    text_cfg_dict = dict(
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=256,
        projection_dim=512,
        hidden_act='quick_gelu',
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )
    motion_cfg_dict = dict(
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        motion_dim=135,
        max_position_embeddings=512,
        projection_dim=512,
        hidden_act='quick_gelu',
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )
    config = MotionCLIPConfig(
        text_config=MotionCLIPTextConfig(**text_cfg_dict),
        motion_config=MotionCLIPMotionConfig(**motion_cfg_dict),
        projection_dim=512,
        logit_scale_init_value=2.6592,
    )
    model = MotionCLIPModel(config)

    print(f'[vm] loading checkpoint {SRC_PTH}')
    raw = torch.load(str(SRC_PTH), map_location='cpu', weights_only=False)
    sd = raw.get('state_dict', raw)
    # Strip motionclip_model. prefix and drop unrelated entries
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('motionclip_model.'):
            new_sd[k[len('motionclip_model.'):]] = v
    new_sd = {k: v for k, v in new_sd.items() if 'position_ids' not in k}

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f'[vm] missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print(f'  missing[:5]: {missing[:5]}')
    if unexpected:
        print(f'  unexpected[:5]: {unexpected[:5]}')

    # Wrap into a small object exposing the same APIs the parity check uses.
    class _VMWrapper:
        def __init__(self, motionclip_model, tokenizer, smpl_pose_processor):
            self.motionclip_model = motionclip_model
            self.tokenizer = tokenizer
            self.smpl_pose_processor = smpl_pose_processor

        def to(self, device):
            self.motionclip_model = self.motionclip_model.to(device)
            if self.smpl_pose_processor is not None:
                self.smpl_pose_processor = self.smpl_pose_processor.to(device)
            return self

        def eval(self):
            self.motionclip_model.eval()
            return self

        def tokenize(self, texts):
            max_len = self.motionclip_model.config.text_config.max_position_embeddings
            return self.tokenizer(
                texts, padding=True, truncation=True, max_length=max_len,
                return_tensors='pt',
            )

        def encode_text(self, input_ids, attn):
            return self.motionclip_model.get_text_features(
                input_ids=input_ids, attention_mask=attn,
            )

        def encode_motion(self, motion, attn):
            return self.motionclip_model.get_motion_features(
                motion_values=motion, attention_mask=attn,
            )

    # Build tokenizer + SMPL processor the same way HF bundle does (same shared assets).
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(str(CLIP_PATH))

    sys.path.insert(0, str(HF_ROOT))
    from hftrainer.models.motion.components.motion_processor.smpl_processor import (
        SMPLPoseProcessor as HFSMPLPoseProcessor,
    )
    smpl_proc = HFSMPLPoseProcessor(
        do_normalize=True,
        stats_file=str(HF_ROOT / 'data/statistic/smplx55_stats_hymotion_aug.json'),
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
        smpl_model=None,
        smooth_model=None,
    )
    return _VMWrapper(model, tok, smpl_proc).eval()


@torch.no_grad()
def parity_check(hf_bundle, vm_trainer, device='cuda'):
    hf_bundle = hf_bundle.to(device)
    vm_trainer = vm_trainer.to(device)

    # ---- Build identical inputs ----
    torch.manual_seed(42)
    B, T, D = 4, 64, 135
    motion_raw = torch.randn(B, T, D, device=device)
    captions = [
        'a person walks forward slowly',
        'someone is running quickly across the room',
        'a person jumps up and down with arms raised',
        'a person sits down on a chair',
    ]
    num_frames = [T] * B

    # ---- HF Trainer path ----
    # Normalize motion using shared SMPLPoseProcessor (same code in both repos).
    motion_norm_hf = hf_bundle.smpl_pose_processor.normalize(motion_raw)
    motion_attn_hf = torch.ones(B, T, device=device)
    enc_hf = hf_bundle.tokenize(captions)
    text_emb_hf = hf_bundle.encode_text(
        enc_hf['input_ids'].to(device),
        enc_hf['attention_mask'].to(device),
    )
    motion_emb_hf = hf_bundle.encode_motion(motion_norm_hf, motion_attn_hf)

    # ---- VersatileMotion path ----
    motion_norm_vm = vm_trainer.smpl_pose_processor.normalize(motion_raw)
    motion_attn_vm = torch.ones(B, T, device=device)
    enc_vm = vm_trainer.tokenize(captions)
    text_emb_vm = vm_trainer.encode_text(
        enc_vm['input_ids'].to(device),
        enc_vm['attention_mask'].to(device),
    )
    motion_emb_vm = vm_trainer.encode_motion(motion_norm_vm, motion_attn_vm)

    # ---- Compare ----
    print()
    print('===== Parity check =====')
    print(f'motion_norm  diff: {(motion_norm_hf - motion_norm_vm).abs().max():.3e}')
    print(f'text_emb     diff: {(text_emb_hf - text_emb_vm).abs().max():.3e}  '
          f'(rel: {((text_emb_hf - text_emb_vm).abs() / (text_emb_vm.abs() + 1e-8)).mean():.3e})')
    print(f'motion_emb   diff: {(motion_emb_hf - motion_emb_vm).abs().max():.3e}  '
          f'(rel: {((motion_emb_hf - motion_emb_vm).abs() / (motion_emb_vm.abs() + 1e-8)).mean():.3e})')
    print(f'text_emb     hf shape: {tuple(text_emb_hf.shape)}, vm shape: {tuple(text_emb_vm.shape)}')

    # Pad-id sanity
    print(f'tokenizer hf pad_max: {enc_hf["input_ids"].shape}, vm: {enc_vm["input_ids"].shape}')
    pass_text = (text_emb_hf - text_emb_vm).abs().max() < 1e-3
    pass_mot = (motion_emb_hf - motion_emb_vm).abs().max() < 1e-3
    print(f'Result: text {"PASS" if pass_text else "FAIL"}, motion {"PASS" if pass_mot else "FAIL"}')
    return pass_text and pass_mot


def main():
    hf = build_hftrainer_bundle()
    vm = load_versatilemotion_model()
    ok = parity_check(hf, vm)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

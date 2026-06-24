#!/usr/bin/env python3
"""Convert versatilemotion mmengine MotionCLIP checkpoint to hftrainer format.

Source format (mmengine .pth):
    state_dict = {
        'meta': {...},
        'state_dict': {
            'motionclip_model.text_model.embeddings.token_embedding.weight': ...,
            'motionclip_model.text_projection.weight': ...,
            'motionclip_model.motion_model.embeddings.motion_projection.weight': ...,
            ...,
            # 'smpl_pose_processor.*' is buffers/zero — discarded here
            # 'tokenizer' is non-tensor — N/A
        },
    }

Target format (hftrainer):
  We strip the 'motionclip_model.' prefix and save a clean MotionCLIPModel
  state_dict in `<out_dir>/motionclip_model.safetensors`. We also save the
  text/motion configs as JSON (`<out_dir>/config.json`) so that
  ``MotionCLIPModel(MotionCLIPConfig.from_pretrained(out_dir))`` works.

Usage:
    python3 tools/convert_motionclip_checkpoint.py \
        --src /apdcephfs_cq11/.../work_dirs/motionclip_base_1p_aug_hq/best_r_precision_top_3_epoch_840.pth \
        --src-config /apdcephfs_cq11/.../configs/motion_clip/motionclip_base_1p_aug_hq.py \
        --out-dir checkpoints/motion_clip/motionclip_base_1p_aug_hq
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='mmengine .pth checkpoint path')
    p.add_argument(
        '--src-config',
        required=False,
        default=None,
        help='Original config .py path (used to copy text/motion configs).',
    )
    p.add_argument('--out-dir', required=True, help='Output directory')
    p.add_argument('--text-config-json', default=None,
                   help='Optional: path to text_config.json (overrides --src-config).')
    p.add_argument('--motion-config-json', default=None,
                   help='Optional: path to motion_config.json (overrides --src-config).')
    args = p.parse_args()

    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[load] {args.src}')
    raw = torch.load(args.src, map_location='cpu', weights_only=False)
    if 'state_dict' in raw:
        sd = raw['state_dict']
    else:
        sd = raw

    print(f'[stats] keys in source state_dict: {len(sd)}')
    sample_keys = list(sd.keys())[:5]
    for k in sample_keys:
        print(f'   sample key: {k}')

    # Strip 'motionclip_model.' from MotionCLIPTrainer-saved checkpoint
    sd = _strip_prefix(sd, 'motionclip_model.')

    # Drop processor/tokenizer entries (non-trainable, recomputed from config)
    sd = {
        k: v for k, v in sd.items()
        if not k.startswith('smpl_pose_processor.')
    }

    # Drop position_ids buffers (not persistent in target model)
    sd = {
        k: v for k, v in sd.items()
        if 'position_ids' not in k
    }

    print(f'[stats] keys after cleanup: {len(sd)}')

    # Verify shape compatibility against a freshly built MotionCLIPModel using
    # the configs from the source config file.
    from hftrainer.models.motion.motion_clip import (
        MotionCLIPConfig,
        MotionCLIPMotionConfig,
        MotionCLIPModel,
        MotionCLIPTextConfig,
    )

    text_cfg_dict, motion_cfg_dict, projection_dim, logit_scale_init = _load_configs_from_source(
        args.src_config, args.text_config_json, args.motion_config_json,
    )
    text_cfg = MotionCLIPTextConfig(**text_cfg_dict)
    motion_cfg = MotionCLIPMotionConfig(**motion_cfg_dict)
    full_cfg = MotionCLIPConfig(
        text_config=text_cfg, motion_config=motion_cfg,
        projection_dim=projection_dim,
        logit_scale_init_value=logit_scale_init,
    )
    model = MotionCLIPModel(full_cfg)

    # Load with strict=False; report any mismatch
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[verify] missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        for k in missing[:10]:
            print(f'   - missing: {k}')
    if unexpected:
        for k in unexpected[:10]:
            print(f'   - unexpected: {k}')

    # Save as safetensors with the same flat key layout as MotionCLIPModel.state_dict()
    final_sd = model.state_dict()
    out_path = out_dir / 'motionclip_model.safetensors'
    save_file(final_sd, str(out_path))
    print(f'[save] {out_path}  ({sum(v.numel() for v in final_sd.values()) * 4 / 1024 / 1024:.1f} MB float32)')

    # Save bundle config JSON for downstream loading.
    bundle_cfg = {
        'text_config': text_cfg.to_dict(),
        'motion_config': motion_cfg.to_dict(),
        'projection_dim': projection_dim,
        'logit_scale_init_value': logit_scale_init,
    }
    cfg_path = out_dir / 'bundle_config.json'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(bundle_cfg, f, indent=2)
    print(f'[save] {cfg_path}')

    print('Done.')


def _load_configs_from_source(src_cfg_path, text_json, motion_json):
    """Return (text_cfg_dict, motion_cfg_dict, projection_dim, logit_scale_init)."""
    if text_json and motion_json:
        with open(text_json, 'r') as f:
            text_cfg = json.load(f)
        with open(motion_json, 'r') as f:
            motion_cfg = json.load(f)
        return text_cfg, motion_cfg, 512, 2.6592

    if not src_cfg_path:
        # Fall back to versatilemotion defaults (motionclip_base_1p_aug_hq).
        clip_vit_b_32 = dict(
            vocab_size=49408, hidden_size=512, intermediate_size=2048,
            num_hidden_layers=12, num_attention_heads=8,
            projection_dim=512, hidden_act='quick_gelu',
            layer_norm_eps=1e-5, attention_dropout=0.0,
            initializer_range=0.02, initializer_factor=1.0,
        )
        text_cfg = dict(
            **{k: v for k, v in clip_vit_b_32.items() if k != 'max_position_embeddings'},
            max_position_embeddings=256,
        )
        motion_cfg = dict(
            hidden_size=512, intermediate_size=2048, num_hidden_layers=12,
            num_attention_heads=8, motion_dim=135, max_position_embeddings=512,
            projection_dim=512, hidden_act='quick_gelu', layer_norm_eps=1e-5,
            attention_dropout=0.0, initializer_range=0.02, initializer_factor=1.0,
        )
        return text_cfg, motion_cfg, 512, 2.6592

    # Otherwise parse the mmengine .py config to extract sub-configs.
    from mmengine.config import Config
    cfg = Config.fromfile(src_cfg_path)
    model_cfg = cfg.model
    text_cfg = dict(model_cfg.get('text_config', {}))
    motion_cfg = dict(model_cfg.get('motion_config', {}))
    projection_dim = int(model_cfg.get('projection_dim', 512))
    logit_scale_init = float(model_cfg.get('logit_scale_init_value', 2.6592))
    return text_cfg, motion_cfg, projection_dim, logit_scale_init


if __name__ == '__main__':
    main()

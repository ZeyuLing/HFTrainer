"""Diagnostic: caption coverage + mask sampler distribution + ckpt sanity.

Run on debug machine via:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer && \
    python3 tools/diag_caption_v2.py [--n_samples 200] [--mode coverage|mask|ckpt|all]

Outputs single-line summary stats so the caller can grep them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np


def _setup():
    """Add repo root to sys.path and silence noisy logs."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path.insert(0, root)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return root


def diag_caption_coverage(n_samples: int = 200) -> dict:
    """Sample N rows from the actual training annotation and report
    caption availability + .pt embedding availability."""
    _setup()
    from hftrainer.datasets.motion.motionhub.transforms.load_text import (
        _caption_path_to_embedding_path,
    )

    anno_file = 'data/annotation/train_hymotion_400h_hq_20260403.json'
    if not os.path.exists(anno_file):
        return {'error': f'anno_file not found: {anno_file}'}
    with open(anno_file) as f:
        anno = json.load(f)

    data_dir = 'data/motionhub'
    if isinstance(anno, dict) and 'data_list' in anno:
        items = anno['data_list']
    else:
        items = anno
    # data_list may be dict (keyed by id) or list
    if isinstance(items, dict):
        keys = list(items.keys())
        rng = np.random.RandomState(0)
        sel = rng.choice(len(keys), size=min(n_samples, len(keys)), replace=False)
        sampled = [items[keys[int(i)]] for i in sel]
    elif isinstance(items, list):
        rng = np.random.RandomState(0)
        sel = rng.choice(len(items), size=min(n_samples, len(items)), replace=False)
        sampled = [items[int(i)] for i in sel]
    else:
        return {'error': f'unexpected items type: {type(items)}'}

    n_total = len(sampled)
    n_has_caption_path = 0
    n_caption_file_exists = 0
    n_pt_path_known = 0
    n_pt_file_exists = 0
    n_pt_loadable = 0
    n_pt_nonempty = 0
    n_caption_list_empty = 0
    sample_pt_paths = []
    n_subset = Counter()
    n_caption_field_used = Counter()

    for item in sampled:
        if not isinstance(item, dict):
            continue
        n_subset[item.get('subset', 'unknown')] += 1
        cap = (
            item.get('hierarchical_caption_path')
            or item.get('caption_path')
            or item.get('caption')
        )
        if cap is None:
            continue
        if isinstance(cap, list):
            cap = cap[0] if cap else None
            if cap is None:
                continue
        if 'hierarchical_caption_path' in item and item['hierarchical_caption_path']:
            n_caption_field_used['hierarchical_caption_path'] += 1
        elif 'caption_path' in item and item['caption_path']:
            n_caption_field_used['caption_path'] += 1
        else:
            n_caption_field_used['caption'] += 1
        n_has_caption_path += 1
        cap_full = cap if os.path.isabs(cap) else os.path.join(data_dir, cap)
        if os.path.exists(cap_full):
            n_caption_file_exists += 1
        pt_path = _caption_path_to_embedding_path(cap_full)
        if pt_path is not None:
            n_pt_path_known += 1
            if os.path.exists(pt_path):
                n_pt_file_exists += 1
                if len(sample_pt_paths) < 3:
                    sample_pt_paths.append(pt_path)
                try:
                    import torch
                    data = torch.load(pt_path, map_location='cpu', weights_only=False)
                    n_pt_loadable += 1
                    res_list = data.get('result', [])
                    if res_list:
                        n_pt_nonempty += 1
                    else:
                        n_caption_list_empty += 1
                except Exception:
                    pass

    return {
        'n_total': n_total,
        'has_caption_path': n_has_caption_path,
        'caption_file_exists': n_caption_file_exists,
        'pt_path_known': n_pt_path_known,
        'pt_file_exists': n_pt_file_exists,
        'pt_loadable': n_pt_loadable,
        'pt_nonempty': n_pt_nonempty,
        'pt_loaded_but_empty': n_caption_list_empty,
        'sample_pt_paths': sample_pt_paths,
        'subset_distribution': dict(n_subset),
        'caption_field_used': dict(n_caption_field_used),
    }


def diag_mask_sampler(n_samples: int = 1000, T: int = 360) -> dict:
    """Sample N masks from the v3 sampler (used by uncond + caption phase2)
    and the v2 sampler with caption_local_046b weights, report stats."""
    _setup()
    from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (
        sample_condition_v3,
        DEFAULT_K_WEIGHTS,
    )
    from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
        sample_condition,
    )

    rng = np.random.RandomState(42)

    def _stats(name, sample_fn):
        densities = []
        pure_t2m = 0
        pure_zero = 0
        K_counter = Counter()
        edit_count = 0
        for _ in range(n_samples):
            mask, edit = sample_fn()
            density = float(mask.mean())  # 1=generate
            densities.append(density)
            if density > 0.999:
                pure_t2m += 1
            if density < 0.001:
                pure_zero += 1
            if edit:
                edit_count += 1
        d = np.array(densities)
        return {
            'name': name,
            'pure_t2m_frac': pure_t2m / n_samples,
            'pure_zero_frac': pure_zero / n_samples,
            'edit_frac': edit_count / n_samples,
            'density_mean': float(d.mean()),
            'density_p10': float(np.percentile(d, 10)),
            'density_p50': float(np.percentile(d, 50)),
            'density_p90': float(np.percentile(d, 90)),
        }

    # caption_local_phase2 uses v3 with overridden k_weights (boosted K=0)
    phase2_k = (0.16, 0.513, 0.233, 0.065, 0.029)
    caption_phase2 = _stats(
        'caption_phase2_v3',
        lambda: sample_condition_v3(T, rng, k_weights=phase2_k, editing_prob=0.15),
    )

    # uncond_local_046b uses v3 with default k_weights
    uncond = _stats(
        'uncond_local_v3_default',
        lambda: sample_condition_v3(T, rng, editing_prob=0.15),
    )

    # caption_local_046b uses v2 with tier2_prob=0.4 + custom tier2_weights
    tier2_weights = {
        'pure_gen': 0.40, 'inbetween': 0.15, 'prefix': 0.10, 'keyframes': 0.10,
        'end_effector': 0.08, 'trajectory': 0.07, 'foot_ground': 0.05, 'edit_repair': 0.05,
    }
    caption_046b = _stats(
        'caption_local_046b_v2',
        lambda: sample_condition(
            T, rng, tier2_prob=0.4, editing_prob=0.15, tier2_weights=tier2_weights,
        ),
    )

    return {'caption_phase2': caption_phase2, 'uncond': uncond, 'caption_046b': caption_046b}


def diag_ckpt(ckpt_path: str) -> dict:
    """Inspect a safetensors / .pt checkpoint and report key info."""
    import torch
    if ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']

    keys = list(sd.keys())
    text_keys = [k for k in keys if 'text' in k.lower() or 'ctxt' in k.lower() or 'vtxt' in k.lower()]
    null_keys = [k for k in keys if 'null' in k.lower()]

    # Compute norms for null embeddings
    null_norms = {}
    for k in null_keys:
        v = sd[k]
        if hasattr(v, 'float') and hasattr(v, 'norm'):
            null_norms[k] = float(v.float().norm().item())

    return {
        'ckpt_path': ckpt_path,
        'total_keys': len(keys),
        'first_5_keys': keys[:5],
        'last_5_keys': keys[-5:],
        'text_related_keys_n': len(text_keys),
        'text_related_keys_sample': text_keys[:5],
        'null_keys': null_keys,
        'null_norms': null_norms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_samples', type=int, default=200)
    p.add_argument('--mode', default='all',
                   choices=['coverage', 'mask', 'ckpt', 'all'])
    p.add_argument('--ckpt',
                   default='work_dirs/hymotion_m2m_v2_caption_local_phase2/'
                           'checkpoint-epoch_1810/model.safetensors')
    args = p.parse_args()

    out = {}
    if args.mode in ('coverage', 'all'):
        print('=== caption coverage ===', flush=True)
        out['coverage'] = diag_caption_coverage(args.n_samples)
        print(json.dumps(out['coverage'], indent=2), flush=True)
    if args.mode in ('mask', 'all'):
        print('=== mask sampler distribution ===', flush=True)
        out['mask'] = diag_mask_sampler(n_samples=2000)
        print(json.dumps(out['mask'], indent=2), flush=True)
    if args.mode in ('ckpt', 'all'):
        print(f'=== ckpt sanity: {args.ckpt} ===', flush=True)
        out['ckpt'] = diag_ckpt(args.ckpt)
        print(json.dumps(out['ckpt'], indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()

"""Forward sanity test: how much does the trained M2M phase2 model
actually depend on caption embedding?

Strategy:
  - Build the bundle with phase2 caption_local_phase2 config
  - Load epoch_1810 weights (no text_encoder needed)
  - Pick 5 .pt caption embedding files from the training data
  - For each: forward (mask=1, x_t=N(0,1), real_caption) vs forward (mask=1, x_t=same, null_ctxt)
  - Measure output velocity norm difference / cosine similarity
  - Compare with: forward (different captions, same x_t) — diversity test

A working caption-conditioned model should show:
  - high diff between real_caption and null_ctxt
  - high diff between two different real captions
  - low diff if the caption is reused

A degenerate model (caption ignored) shows:
  - near-zero diff in all three cases
"""
from __future__ import annotations

import os
import sys
import argparse
import json

import numpy as np
import torch


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path.insert(0, root)


def load_bundle(config_path: str, ckpt_path: str, device='cuda:0'):
    from mmengine import Config
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model_cfg['text_encoder'] = None  # we don't need online encoding

    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    bundle = HyMotionM2MBundle(**model_cfg)
    sd = load_checkpoint(ckpt_path)
    bundle.load_state_dict_selective(sd, strict=False)
    bundle.eval().to(device)
    return bundle


def find_caption_pts(n: int = 5):
    """Find a few existing pre-extracted caption .pt files."""
    base_dirs = [
        'data/hymotion_data/Academic',
        'data/hymotion_data/AcademicRetarget',
        'data/hymotion_data/Taobao',
        'data/hymotion_data/Game',
    ]
    found = []
    for bd in base_dirs:
        if not os.path.isdir(bd):
            continue
        # walk shallow — qwen3_* dirs sit one level deep
        for sub in sorted(os.listdir(bd))[:5]:
            for sub2 in ['qwen3_augmented', 'qwen3_human_checked_short', 'qwen3_improved_simple_short']:
                d = os.path.join(bd, sub, sub2)
                if not os.path.isdir(d):
                    continue
                for root, dirs, files in os.walk(d):
                    for f in files:
                        if f.endswith('.pt'):
                            found.append(os.path.join(root, f))
                            if len(found) >= n * 5:
                                break
                    if len(found) >= n * 5:
                        break
                if len(found) >= n * 5:
                    break
            if len(found) >= n * 5:
                break
        if len(found) >= n * 5:
            break
    return found[:n]


def load_caption_embedding(pt_path: str):
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    res = data.get('result', [])
    if not res:
        return None
    emb = res[0].get('text_embedding')
    if emb is None:
        return None
    cap = res[0].get('caption', '')
    return {
        'caption': cap[:80],
        'text_vec_raw': emb['text_vec_raw'].squeeze(0),    # [1, 768]
        'text_ctxt_raw': emb['text_ctxt_raw'].squeeze(0),  # [seq, 4096]
        'text_ctxt_raw_length': emb['text_ctxt_raw_length'].squeeze(0),  # scalar
    }


def pad_ctxt(ctxt, pad_len=128):
    if ctxt.shape[0] >= pad_len:
        return ctxt[:pad_len]
    pad = pad_len - ctxt.shape[0]
    return torch.nn.functional.pad(ctxt, (0, 0, 0, pad))


@torch.no_grad()
def forward_test(bundle, caption_emb, B=1, T=120, device='cuda:0', seed=0):
    """Run one forward pass with given caption embedding (or None for null).

    Returns: pred velocity tensor (B, T, 198) and intermediate stats.
    """
    torch.manual_seed(seed)
    D = 198
    pad_len = 128

    # x_t starts as pure noise; mask=1 means generate everything (T2M)
    x_t = torch.randn(B, T, D, device=device)
    src_mask = torch.ones(B, T, D, device=device)
    src_motion = torch.zeros(B, T, D, device=device)
    timesteps = torch.full((B,), 0.5, device=device)
    tgt_padding_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    if caption_emb is None:
        # Null: use null_ctxt_input / null_vtxt_feat at 1 token
        vtxt_input = bundle.null_vtxt_feat.expand(B, 1, -1).contiguous()
        ctxt_input = bundle.null_ctxt_input.expand(B, 1, -1).contiguous()
        ctxt_length = torch.ones(B, dtype=torch.long, device=device)
        ctxt_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    else:
        vtxt_input = caption_emb['text_vec_raw'].unsqueeze(0).to(device).contiguous()  # (1, 1, 768)
        ctxt_padded = pad_ctxt(caption_emb['text_ctxt_raw'], pad_len).unsqueeze(0).to(device).contiguous()
        valid_len = int(caption_emb['text_ctxt_raw_length'])
        ctxt_input = ctxt_padded
        ctxt_length = torch.tensor([valid_len], device=device)
        ctxt_mask = torch.zeros(B, pad_len, dtype=torch.bool, device=device)
        ctxt_mask[:, :valid_len] = True

    vace_context = bundle.prepare_vace_input(
        src_motion=src_motion,
        ref_pose=None,
        src_mask=src_mask,
    )
    x_input = torch.cat([x_t, vace_context], dim=-1)
    pred = bundle.predict_flow(
        x_input=x_input,
        ctxt_input=ctxt_input,
        vtxt_input=vtxt_input,
        timesteps=timesteps,
        x_mask_temporal=tgt_padding_mask,
        ctxt_mask_temporal=ctxt_mask,
    )
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py')
    p.add_argument('--ckpt', default='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_1810/model.pt')
    p.add_argument('--n_captions', type=int, default=5)
    p.add_argument('--n_seeds', type=int, default=3)
    args = p.parse_args()

    _setup()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    print(f'loading bundle from {args.ckpt}...', flush=True)
    bundle = load_bundle(args.config, args.ckpt, device=device)
    print(f'  null_vtxt_feat norm: {bundle.null_vtxt_feat.float().norm().item():.4f}')
    print(f'  null_ctxt_input norm: {bundle.null_ctxt_input.float().norm().item():.4f}')

    caption_pts = find_caption_pts(args.n_captions)
    print(f'\nfound {len(caption_pts)} caption .pt files:')
    captions = []
    for pt in caption_pts:
        emb = load_caption_embedding(pt)
        if emb is not None:
            captions.append(emb)
            print(f'  - "{emb["caption"]}" (seq_len={int(emb["text_ctxt_raw_length"])}, ctxt_norm={emb["text_ctxt_raw"].norm():.2f})')
    if not captions:
        print('No captions found!')
        return

    # For each seed, compute pred for: null_caption, each real caption
    print(f'\n=== Forward consistency test (T2M mode, mask=1) ===')
    results = []
    for seed in range(args.n_seeds):
        pred_null = forward_test(bundle, None, seed=seed, device=device)
        preds_cap = [forward_test(bundle, c, seed=seed, device=device) for c in captions]

        # Compare null vs each caption
        diffs_vs_null = []
        cosines_vs_null = []
        for i, p in enumerate(preds_cap):
            diff = (p - pred_null).norm() / pred_null.norm()
            cos = torch.nn.functional.cosine_similarity(
                p.flatten().unsqueeze(0), pred_null.flatten().unsqueeze(0)
            ).item()
            diffs_vs_null.append(float(diff.item()))
            cosines_vs_null.append(cos)

        # Compare two different captions
        diffs_cross = []
        for i in range(len(preds_cap)):
            for j in range(i + 1, len(preds_cap)):
                diff = (preds_cap[i] - preds_cap[j]).norm() / preds_cap[i].norm()
                diffs_cross.append(float(diff.item()))

        results.append({
            'seed': seed,
            'pred_null_norm': float(pred_null.norm().item()),
            'pred_cap_norms': [float(p.norm().item()) for p in preds_cap],
            'rel_diff_null_vs_cap': diffs_vs_null,
            'cosine_null_vs_cap': cosines_vs_null,
            'rel_diff_cap_vs_cap': diffs_cross,
        })

    print(json.dumps(results, indent=2))

    # Aggregate
    all_null_v_cap = [d for r in results for d in r['rel_diff_null_vs_cap']]
    all_cap_v_cap = [d for r in results for d in r['rel_diff_cap_vs_cap']]
    all_cos = [c for r in results for c in r['cosine_null_vs_cap']]
    print(f'\n=== Summary ===')
    print(f'mean rel_diff(null, cap):  {np.mean(all_null_v_cap):.4f}  (high = caption changes output)')
    print(f'mean cosine(null, cap):    {np.mean(all_cos):.4f}  (low = caption changes output)')
    print(f'mean rel_diff(cap_i, cap_j): {np.mean(all_cap_v_cap):.4f}  (high = different captions yield different outputs)')


if __name__ == '__main__':
    main()

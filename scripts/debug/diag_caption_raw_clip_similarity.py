"""Diagnose text conditioning collapse in HyMotion M2M v2 caption models.

Investigates whether CLIP-L embeddings collapse through the vtxt_encoder MLP,
making text-conditioned and null-conditioned adapter signals indistinguishable.

Steps:
  1. Load E2 caption model's null_vtxt_feat (nn.Parameter, trained from randn*0.01)
  2. Load cached CLIP-L embeddings from eval cache
  3. Compare raw CLIP-L text_vec_raw vs null_vtxt_feat BEFORE MLPEncoder
  4. Compare AFTER MLPEncoder (vtxt_encoder)
  5. Compare multiple captions' CLIP-L embeddings against each other
  6. Measure adapter = timestep_feat + vtxt_feat contribution ratio across timesteps

Usage:
  python scripts/debug/diag_caption_raw_clip_similarity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


# ──────────────────────────────────────────────────────────────────────
# Model loading (same pattern as diag_caption_attention_gates.py)
# ──────────────────────────────────────────────────────────────────────

def load_bundle(config_path, checkpoint_path, device='cuda'):
    from mmengine.config import Config
    import hftrainer  # noqa: F401 — triggers registry population
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile(config_path)
    model_cfg = getattr(cfg, 'model', None)
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()
    bundle_type = model_cfg.get('type')
    bundle_cls = MODEL_BUNDLES.get(bundle_type)
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()
    try:
        state_dict = load_checkpoint(checkpoint_path, map_location='cpu')
        print(f'  Loaded checkpoint: {checkpoint_path}')
        bundle.load_state_dict_selective(state_dict)
    except FileNotFoundError:
        print(f'  WARNING: No checkpoint at {checkpoint_path}, using init weights')
    bundle = bundle.to(device)
    return bundle


def load_cache(cache_path):
    """Load the full caption embedding cache. Returns dict[caption_str -> entry]."""
    cache_raw = torch.load(cache_path, map_location='cpu', weights_only=False)
    cache = cache_raw.get('cache', cache_raw)
    print(f'  Cache loaded: {len(cache)} captions')
    return cache


def pick_captions(cache, n=10):
    """Pick N diverse captions from cache for analysis."""
    all_captions = list(cache.keys())
    if len(all_captions) <= n:
        return all_captions
    # Evenly spaced selection for diversity
    indices = np.linspace(0, len(all_captions) - 1, n, dtype=int)
    return [all_captions[i] for i in indices]


# ──────────────────────────────────────────────────────────────────────
# Cosine similarity helpers
# ──────────────────────────────────────────────────────────────────────

def cos_sim(a, b):
    """Cosine similarity between two vectors (flattened)."""
    a_flat = a.flatten().float().cpu()
    b_flat = b.flatten().float().cpu()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def l2_dist(a, b):
    """L2 distance between two vectors (flattened)."""
    return (a.flatten().float().cpu() - b.flatten().float().cpu()).norm().item()


def pairwise_cosine_matrix(vecs):
    """Compute NxN cosine similarity matrix from list of 1D tensors."""
    n = len(vecs)
    stacked = torch.stack([v.flatten().float().cpu() for v in vecs])
    normed = F.normalize(stacked, dim=-1)
    mat = normed @ normed.T
    return mat


# ──────────────────────────────────────────────────────────────────────
# Table printing helpers
# ──────────────────────────────────────────────────────────────────────

def print_header(title):
    w = 90
    print(f'\n{"=" * w}')
    print(f'  {title}')
    print(f'{"=" * w}')


def print_matrix(mat, labels, title, fmt='.4f'):
    """Print a labelled similarity matrix."""
    n = mat.shape[0]
    max_label_len = min(max(len(l) for l in labels), 35)
    trunc_labels = [l[:max_label_len] for l in labels]

    print(f'\n  {title}')
    # Column headers
    header = ' ' * (max_label_len + 4)
    for i in range(n):
        header += f'  [{i:2d}]  '
    print(header)

    for i in range(n):
        row = f'  [{i:2d}] {trunc_labels[i]:<{max_label_len}s}'
        for j in range(n):
            val = mat[i, j].item()
            if i == j:
                row += f'  {"---":>6s} '
            else:
                row += f'  {val:{fmt}} '
        print(row)


def print_comparison_table(captions, raw_sims, enc_sims, raw_dists, enc_dists,
                           null_norm, caption_raw_norms, caption_enc_norms,
                           null_enc_norm):
    """Print per-caption comparison table: raw vs encoded similarity to null."""
    max_cap_len = 50
    print(f'\n  {"#":>3s}  {"Caption":<{max_cap_len}s}  '
          f'{"‖raw‖":>8s}  {"cos(raw,null)":>13s}  {"L2(raw,null)":>12s}  '
          f'{"‖enc‖":>8s}  {"cos(enc,null)":>13s}  {"L2(enc,null)":>12s}  '
          f'{"Δcos":>6s}')
    print(f'  {"-" * 3}  {"-" * max_cap_len}  '
          f'{"-" * 8}  {"-" * 13}  {"-" * 12}  '
          f'{"-" * 8}  {"-" * 13}  {"-" * 12}  '
          f'{"-" * 6}')

    for i, cap in enumerate(captions):
        cap_trunc = cap[:max_cap_len]
        delta = enc_sims[i] - raw_sims[i]
        print(f'  {i:3d}  {cap_trunc:<{max_cap_len}s}  '
              f'{caption_raw_norms[i]:8.4f}  {raw_sims[i]:13.6f}  {raw_dists[i]:12.4f}  '
              f'{caption_enc_norms[i]:8.4f}  {enc_sims[i]:13.6f}  {enc_dists[i]:12.4f}  '
              f'{delta:+6.3f}')

    print(f'\n  null_vtxt_feat:  ‖raw‖ = {null_norm:.6f}   ‖encoded‖ = {null_enc_norm:.6f}')


# ──────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    E2_CONFIG = 'configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py'
    E2_CKPT = 'work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_90'
    CACHE_PATH = 'data/eval/m2m_v2/caption_embeddings/cache.pt'

    NUM_CAPTIONS = 12  # Number of captions to analyse
    TIMESTEPS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0]  # For adapter analysis

    # ── Step 0: Load model and cache ────────────────────────────────
    print('Loading E2 caption model...')
    bundle = load_bundle(E2_CONFIG, E2_CKPT, device)
    model_dtype = next(bundle.motion_transformer.parameters()).dtype

    print('\nLoading caption embedding cache...')
    cache = load_cache(CACHE_PATH)
    captions = pick_captions(cache, NUM_CAPTIONS)

    # Extract key components
    null_vtxt_raw = bundle.null_vtxt_feat.detach().clone()  # shape [1, 1, 768]
    vtxt_encoder = bundle.motion_transformer.vtxt_encoder
    timestep_encoder = bundle.motion_transformer.timestep_encoder

    # ── Step 1: Raw CLIP-L embedding statistics ─────────────────────
    print_header('STEP 1: Raw CLIP-L Embedding Statistics')

    raw_vecs = []
    for cap in captions:
        entry = cache[cap]
        vtxt = entry['text_vec_raw'].float()  # [1, 768] or [768]
        if vtxt.dim() == 1:
            vtxt = vtxt.unsqueeze(0)
        raw_vecs.append(vtxt.squeeze(0))  # store as [768]

    null_raw_flat = null_vtxt_raw.flatten()  # [768]

    print(f'\n  null_vtxt_feat stats:')
    print(f'    shape:   {list(null_vtxt_raw.shape)}')
    print(f'    ‖null‖₂: {null_raw_flat.norm().item():.6f}')
    print(f'    mean:    {null_raw_flat.mean().item():.6f}')
    print(f'    std:     {null_raw_flat.std().item():.6f}')
    print(f'    min:     {null_raw_flat.min().item():.6f}')
    print(f'    max:     {null_raw_flat.max().item():.6f}')

    print(f'\n  CLIP-L raw embedding stats (across {len(captions)} captions):')
    all_norms = [v.norm().item() for v in raw_vecs]
    print(f'    ‖clip‖₂ range: [{min(all_norms):.4f}, {max(all_norms):.4f}]')
    print(f'    ‖clip‖₂ mean:  {np.mean(all_norms):.4f}')
    print(f'    ‖null‖₂:       {null_raw_flat.norm().item():.6f}')
    print(f'    ratio ‖null‖/‖clip_mean‖: {null_raw_flat.norm().item() / np.mean(all_norms):.6f}')

    # ── Step 2: Raw cosine similarity between captions ──────────────
    print_header('STEP 2: Inter-Caption Raw CLIP-L Cosine Similarity')

    raw_cos_mat = pairwise_cosine_matrix(raw_vecs)
    short_labels = [c[:35] for c in captions]
    print_matrix(raw_cos_mat, short_labels, 'Raw CLIP-L pairwise cosine similarity:')

    # Off-diagonal statistics
    n = len(raw_vecs)
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(raw_cos_mat[i, j].item())
    print(f'\n  Off-diagonal statistics (raw CLIP-L inter-caption):')
    print(f'    mean cos:  {np.mean(off_diag):.6f}')
    print(f'    std cos:   {np.std(off_diag):.6f}')
    print(f'    min cos:   {np.min(off_diag):.6f}')
    print(f'    max cos:   {np.max(off_diag):.6f}')

    # ── Step 3: Raw similarity to null_vtxt_feat ────────────────────
    print_header('STEP 3: Raw CLIP-L vs null_vtxt_feat Similarity')

    raw_sims_to_null = []
    raw_dists_to_null = []
    raw_norms = []

    for v in raw_vecs:
        raw_sims_to_null.append(cos_sim(v, null_raw_flat))
        raw_dists_to_null.append(l2_dist(v, null_raw_flat))
        raw_norms.append(v.norm().item())

    print(f'\n  cos(clip_raw, null_vtxt_feat) across {n} captions:')
    print(f'    mean:  {np.mean(raw_sims_to_null):.6f}')
    print(f'    std:   {np.std(raw_sims_to_null):.6f}')
    print(f'    min:   {np.min(raw_sims_to_null):.6f}')
    print(f'    max:   {np.max(raw_sims_to_null):.6f}')

    # ── Step 4: Pass through vtxt_encoder (MLPEncoder) ──────────────
    print_header('STEP 4: After vtxt_encoder (MLPEncoder: 768→1024→1024)')

    with torch.no_grad():
        # Encode null
        null_input = null_vtxt_raw.to(device=device, dtype=model_dtype)
        null_encoded = vtxt_encoder(null_input.float()).detach().cpu().float()
        null_enc_flat = null_encoded.flatten()

        # Encode each caption
        enc_vecs = []
        for v in raw_vecs:
            inp = v.unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)  # [1, 1, 768]
            out = vtxt_encoder(inp.float()).detach().cpu().float()
            enc_vecs.append(out.squeeze(0).squeeze(0))  # [1024]

    print(f'\n  null_vtxt_feat after vtxt_encoder:')
    print(f'    ‖encoded_null‖₂:  {null_enc_flat.norm().item():.6f}')
    print(f'    mean:             {null_enc_flat.mean().item():.6f}')
    print(f'    std:              {null_enc_flat.std().item():.6f}')

    enc_norms = [v.norm().item() for v in enc_vecs]
    print(f'\n  Encoded caption stats:')
    print(f'    ‖enc‖₂ range: [{min(enc_norms):.4f}, {max(enc_norms):.4f}]')
    print(f'    ‖enc‖₂ mean:  {np.mean(enc_norms):.4f}')

    # Encoded pairwise similarity
    enc_cos_mat = pairwise_cosine_matrix(enc_vecs)
    print_matrix(enc_cos_mat, short_labels, 'Encoded (post-vtxt_encoder) pairwise cosine similarity:')

    enc_off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            enc_off_diag.append(enc_cos_mat[i, j].item())
    print(f'\n  Off-diagonal statistics (encoded inter-caption):')
    print(f'    mean cos:  {np.mean(enc_off_diag):.6f}')
    print(f'    std cos:   {np.std(enc_off_diag):.6f}')
    print(f'    min cos:   {np.min(enc_off_diag):.6f}')
    print(f'    max cos:   {np.max(enc_off_diag):.6f}')

    # ── Step 5: Combined table — raw vs encoded similarity to null ──
    print_header('STEP 5: Per-Caption Comparison (Raw vs Encoded vs Null)')

    enc_sims_to_null = [cos_sim(v, null_enc_flat) for v in enc_vecs]
    enc_dists_to_null = [l2_dist(v, null_enc_flat) for v in enc_vecs]

    print_comparison_table(
        captions, raw_sims_to_null, enc_sims_to_null,
        raw_dists_to_null, enc_dists_to_null,
        null_raw_flat.norm().item(), raw_norms, enc_norms,
        null_enc_flat.norm().item(),
    )

    # Collapse diagnostic
    print(f'\n  ★ KEY DIAGNOSTIC:')
    print(f'    Raw cos(caption, null) mean:     {np.mean(raw_sims_to_null):.6f}')
    print(f'    Encoded cos(caption, null) mean: {np.mean(enc_sims_to_null):.6f}')
    delta = np.mean(enc_sims_to_null) - np.mean(raw_sims_to_null)
    if delta > 0.1:
        print(f'    → vtxt_encoder INCREASES similarity to null by {delta:+.4f}')
        print(f'      ⚠ This suggests the MLP is collapsing text features toward null!')
    elif delta < -0.1:
        print(f'    → vtxt_encoder DECREASES similarity to null by {delta:+.4f}')
        print(f'      ✓ MLP is separating text from null.')
    else:
        print(f'    → vtxt_encoder changes similarity by {delta:+.4f} (small effect)')

    print(f'\n    Raw inter-caption cos mean:     {np.mean(off_diag):.6f}')
    print(f'    Encoded inter-caption cos mean: {np.mean(enc_off_diag):.6f}')
    inter_delta = np.mean(enc_off_diag) - np.mean(off_diag)
    if inter_delta > 0.1:
        print(f'    → vtxt_encoder INCREASES inter-caption similarity by {inter_delta:+.4f}')
        print(f'      ⚠ Captions become MORE similar after encoding — diversity loss!')
    else:
        print(f'    → vtxt_encoder changes inter-caption similarity by {inter_delta:+.4f}')

    # ── Step 6: Adapter contribution analysis ───────────────────────
    print_header('STEP 6: Adapter Contribution Analysis (timestep_feat + vtxt_feat)')
    print(f'  adapter = timestep_feat + vtxt_feat')
    print(f'  Question: how much does vtxt_feat contribute to the adapter signal?\n')

    with torch.no_grad():
        # Pick one caption and the null for comparison
        test_caption_idx = 0
        test_raw = raw_vecs[test_caption_idx]
        test_cap_name = captions[test_caption_idx][:60]

        # Encode vtxt for the test caption
        vtxt_input_text = test_raw.unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)
        vtxt_feat_text = vtxt_encoder(vtxt_input_text.float())  # [1, 1, 1024]

        # Encode vtxt for null
        vtxt_feat_null = vtxt_encoder(null_vtxt_raw.to(device=device, dtype=model_dtype).float())

        print(f'  Test caption: "{test_cap_name}"')
        print(f'  ‖vtxt_feat(text)‖₂:  {vtxt_feat_text.float().norm().item():.6f}')
        print(f'  ‖vtxt_feat(null)‖₂:  {vtxt_feat_null.float().norm().item():.6f}')
        print(f'  cos(vtxt_text, vtxt_null): {cos_sim(vtxt_feat_text, vtxt_feat_null):.6f}')
        print()

        # Table header
        print(f'  {"t":>6s}  '
              f'{"‖ts_feat‖":>10s}  {"‖vtxt_text‖":>11s}  {"‖vtxt_null‖":>11s}  '
              f'{"‖adapt_text‖":>12s}  {"‖adapt_null‖":>12s}  '
              f'{"vtxt%_text":>10s}  {"vtxt%_null":>10s}  '
              f'{"cos(a_t,a_n)":>12s}  '
              f'{"‖a_t - a_n‖":>12s}')
        print(f'  {"-" * 6}  '
              f'{"-" * 10}  {"-" * 11}  {"-" * 11}  '
              f'{"-" * 12}  {"-" * 12}  '
              f'{"-" * 10}  {"-" * 10}  '
              f'{"-" * 12}  '
              f'{"-" * 12}')

        for t_val_f in TIMESTEPS:
            t_val = torch.tensor([t_val_f], device=device, dtype=model_dtype)

            # Compute timestep embedding
            ts_feat = timestep_encoder(t_val)  # [1, 1, 1024]

            # Adapter with text
            adapter_text = ts_feat + vtxt_feat_text
            # Adapter with null
            adapter_null = ts_feat + vtxt_feat_null

            ts_norm = ts_feat.float().norm().item()
            vtxt_text_norm = vtxt_feat_text.float().norm().item()
            vtxt_null_norm = vtxt_feat_null.float().norm().item()
            adapt_text_norm = adapter_text.float().norm().item()
            adapt_null_norm = adapter_null.float().norm().item()

            # vtxt contribution as percentage of adapter norm
            vtxt_pct_text = (vtxt_text_norm / adapt_text_norm * 100) if adapt_text_norm > 0 else 0
            vtxt_pct_null = (vtxt_null_norm / adapt_null_norm * 100) if adapt_null_norm > 0 else 0

            # Cosine similarity between text-conditioned and null-conditioned adapters
            adapter_cos = cos_sim(adapter_text, adapter_null)

            # L2 distance between adapters
            adapter_l2 = l2_dist(adapter_text, adapter_null)

            print(f'  {t_val_f:6.3f}  '
                  f'{ts_norm:10.4f}  {vtxt_text_norm:11.4f}  {vtxt_null_norm:11.4f}  '
                  f'{adapt_text_norm:12.4f}  {adapt_null_norm:12.4f}  '
                  f'{vtxt_pct_text:9.2f}%  {vtxt_pct_null:9.2f}%  '
                  f'{adapter_cos:12.6f}  '
                  f'{adapter_l2:12.4f}')

    # ── Step 7: vtxt_encoder weight analysis ────────────────────────
    print_header('STEP 7: vtxt_encoder Weight Analysis')
    print(f'  Architecture: MLPEncoder(768 → 1024, SiLU, 1024 → 1024)')
    print()

    for name, param in vtxt_encoder.named_parameters():
        p = param.detach().float()
        print(f'  {name:<30s}  shape={str(list(p.shape)):<16s}  '
              f'‖w‖={p.norm().item():10.4f}  '
              f'mean={p.mean().item():+10.6f}  '
              f'std={p.std().item():10.6f}  '
              f'max_abs={p.abs().max().item():10.6f}')

    # Compute effective amplification: how much does each layer scale inputs?
    print(f'\n  Effective gain analysis (singular value spectrum):')
    for name, param in vtxt_encoder.named_parameters():
        if 'weight' in name and param.dim() == 2:
            p = param.detach().float()
            svs = torch.linalg.svdvals(p)
            print(f'    {name:<30s}  σ_max={svs[0].item():.4f}  '
                  f'σ_min={svs[-1].item():.6f}  '
                  f'σ_mean={svs.mean().item():.4f}  '
                  f'cond={svs[0].item() / max(svs[-1].item(), 1e-8):.2f}')

    # ── Step 8: Directional analysis ────────────────────────────────
    print_header('STEP 8: Directional Analysis — Does vtxt_encoder Preserve or Destroy Direction?')

    with torch.no_grad():
        # For each pair of captions, measure how the angle between them
        # changes after encoding
        print(f'\n  For each caption pair (i, j):')
        print(f'    raw_angle    = arccos(cos(raw_i, raw_j))')
        print(f'    enc_angle    = arccos(cos(enc_i, enc_j))')
        print(f'    ratio        = enc_angle / raw_angle  (< 1 means directions collapsed)')
        print()

        raw_angles = []
        enc_angles = []
        for i in range(min(n, 8)):
            for j in range(i + 1, min(n, 8)):
                rc = cos_sim(raw_vecs[i], raw_vecs[j])
                ec = cos_sim(enc_vecs[i], enc_vecs[j])
                # Clamp to valid arccos range
                rc = max(-1.0, min(1.0, rc))
                ec = max(-1.0, min(1.0, ec))
                ra = np.degrees(np.arccos(rc))
                ea = np.degrees(np.arccos(ec))
                raw_angles.append(ra)
                enc_angles.append(ea)
                ratio = ea / ra if ra > 0.01 else float('inf')
                print(f'    [{i:2d}] vs [{j:2d}]:  raw_angle={ra:6.2f}°  '
                      f'enc_angle={ea:6.2f}°  ratio={ratio:.4f}  '
                      f'{"← collapsed" if ratio < 0.5 else "← preserved" if ratio > 0.8 else "← partial"}')

        if raw_angles:
            mean_raw = np.mean(raw_angles)
            mean_enc = np.mean(enc_angles)
            print(f'\n    Average: raw_angle={mean_raw:.2f}°  enc_angle={mean_enc:.2f}°  '
                  f'ratio={mean_enc / mean_raw if mean_raw > 0.01 else float("inf"):.4f}')
            if mean_enc / max(mean_raw, 0.01) < 0.5:
                print(f'    ⚠ SEVERE COLLAPSE: encoded angles < 50% of raw angles')
            elif mean_enc / max(mean_raw, 0.01) < 0.8:
                print(f'    ⚠ MODERATE COLLAPSE: encoded angles 50-80% of raw angles')
            else:
                print(f'    ✓ Directions largely preserved after encoding')

    # ── Step 9: Bias dominance check ────────────────────────────────
    print_header('STEP 9: Bias Dominance Check')
    print(f'  If the MLP biases dominate, all inputs map to ~same output.')
    print()

    with torch.no_grad():
        # Pass zero vector through encoder
        zero_input = torch.zeros(1, 1, 768, device=device, dtype=model_dtype)
        zero_encoded = vtxt_encoder(zero_input.float()).detach().cpu().float()
        zero_enc_flat = zero_encoded.flatten()

        print(f'  vtxt_encoder(zeros):')
        print(f'    ‖output‖₂ = {zero_enc_flat.norm().item():.6f}')
        print(f'    This is the pure bias path: Linear₁(0)+b₁ → SiLU → Linear₂(·)+b₂')
        print()

        # Compare: how much of each encoded output is "bias" vs "signal"?
        print(f'  Decomposition: encoded(x) = encoded(0) + [encoded(x) - encoded(0)]')
        print(f'  {"#":>3s}  {"‖encoded(x)‖":>14s}  {"‖bias_part‖":>12s}  '
              f'{"‖signal_part‖":>14s}  {"bias%":>7s}  '
              f'{"cos(enc,bias)":>13s}')
        print(f'  {"-" * 3}  {"-" * 14}  {"-" * 12}  '
              f'{"-" * 14}  {"-" * 7}  '
              f'{"-" * 13}')

        for i, v in enumerate(enc_vecs):
            enc_norm = v.norm().item()
            bias_norm = zero_enc_flat.norm().item()
            signal = v - zero_enc_flat
            signal_norm = signal.norm().item()
            bias_pct = bias_norm / enc_norm * 100 if enc_norm > 0 else 0
            cos_enc_bias = cos_sim(v, zero_enc_flat)
            print(f'  {i:3d}  {enc_norm:14.4f}  {bias_norm:12.4f}  '
                  f'{signal_norm:14.4f}  {bias_pct:6.1f}%  '
                  f'{cos_enc_bias:13.6f}')

        # Null through same analysis
        null_signal = null_enc_flat - zero_enc_flat
        print(f'\n  null:')
        print(f'    ‖encoded(null_vtxt_feat)‖ = {null_enc_flat.norm().item():.4f}')
        print(f'    ‖bias_part‖ = {zero_enc_flat.norm().item():.4f}')
        print(f'    ‖signal_part‖ = {null_signal.norm().item():.4f}')
        print(f'    cos(enc_null, bias) = {cos_sim(null_enc_flat, zero_enc_flat):.6f}')

    # ── Summary ─────────────────────────────────────────────────────
    print_header('SUMMARY')
    print(f'''
  Input space (768-dim CLIP-L):
    ‖null_vtxt_feat‖          = {null_raw_flat.norm().item():.6f}
    ‖clip_embedding‖ (mean)   = {np.mean(all_norms):.4f}
    cos(clip, null) (mean)    = {np.mean(raw_sims_to_null):.6f}
    inter-caption cos (mean)  = {np.mean(off_diag):.6f}

  Output space (1024-dim, after vtxt_encoder):
    ‖encoded_null‖            = {null_enc_flat.norm().item():.6f}
    ‖encoded_clip‖ (mean)     = {np.mean(enc_norms):.4f}
    cos(enc_clip, enc_null)   = {np.mean(enc_sims_to_null):.6f}
    inter-caption cos (mean)  = {np.mean(enc_off_diag):.6f}

  Adapter analysis (adapter = timestep_feat + vtxt_feat):
    See Step 6 table above for per-timestep breakdown.
    If cos(adapter_text, adapter_null) ≈ 1.0 for all timesteps,
    then the model cannot distinguish text from null via the adapter
    signal, and text conditioning has effectively collapsed.

  Diagnosis:''')

    # Auto-diagnosis
    issues = []
    if np.mean(enc_sims_to_null) > 0.95:
        issues.append('CRITICAL: encoded captions nearly identical to encoded null (cos > 0.95)')
    if np.mean(enc_off_diag) > 0.98:
        issues.append('CRITICAL: encoded captions nearly identical to each other (cos > 0.98)')
    if mean_enc / max(mean_raw, 0.01) < 0.3:
        issues.append('SEVERE: angular diversity destroyed by encoder (ratio < 0.3)')
    if null_raw_flat.norm().item() / np.mean(all_norms) < 0.01:
        issues.append(f'NOTE: null_vtxt_feat norm ({null_raw_flat.norm().item():.4f}) is '
                      f'{null_raw_flat.norm().item() / np.mean(all_norms) * 100:.2f}% '
                      f'of avg CLIP norm — very small, may be swamped by MLP bias')

    if issues:
        for iss in issues:
            print(f'    ⚠ {iss}')
    else:
        print(f'    ✓ No obvious collapse detected.')

    print()

    # Cleanup
    del bundle
    torch.cuda.empty_cache()
    print('Done.')


if __name__ == '__main__':
    main()

"""Deep probe into double_block attention weights and gate values.

Hooks into the scaled_dot_product_attention to measure:
1. Motion→Text attention weight magnitudes (are they near-zero?)
2. Gate values from ModulateDiT (are text gates suppressed?)
3. Text feature norm evolution through blocks

Also compares E2 vs parent model to see where collapse happened.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def load_bundle(config_path, checkpoint_path, device='cuda'):
    from mmengine.config import Config
    import hftrainer  # noqa
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
        print(f'  Warning: No checkpoint at {checkpoint_path}')
    bundle = bundle.to(device)
    return bundle


def load_text_embeddings(cache_path, caption, device='cuda'):
    cache_raw = torch.load(cache_path, map_location='cpu', weights_only=False)
    cache = cache_raw.get('cache', cache_raw)
    if caption in cache:
        entry = cache[caption]
    else:
        first_key = next(iter(cache))
        entry = cache[first_key]
        caption = first_key
        print(f'  Using: "{caption[:80]}..."')
    vtxt = entry['text_vec_raw'].float().to(device)
    ctxt = entry['text_ctxt_raw'].float().to(device)
    ctxt_len = entry['text_ctxt_raw_length']
    if vtxt.dim() == 2: vtxt = vtxt.unsqueeze(0)
    if ctxt.dim() == 2: ctxt = ctxt.unsqueeze(0)
    return vtxt, ctxt, ctxt_len, caption


class DeepDoubleBlockProbe:
    """Hooks into double_block internals to measure attention and gate values."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.transformer = bundle.motion_transformer
        self.hooks = []
        self.data = defaultdict(dict)
        self.tag = 'default'

    def _hook_double_block_forward(self, block_idx):
        """Monkey-patch a double_block's forward to capture internal values."""
        block = self.transformer.double_blocks[block_idx]
        original_forward = block.forward
        probe = self

        def patched_forward(*args, **kwargs):
            # Capture modulation outputs
            adapter = kwargs.get('adapter') or args[2]

            # Get motion/text modulation params
            motion_mod_out = block.motion_mod(adapter)
            text_mod_out = block.text_mod(adapter)

            # Split into shift/scale/gate (factor=6 for double blocks)
            feat_dim = motion_mod_out.shape[-1] // 6
            motion_params = motion_mod_out.reshape(-1, 6, feat_dim)
            text_params = text_mod_out.reshape(-1, 6, feat_dim)

            # Store gate values (indices 2=gate_msa, 5=gate_mlp)
            probe.data[probe.tag][f'db{block_idx}_motion_gate_msa'] = motion_params[:, 2, :].detach().float().cpu()
            probe.data[probe.tag][f'db{block_idx}_motion_gate_mlp'] = motion_params[:, 5, :].detach().float().cpu()
            probe.data[probe.tag][f'db{block_idx}_text_gate_msa'] = text_params[:, 2, :].detach().float().cpu()
            probe.data[probe.tag][f'db{block_idx}_text_gate_mlp'] = text_params[:, 5, :].detach().float().cpu()

            # Store shift/scale for analysis
            probe.data[probe.tag][f'db{block_idx}_motion_shift_msa'] = motion_params[:, 0, :].detach().float().cpu()
            probe.data[probe.tag][f'db{block_idx}_motion_scale_msa'] = motion_params[:, 1, :].detach().float().cpu()
            probe.data[probe.tag][f'db{block_idx}_text_shift_msa'] = text_params[:, 0, :].detach().float().cpu()
            probe.data[probe.tag][f'db{block_idx}_text_scale_msa'] = text_params[:, 1, :].detach().float().cpu()

            # Call original forward
            result = original_forward(*args, **kwargs)
            return result

        block.forward = patched_forward
        return original_forward  # So we can restore later

    def _hook_attention_weights(self, block_idx):
        """Hook into the attention computation to capture weights."""
        block = self.transformer.double_blocks[block_idx]
        probe = self

        # We need to hook into the actual attention computation
        # The double_block calls F.scaled_dot_product_attention or manual attention
        # Let's hook the entire block and compute attention weights separately

        def attn_hook(module, input, output):
            # We'll compute attention weights in a separate pass (see below)
            pass

    def setup(self):
        """Setup all hooks."""
        self.originals = {}
        for i in range(len(self.transformer.double_blocks)):
            orig = self._hook_double_block_forward(i)
            self.originals[f'db{i}'] = (self.transformer.double_blocks[i], orig)

    def restore(self):
        """Restore original forward methods."""
        for key, (block, orig_forward) in self.originals.items():
            block.forward = orig_forward

    def clear(self):
        self.data.clear()


def compute_attention_weights_manually(bundle, x_input, vtxt, ctxt, ctxt_mask,
                                       tgt_mask, t_val, L, device):
    """Manually compute Q, K for each double_block and measure attention scores."""
    model_dtype = next(bundle.motion_transformer.parameters()).dtype
    transformer = bundle.motion_transformer

    # Encode inputs
    motion_feat = transformer.input_encoder(x_input.to(dtype=model_dtype))

    # Build adapter
    timestep_feat = transformer.timestep_encoder(t_val.expand(1).to(dtype=model_dtype))
    vtxt_feat = transformer.vtxt_encoder(vtxt.to(dtype=model_dtype).float())
    adapter = timestep_feat + vtxt_feat

    # Encode context text
    ctxt_feat = transformer.ctxt_encoder(ctxt.to(dtype=model_dtype).float())

    # Text refiner
    if hasattr(transformer, 'text_refiner'):
        ctxt_key_padding = transformer._canonical_mask(ctxt_mask)
        refiner_mask = (ctxt_key_padding == 0).to(device)
        ctxt_feat = transformer.text_refiner(x=ctxt_feat, t=t_val.expand(1), mask=refiner_mask)

    text_len = ctxt_feat.shape[1]
    motion_len = motion_feat.shape[1]

    results = {}

    for i, block in enumerate(transformer.double_blocks):
        # Modulation
        motion_mod_out = block.motion_mod(adapter)
        text_mod_out = block.text_mod(adapter)

        feat_dim = motion_mod_out.shape[-1] // 6
        m_shift_msa, m_scale_msa, m_gate_msa, m_shift_mlp, m_scale_mlp, m_gate_mlp = \
            motion_mod_out.reshape(-1, 6, feat_dim).unbind(1)
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = \
            text_mod_out.reshape(-1, 6, feat_dim).unbind(1)

        # Apply modulation to get Q, K
        motion_modulated = motion_feat * (1 + m_scale_msa.unsqueeze(1)) + m_shift_msa.unsqueeze(1)
        text_modulated = ctxt_feat * (1 + t_scale_msa.unsqueeze(1)) + t_shift_msa.unsqueeze(1)

        # Project to QKV
        motion_qkv = block.motion_qkv(motion_modulated)
        text_qkv = block.text_qkv(text_modulated)

        num_heads = block.num_heads
        head_dim = feat_dim // num_heads

        # Reshape: (B, L, 3*feat) -> (3, B, L, H, D)
        from einops import rearrange
        mq, mk, mv = rearrange(motion_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads).unbind(0)
        tq, tk, tv = rearrange(text_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads).unbind(0)

        # Normalize Q/K
        mq = block.motion_q_norm(mq)
        mk = block.motion_k_norm(mk)
        tq = block.text_q_norm(tq)
        tk = block.text_k_norm(tk)

        # Joint Q/K
        q = torch.cat([mq, tq], dim=2)  # (B, H, L_m+L_t, D)
        k = torch.cat([mk, tk], dim=2)

        # Compute raw attention scores
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L_total, L_total)

        # Softmax (ignoring mask for raw analysis)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, L_total, L_total)

        # Extract motion→text attention: rows=motion, cols=text
        # motion rows: 0:motion_len, text cols: motion_len:motion_len+text_len
        m2t_attn = attn_weights[:, :, :motion_len, motion_len:motion_len+text_len]  # (B,H,Lm,Lt)
        m2m_attn = attn_weights[:, :, :motion_len, :motion_len]  # (B,H,Lm,Lm)
        t2t_attn = attn_weights[:, :, motion_len:, motion_len:]  # (B,H,Lt,Lt)

        # Motion→Text attention weight sum (how much attention goes to text)
        m2t_weight_sum = m2t_attn.sum(dim=-1).mean().item()  # avg over heads/positions
        m2m_weight_sum = m2m_attn.sum(dim=-1).mean().item()

        # Store
        results[f'db{i}'] = {
            'motion_to_text_attn_mean': m2t_weight_sum,
            'motion_to_motion_attn_mean': m2m_weight_sum,
            'motion_to_text_attn_max': m2t_attn.max().item(),
            'm2t_attn_per_head': m2t_attn.sum(dim=-1).mean(dim=2)[0].tolist(),  # per-head avg
            'motion_gate_msa_mean': m_gate_msa.abs().mean().item(),
            'motion_gate_msa_std': m_gate_msa.std().item(),
            'text_gate_msa_mean': t_gate_msa.abs().mean().item(),
            'text_gate_msa_std': t_gate_msa.std().item(),
            'motion_gate_mlp_mean': m_gate_mlp.abs().mean().item(),
            'text_gate_mlp_mean': t_gate_mlp.abs().mean().item(),
            'motion_feat_norm': motion_feat.float().norm().item(),
            'text_feat_norm': ctxt_feat.float().norm().item(),
        }

        # Run actual forward to update features for next block
        motion_feat, ctxt_feat = block(
            motion_feat=motion_feat,
            text_feat=ctxt_feat,
            adapter=adapter,
            attn_mask=None,  # simplified, no mask for this diagnostic
        )

    return results


def analyze_model(label, bundle, vtxt, ctxt, ctxt_len, ctxt_mask, null_vtxt, null_ctxt, null_ctxt_mask,
                  L, D, device):
    """Run analysis for one model."""
    model_dtype = next(bundle.motion_transformer.parameters()).dtype
    B = 1

    # Setup inputs
    src_mask = torch.ones(B, L, D, device=device, dtype=model_dtype)
    src_motion = torch.zeros(B, L, D, device=device, dtype=model_dtype)
    vace_context = bundle.prepare_vace_input(src_motion=src_motion, ref_pose=None, src_mask=src_mask)
    tgt_padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    z = torch.randn(B, L, D, device=device, dtype=model_dtype)
    x_input = torch.cat([z, vace_context], dim=-1)

    print(f'\n{"="*80}')
    print(f'  [{label}] ATTENTION WEIGHT & GATE ANALYSIS')
    print(f'{"="*80}')

    for t_val_f in [0.0, 0.5, 0.98]:
        t_val = torch.tensor(t_val_f, device=device, dtype=model_dtype)

        print(f'\n  --- t={t_val_f:.2f} ---')

        with torch.no_grad():
            # Text conditioned
            res_text = compute_attention_weights_manually(
                bundle, x_input, vtxt, ctxt, ctxt_mask, tgt_padding_mask, t_val, L, device
            )
            # Null conditioned
            res_null = compute_attention_weights_manually(
                bundle, x_input, null_vtxt, null_ctxt, null_ctxt_mask, tgt_padding_mask, t_val, L, device
            )

        print(f'\n  {"Block":>6s} | {"M→T attn(text)":>14s} {"M→T attn(null)":>14s} | '
              f'{"M gate_msa":>10s} {"T gate_msa":>10s} {"M gate_mlp":>10s} {"T gate_mlp":>10s} | '
              f'{"Motion norm":>11s} {"Text norm":>11s}')
        print(f'  {"-"*6}-+-{"-"*14}-{"-"*14}-+-{"-"*10}-{"-"*10}-{"-"*10}-{"-"*10}-+-{"-"*11}-{"-"*11}')

        for i in range(len(bundle.motion_transformer.double_blocks)):
            rt = res_text[f'db{i}']
            rn = res_null[f'db{i}']

            print(f'  db{i:3d} | {rt["motion_to_text_attn_mean"]:14.6f} {rn["motion_to_text_attn_mean"]:14.6f} | '
                  f'{rt["motion_gate_msa_mean"]:10.6f} {rt["text_gate_msa_mean"]:10.6f} '
                  f'{rt["motion_gate_mlp_mean"]:10.6f} {rt["text_gate_mlp_mean"]:10.6f} | '
                  f'{rt["motion_feat_norm"]:11.2f} {rt["text_feat_norm"]:11.2f}')

        # Per-head attention analysis for last block
        last_block_idx = len(bundle.motion_transformer.double_blocks) - 1
        print(f'\n  Last double_block (db{last_block_idx}) per-head M→T attention:')
        heads_text = res_text[f'db{last_block_idx}']['m2t_attn_per_head']
        heads_null = res_null[f'db{last_block_idx}']['m2t_attn_per_head']
        for h_idx in range(len(heads_text)):
            bar_t = '█' * int(heads_text[h_idx] * 100)
            bar_n = '█' * int(heads_null[h_idx] * 100)
            print(f'    Head {h_idx:2d}: text={heads_text[h_idx]:.4f} {bar_t} | null={heads_null[h_idx]:.4f} {bar_n}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    E2_CONFIG = 'configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py'
    E2_CKPT = 'work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_90'

    # Also load parent for comparison
    PARENT_CONFIG = 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py'
    PARENT_CKPT = 'work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370'

    CACHE_PATH = 'data/eval/m2m_v2/caption_embeddings/cache.pt'
    caption = 'A person adjusts their stance and performs a golf swing'
    L = 64

    # Load text
    print('Loading text embeddings...')
    vtxt, ctxt, ctxt_len, actual_caption = load_text_embeddings(CACHE_PATH, caption, device)
    print(f'  Caption: "{actual_caption[:80]}..."')

    if isinstance(ctxt_len, torch.Tensor):
        ctxt_length = ctxt_len.long().to(device)
    else:
        ctxt_length = torch.tensor([ctxt_len], dtype=torch.long, device=device)
    ctxt_seq_len = ctxt.shape[1]
    ctxt_mask = torch.arange(ctxt_seq_len, device=device).unsqueeze(0) < ctxt_length.unsqueeze(1)

    # Run E2
    print('\n--- Loading E2 model ---')
    bundle_e2 = load_bundle(E2_CONFIG, E2_CKPT, device)
    model_dtype = next(bundle_e2.motion_transformer.parameters()).dtype
    D = int(bundle_e2.mean.numel())

    null_vtxt = bundle_e2.null_vtxt_feat.to(dtype=model_dtype)
    if null_vtxt.dim() == 2: null_vtxt = null_vtxt.unsqueeze(0)
    null_vtxt_e2 = null_vtxt.expand_as(vtxt.to(dtype=model_dtype))
    null_ctxt = bundle_e2.null_ctxt_input.to(dtype=model_dtype)
    if null_ctxt.dim() == 2: null_ctxt = null_ctxt.unsqueeze(0)
    null_ctxt_e2 = null_ctxt.expand(ctxt.shape[0], ctxt.shape[1], -1).contiguous()
    null_ctxt_mask_e2 = torch.zeros_like(ctxt_mask)
    null_ctxt_mask_e2[:, 0] = True

    analyze_model('E2', bundle_e2, vtxt.to(dtype=model_dtype), ctxt.to(dtype=model_dtype),
                  ctxt_len, ctxt_mask, null_vtxt_e2, null_ctxt_e2, null_ctxt_mask_e2,
                  L, D, device)

    # Gate value comparison with parent
    print(f'\n\n{"="*80}')
    print(f'  GATE VALUE COMPARISON: E2 vs E2 init (text-related layers)')
    print(f'{"="*80}')

    # Extract gate-related weights from E2
    print(f'\n  E2 double_block gate weights:')
    print(f'  {"Layer":<40s} | {"Weight norm":>11s} {"Bias norm":>9s} {"Bias mean":>9s}')
    print(f'  {"-"*40}-+-{"-"*11}-{"-"*9}-{"-"*9}')

    for name, param in bundle_e2.motion_transformer.named_parameters():
        if 'mod' in name and ('linear' in name):
            w_norm = param.data.float().norm().item()
            if 'bias' in name:
                b_mean = param.data.float().mean().item()
                print(f'  {name:<40s} | {w_norm:11.6f} {b_mean:9.6f}')
            else:
                print(f'  {name:<40s} | {w_norm:11.6f}')

    del bundle_e2
    torch.cuda.empty_cache()

    # Run Parent
    print('\n\n--- Loading Parent model ---')
    bundle_parent = load_bundle(PARENT_CONFIG, PARENT_CKPT, device)
    model_dtype = next(bundle_parent.motion_transformer.parameters()).dtype

    null_vtxt = bundle_parent.null_vtxt_feat.to(dtype=model_dtype)
    if null_vtxt.dim() == 2: null_vtxt = null_vtxt.unsqueeze(0)
    null_vtxt_p = null_vtxt.expand_as(vtxt.to(dtype=model_dtype))
    null_ctxt = bundle_parent.null_ctxt_input.to(dtype=model_dtype)
    if null_ctxt.dim() == 2: null_ctxt = null_ctxt.unsqueeze(0)
    null_ctxt_p = null_ctxt.expand(ctxt.shape[0], ctxt.shape[1], -1).contiguous()
    null_ctxt_mask_p = torch.zeros_like(ctxt_mask)
    null_ctxt_mask_p[:, 0] = True

    analyze_model('Parent', bundle_parent, vtxt.to(dtype=model_dtype), ctxt.to(dtype=model_dtype),
                  ctxt_len, ctxt_mask, null_vtxt_p, null_ctxt_p, null_ctxt_mask_p,
                  L, D, device)

    # Parent gate weights
    print(f'\n  Parent double_block gate weights:')
    print(f'  {"Layer":<40s} | {"Weight norm":>11s} {"Bias norm":>9s} {"Bias mean":>9s}')
    print(f'  {"-"*40}-+-{"-"*11}-{"-"*9}-{"-"*9}')

    for name, param in bundle_parent.motion_transformer.named_parameters():
        if 'mod' in name and ('linear' in name):
            w_norm = param.data.float().norm().item()
            if 'bias' in name:
                b_mean = param.data.float().mean().item()
                print(f'  {name:<40s} | {w_norm:11.6f} {b_mean:9.6f}')
            else:
                print(f'  {name:<40s} | {w_norm:11.6f}')

    del bundle_parent
    torch.cuda.empty_cache()

    print('\nDone.')


if __name__ == '__main__':
    main()

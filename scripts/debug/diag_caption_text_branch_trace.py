"""Trace text branch activations through MMDiT to find where collapse happens.

Hooks into double_blocks and single_blocks to compare:
1. Text stream features (ctxt) between text-conditioned vs null-conditioned passes
2. Adapter (timestep + vtxt) contribution: how much does vtxt matter?
3. Modulation parameters (shift/scale/gate) differences
4. Attention output differences

Usage:
    python scripts/debug/diag_caption_text_branch_trace.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from collections import defaultdict


def load_bundle(config_path, checkpoint_path, device='cuda'):
    """Load a model bundle from config + checkpoint."""
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
    """Load pre-computed text embeddings."""
    cache_raw = torch.load(cache_path, map_location='cpu', weights_only=False)
    cache = cache_raw.get('cache', cache_raw)
    if caption in cache:
        entry = cache[caption]
    else:
        first_key = next(iter(cache))
        entry = cache[first_key]
        caption = first_key
        print(f'  Caption not found, using: "{caption[:80]}..."')

    vtxt = entry['text_vec_raw'].float().to(device)
    ctxt = entry['text_ctxt_raw'].float().to(device)
    ctxt_len = entry['text_ctxt_raw_length']
    if vtxt.dim() == 2:
        vtxt = vtxt.unsqueeze(0)
    if ctxt.dim() == 2:
        ctxt = ctxt.unsqueeze(0)
    return vtxt, ctxt, ctxt_len, caption


class ActivationTracer:
    """Hook-based tracer for MMDiT activations."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = defaultdict(dict)
        self.current_tag = 'default'

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # DoubleBlock returns (motion_feat, text_feat)
                self.activations[self.current_tag][name] = tuple(
                    o.detach().float().cpu() if isinstance(o, torch.Tensor) else o
                    for o in output
                )
            elif isinstance(output, torch.Tensor):
                self.activations[self.current_tag][name] = output.detach().float().cpu()
        return hook_fn

    def register_hooks(self):
        transformer = self.model.motion_transformer

        # Hook text_refiner
        if hasattr(transformer, 'text_refiner'):
            h = transformer.text_refiner.register_forward_hook(self._make_hook('text_refiner'))
            self.hooks.append(h)

        # Hook vtxt_encoder
        if hasattr(transformer, 'vtxt_encoder'):
            h = transformer.vtxt_encoder.register_forward_hook(self._make_hook('vtxt_encoder'))
            self.hooks.append(h)

        # Hook ctxt_encoder
        if hasattr(transformer, 'ctxt_encoder'):
            h = transformer.ctxt_encoder.register_forward_hook(self._make_hook('ctxt_encoder'))
            self.hooks.append(h)

        # Hook timestep_encoder
        if hasattr(transformer, 'timestep_encoder'):
            h = transformer.timestep_encoder.register_forward_hook(self._make_hook('timestep_encoder'))
            self.hooks.append(h)

        # Hook input_encoder
        if hasattr(transformer, 'input_encoder'):
            h = transformer.input_encoder.register_forward_hook(self._make_hook('input_encoder'))
            self.hooks.append(h)

        # Hook double_blocks
        for i, block in enumerate(transformer.double_blocks):
            h = block.register_forward_hook(self._make_hook(f'double_block_{i}'))
            self.hooks.append(h)
            # Hook individual sub-layers
            if hasattr(block, 'text_mod'):
                h = block.text_mod.register_forward_hook(self._make_hook(f'double_block_{i}_text_mod'))
                self.hooks.append(h)
            if hasattr(block, 'motion_mod'):
                h = block.motion_mod.register_forward_hook(self._make_hook(f'double_block_{i}_motion_mod'))
                self.hooks.append(h)

        # Hook single_blocks
        for i, block in enumerate(transformer.single_blocks):
            h = block.register_forward_hook(self._make_hook(f'single_block_{i}'))
            self.hooks.append(h)
            if hasattr(block, 'modulation'):
                h = block.modulation.register_forward_hook(self._make_hook(f'single_block_{i}_modulation'))
                self.hooks.append(h)

        # Hook final_layer
        if hasattr(transformer, 'final_layer'):
            h = transformer.final_layer.register_forward_hook(self._make_hook('final_layer'))
            self.hooks.append(h)

        print(f'  Registered {len(self.hooks)} hooks')

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear(self):
        self.activations.clear()


def run_traced_forward(bundle, tracer, x, vace_context, vtxt_input, ctxt_input,
                       ctxt_mask_temporal, tgt_padding_mask, t_val, tag, device):
    """Run a single forward pass with activation tracing."""
    model_dtype = next(bundle.motion_transformer.parameters()).dtype
    tracer.current_tag = tag

    x_input = torch.cat([x, vace_context], dim=-1)

    pred = bundle.predict_flow(
        x_input=x_input.to(dtype=model_dtype),
        ctxt_input=ctxt_input.to(dtype=model_dtype),
        vtxt_input=vtxt_input.to(dtype=model_dtype),
        timesteps=t_val.expand(x.shape[0]).to(dtype=model_dtype),
        x_mask_temporal=tgt_padding_mask,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )

    if bundle.pred_type == 'x1':
        t_eps = 0.05
        pred = (pred - x.to(dtype=model_dtype)) / (1.0 - t_val).clamp_min(t_eps)

    return pred.float().detach().cpu()


def compare_activations(tracer, tag_text, tag_null):
    """Compare activations between text-conditioned and null-conditioned passes."""
    act_text = tracer.activations.get(tag_text, {})
    act_null = tracer.activations.get(tag_null, {})

    common_keys = sorted(set(act_text.keys()) & set(act_null.keys()))

    print(f'\n  {"Layer":<35s} | {"Text norm":>10s} {"Null norm":>10s} | {"Diff norm":>10s} {"CosSim":>8s} {"Ratio":>8s}')
    print(f'  {"-"*35}-+-{"-"*10}-{"-"*10}-+-{"-"*10}-{"-"*8}-{"-"*8}')

    results = {}
    for key in common_keys:
        t_act = act_text[key]
        n_act = act_null[key]

        # Handle tuple outputs (double_blocks return (motion_feat, text_feat))
        if isinstance(t_act, tuple) and isinstance(n_act, tuple):
            for idx, suffix in enumerate(['_motion', '_text']):
                if idx < len(t_act) and isinstance(t_act[idx], torch.Tensor):
                    t_tensor = t_act[idx].flatten().float()
                    n_tensor = n_act[idx].flatten().float()
                    t_norm = t_tensor.norm().item()
                    n_norm = n_tensor.norm().item()
                    diff_norm = (t_tensor - n_tensor).norm().item()
                    if t_norm > 1e-8 and n_norm > 1e-8:
                        cos = torch.nn.functional.cosine_similarity(
                            t_tensor.unsqueeze(0), n_tensor.unsqueeze(0)
                        ).item()
                    else:
                        cos = 0.0
                    ratio = diff_norm / max(t_norm, n_norm, 1e-8)
                    full_key = key + suffix
                    results[full_key] = {
                        'text_norm': t_norm, 'null_norm': n_norm,
                        'diff_norm': diff_norm, 'cos_sim': cos, 'ratio': ratio
                    }
                    print(f'  {full_key:<35s} | {t_norm:10.4f} {n_norm:10.4f} | {diff_norm:10.4f} {cos:8.5f} {ratio:8.5f}')
        elif isinstance(t_act, torch.Tensor) and isinstance(n_act, torch.Tensor):
            t_tensor = t_act.flatten().float()
            n_tensor = n_act.flatten().float()
            t_norm = t_tensor.norm().item()
            n_norm = n_tensor.norm().item()
            diff_norm = (t_tensor - n_tensor).norm().item()
            if t_norm > 1e-8 and n_norm > 1e-8:
                cos = torch.nn.functional.cosine_similarity(
                    t_tensor.unsqueeze(0), n_tensor.unsqueeze(0)
                ).item()
            else:
                cos = 0.0
            ratio = diff_norm / max(t_norm, n_norm, 1e-8)
            results[key] = {
                'text_norm': t_norm, 'null_norm': n_norm,
                'diff_norm': diff_norm, 'cos_sim': cos, 'ratio': ratio
            }
            print(f'  {key:<35s} | {t_norm:10.4f} {n_norm:10.4f} | {diff_norm:10.4f} {cos:8.5f} {ratio:8.5f}')

    return results


def analyze_adapter_contribution(tracer, tag):
    """Analyze how much vtxt contributes to the adapter vs timestep."""
    act = tracer.activations.get(tag, {})
    timestep_feat = act.get('timestep_encoder')
    vtxt_feat = act.get('vtxt_encoder')

    if timestep_feat is not None and vtxt_feat is not None:
        t_norm = timestep_feat.float().norm().item()
        v_norm = vtxt_feat.float().norm().item()
        adapter = timestep_feat.float() + vtxt_feat.float()
        a_norm = adapter.norm().item()

        print(f'\n  === ADAPTER ANALYSIS ({tag}) ===')
        print(f'  timestep_feat norm: {t_norm:.4f}')
        print(f'  vtxt_feat norm:     {v_norm:.4f}')
        print(f'  adapter norm:       {a_norm:.4f}')
        print(f'  vtxt/timestep ratio: {v_norm / max(t_norm, 1e-8):.4f}')
        print(f'  vtxt contribution:  {v_norm / max(a_norm, 1e-8) * 100:.1f}%')

        # Cosine similarity between adapter and timestep-only
        cos = torch.nn.functional.cosine_similarity(
            adapter.flatten().unsqueeze(0),
            timestep_feat.flatten().unsqueeze(0)
        ).item()
        print(f'  adapter vs timestep-only cos: {cos:.6f}')


def analyze_modulation_params(tracer, tag_text, tag_null):
    """Compare modulation parameters (shift/scale/gate) between text and null."""
    act_text = tracer.activations.get(tag_text, {})
    act_null = tracer.activations.get(tag_null, {})

    print(f'\n  === MODULATION PARAMETER COMPARISON ===')
    print(f'  {"Layer":<40s} | {"Text mod norm":>12s} {"Null mod norm":>12s} | {"Diff":>10s} {"CosSim":>8s}')
    print(f'  {"-"*40}-+-{"-"*12}-{"-"*12}-+-{"-"*10}-{"-"*8}')

    for key in sorted(act_text.keys()):
        if '_mod' in key or '_modulation' in key:
            if key in act_null:
                t_mod = act_text[key]
                n_mod = act_null[key]
                if isinstance(t_mod, torch.Tensor) and isinstance(n_mod, torch.Tensor):
                    t_flat = t_mod.flatten().float()
                    n_flat = n_mod.flatten().float()
                    t_norm = t_flat.norm().item()
                    n_norm = n_flat.norm().item()
                    diff = (t_flat - n_flat).norm().item()
                    cos = torch.nn.functional.cosine_similarity(
                        t_flat.unsqueeze(0), n_flat.unsqueeze(0)
                    ).item() if t_norm > 1e-8 and n_norm > 1e-8 else 0.0
                    print(f'  {key:<40s} | {t_norm:12.4f} {n_norm:12.4f} | {diff:10.4f} {cos:8.5f}')


def analyze_text_refiner(tracer, tag_text, tag_null):
    """Compare text_refiner outputs between text and null."""
    act_text = tracer.activations.get(tag_text, {})
    act_null = tracer.activations.get(tag_null, {})

    print(f'\n  === TEXT PROCESSING COMPARISON ===')

    for key in ['ctxt_encoder', 'text_refiner']:
        if key in act_text and key in act_null:
            t_feat = act_text[key].float()
            n_feat = act_null[key].float()

            print(f'\n  [{key}]')
            print(f'    Text shape: {t_feat.shape}, norm: {t_feat.norm():.4f}')
            print(f'    Null shape: {n_feat.shape}, norm: {n_feat.norm():.4f}')

            # Per-token analysis
            if t_feat.dim() == 3:
                for tok_idx in range(min(5, t_feat.shape[1])):
                    t_tok = t_feat[0, tok_idx]
                    n_tok = n_feat[0, tok_idx]
                    cos = torch.nn.functional.cosine_similarity(
                        t_tok.unsqueeze(0), n_tok.unsqueeze(0)
                    ).item() if t_tok.norm() > 1e-8 and n_tok.norm() > 1e-8 else 0.0
                    print(f'    Token {tok_idx}: text_norm={t_tok.norm():.4f}, null_norm={n_tok.norm():.4f}, cos={cos:.5f}')

    for key in ['vtxt_encoder']:
        if key in act_text and key in act_null:
            t_feat = act_text[key].float()
            n_feat = act_null[key].float()
            cos = torch.nn.functional.cosine_similarity(
                t_feat.flatten().unsqueeze(0), n_feat.flatten().unsqueeze(0)
            ).item()
            print(f'\n  [{key}]')
            print(f'    Text shape: {t_feat.shape}, norm: {t_feat.norm():.4f}')
            print(f'    Null shape: {n_feat.shape}, norm: {n_feat.norm():.4f}')
            print(f'    CosSim: {cos:.6f}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    E2_CONFIG = 'configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py'
    E2_CKPT = 'work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_90'
    CACHE_PATH = 'data/eval/m2m_v2/caption_embeddings/cache.pt'
    caption = 'A person adjusts their stance and performs a golf swing'

    L = 64  # frames

    # Load text embeddings
    print('Loading text embeddings...')
    vtxt, ctxt, ctxt_len, actual_caption = load_text_embeddings(CACHE_PATH, caption, device)
    print(f'  Caption: "{actual_caption[:80]}..."')
    print(f'  vtxt: {vtxt.shape}, ctxt: {ctxt.shape}, ctxt_len: {ctxt_len}')

    # Load E2 model
    print('\nLoading E2 model...')
    bundle = load_bundle(E2_CONFIG, E2_CKPT, device)
    model_dtype = next(bundle.motion_transformer.parameters()).dtype
    D = int(bundle.mean.numel())

    print(f'\n  D={D}, pred_type={bundle.pred_type}')
    print(f'  uncondition_mode={bundle.uncondition_mode}')

    # Setup inputs
    B = 1
    src_mask = torch.ones(B, L, D, device=device, dtype=model_dtype)
    src_motion = torch.zeros(B, L, D, device=device, dtype=model_dtype)
    vace_context = bundle.prepare_vace_input(src_motion=src_motion, ref_pose=None, src_mask=src_mask)

    tgt_padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Text inputs
    vtxt_input = vtxt.to(dtype=model_dtype)
    ctxt_input = ctxt.to(dtype=model_dtype)
    if isinstance(ctxt_len, torch.Tensor):
        ctxt_length = ctxt_len.long().to(device)
    else:
        ctxt_length = torch.tensor([ctxt_len], dtype=torch.long, device=device)
    ctxt_seq_len = ctxt_input.shape[1]
    ctxt_mask_temporal = torch.arange(ctxt_seq_len, device=device).unsqueeze(0) < ctxt_length.unsqueeze(1)

    # Null inputs
    null_vtxt = bundle.null_vtxt_feat.to(dtype=model_dtype)
    if null_vtxt.dim() == 2:
        null_vtxt = null_vtxt.unsqueeze(0)
    null_vtxt = null_vtxt.expand_as(vtxt_input)

    null_ctxt = bundle.null_ctxt_input.to(dtype=model_dtype)
    if null_ctxt.dim() == 2:
        null_ctxt = null_ctxt.unsqueeze(0)
    null_ctxt = null_ctxt.expand(ctxt_input.shape[0], ctxt_input.shape[1], -1).contiguous()

    null_ctxt_mask = torch.zeros_like(ctxt_mask_temporal)
    null_ctxt_mask[:, 0] = True

    # Test at multiple timesteps
    test_timesteps = [0.0, 0.25, 0.5, 0.75, 0.98]

    # Use same noise for all comparisons
    z = torch.randn(B, L, D, device=device, dtype=model_dtype)

    # Setup tracer
    tracer = ActivationTracer(bundle)
    tracer.register_hooks()

    for t_val_f in test_timesteps:
        t_val = torch.tensor(t_val_f, device=device, dtype=model_dtype)

        print(f'\n{"="*80}')
        print(f'  TIMESTEP t={t_val_f:.2f}')
        print(f'{"="*80}')

        tracer.clear()

        with torch.no_grad():
            # Text-conditioned forward
            pred_text = run_traced_forward(
                bundle, tracer, z, vace_context,
                vtxt_input, ctxt_input, ctxt_mask_temporal,
                tgt_padding_mask, t_val, 'text', device
            )

            # Null-conditioned forward
            pred_null = run_traced_forward(
                bundle, tracer, z, vace_context,
                null_vtxt, null_ctxt, null_ctxt_mask,
                tgt_padding_mask, t_val, 'null', device
            )

        # Velocity comparison
        v_text = pred_text.float().norm().item() / (L * D) ** 0.5
        v_null = pred_null.float().norm().item() / (L * D) ** 0.5
        v_diff = (pred_text - pred_null).float().norm().item() / (L * D) ** 0.5
        print(f'\n  VELOCITY: text={v_text:.6f}, null={v_null:.6f}, diff={v_diff:.6f}')
        print(f'  TEXT/NULL ratio: {v_text / max(v_null, 1e-8):.4f}')

        # Translation velocity comparison
        v_text_trans = pred_text[..., :3].float().norm().item() / (L * 3) ** 0.5
        v_null_trans = pred_null[..., :3].float().norm().item() / (L * 3) ** 0.5
        print(f'  TRANSL velocity: text={v_text_trans:.6f}, null={v_null_trans:.6f}')

        # 1. Text processing analysis
        analyze_text_refiner(tracer, 'text', 'null')

        # 2. Adapter contribution
        analyze_adapter_contribution(tracer, 'text')
        analyze_adapter_contribution(tracer, 'null')

        # 3. Modulation comparison
        analyze_modulation_params(tracer, 'text', 'null')

        # 4. Full activation comparison
        print(f'\n  === LAYER-BY-LAYER ACTIVATION COMPARISON ===')
        results = compare_activations(tracer, 'text', 'null')

        # 5. Track how motion features diverge through blocks
        print(f'\n  === MOTION FEATURE DIVERGENCE THROUGH BLOCKS ===')
        motion_cos_values = []
        for key in sorted(results.keys()):
            if 'double_block' in key and key.endswith('_motion'):
                r = results[key]
                block_idx = key.split('_')[2]
                motion_cos_values.append((f'DB{block_idx}', r['cos_sim'], r['ratio']))
            elif 'single_block' in key and '_modulation' not in key:
                r = results[key]
                block_idx = key.split('_')[2]
                motion_cos_values.append((f'SB{block_idx}', r['cos_sim'], r['ratio']))

        if motion_cos_values:
            print(f'  {"Block":<8s} {"CosSim":>10s} {"DiffRatio":>10s}')
            for name, cos, ratio in motion_cos_values:
                bar_len = int((1.0 - cos) * 100)
                bar = '█' * min(bar_len, 40)
                print(f'  {name:<8s} {cos:10.6f} {ratio:10.6f}  {bar}')

    # Cleanup
    tracer.remove_hooks()

    # =========================================================================
    # PART 2: Compare two different captions to see if text features differ
    # =========================================================================
    print(f'\n\n{"#"*80}')
    print(f'# PART 2: Caption Sensitivity in Intermediate Features')
    print(f'{"#"*80}')

    # Load a second caption
    cache_raw = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    cache = cache_raw.get('cache', cache_raw)
    captions = list(cache.keys())
    if len(captions) >= 2:
        cap1 = actual_caption
        # Pick a very different caption
        cap2 = None
        for c in captions:
            if c != cap1:
                cap2 = c
                break
        if cap2 is None:
            cap2 = captions[1] if captions[1] != cap1 else captions[0]

        entry1 = cache[cap1]
        entry2 = cache[cap2]

        vtxt1 = entry1['text_vec_raw'].float().to(device)
        ctxt1 = entry1['text_ctxt_raw'].float().to(device)
        ctxt1_len = entry1['text_ctxt_raw_length']
        vtxt2 = entry2['text_vec_raw'].float().to(device)
        ctxt2 = entry2['text_ctxt_raw'].float().to(device)
        ctxt2_len = entry2['text_ctxt_raw_length']

        for v in [vtxt1, vtxt2]:
            if v.dim() == 2:
                v.unsqueeze_(0)
        for c in [ctxt1, ctxt2]:
            if c.dim() == 2:
                c.unsqueeze_(0)

        # Input embedding similarity
        vtxt_cos = torch.nn.functional.cosine_similarity(
            vtxt1.flatten().unsqueeze(0), vtxt2.flatten().unsqueeze(0)
        ).item()
        print(f'\n  Caption 1: "{cap1[:60]}..."')
        print(f'  Caption 2: "{cap2[:60]}..."')
        print(f'  vtxt cosine similarity: {vtxt_cos:.4f}')

        # Now trace both through the model
        tracer2 = ActivationTracer(bundle)
        tracer2.register_hooks()

        t_val = torch.tensor(0.5, device=device, dtype=model_dtype)
        tracer2.clear()

        if isinstance(ctxt1_len, torch.Tensor):
            cl1 = ctxt1_len.long().to(device)
        else:
            cl1 = torch.tensor([ctxt1_len], dtype=torch.long, device=device)
        mask1 = torch.arange(ctxt1.shape[1], device=device).unsqueeze(0) < cl1.unsqueeze(1)

        if isinstance(ctxt2_len, torch.Tensor):
            cl2 = ctxt2_len.long().to(device)
        else:
            cl2 = torch.tensor([ctxt2_len], dtype=torch.long, device=device)
        mask2 = torch.arange(ctxt2.shape[1], device=device).unsqueeze(0) < cl2.unsqueeze(1)

        with torch.no_grad():
            pred_cap1 = run_traced_forward(
                bundle, tracer2, z, vace_context,
                vtxt1.to(dtype=model_dtype), ctxt1.to(dtype=model_dtype), mask1,
                tgt_padding_mask, t_val, 'cap1', device
            )
            pred_cap2 = run_traced_forward(
                bundle, tracer2, z, vace_context,
                vtxt2.to(dtype=model_dtype), ctxt2.to(dtype=model_dtype), mask2,
                tgt_padding_mask, t_val, 'cap2', device
            )

        pred_cos = torch.nn.functional.cosine_similarity(
            pred_cap1.flatten().unsqueeze(0), pred_cap2.flatten().unsqueeze(0)
        ).item()
        print(f'\n  Final prediction cosine: {pred_cos:.6f}')

        print(f'\n  === INTERMEDIATE FEATURE SENSITIVITY TO CAPTION CHANGE ===')
        results_cap = compare_activations(tracer2, 'cap1', 'cap2')

        # Track where caption info is lost
        print(f'\n  === CAPTION SENSITIVITY DECAY THROUGH BLOCKS ===')
        print(f'  (Lower cos = more caption-sensitive, higher = caption-invariant)')
        for key in sorted(results_cap.keys()):
            if ('double_block' in key and key.endswith('_motion')) or \
               ('single_block' in key and '_modulation' not in key):
                r = results_cap[key]
                bar_len = int(r['cos_sim'] * 40)
                bar = '█' * bar_len + '░' * (40 - bar_len)
                print(f'  {key:<35s} cos={r["cos_sim"]:.6f} {bar}')

        tracer2.remove_hooks()

    print('\nDone.')


if __name__ == '__main__':
    main()

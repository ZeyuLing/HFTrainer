"""Per-step ODE velocity analysis for HyMotion M2M v2 caption model (E2).

Traces velocity predictions through the 50-step Euler ODE to understand:
1. Translation velocity magnitude vs timestep
2. CFG contribution (text vs null difference)
3. Whether the model outputs near-constant translations

Also supports running the unconditioned base model for comparison.

Usage:
    # E2 caption model
    python scripts/debug/diag_caption_ode_velocity.py --mode caption

    # Unconditioned base model
    python scripts/debug/diag_caption_ode_velocity.py --mode uncond

    # Both for comparison
    python scripts/debug/diag_caption_ode_velocity.py --mode both
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np


def load_bundle(config_path, checkpoint_path, device='cuda'):
    """Load a model bundle from config + checkpoint."""
    from mmengine.config import Config
    import hftrainer  # noqa: trigger auto-imports
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
        print(f'  Warning: No checkpoint in {checkpoint_path}, using pretrained.')

    bundle = bundle.to(device)
    return bundle


def load_text_embeddings(cache_path, caption, device='cuda'):
    """Load pre-computed text embeddings from cache."""
    cache_raw = torch.load(cache_path, map_location='cpu', weights_only=False)
    cache = cache_raw.get('cache', cache_raw)

    if caption in cache:
        entry = cache[caption]
    else:
        # Use first available caption
        first_key = next(iter(cache))
        entry = cache[first_key]
        caption = first_key
        print(f'  Caption not found, using: "{caption[:80]}..."')

    vtxt = entry['text_vec_raw'].float().to(device)
    ctxt = entry['text_ctxt_raw'].float().to(device)
    ctxt_len = entry['text_ctxt_raw_length']

    # Ensure 3D
    if vtxt.dim() == 2:
        vtxt = vtxt.unsqueeze(0)
    if ctxt.dim() == 2:
        ctxt = ctxt.unsqueeze(0)

    return vtxt, ctxt, ctxt_len, caption


def run_ode_analysis(bundle, vtxt, ctxt, ctxt_len, caption,
                     num_steps=50, guidance_scale=5.0, L=64, device='cuda',
                     label='model'):
    """Run per-step ODE analysis and return diagnostics."""
    B = 1
    D = int(bundle.mean.numel())
    model_dtype = next(bundle.motion_transformer.parameters()).dtype

    print(f'\n{"="*60}')
    print(f'  [{label}] D={D}, pred_type={bundle.pred_type}')
    print(f'  [{label}] uncondition_mode={bundle.uncondition_mode}')
    print(f'  [{label}] vace_condition_mode={bundle.vace_condition_mode}')
    print(f'  [{label}] mean shape={bundle.mean.shape}, std shape={bundle.std.shape}')
    print(f'  [{label}] mean[:3] (transl): {bundle.mean[:3].cpu().numpy()}')
    print(f'  [{label}] std[:3] (transl): {bundle.std[:3].cpu().numpy()}')

    # Check null embeddings
    null_vtxt_norm = bundle.null_vtxt_feat.float().norm().item()
    null_ctxt_norm = bundle.null_ctxt_input.float().norm().item()
    print(f'  [{label}] null_vtxt norm: {null_vtxt_norm:.4f}')
    print(f'  [{label}] null_ctxt norm: {null_ctxt_norm:.4f}')

    do_cfg = guidance_scale > 1.0 and not bundle.uncondition_mode

    # Text inputs
    if do_cfg:
        vtxt_input = vtxt.to(dtype=model_dtype)
        ctxt_input = ctxt.to(dtype=model_dtype)
        if isinstance(ctxt_len, torch.Tensor):
            ctxt_length = ctxt_len.long().to(device)
        else:
            ctxt_length = torch.tensor([ctxt_len], dtype=torch.long, device=device)
        ctxt_seq_len = ctxt_input.shape[1]

        # Build ctxt_mask_temporal
        ctxt_mask_temporal = torch.arange(ctxt_seq_len, device=device).unsqueeze(0) < ctxt_length.unsqueeze(1)

        # Null branch
        null_vtxt = bundle.null_vtxt_feat.to(dtype=model_dtype).expand_as(vtxt_input)
        null_ctxt = bundle.null_ctxt_input.to(dtype=model_dtype).expand(
            ctxt_input.shape[0], ctxt_input.shape[1], -1
        ).contiguous()
        null_ctxt_mask = torch.zeros_like(ctxt_mask_temporal)
        null_ctxt_mask[:, 0] = True

        vtxt_norm = vtxt_input.float().norm().item()
        ctxt_norm = ctxt_input[:, 0].float().norm().item()
        print(f'  [{label}] CFG enabled, scale={guidance_scale}')
        print(f'  [{label}] vtxt norm: {vtxt_norm:.4f}')
        print(f'  [{label}] ctxt[0] norm: {ctxt_norm:.4f}')
    else:
        # Unconditioned: use null text inputs (already 3D if from bundle)
        vtxt_input = bundle.null_vtxt_feat.to(dtype=model_dtype)
        if vtxt_input.dim() == 2:
            vtxt_input = vtxt_input.unsqueeze(0)
        ctxt_input = bundle.null_ctxt_input.to(dtype=model_dtype)
        if ctxt_input.dim() == 2:
            ctxt_input = ctxt_input.unsqueeze(0)
        ctxt_seq_len = ctxt_input.shape[1]
        ctxt_mask_temporal = torch.ones(B, ctxt_seq_len, dtype=torch.bool, device=device)
        print(f'  [{label}] Unconditioned mode (no CFG)')

    # T2M: src_mask=all ones, src_motion=zeros (all generate)
    src_mask = torch.ones(B, L, D, device=device, dtype=model_dtype)
    src_motion = torch.zeros(B, L, D, device=device, dtype=model_dtype)

    # Build VACE context via bundle's prepare_vace_input
    vace_context = bundle.prepare_vace_input(
        src_motion=src_motion,
        ref_pose=None,
        src_mask=src_mask,
    )
    print(f'  [{label}] vace_context shape: {vace_context.shape} (expect (1, {L}, {2*D}))')
    total_input_dim = D + vace_context.shape[-1]
    print(f'  [{label}] total model input dim: {total_input_dim} (expect {3*D})')

    # Padding mask (all valid)
    tgt_padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # ODE setup
    z = torch.randn(B, L, D, device=device, dtype=model_dtype)
    t_schedule = torch.linspace(0, 1, num_steps + 1, device=device, dtype=model_dtype)

    print(f'\n  [{label}] Starting {num_steps}-step Euler ODE...')
    print(f'  {"Step":>4s} {"t":>6s} {"dt":>6s} | {"v_norm":>8s} {"v_trans":>8s} {"v_rot":>8s} {"v_pos":>8s} | {"x_range":>8s} {"x_transl_range":>12s}', end='')
    if do_cfg:
        print(f' | {"null_v":>8s} {"text_v":>8s} {"cfg_diff":>8s}', end='')
    print()

    step_data = []
    x = z.clone()

    with torch.no_grad():
        for i in range(num_steps):
            t_val = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]

            # Build model input: cat([x, vace_context], dim=-1)
            x_input = torch.cat([x, vace_context], dim=-1)

            if do_cfg:
                # Double batch for CFG
                x_input_cfg = torch.cat([x_input, x_input], dim=0)
                ctxt_cfg = torch.cat([null_ctxt, ctxt_input], dim=0)
                vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
                t_cfg = t_val.expand(2)
                mask_cfg = tgt_padding_mask.repeat(2, 1)
                ctxt_mask_cfg = torch.cat([null_ctxt_mask, ctxt_mask_temporal], dim=0)

                x_pred = bundle.predict_flow(
                    x_input=x_input_cfg,
                    ctxt_input=ctxt_cfg,
                    vtxt_input=vtxt_cfg,
                    timesteps=t_cfg,
                    x_mask_temporal=mask_cfg,
                    ctxt_mask_temporal=ctxt_mask_cfg,
                )

                # Velocity conversion if pred_type == 'x1'
                if bundle.pred_type == 'x1':
                    t_eps = 0.05
                    x_doubled = torch.cat([x, x], dim=0)
                    x_pred = (x_pred - x_doubled) / (1.0 - t_val).clamp_min(t_eps)

                pred_null, pred_text = x_pred.chunk(2, dim=0)
                v = pred_null + guidance_scale * (pred_text - pred_null)

                null_v_norm = pred_null.float().norm().item() / (L * D) ** 0.5
                text_v_norm = pred_text.float().norm().item() / (L * D) ** 0.5
                cfg_diff_norm = (pred_text - pred_null).float().norm().item() / (L * D) ** 0.5
            else:
                x_pred = bundle.predict_flow(
                    x_input=x_input,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t_val.expand(B),
                    x_mask_temporal=tgt_padding_mask,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )

                if bundle.pred_type == 'x1':
                    t_eps = 0.05
                    x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)

                v = x_pred
                null_v_norm = text_v_norm = cfg_diff_norm = 0.0

            # Per-component velocity norms (normalized per element)
            v_f = v.float()
            v_norm = v_f.norm().item() / (L * D) ** 0.5
            v_trans = v_f[..., :3].norm().item() / (L * 3) ** 0.5
            v_rot = v_f[..., 3:135].norm().item() / (L * 132) ** 0.5
            v_pos = v_f[..., 135:198].norm().item() / (L * 63) ** 0.5

            # Euler step
            x = x + v * dt

            # State diagnostics
            x_f = x.float()
            x_range = x_f.max().item() - x_f.min().item()
            x_transl = x_f[0, :, :3]  # (L, 3)
            x_transl_range = (x_transl.max(0).values - x_transl.min(0).values).tolist()

            row = {
                'step': i, 't': t_val.item(), 'dt': dt.item(),
                'v_norm': v_norm, 'v_trans': v_trans, 'v_rot': v_rot, 'v_pos': v_pos,
                'x_range': x_range, 'x_transl_range': x_transl_range,
                'null_v': null_v_norm, 'text_v': text_v_norm, 'cfg_diff': cfg_diff_norm,
            }
            step_data.append(row)

            if i % 5 == 0 or i == num_steps - 1:
                tr = x_transl_range
                line = f'  {i:4d} {t_val.item():6.3f} {dt.item():6.4f} | {v_norm:8.4f} {v_trans:8.4f} {v_rot:8.4f} {v_pos:8.4f} | {x_range:8.3f} [{tr[0]:5.3f},{tr[1]:5.3f},{tr[2]:5.3f}]'
                if do_cfg:
                    line += f' | {null_v_norm:8.4f} {text_v_norm:8.4f} {cfg_diff_norm:8.4f}'
                print(line)

    # Final state analysis
    x_final = x.float()
    print(f'\n  [{label}] === FINAL STATE (t=1.0) ===')
    print(f'  [{label}] x_final shape: {x_final.shape}')
    print(f'  [{label}] x_final range: [{x_final.min().item():.4f}, {x_final.max().item():.4f}]')
    print(f'  [{label}] x_final mean: {x_final.mean().item():.4f}')
    print(f'  [{label}] x_final std: {x_final.std().item():.4f}')

    # Denormalize translation
    mean = bundle.mean.float()
    std = bundle.std.float()
    std_safe = torch.where(std < 1e-3, torch.zeros_like(std), std)
    x_denorm = x_final * std_safe + mean

    transl_denorm = x_denorm[0, :, :3]  # (L, 3)
    print(f'\n  [{label}] === DENORMALIZED TRANSLATION ===')
    print(f'  [{label}] transl range per-dim: X=[{transl_denorm[:,0].min():.4f}, {transl_denorm[:,0].max():.4f}]')
    print(f'  [{label}]                       Y=[{transl_denorm[:,1].min():.4f}, {transl_denorm[:,1].max():.4f}]')
    print(f'  [{label}]                       Z=[{transl_denorm[:,2].min():.4f}, {transl_denorm[:,2].max():.4f}]')

    transl_disp = transl_denorm[-1] - transl_denorm[0]
    print(f'  [{label}] total displacement (first->last): X={transl_disp[0]:.4f}m Y={transl_disp[1]:.4f}m Z={transl_disp[2]:.4f}m')
    print(f'  [{label}] total distance: {transl_disp.norm():.4f}m')

    # Per-frame velocity (in physical space)
    transl_vel = (transl_denorm[1:] - transl_denorm[:-1])  # (L-1, 3)
    vel_mag = transl_vel.norm(dim=-1)  # (L-1,)
    print(f'\n  [{label}] === PHYSICAL TRANSLATION VELOCITY ===')
    print(f'  [{label}] per-frame velocity: mean={vel_mag.mean():.6f}m/frame, max={vel_mag.max():.6f}m/frame')
    print(f'  [{label}] at 30fps: mean={vel_mag.mean()*30:.4f}m/s, max={vel_mag.max()*30:.4f}m/s')
    print(f'  [{label}] (reference: walking ~1.4m/s, ~0.047m/frame)')

    # Frame-to-frame normalized x variation
    x_norm_delta = (x_final[0, 1:, :3] - x_final[0, :-1, :3]).norm(dim=-1)  # (L-1,)
    print(f'\n  [{label}] === NORMALIZED TRANSLATION VARIATION ===')
    print(f'  [{label}] per-frame delta (norm space): mean={x_norm_delta.mean():.6f}, max={x_norm_delta.max():.6f}')

    # Rot6d analysis
    rot_denorm = x_denorm[0, :, 3:135].reshape(L, 22, 6)  # (L, 22, 6)
    rot_delta = (rot_denorm[1:] - rot_denorm[:-1]).norm(dim=-1)  # (L-1, 22)
    print(f'\n  [{label}] === ROTATION VARIATION ===')
    print(f'  [{label}] per-frame rot delta (all joints): mean={rot_delta.mean():.6f}, max={rot_delta.max():.6f}')
    for j_name, j_idx in [('Pelvis', 0), ('L_Hip', 1), ('R_Hip', 2), ('L_Knee', 4), ('R_Knee', 5)]:
        jd = rot_delta[:, j_idx]
        print(f'  [{label}]   {j_name:10s}: mean={jd.mean():.6f}, max={jd.max():.6f}')

    return step_data, x_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['caption', 'uncond', 'both'], default='both')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--num-frames', type=int, default=64)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--caption', default='A person adjusts their stance and performs a golf swing')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to cpu')
        device = 'cpu'

    # Configs and checkpoints
    E2_CONFIG = 'configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py'
    E2_CKPT = 'work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_90'

    UNCOND_CONFIG = 'configs/hymotion_m2m/_base_hymotion_m2m_046b.py'
    UNCOND_CKPT = 'work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2930'

    CACHE_PATH = 'data/eval/m2m_v2/caption_embeddings/cache.pt'

    # Load text embeddings (needed for caption mode)
    vtxt, ctxt, ctxt_len, actual_caption = None, None, None, None
    if args.mode in ('caption', 'both'):
        print(f'\nLoading text embeddings...')
        vtxt, ctxt, ctxt_len, actual_caption = load_text_embeddings(
            CACHE_PATH, args.caption, device
        )
        print(f'  Caption: "{actual_caption[:80]}..."')
        print(f'  vtxt shape: {vtxt.shape}, ctxt shape: {ctxt.shape}, ctxt_len: {ctxt_len}')

    results = {}

    # Run E2 caption model
    if args.mode in ('caption', 'both'):
        print(f'\n{"#"*60}')
        print(f'# E2 Caption Model (epoch 90)')
        print(f'{"#"*60}')
        bundle_e2 = load_bundle(E2_CONFIG, E2_CKPT, device)
        e2_data, e2_final = run_ode_analysis(
            bundle_e2, vtxt, ctxt, ctxt_len, actual_caption,
            num_steps=args.num_steps, guidance_scale=args.guidance_scale,
            L=args.num_frames, device=device, label='E2'
        )
        results['E2'] = {
            'data': e2_data, 'final': e2_final.cpu(),
            'mean': bundle_e2.mean.cpu().float(), 'std': bundle_e2.std.cpu().float(),
        }
        del bundle_e2
        torch.cuda.empty_cache()

    # Run unconditioned base model
    if args.mode in ('uncond', 'both'):
        print(f'\n{"#"*60}')
        print(f'# Unconditioned Base Model (epoch 2930)')
        print(f'{"#"*60}')
        bundle_uncond = load_bundle(UNCOND_CONFIG, UNCOND_CKPT, device)
        uncond_data, uncond_final = run_ode_analysis(
            bundle_uncond, vtxt, ctxt, ctxt_len, actual_caption,
            num_steps=args.num_steps, guidance_scale=1.0,  # no CFG for uncond
            L=args.num_frames, device=device, label='UNCOND'
        )
        results['UNCOND'] = {
            'data': uncond_data, 'final': uncond_final.cpu(),
            'mean': bundle_uncond.mean.cpu().float(), 'std': bundle_uncond.std.cpu().float(),
        }
        del bundle_uncond
        torch.cuda.empty_cache()

    # Comparison summary
    if args.mode == 'both' and len(results) == 2:
        print(f'\n{"="*60}')
        print(f'  COMPARISON SUMMARY')
        print(f'{"="*60}')
        for name in ['E2', 'UNCOND']:
            r = results[name]
            f = r['final'].float()
            mean_v = r['mean']
            std_v = r['std']
            std_safe = torch.where(std_v < 1e-3, torch.zeros_like(std_v), std_v)
            x_denorm = f * std_safe + mean_v
            transl = x_denorm[0, :, :3]
            disp = transl[-1] - transl[0]
            vel = (transl[1:] - transl[:-1]).norm(dim=-1)
            print(f'  {name}: x_range=[{f.min():.3f}, {f.max():.3f}], '
                  f'disp={disp.norm():.4f}m, '
                  f'vel_mean={vel.mean()*30:.4f}m/s, '
                  f'vel_max={vel.max()*30:.4f}m/s')

    print('\nDone.')


if __name__ == '__main__':
    main()

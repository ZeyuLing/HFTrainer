"""Test whether CUDA index OOB assertions from SmplxLite corrupt subsequent operations.

Strategy:
1. Build bundle normally (triggers SMPLX CUDA assertions)
2. Build bundle without SMPLX (skip the problematic model)
3. Compare encode_motion + transformer forward results

If results differ -> CUDA corruption confirmed
If results are same -> CUDA corruption ruled out, investigate elsewhere
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from mmengine import Config

import hftrainer.models.motion.prism  # noqa: F401
import hftrainer.datasets  # noqa: F401

CKPT_DIR = 'work_dirs/prism_overfit_100/checkpoint-epoch_999'
CONFIG_PATH = 'work_dirs/prism_overfit_100/20260526_212303/config.py'
BATCH_PATH = 'docs/temp/autodebug_prism_overfit/cached_batch.pt'


def load_model_normal():
    """Load model normally (triggers SMPLX CUDA assertions)."""
    cfg = Config.fromfile(CONFIG_PATH)
    from hftrainer.registry import MODEL_BUNDLES
    bundle = MODEL_BUNDLES.build(cfg.model)
    _load_checkpoint(bundle)
    return bundle


def load_model_no_smplx():
    """Load model without SmplxLiteV437Coco17 (no CUDA assertions)."""
    cfg = Config.fromfile(CONFIG_PATH)
    # Remove smpl_model from config to avoid loading SmplxLite
    cfg.model.smpl_pose_processor.smpl_model = None
    from hftrainer.registry import MODEL_BUNDLES
    bundle = MODEL_BUNDLES.build(cfg.model)
    _load_checkpoint(bundle)
    return bundle


def _load_checkpoint(bundle):
    state = torch.load(f'{CKPT_DIR}/model.pt', map_location='cpu', weights_only=False)
    info = bundle.transformer.load_state_dict(state['transformer'], strict=False)
    print(f"  Transformer: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
    for k, v in state['__bundle_params__'].items():
        if hasattr(bundle, k):
            param = getattr(bundle, k)
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(v)
            else:
                setattr(bundle, k, torch.nn.Parameter(v, requires_grad=False))
    bundle.cuda()
    bundle.eval()


def run_forward(bundle, batch, seed=0):
    """Run one forward pass and return latents + model_pred + loss."""
    motion = batch['motion'].cuda()
    num_frames = batch.get('num_frames')
    if num_frames is not None and isinstance(num_frames, torch.Tensor):
        num_frames = num_frames.cuda()

    with torch.no_grad():
        latents = bundle.encode_motion(motion)
    batch_size, _, latent_frames, latent_joints = latents.shape

    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=latents.device,
    )

    transformer_dtype = next(bundle.transformer.parameters()).dtype
    text_states = batch['t5_text_embeds'].to(device=latents.device, dtype=transformer_dtype)
    text_mask = batch['t5_text_mask'].to(device=latents.device)

    # Force sigma=0.95 for consistent test
    all_sigmas = bundle.scheduler.sigmas
    best_idx = (all_sigmas - 0.95).abs().argmin().item()
    step_indices = torch.full((batch_size,), best_idx, device=latents.device, dtype=torch.long)
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]

    # No conditioning mask (all generate)
    condition_frame_mask_vae = torch.ones(batch_size, 1, latent_frames, latent_joints,
                                           device=latents.device, dtype=torch.bool)

    torch.manual_seed(seed)
    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)

    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    ts_seq = bundle.create_sequence_ts(
        timesteps, condition_frame_mask_vae, transformer_module.config.patch_size
    )
    noisy_latents = noisy_latents.to(dtype=transformer_dtype)

    with torch.no_grad():
        model_pred = bundle.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=text_states,
            timestep=ts_seq,
            hidden_states_mask=padding_mask if num_frames is not None else None,
            encoder_hidden_states_mask=text_mask,
        )

    model_pred = model_pred.float()
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    mask = padding_mask.unsqueeze(1).expand_as(mse).float()
    loss_rot = (mse[:, :, :, 1:] * mask[:, :, :, 1:]).sum() / (mask[:, :, :, 1:].sum() + 1e-6)

    return {
        'latents': latents.cpu(),
        'model_pred': model_pred.cpu(),
        'targets': targets.cpu(),
        'loss_rot': loss_rot.item(),
        'latents_stats': {
            'mean': latents.mean().item(),
            'std': latents.std().item(),
            'min': latents.min().item(),
            'max': latents.max().item(),
        },
        'pred_stats': {
            'mean': model_pred.mean().item(),
            'std': model_pred.std().item(),
            'min': model_pred.min().item(),
            'max': model_pred.max().item(),
        },
        'target_stats': {
            'mean': targets.mean().item(),
            'std': targets.std().item(),
            'min': targets.min().item(),
            'max': targets.max().item(),
        }
    }


def main():
    print("=" * 70)
    print("CUDA CORRUPTION TEST — SMPLX vs No-SMPLX comparison")
    print("=" * 70)

    batch = torch.load(BATCH_PATH, map_location='cpu', weights_only=False)
    print(f"\nBatch: motion={batch['motion'].shape}, num_frames={batch['num_frames'].tolist()}")

    # ====== TEST 1: Load WITHOUT SMPLX (clean CUDA state) ======
    print("\n" + "=" * 70)
    print("TEST 1: Loading model WITHOUT SmplxLite (no CUDA assertions)")
    print("=" * 70)
    bundle_clean = load_model_no_smplx()
    result_clean = run_forward(bundle_clean, batch, seed=0)
    print(f"  loss_rot = {result_clean['loss_rot']:.6f}")
    print(f"  Latents: mean={result_clean['latents_stats']['mean']:.4f}, "
          f"std={result_clean['latents_stats']['std']:.4f}, "
          f"range=[{result_clean['latents_stats']['min']:.2f}, {result_clean['latents_stats']['max']:.2f}]")
    print(f"  Pred: mean={result_clean['pred_stats']['mean']:.4f}, "
          f"std={result_clean['pred_stats']['std']:.4f}")
    print(f"  Target: mean={result_clean['target_stats']['mean']:.4f}, "
          f"std={result_clean['target_stats']['std']:.4f}")

    # Check cosine similarity between pred and target
    pred_flat = result_clean['model_pred'].reshape(-1)
    target_flat = result_clean['targets'].reshape(-1)
    cos_sim = F.cosine_similarity(pred_flat.unsqueeze(0), target_flat.unsqueeze(0)).item()
    print(f"  Cosine(pred, target) = {cos_sim:.4f}")

    # Free memory
    del bundle_clean
    torch.cuda.empty_cache()

    # ====== TEST 2: Load WITH SMPLX (CUDA assertions fire) ======
    print("\n" + "=" * 70)
    print("TEST 2: Loading model WITH SmplxLite (CUDA assertions will fire)")
    print("=" * 70)
    bundle_smplx = load_model_normal()
    result_smplx = run_forward(bundle_smplx, batch, seed=0)
    print(f"  loss_rot = {result_smplx['loss_rot']:.6f}")
    print(f"  Latents: mean={result_smplx['latents_stats']['mean']:.4f}, "
          f"std={result_smplx['latents_stats']['std']:.4f}, "
          f"range=[{result_smplx['latents_stats']['min']:.2f}, {result_smplx['latents_stats']['max']:.2f}]")
    print(f"  Pred: mean={result_smplx['pred_stats']['mean']:.4f}, "
          f"std={result_smplx['pred_stats']['std']:.4f}")
    print(f"  Target: mean={result_smplx['target_stats']['mean']:.4f}, "
          f"std={result_smplx['target_stats']['std']:.4f}")

    pred_flat2 = result_smplx['model_pred'].reshape(-1)
    target_flat2 = result_smplx['targets'].reshape(-1)
    cos_sim2 = F.cosine_similarity(pred_flat2.unsqueeze(0), target_flat2.unsqueeze(0)).item()
    print(f"  Cosine(pred, target) = {cos_sim2:.4f}")

    # ====== COMPARISON ======
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    latent_diff = (result_clean['latents'] - result_smplx['latents']).abs()
    pred_diff = (result_clean['model_pred'] - result_smplx['model_pred']).abs()

    print(f"  Latents max diff: {latent_diff.max().item():.8f}")
    print(f"  Latents mean diff: {latent_diff.mean().item():.8f}")
    print(f"  Model pred max diff: {pred_diff.max().item():.8f}")
    print(f"  Model pred mean diff: {pred_diff.mean().item():.8f}")
    print(f"  Loss diff: {abs(result_clean['loss_rot'] - result_smplx['loss_rot']):.8f}")

    if latent_diff.max().item() > 1e-5:
        print("\n  *** CUDA CORRUPTION CONFIRMED in encode_motion ***")
        print("  → SMPLX assertions corrupt VAE encoding!")
    elif pred_diff.max().item() > 1e-3:
        print("\n  *** CUDA CORRUPTION CONFIRMED in transformer ***")
        print("  → SMPLX assertions corrupt transformer forward pass!")
    else:
        print("\n  ✓ NO CUDA corruption — results are identical")
        print("  → The 10x loss gap has a different cause")

    # ====== DIAGNOSTIC: Is the model actually predicting anything useful? ======
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Model prediction quality")
    print("=" * 70)

    # For a well-trained model at sigma=0.95 (high noise):
    # - target = noise - latents
    # - model should predict something close to target
    # If loss_rot ≈ 0.08, then avg per-element MSE ≈ 0.08
    # That means avg |error| ≈ sqrt(0.08) ≈ 0.28

    print(f"  Expected from training: loss_rot ≈ 0.08-0.10")
    print(f"  Observed: loss_rot = {result_clean['loss_rot']:.4f}")
    print(f"  Gap factor: {result_clean['loss_rot'] / 0.09:.1f}x")

    # What would random prediction give?
    random_pred = torch.randn_like(result_clean['targets'])
    random_mse = F.mse_loss(random_pred, result_clean['targets']).item()
    print(f"\n  Random prediction loss: {random_mse:.4f}")
    print(f"  Model prediction loss:  {result_clean['loss_rot']:.4f}")
    print(f"  Model is {random_mse / result_clean['loss_rot']:.1f}x better than random")

    # What about predicting zeros?
    zero_pred = torch.zeros_like(result_clean['targets'])
    zero_mse = F.mse_loss(zero_pred, result_clean['targets']).item()
    print(f"  Zero prediction loss:   {zero_mse:.4f}")

    # What about predicting -latents (i.e., assuming noise ≈ 0)?
    # target = noise - latents, if noise≈0 then target≈-latents
    neg_latent_mse = F.mse_loss(-result_clean['latents'], result_clean['targets']).item()
    print(f"  Pred=-latents loss:     {neg_latent_mse:.4f}")

    # What about predicting just noise (target=noise-latent, at sigma=0.95 noise dominates)?
    # If x_t = (1-sigma)*latent + sigma*noise, target = noise-latent
    # At sigma=0.95, x_t ≈ noise, and target ≈ noise-latent
    # If model just predicts something proportional to x_t...
    print(f"\n  Note: At sigma=0.95, target=noise-latent where noise is random N(0,1)")
    print(f"  Target std = {result_clean['target_stats']['std']:.4f} (should be ~sqrt(1+latent_var))")
    print(f"  Pred std = {result_clean['pred_stats']['std']:.4f}")

    # ====== Per-channel analysis ======
    print("\n" + "=" * 70)
    print("PER-CHANNEL ANALYSIS")
    print("=" * 70)

    pred = result_clean['model_pred']  # [B, C, T, J]
    tgt = result_clean['targets']
    for c in range(min(16, pred.shape[1])):
        mse_c = F.mse_loss(pred[:, c], tgt[:, c]).item()
        cos_c = F.cosine_similarity(
            pred[:, c].reshape(1, -1), tgt[:, c].reshape(1, -1)
        ).item()
        print(f"  Channel {c:2d}: MSE={mse_c:.4f}, cos={cos_c:.4f}")

    # ====== Check normalization stats ======
    print("\n" + "=" * 70)
    print("NORMALIZATION STATS CHECK")
    print("=" * 70)

    print(f"  latents_mean shape: {bundle_smplx.latents_mean.shape}")
    print(f"  latents_mean values: {bundle_smplx.latents_mean.flatten().cpu().numpy()}")
    print(f"  latents_std values: {bundle_smplx.latents_std.flatten().cpu().numpy()}")

    # Check smpl_pose_processor stats
    if hasattr(bundle_smplx, 'smpl_pose_processor'):
        proc = bundle_smplx.smpl_pose_processor
        if hasattr(proc, 'mean') and proc.mean is not None:
            print(f"\n  smpl_pose_processor.mean: shape={proc.mean.shape}, "
                  f"range=[{proc.mean.min():.4f}, {proc.mean.max():.4f}]")
            print(f"  smpl_pose_processor.std: shape={proc.std.shape}, "
                  f"range=[{proc.std.min():.4f}, {proc.std.max():.4f}]")
            # Check for zeros in std (would cause inf/nan)
            zero_std = (proc.std.abs() < 1e-8).sum().item()
            if zero_std > 0:
                print(f"  *** WARNING: {zero_std} zero values in std! ***")


if __name__ == '__main__':
    main()

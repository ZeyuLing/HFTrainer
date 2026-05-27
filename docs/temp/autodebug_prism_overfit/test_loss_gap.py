"""Pixel-perfect training step replication to find the loss discrepancy.

Training logs show loss_rot ≈ 0.10, but our diagnostic showed MSE ≈ 0.27 at high sigma.
This script replicates the EXACT training step code to identify the gap.
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
from mmengine import Config

import hftrainer.models.motion.prism  # noqa: F401
import hftrainer.datasets  # noqa: F401

CKPT_DIR = 'work_dirs/prism_overfit_100/checkpoint-epoch_999'
CONFIG_PATH = 'work_dirs/prism_overfit_100/20260526_212303/config.py'

def load_model():
    cfg = Config.fromfile(CONFIG_PATH)
    from hftrainer.registry import MODEL_BUNDLES
    bundle = MODEL_BUNDLES.build(cfg.model)

    # Load checkpoint
    state = torch.load(f'{CKPT_DIR}/model.pt', map_location='cpu', weights_only=False)

    # Load transformer weights
    info = bundle.transformer.load_state_dict(state['transformer'], strict=False)
    print(f"Transformer: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
    if info.missing_keys:
        print(f"  First missing: {info.missing_keys[:3]}")

    # Load bundle params
    for k, v in state['__bundle_params__'].items():
        if hasattr(bundle, k):
            param = getattr(bundle, k)
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(v)
            else:
                setattr(bundle, k, torch.nn.Parameter(v, requires_grad=False))
            print(f"  Loaded bundle param: {k} {v.shape}")

    bundle.cuda()
    bundle.eval()
    return bundle, cfg

def load_training_batch(cfg, batch_size=8):
    """Load a training batch from the dataset."""
    from hftrainer.registry import DATASETS
    from torch.utils.data import DataLoader

    dataset_cfg = cfg.train_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)
    print(f"  Dataset: {len(dataset)} samples")

    # Get the collate function
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    return batch

def exact_training_step(bundle, batch, force_sigma=None, force_no_cond=False):
    """Replicate PrismTrainer.train_step() EXACTLY.

    Args:
        force_sigma: if set, force all samples to this sigma value
        force_no_cond: if True, force no frame conditioning (all generate)
    """
    motion = batch['motion'].cuda()
    num_frames = batch.get('num_frames')
    if num_frames is not None and isinstance(num_frames, torch.Tensor):
        num_frames = num_frames.cuda()

    # Encode motion
    with torch.no_grad():
        latents = bundle.encode_motion(motion)
    batch_size, _, latent_frames, latent_joints = latents.shape

    # Padding mask
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=latents.device,
    )

    # Text
    transformer_dtype = next(bundle.transformer.parameters()).dtype
    if 't5_text_embeds' in batch:
        text_states = batch['t5_text_embeds'].to(device=latents.device, dtype=transformer_dtype)
        text_mask = batch['t5_text_mask'].to(device=latents.device)
    else:
        raise ValueError("No t5_text_embeds in batch!")

    # Condition mask
    if force_no_cond:
        condition_frame_mask_vae = torch.ones(batch_size, 1, latent_frames, latent_joints,
                                               device=latents.device, dtype=torch.bool)
    else:
        condition_frame_mask_vae = bundle.create_condition_mask(
            latents,
            frame_condition_rate=0.5,
            condition_num_frames=[1, 5, 9],
            num_frames=num_frames,
        )

    # Sample timesteps
    if force_sigma is not None:
        # Find the step_index closest to desired sigma
        all_sigmas = bundle.scheduler.sigmas
        diffs = (all_sigmas - force_sigma).abs()
        best_idx = diffs.argmin().item()
        step_indices = torch.full((batch_size,), best_idx, device=latents.device, dtype=torch.long)
    else:
        step_indices = torch.randint(0, len(bundle.scheduler.timesteps), (batch_size,), device=latents.device)

    scheduler_timesteps = bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]

    # Add noise
    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)

    # Per-token timesteps
    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    ts_seq = bundle.create_sequence_ts(
        timesteps, condition_frame_mask_vae, transformer_module.config.patch_size
    )

    noisy_latents = noisy_latents.to(dtype=transformer_dtype)

    # Forward
    with torch.no_grad():
        model_pred = bundle.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=text_states,
            timestep=ts_seq,
            hidden_states_mask=padding_mask if num_frames is not None else None,
            encoder_hidden_states_mask=text_mask,
        )

    # Loss - EXACT training formula
    model_pred = model_pred.float()
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')

    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask_exp = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask_exp

    # Separate translation and rotation
    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

    mse_rot = mse[:, :, :, 1:]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

    loss = 0.5 * loss_transl + 0.5 * loss_rot

    from hftrainer.models.motion.prism.bundle import _get_sigmas
    sigmas = _get_sigmas(bundle.scheduler, timesteps, n_dim=4, dtype=latents.dtype)

    return {
        'loss': loss.item(),
        'loss_transl': loss_transl.item(),
        'loss_rot': loss_rot.item(),
        'sigmas': sigmas[:, 0, 0, 0].cpu().numpy(),
        'condition_rate': condition_frame_mask_vae.float().mean().item(),
    }

def main():
    print("=" * 70)
    print("TRAINING STEP REPLICATION - Finding Loss Discrepancy")
    print("=" * 70)

    print("\nLoading model...")
    bundle, cfg = load_model()

    print("\nLoading training batch...")
    batch = load_training_batch(cfg, batch_size=8)
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  motion: {batch['motion'].shape}")
    if 't5_text_embeds' in batch:
        print(f"  t5_text_embeds: {batch['t5_text_embeds'].shape}, dtype={batch['t5_text_embeds'].dtype}")
        print(f"  t5_text_mask sum per sample: {batch['t5_text_mask'].sum(dim=1).tolist()}")

    # PART 1: Random sigma, with training's conditioning (exact replication)
    print("\n" + "=" * 70)
    print("PART 1: Exact training step (eval mode, random sigma, with conditioning)")
    print("=" * 70)

    losses = []
    for i in range(100):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch)
        losses.append(result)
        if i < 3:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_t={result['loss_transl']:.6f} "
                  f"loss_r={result['loss_rot']:.4f} σ_mean={result['sigmas'].mean():.3f} "
                  f"cond={result['condition_rate']:.2f}")

    print(f"\n  100-step average:")
    print(f"    loss = {np.mean([r['loss'] for r in losses]):.4f} ± {np.std([r['loss'] for r in losses]):.4f}")
    print(f"    loss_rot = {np.mean([r['loss_rot'] for r in losses]):.4f}")
    print(f"    loss_transl = {np.mean([r['loss_transl'] for r in losses]):.6f}")
    print(f"    EXPECTED from training: loss≈0.05, loss_rot≈0.10, loss_transl≈0.005")

    # PART 2: Force high sigma only
    print("\n" + "=" * 70)
    print("PART 2: Force HIGH sigma=0.95, no conditioning")
    print("=" * 70)

    losses_hs = []
    for i in range(20):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch, force_sigma=0.95, force_no_cond=True)
        losses_hs.append(result)
        if i < 3:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_r={result['loss_rot']:.4f} σ={result['sigmas'][0]:.4f}")

    print(f"\n  Average at σ≈0.95: loss_rot={np.mean([r['loss_rot'] for r in losses_hs]):.4f}")

    # PART 3: Force low sigma
    print("\n" + "=" * 70)
    print("PART 3: Force LOW sigma=0.05, no conditioning")
    print("=" * 70)

    losses_ls = []
    for i in range(20):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch, force_sigma=0.05, force_no_cond=True)
        losses_ls.append(result)
        if i < 3:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_r={result['loss_rot']:.4f} σ={result['sigmas'][0]:.4f}")

    print(f"\n  Average at σ≈0.05: loss_rot={np.mean([r['loss_rot'] for r in losses_ls]):.4f}")

    # PART 4: Train mode vs eval mode
    print("\n" + "=" * 70)
    print("PART 4: Train vs Eval mode comparison")
    print("=" * 70)

    bundle.eval()
    torch.manual_seed(42)
    result_eval = exact_training_step(bundle, batch)

    bundle.train()
    torch.manual_seed(42)
    result_train = exact_training_step(bundle, batch)

    print(f"  Eval mode: loss={result_eval['loss']:.4f} loss_rot={result_eval['loss_rot']:.4f}")
    print(f"  Train mode: loss={result_train['loss']:.4f} loss_rot={result_train['loss_rot']:.4f}")
    print(f"  Difference: {result_train['loss'] - result_eval['loss']:.6f}")

    # PART 5: Check model output stats
    print("\n" + "=" * 70)
    print("PART 5: Model output diagnostics at σ=0.95")
    print("=" * 70)

    bundle.eval()
    motion = batch['motion'].cuda()
    num_frames = batch.get('num_frames')
    if num_frames is not None and isinstance(num_frames, torch.Tensor):
        num_frames = num_frames.cuda()

    with torch.no_grad():
        latents = bundle.encode_motion(motion)
    bs, C, T, J = latents.shape
    print(f"  Latents: shape={latents.shape}")
    print(f"    mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"    per-channel std: {latents.std(dim=(0,2,3)).cpu().numpy()}")

    # Force sigma=0.95
    from hftrainer.models.motion.prism.bundle import _get_sigmas
    all_sigmas = bundle.scheduler.sigmas
    best_idx = (all_sigmas - 0.95).abs().argmin().item()
    step_indices = torch.full((bs,), best_idx, device=latents.device, dtype=torch.long)
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]
    sigmas = _get_sigmas(bundle.scheduler, timesteps, n_dim=4, dtype=latents.dtype)
    sigma_val = sigmas[0,0,0,0].item()
    print(f"  Sigma: {sigma_val:.4f}")

    torch.manual_seed(0)
    noise = torch.randn_like(latents)
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    targets = noise - latents

    print(f"  Noise: mean={noise.mean():.4f}, std={noise.std():.4f}")
    print(f"  Targets (noise-latent): mean={targets.mean():.4f}, std={targets.std():.4f}")

    # Forward pass
    condition_frame_mask_vae = torch.ones(bs, 1, T, J, device=latents.device, dtype=torch.bool)
    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    ts_seq = bundle.create_sequence_ts(
        timesteps, condition_frame_mask_vae, transformer_module.config.patch_size
    )
    transformer_dtype = next(bundle.transformer.parameters()).dtype
    text_states = batch['t5_text_embeds'].to(device=latents.device, dtype=transformer_dtype)
    text_mask = batch['t5_text_mask'].to(device=latents.device)
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames, batch_size=bs,
        latent_frames=T, latent_joints=J, device=latents.device,
    )

    with torch.no_grad():
        model_pred = bundle.transformer(
            hidden_states=noisy_latents.to(dtype=transformer_dtype),
            encoder_hidden_states=text_states,
            timestep=ts_seq,
            hidden_states_mask=padding_mask if num_frames is not None else None,
            encoder_hidden_states_mask=text_mask,
        )
    model_pred = model_pred.float()

    print(f"  Model pred: mean={model_pred.mean():.4f}, std={model_pred.std():.4f}")

    # MSE vs various targets
    mask = padding_mask.unsqueeze(1).expand(bs, C, T, J).float()
    mse_vel = ((model_pred - targets)**2 * mask).sum() / mask.sum()
    mse_noise = ((model_pred - noise)**2 * mask).sum() / mask.sum()
    mse_latent = ((model_pred - latents)**2 * mask).sum() / mask.sum()

    print(f"  MSE vs velocity (target): {mse_vel.item():.4f}")
    print(f"  MSE vs noise: {mse_noise.item():.4f}")
    print(f"  MSE vs clean latent: {mse_latent.item():.4f}")

    # Cosine similarity
    def cosine_flat(a, b, mask):
        a_flat = (a * mask).reshape(-1)
        b_flat = (b * mask).reshape(-1)
        return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    print(f"  Cosine vs velocity: {cosine_flat(model_pred, targets, mask):.4f}")
    print(f"  Cosine vs noise: {cosine_flat(model_pred, noise, mask):.4f}")
    print(f"  Cosine vs latent: {cosine_flat(model_pred, latents, mask):.4f}")

    # Per-sample loss
    print(f"\n  Per-sample loss_rot at σ≈0.95:")
    for s in range(min(bs, 8)):
        mse_s = F.mse_loss(model_pred[s, :, :, 1:], targets[s, :, :, 1:]).item()
        print(f"    sample {s}: mse_rot={mse_s:.4f}")

    # PART 6: Try loading a later checkpoint to see if loss actually decreased
    print("\n" + "=" * 70)
    print("PART 6: Compare checkpoints (epoch_999 vs epoch_1199)")
    print("=" * 70)

    # Load epoch_1199 transformer
    import os
    ckpt_1199 = 'work_dirs/prism_overfit_100/checkpoint-epoch_1199/model.pt'
    if os.path.exists(ckpt_1199):
        state_1199 = torch.load(ckpt_1199, map_location='cuda', weights_only=False)
        bundle.transformer.load_state_dict(state_1199['transformer'], strict=True)
        for k, v in state_1199['__bundle_params__'].items():
            if hasattr(bundle, k):
                getattr(bundle, k).data.copy_(v.cuda())
        print("  Loaded epoch_1199")

        bundle.eval()
        losses_1199 = []
        for i in range(20):
            torch.manual_seed(i)
            result = exact_training_step(bundle, batch)
            losses_1199.append(result)

        print(f"  epoch_1199: loss={np.mean([r['loss'] for r in losses_1199]):.4f} "
              f"loss_rot={np.mean([r['loss_rot'] for r in losses_1199]):.4f}")
    else:
        print(f"  epoch_1199 not found")

    print("\n" + "=" * 70)
    print("DONE - Summary")
    print("=" * 70)
    print(f"  Training logs report: loss≈0.05, loss_rot≈0.10, loss_transl≈0.005")
    print(f"  Our replication (epoch_999): loss={np.mean([r['loss'] for r in losses]):.4f}, "
          f"loss_rot={np.mean([r['loss_rot'] for r in losses]):.4f}")
    gap = np.mean([r['loss_rot'] for r in losses]) / 0.10
    print(f"  GAP FACTOR: {gap:.1f}x")

if __name__ == '__main__':
    main()

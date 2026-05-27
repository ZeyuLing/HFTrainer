"""Pixel-perfect training step replication to find the loss discrepancy.
Uses a pre-cached batch to avoid DataLoader hangs.
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
BATCH_PATH = 'docs/temp/autodebug_prism_overfit/cached_batch.pt'


def load_model():
    cfg = Config.fromfile(CONFIG_PATH)
    from hftrainer.registry import MODEL_BUNDLES
    bundle = MODEL_BUNDLES.build(cfg.model)

    state = torch.load(f'{CKPT_DIR}/model.pt', map_location='cpu', weights_only=False)
    info = bundle.transformer.load_state_dict(state['transformer'], strict=False)
    print(f"Transformer: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")

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


def exact_training_step(bundle, batch, force_sigma=None, force_no_cond=False):
    """Replicate PrismTrainer.train_step() EXACTLY."""
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
    if 't5_text_embeds' in batch:
        text_states = batch['t5_text_embeds'].to(device=latents.device, dtype=transformer_dtype)
        text_mask = batch['t5_text_mask'].to(device=latents.device)
    else:
        raise ValueError("No t5_text_embeds in batch!")

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

    if force_sigma is not None:
        all_sigmas = bundle.scheduler.sigmas
        diffs = (all_sigmas - force_sigma).abs()
        best_idx = diffs.argmin().item()
        step_indices = torch.full((batch_size,), best_idx, device=latents.device, dtype=torch.long)
    else:
        step_indices = torch.randint(0, len(bundle.scheduler.timesteps), (batch_size,), device=latents.device)

    scheduler_timesteps = bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]

    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)

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

    # Loss - EXACT training formula
    model_pred = model_pred.float()
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')

    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask_exp = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask_exp

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

    print("\nLoading cached batch...")
    batch = torch.load(BATCH_PATH, map_location='cpu', weights_only=False)
    print(f"  motion: {batch['motion'].shape}")
    print(f"  t5_text_embeds: {batch['t5_text_embeds'].shape}, dtype={batch['t5_text_embeds'].dtype}")
    print(f"  num_frames: {batch['num_frames'].tolist()}")

    # PART 1: Random sigma, with training's conditioning (exact replication)
    print("\n" + "=" * 70)
    print("PART 1: Exact training step (eval mode, random sigma, with conditioning)")
    print("=" * 70)

    losses = []
    for i in range(50):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch)
        losses.append(result)
        if i < 5:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_t={result['loss_transl']:.6f} "
                  f"loss_r={result['loss_rot']:.4f} σ_mean={result['sigmas'].mean():.3f} "
                  f"cond={result['condition_rate']:.2f}")

    print(f"\n  50-step average:")
    print(f"    loss = {np.mean([r['loss'] for r in losses]):.4f} ± {np.std([r['loss'] for r in losses]):.4f}")
    print(f"    loss_rot = {np.mean([r['loss_rot'] for r in losses]):.4f}")
    print(f"    loss_transl = {np.mean([r['loss_transl'] for r in losses]):.6f}")
    print(f"    EXPECTED from training: loss≈0.05, loss_rot≈0.10, loss_transl≈0.005")

    # PART 2: Force high sigma only, no conditioning
    print("\n" + "=" * 70)
    print("PART 2: Force HIGH sigma=0.95, no conditioning")
    print("=" * 70)

    losses_hs = []
    for i in range(10):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch, force_sigma=0.95, force_no_cond=True)
        losses_hs.append(result)
        if i < 3:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_r={result['loss_rot']:.4f} σ={result['sigmas'][0]:.4f}")

    print(f"\n  Average at σ≈0.95 no cond: loss_rot={np.mean([r['loss_rot'] for r in losses_hs]):.4f}")

    # PART 3: Force low sigma, no conditioning
    print("\n" + "=" * 70)
    print("PART 3: Force LOW sigma=0.05, no conditioning")
    print("=" * 70)

    losses_ls = []
    for i in range(10):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch, force_sigma=0.05, force_no_cond=True)
        losses_ls.append(result)
        if i < 3:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_r={result['loss_rot']:.4f} σ={result['sigmas'][0]:.4f}")

    print(f"\n  Average at σ≈0.05 no cond: loss_rot={np.mean([r['loss_rot'] for r in losses_ls]):.4f}")

    # PART 4: Force high sigma WITH conditioning (training config)
    print("\n" + "=" * 70)
    print("PART 4: Force HIGH sigma=0.95, WITH conditioning")
    print("=" * 70)

    losses_hc = []
    for i in range(10):
        torch.manual_seed(i)
        result = exact_training_step(bundle, batch, force_sigma=0.95, force_no_cond=False)
        losses_hc.append(result)
        if i < 3:
            print(f"  seed={i}: loss={result['loss']:.4f} loss_r={result['loss_rot']:.4f} "
                  f"σ={result['sigmas'][0]:.4f} cond={result['condition_rate']:.2f}")

    print(f"\n  Average at σ≈0.95 with cond: loss_rot={np.mean([r['loss_rot'] for r in losses_hc]):.4f}")

    # PART 5: Train mode vs eval mode comparison
    print("\n" + "=" * 70)
    print("PART 5: Train vs Eval mode comparison")
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

    # PART 6: Check sigma distribution - what fraction of steps see low sigma?
    print("\n" + "=" * 70)
    print("PART 6: Sigma distribution analysis")
    print("=" * 70)

    all_sigmas_sched = bundle.scheduler.sigmas
    print(f"  Scheduler sigmas: min={all_sigmas_sched.min():.4f}, max={all_sigmas_sched.max():.4f}")
    print(f"  Median={all_sigmas_sched.median():.4f}, P25={all_sigmas_sched.quantile(0.25):.4f}, P75={all_sigmas_sched.quantile(0.75):.4f}")
    # What fraction of sigmas are < 0.5?
    frac_low = (all_sigmas_sched < 0.5).float().mean().item()
    frac_mid = ((all_sigmas_sched >= 0.5) & (all_sigmas_sched < 0.8)).float().mean().item()
    frac_high = (all_sigmas_sched >= 0.8).float().mean().item()
    print(f"  σ < 0.5: {frac_low*100:.1f}%, 0.5 ≤ σ < 0.8: {frac_mid*100:.1f}%, σ ≥ 0.8: {frac_high*100:.1f}%")

    # PART 7: Loss at various sigma levels
    print("\n" + "=" * 70)
    print("PART 7: Loss vs sigma sweep (with conditioning)")
    print("=" * 70)

    for sigma in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        losses_s = []
        for i in range(5):
            torch.manual_seed(i)
            result = exact_training_step(bundle, batch, force_sigma=sigma, force_no_cond=False)
            losses_s.append(result)
        avg_rot = np.mean([r['loss_rot'] for r in losses_s])
        avg_cond = np.mean([r['condition_rate'] for r in losses_s])
        print(f"  σ={sigma:.2f}: loss_rot={avg_rot:.4f} (cond_rate={avg_cond:.2f})")

    # PART 8: Understand conditioning mask contribution
    print("\n" + "=" * 70)
    print("PART 8: Conditioning mask analysis")
    print("=" * 70)

    # With frame_condition_rate=0.5, what's the effective mask rate?
    # condition_frame_mask_vae: True=generate, False=keep (conditioned)
    motion = batch['motion'].cuda()
    num_frames = batch['num_frames'].cuda()
    with torch.no_grad():
        latents = bundle.encode_motion(motion)
    bs, C, T, J = latents.shape
    print(f"  Latent shape: [{bs}, {C}, {T}, {J}]")

    cond_rates = []
    for i in range(20):
        torch.manual_seed(i)
        mask = bundle.create_condition_mask(
            latents,
            frame_condition_rate=0.5,
            condition_num_frames=[1, 5, 9],
            num_frames=num_frames,
        )
        # mask: [B, 1, T, J], True=generate
        rate = mask.float().mean().item()
        cond_rates.append(rate)

    print(f"  Condition mask generate rate (True%): {np.mean(cond_rates):.3f} ± {np.std(cond_rates):.3f}")
    print(f"  → ~{np.mean(cond_rates)*100:.1f}% of tokens are generated (contribute to loss)")
    print(f"  → ~{(1 - np.mean(cond_rates))*100:.1f}% of tokens are conditioned (loss masked out)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Training logs:     loss≈0.05, loss_rot≈0.10")
    print(f"  Our replication:   loss={np.mean([r['loss'] for r in losses]):.4f}, "
          f"loss_rot={np.mean([r['loss_rot'] for r in losses]):.4f}")
    gap = np.mean([r['loss_rot'] for r in losses]) / 0.10
    print(f"  GAP FACTOR: {gap:.2f}x")
    if gap < 1.5:
        print("  ✓ Replication matches training! The model IS learning correctly.")
        print("  → The inference issue is in the denoising loop, not the model weights.")
    else:
        print("  ✗ Replication does NOT match training!")
        print("  → There's a systematic difference in how we run the model vs training.")


if __name__ == '__main__':
    main()

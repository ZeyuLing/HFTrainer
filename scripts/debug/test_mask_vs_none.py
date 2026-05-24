"""Empirical test: hidden_states_mask=ones vs hidden_states_mask=None.

Training passes None when there's no padding.
Inference always passes ones(B, T_lat, J).

If outputs differ: THIS IS THE BUG.
If outputs are identical: bug is elsewhere.

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/test_mask_vs_none.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gc
import torch
from mmengine.config import Config

import hftrainer  # noqa
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint


def main():
    device = torch.device('cuda')
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    print("=" * 70)
    print("TEST: hidden_states_mask=ones vs hidden_states_mask=None")
    print("=" * 70)

    # Load bundle
    print("\n[1] Loading bundle...")
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # Encode text
    print("\n[2] Encoding text...")
    prompt = "a person walks forward slowly"
    text_states = bundle.encode_prompt(prompt, max_sequence_length=256, dtype=torch.bfloat16).to(device)

    # Move transformer
    print("\n[3] Moving transformer to GPU...")
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    bundle.transformer.eval()
    torch.cuda.empty_cache()

    # Setup inputs
    batch_size = 1
    T_lat = 33
    J = 23
    N = T_lat * J  # 759
    C = bundle.transformer.config.in_channels  # 16

    torch.manual_seed(42)
    latents = torch.randn(batch_size, C, T_lat, J, device=device, dtype=torch.bfloat16)

    # Timestep (mid-range)
    t_val = 500.0
    timestep = torch.full((batch_size, N), t_val, device=device, dtype=torch.bfloat16)

    # Motion mask = all ones (what inference uses)
    motion_mask_ones = torch.ones(batch_size, T_lat, J, device=device)

    print(f"\n[4] Running transformer with mask=ones vs mask=None...")
    print(f"    Input: latents shape={latents.shape}, timestep shape={timestep.shape}")
    print(f"    Motion mask shape: {motion_mask_ones.shape}")

    # Test 1: With ones mask (inference behavior)
    with torch.no_grad():
        pred_with_mask = bundle.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask_ones,
        ).float()

    # Test 2: With None mask (training behavior when no padding)
    with torch.no_grad():
        pred_no_mask = bundle.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=None,
        ).float()

    # Compare
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")

    diff = (pred_with_mask - pred_no_mask).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (pred_no_mask.abs() + 1e-8)).mean().item()

    print(f"\n  pred_with_mask (inference): std={pred_with_mask.std():.6f}, mean={pred_with_mask.mean():.6f}")
    print(f"  pred_no_mask  (training):  std={pred_no_mask.std():.6f}, mean={pred_no_mask.mean():.6f}")
    print(f"\n  Max absolute diff:  {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")
    print(f"  Mean relative diff: {rel_diff:.8f}")
    print(f"  Std ratio (mask/none): {pred_with_mask.std() / pred_no_mask.std():.6f}")

    cos_sim = torch.nn.functional.cosine_similarity(
        pred_with_mask.flatten().unsqueeze(0),
        pred_no_mask.flatten().unsqueeze(0)
    ).item()
    print(f"  Cosine similarity: {cos_sim:.8f}")

    if max_diff < 1e-3:
        print(f"\n  ✓ IDENTICAL (within bf16 precision)")
        print(f"  → hidden_states_mask is NOT the bug")
    elif max_diff < 0.1:
        print(f"\n  ⚠️  SMALL DIFFERENCE (numerical precision issue)")
        print(f"  → Unlikely to cause major deformation")
    else:
        print(f"\n  ❌ SIGNIFICANT DIFFERENCE!")
        print(f"  → hidden_states_mask IS likely the bug!")
        print(f"  → Fix: pass hidden_states_mask=None in inference pipeline")

    # Also test: what about encoder_hidden_states_mask?
    # Training explicitly passes encoder_hidden_states_mask=None
    # Let's verify this is default behavior
    print(f"\n\n{'='*70}")
    print(f"TEST 2: Multiple timestep values")
    print(f"{'='*70}")

    for t_val in [100.0, 500.0, 900.0]:
        timestep = torch.full((batch_size, N), t_val, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            p1 = bundle.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=text_states,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask_ones,
            ).float()

            p2 = bundle.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=text_states,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=None,
            ).float()

        diff = (p1 - p2).abs()
        ratio = p1.std() / p2.std()
        print(f"  t={t_val:6.1f}: mask_std={p1.std():.4f}, none_std={p2.std():.4f}, "
              f"ratio={ratio:.4f}, max_diff={diff.max():.6f}, cos={torch.nn.functional.cosine_similarity(p1.flatten().unsqueeze(0), p2.flatten().unsqueeze(0)).item():.6f}")

    # Test 3: Also check if training actually passes None or a mask
    print(f"\n\n{'='*70}")
    print(f"TEST 3: Check training code - does num_frames=None happen?")
    print(f"{'='*70}")
    print(f"  Training code: hidden_states_mask=padding_mask if num_frames is not None else None")
    print(f"  → When num_frames is None (no variable-length info): mask = None")
    print(f"  → When num_frames is set: mask = padding_mask (could be all-ones for max-length samples)")
    print(f"  → Inference: ALWAYS uses ones(B, T_lat, J)")
    print(f"")
    print(f"  IF max_diff is significant above, the fix is:")
    print(f"  → Change inference to pass hidden_states_mask=None")


if __name__ == "__main__":
    main()

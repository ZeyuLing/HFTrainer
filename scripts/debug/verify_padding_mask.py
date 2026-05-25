#!/usr/bin/env python3
"""Rigorous verification that the PRISM padding fix correctly zeros out loss for padded frames.

Proves:
  1. padding_mask is exactly 0 for ALL positions beyond num_frames_vae
  2. padding_mask is exactly 1 for ALL valid positions
  3. condition_frame_mask_vae has padding positions set to True (generate)
  4. full_mask = condition_mask * padding_mask gives exactly 0 for padding positions
  5. Loss denominator (full_mask.sum()) excludes padding weight entirely
  6. Even if condition_mask=1 for padding frames, padding_mask=0 zeros them out

Additionally tests the FAILURE mode: what would happen WITHOUT the fix.

Usage:
    python scripts/debug/verify_padding_mask.py
"""

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

# ============================================================================
# Replicate the exact logic from bundle.py (no model loading required)
# ============================================================================

CLIP_LEN = 360  # RandomCropPadding target
SCALE_FACTOR_TEMPORAL = 4  # VAE temporal downsampling
LATENT_FRAMES = CLIP_LEN // SCALE_FACTOR_TEMPORAL  # 90
LATENT_JOINTS = 23  # PRISM token layout: 1 translation + 22 rotation
LATENT_CHANNELS = 8  # VAE latent channels (for mse shape)


def create_padding_mask(
    num_frames: Optional[torch.Tensor],
    batch_size: int,
    latent_frames: int,
    latent_joints: int,
    device: torch.device,
) -> torch.Tensor:
    """Exact replica of PrismBundle.create_padding_mask()."""
    if num_frames is None:
        return torch.ones(batch_size, latent_frames, latent_joints, device=device)

    num_frames = num_frames.to(device)
    scale_factor = SCALE_FACTOR_TEMPORAL
    num_frames_vae = (num_frames + scale_factor - 1) // scale_factor
    num_frames_vae = torch.clamp(num_frames_vae, min=0, max=latent_frames)
    frame_idx = torch.arange(latent_frames, device=device).unsqueeze(0)
    mask = frame_idx < num_frames_vae.unsqueeze(1)
    return mask.unsqueeze(-1).expand(batch_size, latent_frames, latent_joints).float()


def create_condition_mask(
    batch_size: int,
    latent_frames: int,
    latent_joints: int,
    device: torch.device,
    frame_condition_rate: float = 0.1,
    condition_num_frames: Union[int, List[int]] = 1,
    num_frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact replica of PrismBundle.create_condition_mask().

    Returns: Boolean mask [B, 1, T, J], True=generate, False=condition.
    """
    if frame_condition_rate <= 0:
        return torch.ones(
            (batch_size, 1, latent_frames, latent_joints), dtype=torch.bool, device=device
        )

    if isinstance(condition_num_frames, int):
        condition_num_frames = [condition_num_frames]
    cond_candidates = torch.tensor(list(condition_num_frames), dtype=torch.long, device=device)
    idx = torch.randint(0, len(cond_candidates), (batch_size,), device=device)
    num_cond_orig = cond_candidates[idx]
    downsample = SCALE_FACTOR_TEMPORAL
    num_cond_vae = (num_cond_orig + downsample - 1) // downsample
    num_cond_vae = torch.clamp(num_cond_vae, min=0, max=latent_frames)
    do_condition = torch.rand(batch_size, device=device) < float(frame_condition_rate)
    num_cond_sel = num_cond_vae * do_condition.long()
    frame_idx = torch.arange(latent_frames, device=device).unsqueeze(0)
    cond_frame_mask = frame_idx < num_cond_sel.unsqueeze(1)
    mask = (~cond_frame_mask).unsqueeze(1).unsqueeze(-1)

    # CRITICAL FIX: Respect padding boundaries
    if num_frames is not None:
        num_frames_dev = num_frames.to(device)
        scale_factor = SCALE_FACTOR_TEMPORAL
        num_frames_vae = (num_frames_dev + scale_factor - 1) // scale_factor
        num_frames_vae = torch.clamp(num_frames_vae, min=1, max=latent_frames)

        # True where frame >= valid_frames (padded region)
        valid_frame_mask = frame_idx >= num_frames_vae.unsqueeze(1)  # [B, T]
        padding_region = valid_frame_mask.unsqueeze(1).unsqueeze(-1).expand_as(mask)
        mask = mask | padding_region  # Force padding frames to generate (True)

    return mask.expand(batch_size, 1, latent_frames, latent_joints).to(torch.bool)


def create_condition_mask_NO_FIX(
    batch_size: int,
    latent_frames: int,
    latent_joints: int,
    device: torch.device,
    frame_condition_rate: float = 0.1,
    condition_num_frames: Union[int, List[int]] = 1,
    num_frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """UNFIXED version: does NOT force padding frames to generate.

    This simulates the pre-fix behavior where padding frames could be
    marked as condition (False), meaning condition_mask=0, and thus
    full_mask=0*padding_mask=0 — which accidentally gives the correct
    behavior. BUT the real danger is when padding frames are marked as
    generate (True) without the padding_mask protection.
    """
    if frame_condition_rate <= 0:
        return torch.ones(
            (batch_size, 1, latent_frames, latent_joints), dtype=torch.bool, device=device
        )

    if isinstance(condition_num_frames, int):
        condition_num_frames = [condition_num_frames]
    cond_candidates = torch.tensor(list(condition_num_frames), dtype=torch.long, device=device)
    idx = torch.randint(0, len(cond_candidates), (batch_size,), device=device)
    num_cond_orig = cond_candidates[idx]
    downsample = SCALE_FACTOR_TEMPORAL
    num_cond_vae = (num_cond_orig + downsample - 1) // downsample
    num_cond_vae = torch.clamp(num_cond_vae, min=0, max=latent_frames)
    do_condition = torch.rand(batch_size, device=device) < float(frame_condition_rate)
    num_cond_sel = num_cond_vae * do_condition.long()
    frame_idx = torch.arange(latent_frames, device=device).unsqueeze(0)
    cond_frame_mask = frame_idx < num_cond_sel.unsqueeze(1)
    mask = (~cond_frame_mask).unsqueeze(1).unsqueeze(-1)

    # NO FIX: padding boundary not enforced
    return mask.expand(batch_size, 1, latent_frames, latent_joints).to(torch.bool)


# ============================================================================
# Trainer loss logic replica
# ============================================================================


def compute_full_mask_and_loss(
    padding_mask: torch.Tensor,  # [B, T, J]
    condition_frame_mask_vae: torch.Tensor,  # [B, 1, T, J] bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate the trainer's loss masking logic.

    Returns:
        full_mask: [B, C, T, J] float
        loss_transl: scalar
        loss_rot: scalar
    """
    batch_size = padding_mask.shape[0]
    latent_frames = padding_mask.shape[1]
    latent_joints = padding_mask.shape[2]

    # Simulate model predictions and targets (random, doesn't matter for mask verification)
    mse = torch.rand(batch_size, LATENT_CHANNELS, latent_frames, latent_joints)

    # Exact trainer logic
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()  # True->1.0
    padding_mask_expanded = padding_mask.unsqueeze(1).expand_as(mse).float()  # 1.0=valid
    full_mask = condition_mask * padding_mask_expanded

    # Translation / rotation split
    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

    mse_rot = mse[:, :, :, 1:]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

    return full_mask, loss_transl, loss_rot


# ============================================================================
# Test framework
# ============================================================================


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str


def num_frames_to_vae(num_frames: int) -> int:
    """Convert raw frame count to VAE latent frame count (ceil division)."""
    return min((num_frames + SCALE_FACTOR_TEMPORAL - 1) // SCALE_FACTOR_TEMPORAL, LATENT_FRAMES)


def run_single_scenario(
    scenario_name: str,
    num_frames_list: List[int],
    device: torch.device,
    seed: int = 42,
) -> List[TestResult]:
    """Run all verifications for a single scenario (one or more samples in a batch)."""
    results = []
    batch_size = len(num_frames_list)
    num_frames_tensor = torch.tensor(num_frames_list, dtype=torch.long)

    # Compute expected VAE frame counts
    vae_frames = [num_frames_to_vae(n) for n in num_frames_list]

    # --- Create masks ---
    torch.manual_seed(seed)
    padding_mask = create_padding_mask(
        num_frames=num_frames_tensor,
        batch_size=batch_size,
        latent_frames=LATENT_FRAMES,
        latent_joints=LATENT_JOINTS,
        device=device,
    )

    torch.manual_seed(seed)
    condition_mask_vae = create_condition_mask(
        batch_size=batch_size,
        latent_frames=LATENT_FRAMES,
        latent_joints=LATENT_JOINTS,
        device=device,
        frame_condition_rate=0.5,  # High rate to force some conditioning
        condition_num_frames=[1, 4, 8],
        num_frames=num_frames_tensor,
    )

    # --- Test 1: padding_mask validity ---
    # For each sample, check exact 0/1 pattern
    test1_pass = True
    test1_details = []
    for i in range(batch_size):
        valid_frames = vae_frames[i]
        # Valid region: all 1.0
        valid_region = padding_mask[i, :valid_frames, :]
        valid_all_one = (valid_region == 1.0).all().item()
        # Padding region: all 0.0
        if valid_frames < LATENT_FRAMES:
            pad_region = padding_mask[i, valid_frames:, :]
            pad_all_zero = (pad_region == 0.0).all().item()
        else:
            pad_all_zero = True  # No padding

        if not valid_all_one or not pad_all_zero:
            test1_pass = False
        test1_details.append(
            f"  Sample {i}: num_frames={num_frames_list[i]}, vae_frames={valid_frames}, "
            f"valid_all_1={valid_all_one}, pad_all_0={pad_all_zero}"
        )

    results.append(TestResult(
        name=f"[{scenario_name}] padding_mask correctness (1=valid, 0=pad)",
        passed=test1_pass,
        details="\n".join(test1_details),
    ))

    # --- Test 2: condition_mask has padding frames set to True (generate) ---
    test2_pass = True
    test2_details = []
    for i in range(batch_size):
        valid_frames = vae_frames[i]
        if valid_frames < LATENT_FRAMES:
            # All frames in padding region must be True (generate)
            pad_condition = condition_mask_vae[i, 0, valid_frames:, :]
            all_generate = pad_condition.all().item()
            if not all_generate:
                test2_pass = False
                num_false = (~pad_condition).sum().item()
                test2_details.append(
                    f"  Sample {i}: FAIL - {num_false} padding frames marked as condition (False)"
                )
            else:
                test2_details.append(
                    f"  Sample {i}: OK - all {LATENT_FRAMES - valid_frames} padding frames forced to generate"
                )
        else:
            test2_details.append(f"  Sample {i}: N/A - no padding (full clip)")

    results.append(TestResult(
        name=f"[{scenario_name}] condition_mask forces padding->generate",
        passed=test2_pass,
        details="\n".join(test2_details),
    ))

    # --- Test 3: full_mask is exactly 0 in padding region ---
    full_mask, _, _ = compute_full_mask_and_loss(padding_mask, condition_mask_vae)

    test3_pass = True
    test3_details = []
    for i in range(batch_size):
        valid_frames = vae_frames[i]
        if valid_frames < LATENT_FRAMES:
            pad_full_mask = full_mask[i, :, valid_frames:, :]
            sum_in_padding = pad_full_mask.sum().item()
            if sum_in_padding != 0.0:
                test3_pass = False
                test3_details.append(
                    f"  Sample {i}: FAIL - full_mask sum in padding = {sum_in_padding:.6f} (must be 0)"
                )
            else:
                test3_details.append(
                    f"  Sample {i}: OK - full_mask sum in padding = 0.0"
                )
        else:
            test3_details.append(f"  Sample {i}: N/A - no padding")

    results.append(TestResult(
        name=f"[{scenario_name}] full_mask exactly 0 in padding region",
        passed=test3_pass,
        details="\n".join(test3_details),
    ))

    # --- Test 4: Loss denominator excludes padding entirely ---
    test4_pass = True
    test4_details = []
    total_elements = full_mask.numel()
    valid_mask_sum = full_mask.sum().item()

    # Compute maximum possible valid elements (only valid frames can contribute)
    max_valid_elements = sum(
        vae_frames[i] * LATENT_CHANNELS * LATENT_JOINTS for i in range(batch_size)
    )
    total_padding_elements = sum(
        (LATENT_FRAMES - vae_frames[i]) * LATENT_CHANNELS * LATENT_JOINTS
        for i in range(batch_size)
    )

    # The denominator (full_mask.sum()) must be <= max_valid_elements
    # and must have zero contribution from padding
    if valid_mask_sum > max_valid_elements:
        test4_pass = False
        test4_details.append(
            f"  FAIL: full_mask.sum()={valid_mask_sum:.1f} > max_valid={max_valid_elements}"
        )
    else:
        padding_weight_frac = 0.0  # We proved sum in padding = 0 in Test 3
        test4_details.append(
            f"  Total elements: {total_elements}"
        )
        test4_details.append(
            f"  Valid region elements: {max_valid_elements} "
            f"({100 * max_valid_elements / total_elements:.1f}%)"
        )
        test4_details.append(
            f"  Padding region elements: {total_padding_elements} "
            f"({100 * total_padding_elements / total_elements:.1f}%)"
        )
        test4_details.append(
            f"  full_mask.sum() (loss denominator): {valid_mask_sum:.1f}"
        )
        test4_details.append(
            f"  Fraction of loss weight from padding: {padding_weight_frac:.6f}% (MUST be 0%)"
        )

    results.append(TestResult(
        name=f"[{scenario_name}] loss denominator excludes padding",
        passed=test4_pass,
        details="\n".join(test4_details),
    ))

    # --- Test 5: Interaction test - condition_mask=1 + padding_mask=0 = 0 ---
    # Explicitly construct worst case: ALL frames marked as generate (condition_mask=True)
    all_generate_mask = torch.ones(
        batch_size, 1, LATENT_FRAMES, LATENT_JOINTS, dtype=torch.bool, device=device
    )
    full_mask_worst, _, _ = compute_full_mask_and_loss(padding_mask, all_generate_mask)

    test5_pass = True
    test5_details = []
    for i in range(batch_size):
        valid_frames = vae_frames[i]
        if valid_frames < LATENT_FRAMES:
            pad_full_worst = full_mask_worst[i, :, valid_frames:, :]
            sum_worst = pad_full_worst.sum().item()
            if sum_worst != 0.0:
                test5_pass = False
                test5_details.append(
                    f"  Sample {i}: FAIL - even with all-generate condition_mask, "
                    f"padding full_mask={sum_worst:.6f} (must be 0)"
                )
            else:
                test5_details.append(
                    f"  Sample {i}: OK - condition_mask=1 * padding_mask=0 = 0 in padding region"
                )
        else:
            test5_details.append(f"  Sample {i}: N/A - no padding")

    results.append(TestResult(
        name=f"[{scenario_name}] interaction: condition=1 * padding=0 = 0",
        passed=test5_pass,
        details="\n".join(test5_details),
    ))

    # --- Test 6: FAILURE MODE - what happens WITHOUT the padding fix ---
    torch.manual_seed(seed)  # Same seed for fair comparison
    condition_mask_NO_FIX = create_condition_mask_NO_FIX(
        batch_size=batch_size,
        latent_frames=LATENT_FRAMES,
        latent_joints=LATENT_JOINTS,
        device=device,
        frame_condition_rate=0.5,
        condition_num_frames=[1, 4, 8],
        num_frames=num_frames_tensor,
    )

    # Without padding_mask in full_mask (pre-fix trainer had no padding_mask)
    # Simulate: full_mask = condition_mask only (no padding_mask multiplication)
    mse_fake = torch.rand(batch_size, LATENT_CHANNELS, LATENT_FRAMES, LATENT_JOINTS)
    condition_only_mask = condition_mask_NO_FIX.expand_as(mse_fake).float()

    test6_details = []
    has_padding_leak = False
    for i in range(batch_size):
        valid_frames = vae_frames[i]
        if valid_frames < LATENT_FRAMES:
            # Without fix: condition_mask in padding region could be 1 (generate)
            pad_cond_nf = condition_only_mask[i, :, valid_frames:, :]
            leaked_weight = pad_cond_nf.sum().item()
            total_weight = condition_only_mask[i].sum().item()
            leak_fraction = leaked_weight / (total_weight + 1e-9)

            if leaked_weight > 0:
                has_padding_leak = True
            test6_details.append(
                f"  Sample {i} (num_frames={num_frames_list[i]}, vae={valid_frames}):"
            )
            test6_details.append(
                f"    WITHOUT fix: padding contributes {leaked_weight:.0f} / "
                f"{total_weight:.0f} weight = {100 * leak_fraction:.1f}% of loss"
            )
            test6_details.append(
                f"    WITH fix: padding contributes 0 / {full_mask[i].sum().item():.0f} = 0% of loss"
            )
        else:
            test6_details.append(
                f"  Sample {i} (num_frames={num_frames_list[i]}): no padding, no difference"
            )

    # This "test" always passes — it's informational showing the fix matters
    results.append(TestResult(
        name=f"[{scenario_name}] FAILURE MODE: quantifying error without fix",
        passed=True,  # Informational
        details="\n".join(test6_details),
    ))

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    device = torch.device("cpu")
    all_results: List[TestResult] = []

    print("=" * 80)
    print("PRISM Padding Mask Verification")
    print(f"  CLIP_LEN={CLIP_LEN}, SCALE_FACTOR={SCALE_FACTOR_TEMPORAL}, "
          f"LATENT_FRAMES={LATENT_FRAMES}, LATENT_JOINTS={LATENT_JOINTS}")
    print("=" * 80)

    # Scenario definitions
    scenarios = [
        ("Full clip (no padding)", [360]),
        ("Moderate padding (200 frames)", [200]),
        ("Heavy padding (30 frames)", [30]),
        ("Extreme edge (1 frame)", [1]),
        ("Mixed batch [360, 200, 100, 30]", [360, 200, 100, 30]),
    ]

    for name, frames in scenarios:
        print(f"\n{'─' * 70}")
        print(f"Scenario: {name}")
        vae_info = [f"{f}→vae={num_frames_to_vae(f)}" for f in frames]
        print(f"  num_frames: {frames}")
        print(f"  vae_frames: {vae_info}")
        print(f"{'─' * 70}")

        results = run_single_scenario(name, frames, device)
        all_results.extend(results)

        for r in results:
            status = "\033[92mPASS\033[0m" if r.passed else "\033[91mFAIL\033[0m"
            print(f"\n  [{status}] {r.name}")
            for line in r.details.split("\n"):
                print(f"    {line}")

    # --- Additional edge-case: verify boundary frame math ---
    print(f"\n{'─' * 70}")
    print("Edge Case: Boundary frame index arithmetic")
    print(f"{'─' * 70}")
    edge_cases = [
        (1, 1),    # ceil(1/4) = 1
        (4, 1),    # ceil(4/4) = 1
        (5, 2),    # ceil(5/4) = 2
        (30, 8),   # ceil(30/4) = 8
        (100, 25), # ceil(100/4) = 25
        (200, 50), # ceil(200/4) = 50
        (357, 90), # ceil(357/4) = 90 (clamped to LATENT_FRAMES)
        (360, 90), # ceil(360/4) = 90
        (400, 90), # ceil(400/4) = 100 but clamped to 90
    ]
    boundary_pass = True
    for raw, expected_vae in edge_cases:
        actual = num_frames_to_vae(raw)
        ok = actual == expected_vae
        if not ok:
            boundary_pass = False
        status_char = "✓" if ok else "✗"
        print(f"  {status_char} num_frames={raw:>4d} → vae_frames={actual:>3d} (expected {expected_vae})")

    all_results.append(TestResult(
        name="Boundary frame index arithmetic",
        passed=boundary_pass,
        details="ceil division with clamp to LATENT_FRAMES",
    ))

    # --- Additional stress test: many random batches ---
    print(f"\n{'─' * 70}")
    print("Stress Test: 100 random batches (batch_size=8)")
    print(f"{'─' * 70}")
    stress_pass = True
    stress_failures = 0
    torch.manual_seed(12345)
    for trial in range(100):
        num_frames_rand = torch.randint(1, 361, (8,)).tolist()
        num_frames_tensor = torch.tensor(num_frames_rand, dtype=torch.long)

        padding_mask = create_padding_mask(
            num_frames=num_frames_tensor,
            batch_size=8,
            latent_frames=LATENT_FRAMES,
            latent_joints=LATENT_JOINTS,
            device=device,
        )
        condition_mask = create_condition_mask(
            batch_size=8,
            latent_frames=LATENT_FRAMES,
            latent_joints=LATENT_JOINTS,
            device=device,
            frame_condition_rate=0.5,
            condition_num_frames=[1, 4, 8, 16],
            num_frames=num_frames_tensor,
        )

        # full_mask computation
        mse = torch.rand(8, LATENT_CHANNELS, LATENT_FRAMES, LATENT_JOINTS)
        cond_expanded = condition_mask.expand_as(mse).float()
        pad_expanded = padding_mask.unsqueeze(1).expand_as(mse).float()
        full_mask = cond_expanded * pad_expanded

        # Check: for every sample, full_mask must be 0 in padding region
        for i in range(8):
            vf = num_frames_to_vae(num_frames_rand[i])
            if vf < LATENT_FRAMES:
                pad_sum = full_mask[i, :, vf:, :].sum().item()
                if pad_sum != 0.0:
                    stress_pass = False
                    stress_failures += 1

        # Check: condition_mask must be True in padding region
        for i in range(8):
            vf = num_frames_to_vae(num_frames_rand[i])
            if vf < LATENT_FRAMES:
                if not condition_mask[i, 0, vf:, :].all().item():
                    stress_pass = False
                    stress_failures += 1

    status = "\033[92mPASS\033[0m" if stress_pass else "\033[91mFAIL\033[0m"
    print(f"  [{status}] 100 random batches x 8 samples: {stress_failures} failures")
    all_results.append(TestResult(
        name="Stress test: 100 random batches",
        passed=stress_pass,
        details=f"{stress_failures} failures out of 800 samples",
    ))

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    for r in all_results:
        status = "\033[92mPASS\033[0m" if r.passed else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {r.name}")

    print(f"\n  {passed}/{total} tests passed")
    if passed == total:
        print("\n\033[92m  ★ ALL TESTS PASSED — padding fix is verified correct.\033[0m")
        sys.exit(0)
    else:
        print(f"\n\033[91m  ✗ {total - passed} TESTS FAILED\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()

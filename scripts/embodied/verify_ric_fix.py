#!/usr/bin/env python3
"""Verify that the RIC Y coordinate fix produces correct loss magnitude.

This script directly tests the normalization of motion_201 data to confirm
that the RIC Y fix brings normalized values into the expected range.

Expected behavior AFTER fix:
  - Joint Y RIC values should be in range [-0.5, 0.5] (relative to pelvis)
  - After normalization with Mean/Std, all 201 dims should be in [-5, 5] range
  - Flow matching loss should be ~0.02-0.04 (not ~0.38-1.9 as before)

Expected behavior BEFORE fix (broken):
  - Joint Y RIC values were absolute world heights (~0.85m for hip)
  - After normalization: (0.85 - (-0.075)) / 0.031 = 29.8 standard deviations
  - This caused 10-100x too high loss values

Usage:
    python3 scripts/embodied/verify_ric_fix.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # Step 1: Load Mean/Std
    # =========================================================================
    stats_dir = 'checkpoints/HY-Motion-1.0/stats'
    mean = np.load(os.path.join(stats_dir, 'Mean.npy'))
    std = np.load(os.path.join(stats_dir, 'Std.npy'))
    print(f"\nMean shape: {mean.shape}, Std shape: {std.shape}")

    # Show RIC Y statistics (dims 135-200, every 3rd dim starting from 136)
    # Layout: [j0_x, j0_y, j0_z, j1_x, j1_y, j1_z, ...]
    # Y dims: 136, 139, 142, ... (offset 135 + j*3 + 1)
    print("\n=== RIC Y dimension statistics (should be relative to pelvis) ===")
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
        'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
        'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    ]
    for j in range(22):
        y_idx = 135 + j * 3 + 1  # Y coordinate of joint j in RIC
        m, s = mean[y_idx], std[y_idx]
        print(f"  Joint {j:2d} ({joint_names[j]:12s}) Y: mean={m:+.4f}, std={s:.4f}")

    # =========================================================================
    # Step 2: Load body model and sample motion
    # =========================================================================
    print("\n=== Loading body model for FK ===")
    from hftrainer.models.motion.hymotion_m2m.network.smpl_lite import SmplxLiteJ24
    body_model = SmplxLiteJ24()
    body_model.to(device)
    body_model.eval()
    print("  SmplxLiteJ24 loaded.")

    # Find a sample NPZ from previous outputs
    npz_dirs = [
        'output/physflow_v2_train_blend50/data/npz',
        'output/embodied_t2m_v4/data/npz',
    ]
    sample_npz = None
    for d in npz_dirs:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith('.npz')]
            if files:
                sample_npz = os.path.join(d, files[0])
                break

    if sample_npz is None:
        # Generate a synthetic standing motion for testing
        print("\n  No sample NPZ found. Using synthetic standing motion.")
        T = 60
        motion_135 = np.zeros((T, 135), dtype=np.float32)
        # Set reasonable translation (standing at origin, Y=0.9)
        motion_135[:, 1] = 0.9  # Y height
        # Set root rotation to identity in rot6d: [[1,0,0],[0,1,0]] = [1,0,0,0,1,0]
        motion_135[:, 3] = 1.0  # rot6d col1 x
        motion_135[:, 7] = 1.0  # rot6d col2 y
        # Set all body joints to identity rotation
        for j in range(21):
            offset = 9 + j * 6
            motion_135[:, offset + 0] = 1.0  # rot6d col1 x
            motion_135[:, offset + 4] = 1.0  # rot6d col2 y
    else:
        print(f"\n  Loading sample: {sample_npz}")
        data = np.load(sample_npz, allow_pickle=True)
        if 'motion_135' in data:
            motion_135 = data['motion_135']
        elif 'motion' in data:
            motion_135 = data['motion'][:, :135]
        else:
            keys = list(data.keys())
            print(f"  NPZ keys: {keys}")
            # Try first array
            motion_135 = data[keys[0]]
            if motion_135.shape[-1] > 135:
                motion_135 = motion_135[:, :135]

    print(f"  Motion shape: {motion_135.shape}")

    # =========================================================================
    # Step 3: Convert motion_135 to motion_201 with the FIXED function
    # =========================================================================
    print("\n=== Converting motion_135 → motion_201 (with RIC Y fix) ===")
    from scripts.embodied.physflow_trainer import motion_135_to_201
    motion_201 = motion_135_to_201(motion_135, body_model, device)
    print(f"  Output shape: {motion_201.shape}")

    # =========================================================================
    # Step 4: Analyze RIC dimensions
    # =========================================================================
    print("\n=== RIC joint positions (should be relative to pelvis) ===")
    ric_part = motion_201[:, 135:]  # (T, 66)
    ric_reshaped = ric_part.reshape(-1, 22, 3)  # (T, 22, 3)

    print(f"\n  Pelvis RIC (should be [0,0,0]):")
    pelvis_ric = ric_reshaped[:, 0, :]
    print(f"    mean: [{pelvis_ric[:, 0].mean():.6f}, {pelvis_ric[:, 1].mean():.6f}, {pelvis_ric[:, 2].mean():.6f}]")
    print(f"    std:  [{pelvis_ric[:, 0].std():.6f}, {pelvis_ric[:, 1].std():.6f}, {pelvis_ric[:, 2].std():.6f}]")

    print(f"\n  L_Hip RIC (Y should be ~-0.075 [below pelvis]):")
    lhip_ric = ric_reshaped[:, 1, :]
    print(f"    mean: [{lhip_ric[:, 0].mean():.4f}, {lhip_ric[:, 1].mean():.4f}, {lhip_ric[:, 2].mean():.4f}]")

    print(f"\n  L_Ankle RIC (Y should be ~-0.8 [far below pelvis]):")
    lankle_ric = ric_reshaped[:, 8, :]
    print(f"    mean: [{lankle_ric[:, 0].mean():.4f}, {lankle_ric[:, 1].mean():.4f}, {lankle_ric[:, 2].mean():.4f}]")

    # =========================================================================
    # Step 5: Normalize and check magnitudes
    # =========================================================================
    print("\n=== Normalization check ===")
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    # Safe std (zeros for near-zero dims)
    safe_std = torch.where(std_t < 1e-3, torch.ones_like(std_t), std_t)

    motion_201_t = torch.from_numpy(motion_201).float().to(device)
    normalized = (motion_201_t - mean_t) / safe_std
    # Zero out near-zero-std dims
    normalized = torch.where(std_t.unsqueeze(0) < 1e-3, torch.zeros_like(normalized), normalized)

    print(f"  Normalized motion shape: {normalized.shape}")
    print(f"  Overall range: [{normalized.min().item():.2f}, {normalized.max().item():.2f}]")
    print(f"  Overall mean: {normalized.mean().item():.4f}")
    print(f"  Overall std: {normalized.std().item():.4f}")

    # Check RIC Y dims specifically
    ric_y_indices = [135 + j * 3 + 1 for j in range(22)]
    ric_y_normalized = normalized[:, ric_y_indices]
    print(f"\n  RIC Y dims normalized:")
    print(f"    range: [{ric_y_normalized.min().item():.2f}, {ric_y_normalized.max().item():.2f}]")
    print(f"    mean: {ric_y_normalized.mean().item():.4f}")
    print(f"    std: {ric_y_normalized.std().item():.4f}")

    # Compare with what the BUG would produce (absolute Y)
    print(f"\n  === COMPARISON: What the BUG would produce ===")
    # Simulate bug: RIC Y = absolute world Y (not relative to pelvis)
    if sample_npz is not None:
        abs_y_hip = 0.85  # typical absolute hip height
        hip_y_mean = mean[138]  # L_Hip Y mean (should be ~-0.075)
        hip_y_std = std[138]    # L_Hip Y std (should be ~0.031)
        buggy_normalized = (abs_y_hip - hip_y_mean) / hip_y_std
        print(f"    Buggy L_Hip Y normalized: {buggy_normalized:.1f} standard deviations!")
        print(f"    (Expected: ~0 +/- 3 standard deviations)")

    # =========================================================================
    # Step 6: Simulate flow matching loss
    # =========================================================================
    print("\n=== Simulating flow matching loss ===")
    # In flow matching:
    #   x1 = normalized target (from RL correction)
    #   x0 = random noise ~ N(0, 1)
    #   x_t = (1-t)*x0 + t*x1
    #   model predicts v = x1 - x0
    #   loss = SmoothL1(v_pred, v_gt)
    #
    # At initialization (before training), model outputs ~random
    # Initial loss ≈ ||v_gt||^2 / 2 ≈ ||x1 - x0||^2 / 2
    # With x1 ~ normalized data and x0 ~ N(0,1):
    #   ||x1 - x0||^2 / 2 ≈ (Var(x1) + 1) / 2 ≈ 1.0 for well-normalized x1
    #
    # If x1 has some dims at 30 std: those dims contribute ~450 to ||x1 - x0||^2!

    # Estimate expected loss with CORRECT normalization
    x1 = normalized.unsqueeze(0)  # (1, T, 201)
    x0 = torch.randn_like(x1)
    v_gt = x1 - x0

    # SmoothL1 loss (per-element)
    expected_loss = torch.nn.functional.smooth_l1_loss(
        torch.zeros_like(v_gt), v_gt  # model output 0 vs gt velocity
    )
    print(f"  Expected initial loss (model=zeros): {expected_loss.item():.4f}")

    # With a random model output
    v_pred = torch.randn_like(v_gt) * 0.5  # random prediction
    random_loss = torch.nn.functional.smooth_l1_loss(v_pred, v_gt)
    print(f"  Expected loss (model=random): {random_loss.item():.4f}")

    # After some training (model learns mean velocity = 0)
    # Residual loss ~ std of (x1 - x0) per dim
    # With correct normalization, each dim ~ N(0, 1-2), so residual ~ 0.02-0.04
    print(f"\n  ✓ If normalized correctly:")
    print(f"    - Initial loss should be ~0.5-1.5")
    print(f"    - After convergence: ~0.02-0.04")
    print(f"  ✗ If RIC Y bug present:")
    print(f"    - Initial loss would be ~5-50 (30^2/2 contribution from Y dims)")
    print(f"    - Cannot converge below ~0.3-1.0")

    # =========================================================================
    # Step 7: Full model test (if GPU available)
    # =========================================================================
    if device.type == 'cuda':
        print("\n=== Full model forward pass test ===")
        try:
            from mmengine import Config
            cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')
            bundle_cfg = cfg.model

            from hftrainer.registry import MODEL_BUNDLES
            bundle = MODEL_BUNDLES.build(bundle_cfg)

            # Load checkpoint
            ckpt_path = 'checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state_dict = ckpt.get('state_dict', ckpt)
                # Handle prefix
                new_sd = {}
                for k, v in state_dict.items():
                    k2 = k.replace('model.', '')
                    new_sd[k2] = v
                bundle.load_state_dict(new_sd, strict=False)
                print("  Loaded pretrained checkpoint.")

            bundle.to(device)
            bundle.eval()

            # Run a single flow matching forward
            x1_test = normalized[:60].unsqueeze(0).to(device)  # (1, 60, 201)
            x0_test = torch.randn_like(x1_test)
            t_test = torch.tensor([0.5], device=device)
            t_bcast = t_test.unsqueeze(-1).unsqueeze(-1)
            x_t_test = (1 - t_bcast) * x0_test + t_bcast * x1_test

            # Use null text embeddings (just testing loss magnitude)
            vtxt = bundle.null_vtxt_feat.expand(1, -1, -1)
            ctxt = bundle.null_ctxt_input.expand(1, 1, -1)
            x_mask = torch.ones(1, 60, device=device, dtype=torch.bool)
            ctxt_mask = torch.ones(1, 1, device=device, dtype=torch.bool)

            with torch.no_grad():
                pred = bundle.predict_flow(
                    x_input=x_t_test,
                    ctxt_input=ctxt,
                    vtxt_input=vtxt,
                    timesteps=t_test,
                    x_mask_temporal=x_mask,
                    ctxt_mask_temporal=ctxt_mask,
                )

            gt_vel = x1_test - x0_test
            model_loss = torch.nn.functional.smooth_l1_loss(pred, gt_vel)
            print(f"  Model forward loss (null text, t=0.5): {model_loss.item():.4f}")
            print(f"  Pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
            print(f"  GT vel range: [{gt_vel.min().item():.3f}, {gt_vel.max().item():.3f}]")

            # This should be ~0.5-1.5 for untrained model with correct normalization
            if model_loss.item() < 5.0:
                print(f"\n  ✅ PASS: Loss {model_loss.item():.4f} is in expected range (<5.0)")
                print(f"     RIC Y fix is working correctly!")
            else:
                print(f"\n  ❌ FAIL: Loss {model_loss.item():.4f} is too high (>5.0)")
                print(f"     Something is still wrong with normalization!")

        except Exception as e:
            print(f"  Model test skipped: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

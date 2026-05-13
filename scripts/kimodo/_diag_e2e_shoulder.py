#!/usr/bin/env python3
"""End-to-end shoulder collapse diagnosis.

Runs KIMODO inference on a single sample (E2 both_1f — first+last frame given)
and compares INPUT retargeted vs OUTPUT shoulder positions at the constraint
frames. If shoulder collapse exists in the output, it's a KIMODO model artifact;
if not, it's a retargeting/pipeline bug.

Also computes:
  1. Input retargeted SOMA30 shoulder heights (sanity check)
  2. KIMODO output SOMA77 shoulder heights (via index mapping)
  3. Frame-by-frame shoulder Y delta (input vs output)
  4. Constraint frame fidelity (are constraint frames preserved?)

Usage (on GPU via Taiji):
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/kimodo/_diag_e2e_shoulder.py
"""
import os, sys, json, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIMODO_ROOT = os.path.join(PROJECT_ROOT, "ref_repo", "KIMODO", "kimodo")
sys.path.insert(0, KIMODO_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def main():
    import torch
    from kimodo.skeleton.definitions import SOMASkeleton30, SMPLXSkeleton22

    smplx22 = SMPLXSkeleton22()
    soma30 = SOMASkeleton30()
    soma_names = soma30.bone_order_names
    idx30 = {n: i for i, n in enumerate(soma_names)}

    # SOMA77 indices for shoulder joints
    from scripts.kimodo.run_kimodo_all_tasks import (
        SOMA77_TO_SMPL22, SMPLX22_NAMES, smpl22_to_soma30_retarget,
        evaluate_sample, soma77_to_smpl22, KIMODO_MODEL,
    )
    # SMPL22 shoulder indices: left_shoulder=16, right_shoulder=17
    # Their SOMA77 indices from the mapping:
    SMPL22_LEFT_SHOULDER = 16  # -> SOMA77[12] = LeftArm
    SMPL22_RIGHT_SHOULDER = 17  # -> SOMA77[40] = RightArm
    SMPL22_LEFT_ELBOW = 18
    SMPL22_RIGHT_ELBOW = 19
    SMPL22_CHEST = 9  # spine3 -> Chest
    SMPL22_NECK = 12

    # ----------------------------------------------------------------
    # 1. Load a test clip
    # ----------------------------------------------------------------
    ann_path = os.path.join(PROJECT_ROOT, "data", "annotation", "train_hymotion_400h.json")
    if not os.path.exists(ann_path):
        print(f"SKIP: annotation not found at {ann_path}")
        return

    with open(ann_path) as f:
        ann = json.load(f)
    data_list = ann["data_list"]
    ann_dir = os.path.dirname(ann_path)

    # Find a medium-length clip (captions not needed for uncond KIMODO)
    test_clip = None
    for key, item in list(data_list.items())[:2000]:
        nf = item.get("num_frames", 0)
        if nf < 60 or nf > 120:
            continue
        smplx_path = os.path.normpath(os.path.join(ann_dir, item["smplx_path"]))
        if os.path.exists(smplx_path):
            caption = item.get("caption_en", item.get("action_name", ""))
            test_clip = (key, smplx_path, caption, nf)
            break

    if test_clip is None:
        print("SKIP: no suitable test clip found")
        return

    key, path, caption, num_frames = test_clip
    print(f"Test clip: {key}")
    print(f"  Path: {path}")
    print(f"  Caption: {caption}")
    print(f"  Frames: {num_frames}")

    # Load SMPLX data
    data = np.load(path, allow_pickle=True)
    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    T = min(poses.shape[0], num_frames, 100)
    poses = poses[:T]
    trans = trans[:T]

    # Build 135-dim motion
    from kimodo.geometry import axis_angle_to_matrix

    body_aa = poses[:, :66].reshape(T, 22, 3)
    rot_mats = axis_angle_to_matrix(torch.from_numpy(body_aa).float())
    rot6d = rot_mats[:, :, :, :2].permute(0, 1, 3, 2).reshape(T, 22, 6)
    motion_135 = np.zeros((T, 135), dtype=np.float32)
    motion_135[:, :3] = trans
    motion_135[:, 3:135] = rot6d.numpy().reshape(T, 132)

    bone_offsets = np.zeros((22, 3), dtype=np.float32)
    neutral_j = smplx22.neutral_joints.numpy()
    parents = smplx22.joint_parents
    for j in range(22):
        if parents[j] != j:
            bone_offsets[j] = neutral_j[j] - neutral_j[parents[j]]

    # ----------------------------------------------------------------
    # 2. Retarget to SOMA30
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INPUT RETARGETED SOMA30 SHOULDER CHECK")
    print("=" * 60)

    soma_rots, soma_pos = smpl22_to_soma30_retarget(motion_135, bone_offsets)

    # Check input shoulder heights
    chest_y = soma_pos[:, idx30["Chest"], 1].mean().item()
    left_arm_y = soma_pos[:, idx30["LeftArm"], 1].mean().item()
    right_arm_y = soma_pos[:, idx30["RightArm"], 1].mean().item()
    left_forearm_y = soma_pos[:, idx30["LeftForeArm"], 1].mean().item()
    right_forearm_y = soma_pos[:, idx30["RightForeArm"], 1].mean().item()

    print(f"  Mean Chest Y:         {chest_y:.4f}m")
    print(f"  Mean LeftArm Y:       {left_arm_y:.4f}m  (delta from chest: {left_arm_y - chest_y:.4f}m)")
    print(f"  Mean RightArm Y:      {right_arm_y:.4f}m  (delta from chest: {right_arm_y - chest_y:.4f}m)")
    print(f"  Mean LeftForeArm Y:   {left_forearm_y:.4f}m")
    print(f"  Mean RightForeArm Y:  {right_forearm_y:.4f}m")

    # Input bone lengths
    left_upper_len = torch.norm(soma_pos[:, idx30["LeftForeArm"]] - soma_pos[:, idx30["LeftArm"]], dim=-1).mean().item()
    right_upper_len = torch.norm(soma_pos[:, idx30["RightForeArm"]] - soma_pos[:, idx30["RightArm"]], dim=-1).mean().item()
    print(f"  Left upper arm len:   {left_upper_len:.4f}m")
    print(f"  Right upper arm len:  {right_upper_len:.4f}m")

    # Also get SMPLX GT positions for comparison
    from hftrainer.pipelines.motion.differentiable_fk import (
        differentiable_fk, rot6d_to_rotmat_row_major,
    )
    motion_t = torch.from_numpy(motion_135).float()
    offsets_t = torch.from_numpy(bone_offsets).float()
    rot6d_t = motion_t[:, 3:135].reshape(T, 22, 6)
    local_rotmat = rot6d_to_rotmat_row_major(rot6d_t)
    gt_pos_22, _ = differentiable_fk(local_rotmat, motion_t[:, :3], offsets_t)
    gt_pos_22_np = gt_pos_22.numpy()

    gt_l_shoulder_y = gt_pos_22_np[:, SMPL22_LEFT_SHOULDER, 1].mean()
    gt_r_shoulder_y = gt_pos_22_np[:, SMPL22_RIGHT_SHOULDER, 1].mean()
    gt_chest_y = gt_pos_22_np[:, SMPL22_CHEST, 1].mean()
    print(f"\n  GT SMPL22 Chest Y:    {gt_chest_y:.4f}m")
    print(f"  GT SMPL22 LShoulder Y:{gt_l_shoulder_y:.4f}m  (delta: {gt_l_shoulder_y - gt_chest_y:.4f}m)")
    print(f"  GT SMPL22 RShoulder Y:{gt_r_shoulder_y:.4f}m  (delta: {gt_r_shoulder_y - gt_chest_y:.4f}m)")

    # ----------------------------------------------------------------
    # 3. Run KIMODO inference
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING KIMODO INFERENCE (E2 both_1f)")
    print("=" * 60)

    from kimodo import load_model
    print(f"  Loading model: {KIMODO_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(KIMODO_MODEL, device=device)
    skeleton = model.skeleton
    print(f"  Model loaded on {device}")

    pred_pos_22, metrics, extra = evaluate_sample(
        model, skeleton,
        soma_rots, soma_pos,
        gt_pos_22_np,
        caption, T,
        task_id='E2', setting='both_1f',
        fps=30,
        motion_135=motion_135,
        bone_offsets=bone_offsets,
    )

    if pred_pos_22 is None:
        print("  KIMODO inference FAILED!")
        return

    print(f"  Inference time: {metrics.get('inference_time', '?')}s")
    print(f"  MPJPE: {metrics.get('mpjpe_pos', '?')}")
    print(f"  Y anchor delta: {metrics.get('y_anchor_delta', '?')}")

    # ----------------------------------------------------------------
    # 4. Compare shoulders: INPUT vs OUTPUT
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SHOULDER COMPARISON: INPUT GT vs KIMODO OUTPUT")
    print("=" * 60)

    # pred_pos_22 is (T, 22, 3)
    pred_chest_y = pred_pos_22[:, SMPL22_CHEST, 1].mean()
    pred_l_shoulder_y = pred_pos_22[:, SMPL22_LEFT_SHOULDER, 1].mean()
    pred_r_shoulder_y = pred_pos_22[:, SMPL22_RIGHT_SHOULDER, 1].mean()
    pred_l_elbow_y = pred_pos_22[:, SMPL22_LEFT_ELBOW, 1].mean()
    pred_r_elbow_y = pred_pos_22[:, SMPL22_RIGHT_ELBOW, 1].mean()

    print(f"  {'Joint':<25s} {'GT Y':>10s} {'Pred Y':>10s} {'Delta':>10s}")
    print(f"  {'-'*55}")
    comparisons = [
        ("Chest (spine3)", gt_chest_y, pred_chest_y),
        ("LeftShoulder", gt_l_shoulder_y, pred_l_shoulder_y),
        ("RightShoulder", gt_r_shoulder_y, pred_r_shoulder_y),
        ("LeftElbow", gt_pos_22_np[:, SMPL22_LEFT_ELBOW, 1].mean(), pred_l_elbow_y),
        ("RightElbow", gt_pos_22_np[:, SMPL22_RIGHT_ELBOW, 1].mean(), pred_r_elbow_y),
    ]
    for name, gt_y, pred_y in comparisons:
        delta = pred_y - gt_y
        flag = " *** COLLAPSED ***" if delta < -0.05 else ""
        print(f"  {name:<25s} {gt_y:>10.4f} {pred_y:>10.4f} {delta:>10.4f}{flag}")

    # Shoulder-to-chest relative heights
    gt_l_rel = gt_l_shoulder_y - gt_chest_y
    gt_r_rel = gt_r_shoulder_y - gt_chest_y
    pred_l_rel = pred_l_shoulder_y - pred_chest_y
    pred_r_rel = pred_r_shoulder_y - pred_chest_y
    print(f"\n  Shoulder relative to chest:")
    print(f"    GT  LeftShoulder-Chest:  {gt_l_rel:.4f}m")
    print(f"    Pred LeftShoulder-Chest: {pred_l_rel:.4f}m  (delta: {pred_l_rel - gt_l_rel:.4f}m)")
    print(f"    GT  RightShoulder-Chest: {gt_r_rel:.4f}m")
    print(f"    Pred RightShoulder-Chest:{pred_r_rel:.4f}m  (delta: {pred_r_rel - gt_r_rel:.4f}m)")

    # Upper arm bone lengths
    pred_l_upper = np.linalg.norm(
        pred_pos_22[:, SMPL22_LEFT_ELBOW] - pred_pos_22[:, SMPL22_LEFT_SHOULDER], axis=-1).mean()
    pred_r_upper = np.linalg.norm(
        pred_pos_22[:, SMPL22_RIGHT_ELBOW] - pred_pos_22[:, SMPL22_RIGHT_SHOULDER], axis=-1).mean()
    gt_l_upper = np.linalg.norm(
        gt_pos_22_np[:, SMPL22_LEFT_ELBOW] - gt_pos_22_np[:, SMPL22_LEFT_SHOULDER], axis=-1).mean()
    gt_r_upper = np.linalg.norm(
        gt_pos_22_np[:, SMPL22_RIGHT_ELBOW] - gt_pos_22_np[:, SMPL22_RIGHT_SHOULDER], axis=-1).mean()

    print(f"\n  Upper arm bone lengths:")
    print(f"    GT  Left:  {gt_l_upper:.4f}m")
    print(f"    Pred Left: {pred_l_upper:.4f}m  (err: {abs(pred_l_upper - gt_l_upper) * 100:.2f}cm)")
    print(f"    GT  Right: {gt_r_upper:.4f}m")
    print(f"    Pred Right:{pred_r_upper:.4f}m  (err: {abs(pred_r_upper - gt_r_upper) * 100:.2f}cm)")

    # ----------------------------------------------------------------
    # 5. Constraint frame fidelity check (frame 0 and last frame)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONSTRAINT FRAME FIDELITY (both_1f: frame 0 and last)")
    print("=" * 60)

    for fi, flabel in [(0, "Frame 0 (constraint)"), (T - 1, f"Frame {T-1} (constraint)")]:
        if fi >= pred_pos_22.shape[0] or fi >= gt_pos_22_np.shape[0]:
            continue
        per_joint_err = np.linalg.norm(pred_pos_22[fi] - gt_pos_22_np[fi], axis=-1)
        print(f"\n  {flabel}:")
        print(f"    Mean MPJPE: {per_joint_err.mean() * 100:.2f}cm")
        print(f"    Max MPJPE:  {per_joint_err.max() * 100:.2f}cm")
        print(f"    L.Shoulder err: {per_joint_err[SMPL22_LEFT_SHOULDER] * 100:.2f}cm")
        print(f"    R.Shoulder err: {per_joint_err[SMPL22_RIGHT_SHOULDER] * 100:.2f}cm")
        print(f"    Chest err:      {per_joint_err[SMPL22_CHEST] * 100:.2f}cm")

    # ----------------------------------------------------------------
    # 6. Frame-by-frame shoulder Y comparison
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FRAME-BY-FRAME SHOULDER Y DELTA (pred - GT)")
    print("=" * 60)

    n_show = min(T, pred_pos_22.shape[0])
    l_deltas = pred_pos_22[:n_show, SMPL22_LEFT_SHOULDER, 1] - gt_pos_22_np[:n_show, SMPL22_LEFT_SHOULDER, 1]
    r_deltas = pred_pos_22[:n_show, SMPL22_RIGHT_SHOULDER, 1] - gt_pos_22_np[:n_show, SMPL22_RIGHT_SHOULDER, 1]

    print(f"  Frame {'L.Shoulder dY':>15s} {'R.Shoulder dY':>15s}")
    for i in range(0, n_show, max(1, n_show // 10)):
        flag = " ***" if l_deltas[i] < -0.05 or r_deltas[i] < -0.05 else ""
        print(f"  {i:5d} {l_deltas[i]:>15.4f} {r_deltas[i]:>15.4f}{flag}")

    l_collapse_pct = (l_deltas < -0.05).sum() / len(l_deltas) * 100
    r_collapse_pct = (r_deltas < -0.05).sum() / len(r_deltas) * 100
    print(f"\n  L.Shoulder collapsed (>5cm below GT): {l_collapse_pct:.1f}% of frames")
    print(f"  R.Shoulder collapsed (>5cm below GT): {r_collapse_pct:.1f}% of frames")
    print(f"  L.Shoulder mean delta: {l_deltas.mean():.4f}m")
    print(f"  R.Shoulder mean delta: {r_deltas.mean():.4f}m")

    # ----------------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    shoulder_collapse_threshold = -0.03  # 3cm
    has_input_collapse = (left_arm_y - chest_y < shoulder_collapse_threshold or
                          right_arm_y - chest_y < shoulder_collapse_threshold)
    has_output_collapse = (pred_l_rel - gt_l_rel < shoulder_collapse_threshold or
                           pred_r_rel - gt_r_rel < shoulder_collapse_threshold)
    has_proportion_error = (abs(pred_l_upper - gt_l_upper) > 0.03 or
                            abs(pred_r_upper - gt_r_upper) > 0.03)

    if has_input_collapse:
        print("  ❌ INPUT retargeted SOMA30 has shoulder collapse → retargeting bug")
    else:
        print("  ✓ INPUT retargeted SOMA30 shoulders are correct")

    if has_output_collapse:
        print("  ❌ KIMODO OUTPUT has shoulder collapse relative to GT")
        print("    → KIMODO model produces collapsed shoulders (model quality issue)")
    else:
        print("  ✓ KIMODO OUTPUT shoulders match GT within 3cm")

    if has_proportion_error:
        print("  ❌ Upper arm bone length error > 3cm → SOMA77→SMPL22 mapping issue")
    else:
        print("  ✓ Upper arm bone lengths match within 3cm")

    if not has_input_collapse and has_output_collapse:
        print("\n  CONCLUSION: Shoulder collapse is a KIMODO MODEL artifact,")
        print("  not a retargeting bug. The input constraints are correct but")
        print("  KIMODO's diffusion process does not preserve shoulder structure.")
    elif not has_input_collapse and not has_output_collapse:
        print("\n  CONCLUSION: No shoulder collapse detected in this sample.")
        print("  The issue may be sample-dependent — try more samples or")
        print("  longer sequences.")
    elif has_input_collapse:
        print("\n  CONCLUSION: Input retargeting has a shoulder bug.")

    print("\nDone.")


if __name__ == "__main__":
    main()

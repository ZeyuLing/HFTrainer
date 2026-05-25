#!/usr/bin/env python3
"""DEFINITIVE TEST: Directly call run_rl_tracker() and report survival steps.

This test answers the critical question:
  Does run_rl_tracker() ITSELF still fall at step 62,
  or is the "62-step" number stale (from before fixes were applied)?

Strategy:
  1. Prepare ref_qpos EXACTLY as process_single_motion() does (using raw model)
  2. Call run_rl_tracker() directly with those inputs
  3. Also prepare ref_qpos as test_init_diff does (using patched model)
  4. Call run_rl_tracker() with those inputs too
  5. Compare results

If BOTH survive ~148 steps: the "62-step fall" was stale; fixes already worked.
If BOTH fall at ~62: there's still a bug inside run_rl_tracker().
If only one fails: the ref_qpos preparation differs between raw/patched models.
"""

import numpy as np
import mujoco
import sys
import os
import yaml
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Paths
MJCF_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
YAML_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
NPZ_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/v4_walk_005.npz"
ONNX_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"


def prepare_ref_qpos_process_single_motion_style():
    """Prepare ref_qpos exactly as process_single_motion() does.

    Uses RAW model (no floor, no patching) for body_pos_1 and height_shift.
    """
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # Load RAW model (same as process_single_motion lines 1243-1245)
    _temp_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    _temp_data = mujoco.MjData(_temp_model)
    body_pos_1 = _temp_model.body_pos[1].copy()

    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # Height fix: same bilateral foot grounding as process_single_motion
    left_foot_body_ids = set()
    right_foot_body_ids = set()
    for bid in range(1, _temp_model.nbody):
        bname = mujoco.mj_id2name(_temp_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_foot_body_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_foot_body_ids.add(bid)

    _temp_data.qpos[:] = ref_qpos[0]
    _temp_data.qvel[:] = 0.0
    mujoco.mj_forward(_temp_model, _temp_data)

    def _compute_lowest_geom_z(body_id_set, model_t, data_t):
        min_z = float("inf")
        for gid in range(model_t.ngeom):
            body_id = model_t.geom_bodyid[gid]
            if body_id not in body_id_set:
                continue
            gtype = int(model_t.geom_type[gid])
            gsize = model_t.geom_size[gid].copy()
            gxpos = data_t.geom_xpos[gid].copy()
            gxmat = data_t.geom_xmat[gid].reshape(3, 3)
            if gtype == 6:  # box
                half_extents = gsize[:3]
                z_extent = (abs(gxmat[2, 0]) * half_extents[0] +
                            abs(gxmat[2, 1]) * half_extents[1] +
                            abs(gxmat[2, 2]) * half_extents[2])
                geom_bottom_z = gxpos[2] - z_extent
            elif gtype == 5:  # capsule
                radius = gsize[0]
                half_len = gsize[1]
                z_extent = abs(gxmat[2, 2]) * half_len + radius
                geom_bottom_z = gxpos[2] - z_extent
            elif gtype == 3:  # sphere
                geom_bottom_z = gxpos[2] - gsize[0]
            else:
                geom_bottom_z = gxpos[2]
            if geom_bottom_z < min_z:
                min_z = geom_bottom_z
        return min_z

    left_min_z = _compute_lowest_geom_z(left_foot_body_ids, _temp_model, _temp_data)
    right_min_z = _compute_lowest_geom_z(right_foot_body_ids, _temp_model, _temp_data)

    # Same logic as process_single_motion (lines 1353-1376)
    FOOT_SWING_THRESHOLD = 0.08
    TARGET_GEOM_SURFACE_Z = 0.0
    foot_height_diff = abs(left_min_z - right_min_z)
    if foot_height_diff > FOOT_SWING_THRESHOLD:
        grounding_ref_z = min(left_min_z, right_min_z)
    else:
        grounding_ref_z = min(left_min_z, right_min_z)

    height_shift = TARGET_GEOM_SURFACE_Z - grounding_ref_z
    if abs(height_shift) > 0.0001:
        ref_qpos[:, 2] += height_shift

    print(f"  [process_single_motion style] height_shift = {height_shift:+.6f}")
    print(f"  [process_single_motion style] root_h[0] = {ref_qpos[0, 2]:.6f}")
    print(f"  [process_single_motion style] body_pos_1 = {body_pos_1}")

    del _temp_model, _temp_data
    return ref_qpos, fps


def prepare_ref_qpos_test_physics_style():
    """Prepare ref_qpos as test_init_diff/compare_runtime_steps does.

    Uses PATCHED model (with floor, config D) for body_pos_1 and height_shift.
    """
    from test_physics_configs import load_model_with_config
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    model, data, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1 = model.body_pos[1].copy()

    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # Same bilateral foot grounding
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_foot_ids = set()
    right_foot_ids = set()
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_foot_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_foot_ids.add(bid)

    def _lowest_geom_z(body_id_set):
        min_z = float("inf")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in body_id_set:
                continue
            gtype = int(model.geom_type[gid])
            gsize = model.geom_size[gid]
            gxpos = data.geom_xpos[gid]
            gxmat = data.geom_xmat[gid].reshape(3, 3)
            if gtype == 5:
                z_ext = abs(gxmat[2, 2]) * gsize[1] + gsize[0]
                bottom = gxpos[2] - z_ext
            elif gtype == 3:
                bottom = gxpos[2] - gsize[0]
            elif gtype == 6:
                z_ext = sum(abs(gxmat[2, j]) * gsize[j] for j in range(3))
                bottom = gxpos[2] - z_ext
            else:
                bottom = gxpos[2]
            min_z = min(min_z, bottom)
        return min_z

    left_min = _lowest_geom_z(left_foot_ids)
    right_min = _lowest_geom_z(right_foot_ids)
    grounding_ref_z = min(left_min, right_min)
    height_shift = 0.0 - grounding_ref_z
    ref_qpos[:, 2] += height_shift

    print(f"  [test_physics style] height_shift = {height_shift:+.6f}")
    print(f"  [test_physics style] root_h[0] = {ref_qpos[0, 2]:.6f}")
    print(f"  [test_physics style] body_pos_1 = {body_pos_1}")

    del model, data
    return ref_qpos, fps


def main():
    from run_smpl_rl_tracker import run_rl_tracker

    with open(YAML_PATH) as f:
        yaml_meta = yaml.safe_load(f)

    print("=" * 70)
    print("  DEFINITIVE TEST: Direct call to run_rl_tracker()")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: process_single_motion style ref_qpos → run_rl_tracker()
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TEST 1: process_single_motion style → run_rl_tracker()")
    print("─" * 70)
    ref_qpos_1, fps_1 = prepare_ref_qpos_process_single_motion_style()
    print(f"  Calling run_rl_tracker() with {ref_qpos_1.shape[0]} frames @ {fps_1}fps...")
    sim_qpos_1, stats_1 = run_rl_tracker(
        ref_qpos=ref_qpos_1,
        motion_fps=fps_1,
        onnx_path=ONNX_PATH,
        mjcf_path=MJCF_PATH,
        yaml_meta=yaml_meta,
    )
    print(f"\n  RESULT 1: status={stats_1['status']}, "
          f"steps={stats_1['actual_sim_steps']}, "
          f"root_h_min={stats_1.get('root_height_min', 'N/A')}")
    if stats_1['status'] == 'fell':
        print(f"  FELL at step {stats_1['fall_frame']}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: test_physics style ref_qpos → run_rl_tracker()
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TEST 2: test_physics style → run_rl_tracker()")
    print("─" * 70)
    ref_qpos_2, fps_2 = prepare_ref_qpos_test_physics_style()
    print(f"  Calling run_rl_tracker() with {ref_qpos_2.shape[0]} frames @ {fps_2}fps...")
    sim_qpos_2, stats_2 = run_rl_tracker(
        ref_qpos=ref_qpos_2,
        motion_fps=fps_2,
        onnx_path=ONNX_PATH,
        mjcf_path=MJCF_PATH,
        yaml_meta=yaml_meta,
    )
    print(f"\n  RESULT 2: status={stats_2['status']}, "
          f"steps={stats_2['actual_sim_steps']}, "
          f"root_h_min={stats_2.get('root_height_min', 'N/A')}")
    if stats_2['status'] == 'fell':
        print(f"  FELL at step {stats_2['fall_frame']}")

    # ═══════════════════════════════════════════════════════════════
    # Compare ref_qpos between the two preparations
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  REF_QPOS COMPARISON (raw model vs patched model preparation)")
    print("─" * 70)
    max_diff = np.abs(ref_qpos_1 - ref_qpos_2).max()
    print(f"  max |ref_qpos_1 - ref_qpos_2| = {max_diff:.2e}")
    if max_diff > 1e-8:
        # Find where they differ
        frame_diffs = np.abs(ref_qpos_1 - ref_qpos_2).max(axis=1)
        worst_frame = np.argmax(frame_diffs)
        print(f"  Worst frame: {worst_frame}, diff={frame_diffs[worst_frame]:.2e}")
        print(f"  ref_qpos_1[0][:7] = {ref_qpos_1[0][:7]}")
        print(f"  ref_qpos_2[0][:7] = {ref_qpos_2[0][:7]}")
        # Check if height differs
        h_diff = np.abs(ref_qpos_1[:, 2] - ref_qpos_2[:, 2]).max()
        print(f"  max height (Z) difference = {h_diff:.2e}")
    else:
        print(f"  ref_qpos preparations are IDENTICAL")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  TEST 1 (process_single_motion style): {stats_1['actual_sim_steps']} steps — {stats_1['status']}")
    print(f"  TEST 2 (test_physics style):          {stats_2['actual_sim_steps']} steps — {stats_2['status']}")
    print()

    s1 = stats_1['actual_sim_steps']
    s2 = stats_2['actual_sim_steps']

    if s1 >= 140 and s2 >= 140:
        print("  ✓ BOTH survived ~148 steps!")
        print("  → The '62-step fall' was STALE (from before fixes)")
        print("  → run_rl_tracker() is FIXED and working correctly")
    elif s1 < 80 and s2 < 80:
        print("  ✗ BOTH still fall early!")
        print("  → There's a REMAINING BUG inside run_rl_tracker()")
        print("  → Need further investigation")
    elif abs(s1 - s2) > 20:
        print("  ! Results DIFFER between preparations!")
        print("  → The ref_qpos preparation (raw vs patched model) matters")
        print(f"  → Difference: {abs(s1-s2)} steps")
    else:
        print(f"  Both survive roughly the same number of steps (~{(s1+s2)//2})")
        if s1 < 140:
            print("  → run_rl_tracker() still has room for improvement")


if __name__ == "__main__":
    main()

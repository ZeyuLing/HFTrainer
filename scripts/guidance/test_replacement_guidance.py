#!/usr/bin/env python3
"""Compare M2M inference: none vs all vs skip_last replacement guidance.

Usage:
    python scripts/test_replacement_guidance.py
"""
import os, sys, time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "work_dirs/hymotion_m2m_completion_uncond_fm_046b/checkpoint-epoch_66",
)
CONFIG = os.path.join(
    PROJECT_ROOT,
    "work_dirs/hymotion_m2m_completion_uncond_fm_046b/20260326_020518/config.py",
)
SAMPLE = os.path.join(
    PROJECT_ROOT,
    "data/motionhub/motionx/motion_data/smplx_55/perform/"
    "Analysis_of_Basic_Calligraphy_3_clip1.npz",
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/test/replacement_guidance_v3")

NUM_JOINT_GROUPS = 23
TRANSL_DIM = 3
JOINT_ROT_DIM = 6
TOTAL_DIM = 135


def expand_grid_to_mask(grid):
    mask = torch.from_numpy(grid.astype(np.float32))
    transl_mask = mask[:, 0:1].repeat(1, TRANSL_DIM)
    joint_mask = mask[:, 1:].repeat_interleave(JOINT_ROT_DIM, dim=-1)
    return torch.cat([transl_mask, joint_mask], dim=-1)


def build_m3_inbetween(T):
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    t_start = max(1, int(T * 0.2))
    t_end = max(t_start + 1, int(T * 0.8))
    grid[t_start:t_end, :] = 1.0
    return expand_grid_to_mask(grid)


def build_m3_prediction(T):
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    t_split = max(1, int(T * 0.3))
    grid[t_split:, :] = 1.0
    return expand_grid_to_mask(grid)


def build_m4_upper_body(T):
    UPPER = [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[:, UPPER] = 1.0
    return expand_grid_to_mask(grid)


def build_m6_keyframe(T):
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    for kf in sorted(set([0, T // 3, 2 * T // 3, T - 1])):
        grid[kf, :] = 0.0
    return expand_grid_to_mask(grid)


TEST_CASES = [
    ("m3_inbetween", build_m3_inbetween),
    ("m3_prediction", build_m3_prediction),
    ("m4_upper_body", build_m4_upper_body),
    ("m6_keyframe", build_m6_keyframe),
]

MODES = ["none", "all", "skip_last", "flow_interp"]


def compute_metrics(original, repaired, mask):
    T = original.shape[0]
    orig_trans = original[:, :3].numpy()
    rep_trans = repaired[:, :3].numpy()
    orig_trans_vel = np.linalg.norm(np.diff(orig_trans, axis=0), axis=-1)
    rep_trans_vel = np.linalg.norm(np.diff(rep_trans, axis=0), axis=-1)
    orig_pose = original[:, 3:].numpy()
    rep_pose = repaired[:, 3:].numpy()
    orig_pose_vel = np.linalg.norm(np.diff(orig_pose, axis=0), axis=-1)
    rep_pose_vel = np.linalg.norm(np.diff(rep_pose, axis=0), axis=-1)

    unmasked = mask < 0.5
    unmasked_diff = (repaired - original)[unmasked].abs().max().item() if unmasked.any() else -1

    mask_any = mask.any(dim=-1).float().numpy()
    transitions = np.where(np.abs(np.diff(mask_any)) > 0.5)[0]
    bj_trans = bj_pose = 0.0
    for t in transitions:
        if t + 1 < T:
            bj_trans = max(bj_trans, np.linalg.norm(rep_trans[t + 1] - rep_trans[t]))
            bj_pose = max(bj_pose, np.linalg.norm(rep_pose[t + 1] - rep_pose[t]))
    return {
        "rep_trans_vel_max": float(rep_trans_vel.max()),
        "rep_pose_vel_max": float(rep_pose_vel.max()),
        "orig_trans_vel_max": float(orig_trans_vel.max()),
        "orig_pose_vel_max": float(orig_pose_vel.max()),
        "unmasked_diff_max": unmasked_diff,
        "boundary_jump_trans": bj_trans,
        "boundary_jump_pose": bj_pose,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda:0"

    from motion_annot_web.m2m_database.hftrainer_repair_runtime import (
        CompletionRepairRuntime,
        load_npz_as_motion,
        motion_135_to_npz_format,
        _save_repaired_npz,
    )

    print("Loading model...")
    runtime = CompletionRepairRuntime(
        checkpoint_path=CHECKPOINT,
        config_path=CONFIG,
        device=device,
        validation_steps=50,
    )
    bundle = runtime.pipeline.bundle
    motion_135, num_frames, fps, abs_trans_frame0 = load_npz_as_motion(SAMPLE)
    orig_data = dict(np.load(SAMPLE, allow_pickle=True))
    T = min(num_frames, 360)
    print(f"Sample: T={num_frames}, FPS={fps}\n")

    def run_once(mask_135, mode):
        runtime.pipeline.replacement_guidance = mode
        msk = mask_135[:T].unsqueeze(0).to(device)
        motion_norm = bundle.normalize_motion(motion_135[:T].unsqueeze(0).to(device))
        motion_norm = motion_norm * (1 - msk)
        if T < 360:
            pad = 360 - T
            motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad), value=0)
            msk = torch.nn.functional.pad(msk, (0, 0, 0, pad), value=0)
        batch = {
            "src_motion": motion_norm,
            "src_mask": msk,
            "src_length": [T],
            "tgt_length": [T],
        }
        with torch.no_grad():
            result = runtime.pipeline(batch)
        latent = result["latent"][0, :T].cpu()
        raw = bundle.denormalize_motion(latent.unsqueeze(0).to(device))[0].cpu()
        mask_crop = mask_135[:T]
        combined = motion_135[:T] * (1 - mask_crop) + raw * mask_crop
        return combined

    hdr = (
        f"{'Task':<18} {'Mode':<12} {'TransVelMax':>12} {'PoseVelMax':>12} "
        f"{'UnmaskDiff':>11} {'BndJmpTr':>10} {'BndJmpPo':>10} {'Time':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, mask_fn in TEST_CASES:
        mask_135 = mask_fn(T)
        for mode in MODES:
            torch.manual_seed(42)
            t0 = time.time()
            repaired = run_once(mask_135, mode)
            elapsed = time.time() - t0
            m = compute_metrics(motion_135[:T], repaired, mask_135[:T])

            print(
                f"{name:<18} {mode:<12} "
                f"{m['rep_trans_vel_max']:>12.4f} {m['rep_pose_vel_max']:>12.4f} "
                f"{m['unmasked_diff_max']:>11.6f} {m['boundary_jump_trans']:>10.4f} "
                f"{m['boundary_jump_pose']:>10.4f} {elapsed:>6.1f}s"
            )

            out_path = os.path.join(OUTPUT_DIR, f"{name}_{mode}.npz")
            aa, trans = motion_135_to_npz_format(repaired, abs_trans_frame0)
            _save_repaired_npz(out_path, aa, trans, orig_data, fps)

        print()

    runtime.pipeline.replacement_guidance = "none"
    print(f"NPZ files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run KIMODO on the Base Pose / keypose edit demo.

KIMODO does not output HyMotion's 135-dim SMPL representation. We save its
native SOMA-77 mesh-driving tensors so score_m2m can render it through the
existing KIMODO mesh_sequence path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kimodo import run_kimodo_all_tasks as kimodo_tasks  # noqa: E402
from scripts.eval.eval_keyframe_pose_guidance import (  # noqa: E402
    AFTER_DIR,
    BEFORE_DIR,
    MIN_KEYPOSE_DIFF,
    NUM_KEYPOSES,
    load_before_after_pairs,
    select_keyposes,
)
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np  # noqa: E402


def _metrics_from_positions(pred_pos: np.ndarray, before: np.ndarray, after: np.ndarray,
                            keypose_indices: list[int], bone_offsets: np.ndarray) -> dict:
    gt_pos = motion135_to_positions_np(after, bone_offsets)
    before_pos = motion135_to_positions_np(before, bone_offsets)
    n = min(pred_pos.shape[0], gt_pos.shape[0], before_pos.shape[0])
    pred = pred_pos[:n]
    gt = gt_pos[:n]
    src = before_pos[:n]
    kp = [k for k in keypose_indices if 0 <= k < n]
    global_mpjpe = float(np.linalg.norm(pred - gt, axis=-1).mean())
    src_mpjpe = float(np.linalg.norm(pred - src, axis=-1).mean())
    kf_mpjpe = float(np.linalg.norm(pred[kp] - gt[kp], axis=-1).mean()) if kp else 0.0
    acc = np.diff(pred, n=2, axis=0)
    smooth = float(np.linalg.norm(acc, axis=-1).mean()) if acc.shape[0] else 0.0
    feet = pred[:, [7, 8, 10, 11]]
    vel = np.diff(feet[..., [0, 2]], axis=0)
    foot_skating = float(np.linalg.norm(vel, axis=-1).mean()) if vel.shape[0] else 0.0
    return {
        "kf_mpjpe": kf_mpjpe,
        "global_mpjpe": global_mpjpe,
        "src_mpjpe": src_mpjpe,
        "overall_smoothness": smooth,
        "foot_skating": foot_skating,
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = ["kf_mpjpe", "global_mpjpe", "src_mpjpe", "overall_smoothness", "foot_skating"]
    return {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}


def _constraint_frame_indices(
    T: int,
    kp_indices: list[int],
    radius: int,
    context_stride: int | None = None,
) -> list[int]:
    frames = {0, T - 1}
    if context_stride is not None and context_stride > 0:
        frames.update(range(0, T, int(context_stride)))
        frames.add(T - 1)
    for ki in kp_indices:
        frames.add(max(0, int(ki) - radius))
        frames.add(min(T - 1, int(ki) + radius))
        frames.add(int(ki))
    return sorted(frames)


def _model_num_frames(T: int, fps_input: int, model_fps: int) -> int:
    return max(10, int((float(T) / float(fps_input)) * float(model_fps)))


def _input_frame_to_model_frame(frame_idx: int, fps_input: int, model_fps: int,
                                num_model_frames: int) -> int:
    out = int(round((float(frame_idx) / float(fps_input)) * float(model_fps)))
    return max(0, min(num_model_frames - 1, out))


def _resample_positions_linear(pos: torch.Tensor, num_frames: int) -> torch.Tensor:
    if pos.shape[0] == num_frames:
        return pos
    import torch.nn.functional as F

    x = pos.permute(1, 2, 0).reshape(1, -1, pos.shape[0])
    y = F.interpolate(x, size=num_frames, mode="linear", align_corners=True)
    return y.reshape(pos.shape[1], pos.shape[2], num_frames).permute(2, 0, 1)


def _resample_rots_nearest(rots: torch.Tensor, num_frames: int) -> torch.Tensor:
    if rots.shape[0] == num_frames:
        return rots
    idx = torch.linspace(0, rots.shape[0] - 1, num_frames, device=rots.device)
    idx = torch.round(idx).long().clamp(0, rots.shape[0] - 1)
    return rots[idx]


def _sample_context_frames(T: int, context_stride: int | None) -> list[int]:
    if context_stride is None or context_stride <= 0:
        return []
    frames = set(range(0, T, int(context_stride)))
    frames.add(T - 1)
    return sorted(frames)


def _build_fullbody_rot_constraint(skeleton, frame_indices, kp_indices,
                                   before_soma_rots, before_soma_pos,
                                   after_soma_rots, after_soma_pos):
    FullBodyConstraintSet = kimodo_tasks._make_fullbody_with_rot_constraint_set()
    kp_set = set(int(k) for k in kp_indices)
    rots = []
    pos = []
    for f in frame_indices:
        if f in kp_set:
            rots.append(after_soma_rots[f])
            root_delta = before_soma_pos[f, skeleton.root_idx] - after_soma_pos[f, skeleton.root_idx]
            pos.append(after_soma_pos[f] + root_delta)
        else:
            rots.append(before_soma_rots[f])
            pos.append(before_soma_pos[f])
    rots_t = torch.stack(rots, dim=0)
    pos_t = torch.stack(pos, dim=0)
    smooth_root = pos_t[:, skeleton.root_idx, :][:, [0, 2]]
    return [FullBodyConstraintSet(
        skeleton=skeleton,
        frame_indices=torch.tensor(frame_indices, dtype=torch.long),
        global_joints_positions=pos_t,
        global_joints_rots=rots_t,
        smooth_root_2d=smooth_root,
    )]


def _build_fullbody_pos_constraint(skeleton, frame_indices, kp_indices,
                                   before_soma_rots, before_soma_pos,
                                   after_soma_rots, after_soma_pos):
    from kimodo.constraints import FullBodyConstraintSet

    kp_set = set(int(k) for k in kp_indices)
    rots = []
    pos = []
    for f in frame_indices:
        if f in kp_set:
            rots.append(after_soma_rots[f])
            root_delta = before_soma_pos[f, skeleton.root_idx] - after_soma_pos[f, skeleton.root_idx]
            pos.append(after_soma_pos[f] + root_delta)
        else:
            rots.append(before_soma_rots[f])
            pos.append(before_soma_pos[f])
    rots_t = torch.stack(rots, dim=0)
    pos_t = torch.stack(pos, dim=0)
    smooth_root = pos_t[:, skeleton.root_idx, :][:, [0, 2]]
    return [FullBodyConstraintSet(
        skeleton=skeleton,
        frame_indices=torch.tensor(frame_indices, dtype=torch.long),
        global_joints_positions=pos_t,
        global_joints_rots=rots_t,
        smooth_root_2d=smooth_root,
    )]


def _build_base_pose_constraints(skeleton, before_soma_rots, before_soma_pos,
                                 after_soma_rots, after_soma_pos,
                                 hard_frame_indices: list[int],
                                 context_frame_indices: list[int],
                                 kp_indices: list[int],
                                 context_mode: str):
    if context_mode == "none" or not context_frame_indices:
        return _build_fullbody_rot_constraint(
            skeleton,
            hard_frame_indices,
            kp_indices,
            before_soma_rots,
            before_soma_pos,
            after_soma_rots,
            after_soma_pos,
        )
    if context_mode == "fullbody_rot":
        merged = sorted(set(hard_frame_indices).union(context_frame_indices))
        return _build_fullbody_rot_constraint(
            skeleton,
            merged,
            kp_indices,
            before_soma_rots,
            before_soma_pos,
            after_soma_rots,
            after_soma_pos,
        )
    if context_mode in {"fullbody_pos", "fullbody_pos_only"}:
        merged = sorted(set(hard_frame_indices).union(context_frame_indices))
        constraints = _build_fullbody_pos_constraint(
            skeleton,
            merged,
            kp_indices,
            before_soma_rots,
            before_soma_pos,
            after_soma_rots,
            after_soma_pos,
        )
        if context_mode == "fullbody_pos_only":
            return constraints
        kp_frames = [int(f) for f in kp_indices if 0 <= int(f) < before_soma_pos.shape[0]]
        if kp_frames:
            constraints.extend(_build_fullbody_rot_constraint(
                skeleton,
                kp_frames,
                kp_indices,
                before_soma_rots,
                before_soma_pos,
                after_soma_rots,
                after_soma_pos,
            ))
        return constraints
    constraints = _build_fullbody_rot_constraint(
        skeleton,
        hard_frame_indices,
        kp_indices,
        before_soma_rots,
        before_soma_pos,
        after_soma_rots,
        after_soma_pos,
    )
    if context_mode == "root2d":
        from kimodo.constraints import Root2DConstraintSet, compute_global_heading

        hard = set(int(f) for f in hard_frame_indices)
        root_frames = [int(f) for f in context_frame_indices if int(f) not in hard]
        if not root_frames:
            return constraints
        frame_t = torch.tensor(root_frames, dtype=torch.long)
        context_pos = before_soma_pos[frame_t]
        constraints.append(Root2DConstraintSet(
            skeleton=skeleton,
            frame_indices=frame_t,
            smooth_root_2d=context_pos[:, skeleton.root_idx, :][:, [0, 2]],
            global_root_heading=compute_global_heading(context_pos, skeleton),
        ))
        return constraints
    raise ValueError(f"unknown context_mode: {context_mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-cases", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed. The effective seed is seed + case index.",
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--radius", type=int, default=30)
    parser.add_argument(
        "--context-stride",
        type=int,
        default=0,
        help=(
            "Also constrain original-motion frames every N frames. "
            "0 keeps the old sparse policy."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=["none", "root2d", "fullbody_pos", "fullbody_pos_only", "fullbody_rot"],
        default="fullbody_pos_only",
        help=(
            "How to use dense context frames. fullbody_pos_only preserves pose/root "
            "positions without hard-snapping global rotations; fullbody_pos additionally "
            "pins rotations at edit keyframes; fullbody_rot pins dense rotations and can jitter."
        ),
    )
    parser.add_argument("--output-dir", default="output/eval_keyframe_pose_v3/local_rot/kimodo_base_pose_r30")
    args = parser.parse_args()

    kimodo_tasks.DIFFUSION_STEPS = int(args.num_steps)
    # Base Pose Edit constrains every frame's global position when
    # context_stride=1. KIMODO's generic long-motion split blends segment
    # transitions after feature hard-paste, which can move constrained roots
    # at the segment boundary. Keep this task single-pass so condition frames
    # remain exact in the inference output.
    kimodo_tasks.KIMODO_SAFE_LEN = 10000
    before_dir = PROJECT_ROOT / BEFORE_DIR
    after_dir = PROJECT_ROOT / AFTER_DIR
    pairs = load_before_after_pairs(str(before_dir), str(after_dir), max_pairs=args.num_cases)
    if not pairs:
        raise RuntimeError("no before/after pairs loaded")
    end_idx = len(pairs) if args.end_idx is None else min(int(args.end_idx), len(pairs))
    start_idx = max(0, int(args.start_idx))
    pairs = pairs[start_idx:end_idx]
    if not pairs:
        raise RuntimeError(f"empty shard: start={start_idx}, end={end_idx}")

    from kimodo import load_model

    print(f"Loading KIMODO model: {kimodo_tasks.KIMODO_MODEL}", flush=True)
    model = load_model(kimodo_tasks.KIMODO_MODEL, device=args.device)
    skeleton = model.skeleton
    model_fps = int(model.fps)
    bone_offsets = torch.load(PROJECT_ROOT / "data/hymotion_m2m_data/bone_offsets_22.pt", map_location="cpu").numpy()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for local_idx, pair in enumerate(pairs):
        case_idx = start_idx + local_idx
        before = pair["before_motion"]
        after = pair["after_motion"]
        kp_indices, diffs = select_keyposes(before, after, k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF)
        T = int(pair["num_frames"])
        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'
        try:
            np.random.seed(int(args.seed) + case_idx)
            torch.manual_seed(int(args.seed) + case_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed) + case_idx)

            before_rots, before_pos = kimodo_tasks.smpl22_to_soma30_retarget(before, bone_offsets)
            after_rots, after_pos = kimodo_tasks.smpl22_to_soma30_retarget(after, bone_offsets)

            anchor = max(0, min(T - 1, int(kp_indices[0]) if kp_indices else 0))
            R_yaw, t_xz, heading0 = kimodo_tasks.kimodo_compute_canon_transform(
                before_pos, skeleton, anchor_frame=anchor
            )
            before_rots_c, before_pos_c = kimodo_tasks.kimodo_apply_canon(before_rots, before_pos, R_yaw, t_xz)
            after_rots_c, after_pos_c = kimodo_tasks.kimodo_apply_canon(after_rots, after_pos, R_yaw, t_xz)
            num_model_frames = _model_num_frames(T, int(args.fps), model_fps)
            before_pos_m = _resample_positions_linear(before_pos_c, num_model_frames)
            after_pos_m = _resample_positions_linear(after_pos_c, num_model_frames)
            before_rots_m = _resample_rots_nearest(before_rots_c, num_model_frames)
            after_rots_m = _resample_rots_nearest(after_rots_c, num_model_frames)
            kp_indices_m = sorted({
                _input_frame_to_model_frame(k, int(args.fps), model_fps, num_model_frames)
                for k in kp_indices
            })
            radius_m = max(1, int(round(float(args.radius) * float(model_fps) / float(args.fps))))
            context_stride = int(args.context_stride) if args.context_stride > 0 else None
            context_stride_m = (
                max(1, int(round(float(context_stride) * float(model_fps) / float(args.fps))))
                if context_stride is not None else None
            )
            hard_frame_indices = _constraint_frame_indices(
                num_model_frames,
                kp_indices_m,
                radius_m,
                context_stride=None,
            )
            context_frame_indices = _sample_context_frames(num_model_frames, context_stride_m)
            all_constraint_frames = sorted(set(hard_frame_indices).union(context_frame_indices))
            constraints = _build_base_pose_constraints(
                skeleton, before_rots_m, before_pos_m, after_rots_m, after_pos_m,
                hard_frame_indices, context_frame_indices, kp_indices_m, args.context_mode
            )

            t0 = time.time()
            pred_pos, k_metrics, soma_data = kimodo_tasks._run_kimodo_with_constraints(
                model, skeleton, constraints, "", T,
                motion135_to_positions_np(after, bone_offsets),
                fps=int(args.fps),
                constraints_fps=model_fps,
                constraints_T=num_model_frames,
                canon_transform=(R_yaw, t_xz),
            )
            if pred_pos is None or not soma_data:
                raise RuntimeError(f"KIMODO failed: {k_metrics}")
            metrics = _metrics_from_positions(pred_pos, before, after, kp_indices, bone_offsets)
            row = {
                "case_key": case_key,
                "filename": pair["filename"],
                "num_frames": T,
                "keypose_indices": kp_indices,
                "constraint_frames": all_constraint_frames,
                "constraint_frames_input_fps": [
                    int(round((float(f) / float(model_fps)) * float(args.fps)))
                    for f in all_constraint_frames
                ],
                "hard_constraint_frames": hard_frame_indices,
                "context_frames": context_frame_indices,
                "radius": args.radius,
                "context_stride": context_stride or 0,
                "context_mode": args.context_mode,
                "constraint_fps": model_fps,
                "seed": int(args.seed) + case_idx,
                "elapsed_sec": time.time() - t0,
                **metrics,
                **{f"kimodo_{k}": v for k, v in k_metrics.items()},
            }
            rows.append(row)
            np.savez_compressed(
                out_dir / f"{case_key}.npz",
                posed_joints=soma_data["posed_joints"],
                global_rot_mats=soma_data.get("global_rot_mats"),
                output_positions=pred_pos.astype(np.float32),
                before_motion=before,
                after_motion=after,
                keypose_indices=np.array(kp_indices, dtype=np.int64),
                constraint_frames=np.array(all_constraint_frames, dtype=np.int64),
                constraint_frames_input_fps=np.array([
                    int(round((float(f) / float(model_fps)) * float(args.fps)))
                    for f in all_constraint_frames
                ], dtype=np.int64),
                hard_constraint_frames=np.array(hard_frame_indices, dtype=np.int64),
                context_frames=np.array(context_frame_indices, dtype=np.int64),
                constraint_fps=np.array(model_fps, dtype=np.int64),
                context_mode=args.context_mode,
                correction_diffs=diffs,
            )
            print(
                f"KIMODO {case_key}: glob={metrics['global_mpjpe']:.4f} "
                f"kf={metrics['kf_mpjpe']:.4f} smooth={metrics['overall_smoothness']:.4f} "
                f"foot={metrics['foot_skating']:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[KIMODO] {case_key} failed: {e}", flush=True)
            traceback.print_exc()

    result = {"aggregate": _aggregate(rows) if rows else {}, "cases": rows}
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()

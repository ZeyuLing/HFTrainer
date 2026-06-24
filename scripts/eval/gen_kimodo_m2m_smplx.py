#!/usr/bin/env python3
"""KIMODO-SMPLX native M2M (motion-conditioned) generation on HumanML3D.

This is the SMPL-X-skeleton counterpart of the legacy SOMA-coupled
``scripts/kimodo/run_kimodo_all_tasks.py``.  Instead of bridging GT motion
through the SOMA-30 skeleton (retarget in / retarget out), it builds the task
constraints **directly on the native KIMODO SMPL-X 22-joint skeleton** from the
HumanML3D-272 ground truth converted to ``motion_135`` (lossless, since SMPL-X
shares the SMPL 22-joint kinematic tree), exactly like
``gen_kimodo_tp2m_smplx.py`` does for the prefix (TP2M) task.

Supported frame-subset constraint families (one ``--task``):

* ``inbetween``  : keep first + last frame (E2 both_1f, minimal in-betweening).
* ``prediction`` : keep first ``ceil(0.2 T)`` frames    (E2 pre20).
* ``backcast``   : keep last  ``ceil(0.2 T)`` frames     (E2 post20).
* ``clip``       : keep first+last ``ceil(0.2 T)`` frames (E2 mid60).

Each constrained frame uses a full-body ``FullBodyConstraintSet`` on the SMPL-X
skeleton; KIMODO solves the remaining frames.

Output: one ``<sid>.npz`` per sample containing the native KIMODO ``motion_135``
prediction together with the ``gt_motion_135`` and ``src_mask`` needed by the
downstream MIB evaluators (distribution metrics via
``repack_pred_to_272ids.py`` + ``eval_motionstreamer_272.py``; position metrics
via the 272-ric MPJPE/[P]-MPJPE/jitter/foot computation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/eval") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/eval"))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from hftrainer.motion.skeleton.fk import rot6d_to_rotmat_row_major  # noqa: E402
from scripts.eval.gen_kimodo_t2m_positions import (  # noqa: E402
    CacheOnlyTextEncoder,
    _fit_length,
    _load_corpus_jobs,
    _load_humanml3d_jobs,
    _load_kimodo_model,
    _resample_nearest,
    _resample_positions,
    _select_shard,
    _split_num_frames,
    _to_numpy_sequence,
)


DEFAULT_KIMODO_ARTIFACT = REPO / "checkpoints/kimodo/hftrainer_smplx_rp"
DEFAULT_KIMODO_MODEL_NAME = "Kimodo-SMPLX-RP-v1"
DIFFUSION_STEPS = 100

# Task families -------------------------------------------------------------
FRAME_SUBSET_TASKS = ("inbetween", "prediction", "backcast", "clip")
KEYFRAME_TASKS = ("keyframe",)
TRAJ_TASKS = ("traj_xz_dense", "traj_xz_sparse")
# Body-part settings that appear as KIMODO rows in tab_spatial_completion
# (coarse: upper/lower/arms; fine: spine_only/left|right arm|leg/feet_only/no_feet).
SPATIAL_PARTS = (
    "upper", "lower", "arms_only", "spine_only",
    "left_arm", "right_arm", "left_leg", "right_leg", "feet_only", "no_feet",
)
SPATIAL_TASKS = tuple(f"spatial_{p}" for p in SPATIAL_PARTS)
ALL_TASKS = FRAME_SUBSET_TASKS + KEYFRAME_TASKS + TRAJ_TASKS + SPATIAL_TASKS

# Control tasks (keyframe / trajectory / spatial) measure the *residual* control
# error of KIMODO's optimizer, so the observed frames/joints must NOT be hard
# imputed back to GT (that would zero the very error these tables report). Only
# the frame-subset temporal tasks use the hard-clean condition (matching ours).
CLEAN_CONDITION_DEFAULT = {t: True for t in FRAME_SUBSET_TASKS}
for _t in KEYFRAME_TASKS + TRAJ_TASKS + SPATIAL_TASKS:
    CLEAN_CONDITION_DEFAULT[_t] = False

_BONE22_PATH = REPO / "data/hymotion_m2m_data/bone_offsets_22.pt"


def _rotmat_to_rot6d_row_major_np(rotmat: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat, dtype=np.float32)
    col6d = np.concatenate([rotmat[..., 0:3, 0], rotmat[..., 0:3, 1]], axis=-1)
    return col6d[..., [0, 3, 1, 4, 2, 5]]


def _motion135_to_local_root(motion_135: np.ndarray, device: torch.device):
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32)).to(device)
    root = motion[:, :3]
    rot6d = motion[:, 3:135].reshape(len(motion), 22, 6)
    local = rot6d_to_rotmat_row_major(rot6d)
    return local, root


def _frame_indices_for_task(task: str, T: int) -> np.ndarray:
    """Return the GT condition frame indices for a frame-subset task."""
    keep_start = keep_end = 0
    if task == "inbetween":
        keep_start = keep_end = 1
    elif task == "prediction":
        keep_start = max(1, math.ceil(T * 0.20))
    elif task == "backcast":
        keep_end = max(1, math.ceil(T * 0.20))
    elif task == "clip":
        keep_start = max(1, math.ceil(T * 0.20))
        keep_end = max(1, math.ceil(T * 0.20))
    else:
        raise ValueError(f"unsupported frame-subset task: {task}")
    keep_start = max(0, min(keep_start, T))
    keep_end = max(0, min(keep_end, T - keep_start))
    frames: List[int] = []
    if keep_start > 0:
        frames.extend(range(keep_start))
    if keep_end > 0:
        frames.extend(range(T - keep_end, T))
    if not frames:
        frames = [0]
    return np.asarray(sorted(set(frames)), dtype=np.int64)


def _make_fullbody_with_rot_class():
    """FullBodyConstraintSet that ALSO pins global joint rotations at the
    observed frames (the base class only pins positions + smooth root + heading,
    which lets wrists/hands free-rotate at condition frames -> boundary jumps).
    Mirrors the SOMA harness' ``_make_fullbody_with_rot_constraint_set``."""
    from hftrainer.models.motion.kimodo.network.constraints import (
        FullBodyConstraintSet, create_pairs,
    )

    class FullBodyWithRotConstraintSet(FullBodyConstraintSet):
        def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
            super().update_constraints(data_dict, index_dict)
            joints = torch.arange(self.skeleton.nbjoints,
                                  device=self.frame_indices.device)
            indices = create_pairs(self.frame_indices, joints)
            data_dict["global_joints_rots"].append(
                self.global_joints_rots.reshape(-1, 3, 3))
            index_dict["global_joints_rots"].append(indices)

    return FullBodyWithRotConstraintSet


def _make_part_with_rot_class(kept_idx, anchor_root_xz: bool, anchor_heading: bool):
    """Subset analog of FullBodyWithRot for E10 body-part rotation control: pin
    ONLY ``kept_idx`` joints' global rotations (+ positions) over all frames,
    optionally anchoring the root xz / heading. Mirrors the SOMA harness'
    ``_make_part_with_rot_constraint_set`` (native SMPL-X joint indices, no SOMA
    remap)."""
    from hftrainer.models.motion.kimodo.network.constraints import (
        FullBodyConstraintSet, create_pairs,
    )
    _kept = torch.as_tensor(list(kept_idx), dtype=torch.long)
    _aroot = bool(anchor_root_xz)
    _ahead = bool(anchor_heading)

    class PartBodyWithRotConstraintSet(FullBodyConstraintSet):
        name = "partbody_rot"

        def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
            ki = _kept.to(self.frame_indices.device)
            pairs = create_pairs(self.frame_indices, ki)
            data_dict["global_joints_positions"].append(
                self.global_joints_positions[:, ki].reshape(-1, 3))
            index_dict["global_joints_positions"].append(pairs)
            data_dict["global_joints_rots"].append(
                self.global_joints_rots[:, ki].reshape(-1, 3, 3))
            index_dict["global_joints_rots"].append(pairs)
            if _aroot:
                data_dict["smooth_root_2d"].append(self.smooth_root_2d)
                index_dict["smooth_root_2d"].append(self.frame_indices)
                data_dict["root_y_pos"].append(self.root_y_pos)
                index_dict["root_y_pos"].append(self.frame_indices)
            if _ahead:
                data_dict["global_root_heading"].append(self.global_root_heading)
                index_dict["global_root_heading"].append(self.frame_indices)

    return PartBodyWithRotConstraintSet


def _fk_from_motion(model, gt_motion_135: np.ndarray):
    device = torch.device(model.device)
    skeleton = model.skeleton.to(device)
    local, root = _motion135_to_local_root(gt_motion_135, device)
    global_rots, positions, _ = skeleton.fk(local, root)
    return skeleton, device, local, root, global_rots, positions


def _build_task_constraint(task: str, model, gt_motion_135: np.ndarray,
                           bone_offsets: np.ndarray | None):
    """Dispatch the SMPL-X-native constraint set for ``task``.

    Returns ``(constraints, gt_local_np, gt_root_np, meta)`` where ``meta`` holds
    the per-task supervision used downstream (condition frame indices, the
    part-level mask, constrained coordinates, etc.).
    """
    from hftrainer.models.motion.kimodo.network.constraints import (
        FullBodyConstraintSet, Root2DConstraintSet,
    )
    skeleton, device, local, root, global_rots, positions = _fk_from_motion(
        model, gt_motion_135)
    T = int(len(gt_motion_135))
    meta: dict = {"task": task}

    def _fi(frames_np):
        return torch.from_numpy(np.asarray(frames_np, dtype=np.int64)).to(
            device=device, dtype=torch.long)

    if task in FRAME_SUBSET_TASKS:
        frame_idx_np = _frame_indices_for_task(task, T)
        fi = _fi(frame_idx_np)
        cons = FullBodyConstraintSet(
            skeleton, frame_indices=fi,
            global_joints_positions=positions[fi],
            global_joints_rots=global_rots[fi],
            smooth_root_2d=positions[fi, skeleton.root_idx, :][:, [0, 2]],
            to_crop=False)
        meta["condition_frames"] = frame_idx_np

    elif task in KEYFRAME_TASKS:
        from hftrainer.evaluation.motion.m2m_eval_tasks import (
            detect_keyframes_from_motion,
        )
        if bone_offsets is None:
            raise ValueError("keyframe task needs bone_offsets")
        kf = detect_keyframes_from_motion(
            np.asarray(gt_motion_135, dtype=np.float32),
            np.asarray(bone_offsets, dtype=np.float32),
            sparse=True, target_density=1.0 / 30.0, peak_distance=10)
        kf = sorted({int(f) for f in kf if 0 <= int(f) < T})
        if not kf:
            kf = [0, T - 1]
        frame_idx_np = np.asarray(kf, dtype=np.int64)
        fi = _fi(frame_idx_np)
        FullBodyWithRot = _make_fullbody_with_rot_class()
        cons = FullBodyWithRot(
            skeleton, frame_indices=fi,
            global_joints_positions=positions[fi],
            global_joints_rots=global_rots[fi],
            smooth_root_2d=positions[fi, skeleton.root_idx, :][:, [0, 2]],
            to_crop=False)
        meta["condition_frames"] = frame_idx_np

    elif task in TRAJ_TASKS:
        K = 30 if task.endswith("sparse") else 1
        frame_idx_np = np.asarray(list(range(0, T, K)), dtype=np.int64)
        fi = _fi(frame_idx_np)
        root_xz = positions[:, skeleton.root_idx, :][:, [0, 2]]
        cons = Root2DConstraintSet(
            skeleton, frame_indices=fi,
            smooth_root_2d=root_xz[fi], global_root_heading=None)
        meta["condition_frames"] = frame_idx_np
        meta["traj_coords"] = "xz"

    elif task in SPATIAL_TASKS:
        from hftrainer.evaluation.motion.m2m_eval_tasks import build_part_level_mask
        key = task.split("spatial_", 1)[1]
        m = np.asarray(build_part_level_mask(T=1, D=135, keep_part=key))[0]  # (135,) 0=keep
        transl_kept = bool(m[0:3].max() == 0)
        kept_smpl = [j for j in range(22)
                     if m[3 + 6 * j: 3 + 6 * (j + 1)].max() == 0]
        pelvis_kept = (0 in kept_smpl)
        if not kept_smpl:
            kept_smpl = [int(skeleton.root_idx)]
        anchor_root = transl_kept or (not transl_kept and not pelvis_kept)
        anchor_head = pelvis_kept or (not transl_kept and not pelvis_kept)
        frame_idx_np = np.arange(T, dtype=np.int64)
        fi = _fi(frame_idx_np)
        PartCS = _make_part_with_rot_class(kept_smpl, anchor_root, anchor_head)
        cons = PartCS(
            skeleton, frame_indices=fi,
            global_joints_positions=positions[fi],
            global_joints_rots=global_rots[fi],
            smooth_root_2d=positions[fi, skeleton.root_idx, :][:, [0, 2]])
        # full (T,135) part mask for compute_rotation_ctrl_error downstream.
        meta["part_mask_135"] = np.broadcast_to(m, (T, 135)).astype(np.float32)
        meta["kept_joints"] = np.asarray(kept_smpl, dtype=np.int64)
        meta["condition_frames"] = frame_idx_np  # all frames observed (subset of joints)
    else:
        raise ValueError(f"unsupported task: {task}")

    return [cons], local.detach().cpu().numpy(), root.detach().cpu().numpy(), meta


def _recompute_debug_arrays(model, local_np: np.ndarray, root_np: np.ndarray):
    device = torch.device(model.device)
    skeleton = model.skeleton.to(device)
    local = torch.from_numpy(np.asarray(local_np, dtype=np.float32)).to(device)
    root = torch.from_numpy(np.asarray(root_np, dtype=np.float32)).to(device)
    global_rots, posed, _ = skeleton.fk(local, root)
    return {
        "local_rot_mats": local.detach().cpu().numpy().astype(np.float32),
        "global_rot_mats": global_rots.detach().cpu().numpy().astype(np.float32),
        "root_positions": root.detach().cpu().numpy().astype(np.float32),
        "posed_joints": posed.detach().cpu().numpy().astype(np.float32),
    }


def _run_one(
    model,
    task: str,
    caption: str,
    gt_motion_135: np.ndarray,
    bone_offsets: np.ndarray | None,
    target_fps: float,
    cfg: float,
    postprocess: bool,
    *,
    force_clean_condition: bool = True,
    max_segment_frames: int | None = None,
):
    num_frames_30 = int(len(gt_motion_135))
    model_fps = float(model.fps)
    model_frames = max(10, int(round(num_frames_30 * model_fps / target_fps)))

    constraints, gt_local, gt_root, meta = _build_task_constraint(
        task, model, gt_motion_135, bone_offsets)
    frame_idx_np = meta.get("condition_frames", np.asarray([0], dtype=np.int64))
    seg_lens = _split_num_frames(model_frames, safe_len=max_segment_frames)
    is_multi = len(seg_lens) > 1
    if is_multi:
        # Frame-subset constraints index the full clip; keep one segment so the
        # constrained indices stay valid. (Test clips are <= a few hundred frames.)
        seg_lens = [model_frames]
        is_multi = False
    prompts = [caption]
    constraint_arg = [constraints]

    output = model(
        prompts,
        seg_lens,
        num_denoising_steps=DIFFUSION_STEPS,
        cfg_weight=[cfg, cfg],
        num_samples=1,
        return_numpy=True,
        multi_prompt=is_multi,
        constraint_lst=constraint_arg,
        post_processing=postprocess,
    )
    local = _to_numpy_sequence(output.get("local_rot_mats"))
    global_rot = _to_numpy_sequence(output.get("global_rot_mats"))
    root = _to_numpy_sequence(output.get("root_positions"))
    posed = _to_numpy_sequence(output.get("posed_joints"))
    if local is None:
        raise KeyError("KIMODO output has no local_rot_mats")
    if root is None:
        if posed is None:
            raise KeyError("KIMODO output has neither root_positions nor posed_joints")
        root = posed[:, 0]

    if abs(model_fps - target_fps) > 1e-6:
        local = _resample_nearest(local, num_frames_30)
        root = _resample_positions(root[:, None, :], num_frames_30)[:, 0]
        if global_rot is not None:
            global_rot = _resample_nearest(global_rot, num_frames_30)
        if posed is not None:
            posed = _resample_positions(posed, num_frames_30)

    local = _fit_length(local, num_frames_30)
    root = _fit_length(root, num_frames_30)

    # Hard-clean condition ONLY for frame-subset temporal tasks (matches ours'
    # hard-constraint interface). Control tasks (keyframe / traj / spatial) must
    # keep KIMODO's solved frames so the residual control error is measured.
    if force_clean_condition and task in FRAME_SUBSET_TASKS:
        local[frame_idx_np] = gt_local[frame_idx_np]
        root[frame_idx_np] = gt_root[frame_idx_np]
    payload = _recompute_debug_arrays(model, local, root)

    payload["motion_135"] = np.concatenate(
        [
            payload["root_positions"],
            _rotmat_to_rot6d_row_major_np(payload["local_rot_mats"]).reshape(num_frames_30, 132),
        ],
        axis=1,
    ).astype(np.float32)
    payload["_meta"] = meta
    return payload


def _load_gt_motion(gt_dir: Path, sid: str) -> np.ndarray:
    path = gt_dir / f"{sid}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return humanml272_to_motion135(np.load(str(path)).astype(np.float32))


def _load_ids_filter(path: str | None):
    if not path:
        return None
    ids = set()
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s:
            ids.add(s)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(ALL_TASKS))
    parser.add_argument("--humanml3d-272", default="data/evaluators/humanml3d_272")
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_KIMODO_ARTIFACT))
    parser.add_argument("--model-name", default=DEFAULT_KIMODO_MODEL_NAME)
    parser.add_argument("--out-dir", required=True, help="per-sid <sid>.npz output dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--corpus", default=None, help="JSONL id/prompt/length bank.")
    parser.add_argument("--ids", default=None, help="optional id whitelist file")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=100000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--max-segment-frames", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--no-force-clean-condition", action="store_true")
    parser.add_argument("--text-feature-cache-dir", default=None)
    parser.add_argument("--text-feature-namespace", default=None)
    parser.add_argument("--text-feature-encoder-id", default="LLM2VecEncoder")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h3d = Path(args.humanml3d_272)
    gt_dir = Path(args.gt_dir) if args.gt_dir else h3d / "motion_data"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bone_offsets = None
    if args.task in KEYFRAME_TASKS and _BONE22_PATH.exists():
        bone_offsets = torch.load(str(_BONE22_PATH), map_location="cpu").float().numpy()

    if args.corpus:
        all_jobs = _load_corpus_jobs(Path(args.corpus), args.min_len, args.max_len, args.max_samples)
    else:
        all_jobs = _load_humanml3d_jobs(h3d, args.min_len, args.max_len, args.max_samples)

    id_filter = _load_ids_filter(args.ids)
    if id_filter is not None:
        all_jobs = [j for j in all_jobs if j[0] in id_filter]

    jobs: List[Tuple[str, str, int]] = _select_shard(all_jobs, args.num_shards, args.shard_index)
    print(
        f"[setup] task={args.task} total={len(all_jobs)} "
        f"shard={args.shard_index}/{args.num_shards} jobs={len(jobs)}",
        flush=True,
    )

    use_feature_cache = bool(args.text_feature_cache_dir and args.text_feature_namespace)
    if use_feature_cache:
        os.environ["TEXT_ENCODER"] = "dummy"
        os.environ["TEXT_ENCODER_MODE"] = "local"

    global DIFFUSION_STEPS
    DIFFUSION_STEPS = int(args.diffusion_steps)
    model, bundle, kimodo_file = _load_kimodo_model(args, use_feature_cache=use_feature_cache)
    if use_feature_cache:
        model.text_encoder = CacheOnlyTextEncoder(
            namespace=args.text_feature_namespace,
            cache_dir=args.text_feature_cache_dir,
            encoder_id=args.text_feature_encoder_id,
            device=args.device,
        )
        print("[setup] using cached text features "
              f"{Path(args.text_feature_cache_dir) / args.text_feature_namespace}", flush=True)
    skeleton_type = type(model.skeleton).__name__
    print(f"[setup] KIMODO loaded fps={model.fps} skeleton={skeleton_type} "
          f"source={args.model_path or args.model_name}", flush=True)
    if "SMPL" not in skeleton_type.upper():
        raise RuntimeError(f"expected SMPLX KIMODO skeleton, got {skeleton_type}")

    ok = skipped = failed = 0
    for i, (sid, caption, length) in enumerate(jobs):
        out_file = out / f"{sid}.npz"
        if args.skip_existing and out_file.exists():
            skipped += 1
            continue
        try:
            gt_motion_135 = _load_gt_motion(gt_dir, sid)
            T = int(len(gt_motion_135))
            payload = _run_one(
                model,
                args.task,
                caption,
                gt_motion_135,
                bone_offsets,
                args.fps,
                args.cfg,
                args.postprocess,
                force_clean_condition=not args.no_force_clean_condition,
                max_segment_frames=args.max_segment_frames,
            )
            meta = payload["_meta"]
            frame_idx_np = np.asarray(meta.get("condition_frames", [0]), dtype=np.int64)
            pred_135 = payload["motion_135"]
            if not np.isfinite(pred_135).all() or pred_135.shape != (T, 135):
                raise ValueError(f"bad motion_135 shape/range: {pred_135.shape}")
            # src_mask: 1 = generated frame, 0 = observed (condition) frame. For
            # spatial joint-subset control every frame is "observed" (subset of
            # joints) so the frame-level mask is all-zero and the joint-level
            # supervision lives in part_mask_135 instead.
            src_mask = np.ones((T, 1), dtype=np.float32)
            if args.task in SPATIAL_TASKS:
                src_mask[:] = 0.0
            else:
                src_mask[frame_idx_np] = 0.0
            save_kwargs = dict(
                motion_135=pred_135.astype(np.float32),
                gt_motion_135=gt_motion_135.astype(np.float32),
                src_mask=src_mask,
                posed_joints=payload["posed_joints"][:, :22].astype(np.float32),
                condition_frames=frame_idx_np.astype(np.int32),
                caption=np.array(caption, dtype=object),
                sample_id=np.array(sid, dtype=object),
                task=np.array(args.task, dtype=object),
            )
            if "part_mask_135" in meta:
                save_kwargs["part_mask_135"] = meta["part_mask_135"].astype(np.float32)
                save_kwargs["kept_joints"] = meta["kept_joints"].astype(np.int32)
            if "traj_coords" in meta:
                save_kwargs["traj_coords"] = np.array(meta["traj_coords"], dtype=object)
            np.savez_compressed(str(out_file), **save_kwargs)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
            print(f"[progress] {i+1}/{len(jobs)} ok={ok} skipped={skipped} failed={failed}", flush=True)

    summary = {
        "task": args.task,
        "all_jobs": len(all_jobs),
        "jobs": len(jobs),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model_name": getattr(bundle, "resolved_model_name", args.model_name),
        "skeleton_type": skeleton_type,
        "cfg": args.cfg,
        "diffusion_steps": DIFFUSION_STEPS,
        "kimodo_import": kimodo_file,
    }
    (out / f"_summary_{args.task}_shard{args.shard_index}of{args.num_shards}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

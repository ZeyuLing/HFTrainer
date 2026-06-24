#!/usr/bin/env python3
"""Retarget KIMODO/SOMA outputs to SMPL ``motion_135``.

KIMODO exports native SOMA rotations in ``global_rot_mats``.  When those are
available this script uses the public :mod:`hftrainer.motion.retarget.smpl_soma`
rotation-transfer operator.  Position-only IK is only a fallback for legacy
files that do not contain SOMA rotations.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts/eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/eval"))

from hml263_to_smpl_ik import (  # noqa: E402
    N_JOINTS,
    estimate_local_rotations,
    load_smpl_rest,
    matrix_to_rot6d_rowmajor,
    refine_smpl_fit,
    smpl_forward_22,
)
from hftrainer.motion.retarget.smpl_soma import (  # noqa: E402
    KIMODOSOMAToSMPLRetargeter,
    SOMAToSMPLIKConfig,
)


FOOT_HEIGHT_JOINTS = np.array([7, 8, 10, 11], dtype=np.int64)

SOMA77_IDX = {
    "Neck2": 5,
    "Head": 6,
    "HeadEnd": 7,
    "Jaw": 8,
    "LeftEye": 9,
    "RightEye": 10,
    "LeftHand": 14,
    "LeftHandThumbEnd": 18,
    "LeftHandMiddleEnd": 28,
    "RightHand": 42,
    "RightHandThumbEnd": 46,
    "RightHandMiddleEnd": 56,
    "LeftToeBase": 70,
    "LeftToeEnd": 71,
    "RightToeBase": 75,
    "RightToeEnd": 76,
}


def _make_joint_fit_weights(preset: str) -> np.ndarray | None:
    if preset == "uniform":
        return None
    weights = np.ones(N_JOINTS, dtype=np.float32)
    if preset == "relaxed_torso":
        # SOMA and SMPL differ most around the upper spine/head landmarks.
        # Downweight those positional targets so the optimizer prefers a
        # plausible SMPL mesh over twisting the torso/neck to chase a few cm.
        for j in [6, 9, 12, 15]:  # spine2, spine3, neck, head
            weights[j] = 0.15
        for j in [3, 13, 14]:  # spine1 and collar joints
            weights[j] = 0.4
        return weights
    if preset == "relaxed_upper":
        # Diagnostic/visual retarget preset: keep pelvis/hips/legs reliable,
        # but stop upper-body SOMA/SMPL proportion mismatch from contorting the
        # SMPL torso, neck, head, and toe/hand leaves.
        for j in [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
            weights[j] = 0.08
        for j in [10, 11]:  # foot leaf/toe-base positions
            weights[j] = 0.25
        return weights
    raise ValueError(f"unknown joint fit weight preset: {preset}")


def _load_gmm_pose_prior(device: torch.device):
    prior_path = (
        PROJECT_ROOT
        / "ref_repo/FlowMDM/utils/visualize/joints2smpl/src/prior.py"
    )
    prior_folder = (
        PROJECT_ROOT
        / "ref_repo/FlowMDM/utils/visualize/joints2smpl/smpl_models"
    )
    spec = importlib.util.spec_from_file_location("flowmdm_joints2smpl_prior", prior_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prior module from {prior_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = module.MaxMixturePrior(
        prior_folder=str(prior_folder),
        num_gaussians=8,
        dtype=torch.float32,
    ).to(device)
    prior.eval()
    for param in prior.parameters():
        param.requires_grad_(False)
    return prior


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return target joints and caption from a KIMODO output file."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32), None, ""
    with np.load(path, allow_pickle=True) as data:
        if "positions" not in data.files:
            raise KeyError(f"{path} has no 'positions' key")
        soma77 = np.asarray(data["posed_joints"], dtype=np.float32) if "posed_joints" in data.files else None
        caption = ""
        if "caption" in data.files:
            try:
                caption = str(np.asarray(data["caption"]).item())
            except Exception:
                caption = str(data["caption"])
        return np.asarray(data["positions"], dtype=np.float32), soma77, caption


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = n[..., 0] > eps
    return v / np.maximum(n, eps), valid


def _append_guide(
    rest_vecs: list[np.ndarray],
    target_vecs: list[np.ndarray],
    weights: list[float],
    rest_vec: np.ndarray,
    target_vec: np.ndarray,
    weight: float,
) -> None:
    if weight <= 0:
        return
    if np.linalg.norm(target_vec) < 1e-6:
        return
    scale = max(float(np.linalg.norm(rest_vec)), 1e-3)
    rest_vecs.append(np.asarray(rest_vec, dtype=np.float64))
    target_vecs.append(np.asarray(target_vec, dtype=np.float64) / np.linalg.norm(target_vec) * scale)
    weights.append(float(weight))


def _add_soma77_orientation_guides(
    joint_index: int,
    soma: np.ndarray,
    rest_vecs: list[np.ndarray],
    target_vecs: list[np.ndarray],
    weights: list[float],
    head_weight: float,
    leaf_weight: float,
) -> None:
    """Add virtual children for SMPL leaves using SOMA77 end-effectors.

    A 22-joint body skeleton fixes leaf positions but not leaf orientation.
    Head, wrist, and foot mesh artifacts therefore remain invisible to MPJPE.
    SOMA77 provides extra points around those leaves; we use them only as weak
    orientation guides for the local rotation initializer.
    """
    if joint_index == 15:  # SMPL head
        head = soma[SOMA77_IDX["Head"]]
        eye_mid = 0.5 * (soma[SOMA77_IDX["LeftEye"]] + soma[SOMA77_IDX["RightEye"]])
        # SMPL rest frame is X-left, Y-up, Z-forward.
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, 0.08, 0.0]), soma[SOMA77_IDX["HeadEnd"]] - head, head_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.08, 0.0, 0.0]), soma[SOMA77_IDX["LeftEye"]] - soma[SOMA77_IDX["RightEye"]], head_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, 0.0, 0.08]), eye_mid - head, 0.5 * head_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, -0.05, 0.06]), soma[SOMA77_IDX["Jaw"]] - head, 0.35 * head_weight)
    elif joint_index == 20:  # left wrist
        hand = soma[SOMA77_IDX["LeftHand"]]
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.10, 0.0, 0.0]), soma[SOMA77_IDX["LeftHandMiddleEnd"]] - hand, leaf_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.06, -0.04, 0.04]), soma[SOMA77_IDX["LeftHandThumbEnd"]] - hand, 0.5 * leaf_weight)
    elif joint_index == 21:  # right wrist
        hand = soma[SOMA77_IDX["RightHand"]]
        _append_guide(rest_vecs, target_vecs, weights, np.array([-0.10, 0.0, 0.0]), soma[SOMA77_IDX["RightHandMiddleEnd"]] - hand, leaf_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([-0.06, -0.04, 0.04]), soma[SOMA77_IDX["RightHandThumbEnd"]] - hand, 0.5 * leaf_weight)
    elif joint_index == 10:  # left foot/toe
        _append_guide(
            rest_vecs,
            target_vecs,
            weights,
            np.array([0.0, 0.0, 0.12]),
            soma[SOMA77_IDX["LeftToeEnd"]] - soma[SOMA77_IDX["LeftToeBase"]],
            leaf_weight,
        )
    elif joint_index == 11:  # right foot/toe
        _append_guide(
            rest_vecs,
            target_vecs,
            weights,
            np.array([0.0, 0.0, 0.12]),
            soma[SOMA77_IDX["RightToeEnd"]] - soma[SOMA77_IDX["RightToeBase"]],
            leaf_weight,
        )


def estimate_local_rotations_with_soma77_guides(
    target_joints: np.ndarray,
    soma77: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
    head_guide_weight: float = 1.0,
    leaf_guide_weight: float = 0.35,
) -> np.ndarray:
    target_joints = np.asarray(target_joints, dtype=np.float64)
    soma77 = np.asarray(soma77, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_JOINTS], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(N_JOINTS)]
    for j in range(1, N_JOINTS):
        parent = int(parents[j])
        if 0 <= parent < N_JOINTS:
            children[parent].append(j)

    offsets = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for j in range(1, N_JOINTS):
        offsets[j] = rest_joints[j] - rest_joints[int(parents[j])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))
    for t, joints in enumerate(target_joints):
        soma = soma77[min(t, len(soma77) - 1)]
        for j in range(N_JOINTS):
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            rest_vecs = [offsets[c] for c in children[j]]
            target_vecs = [joints[c] - joints[j] for c in children[j]]
            weights = [1.0] * len(rest_vecs)
            if orientation_mode == "parent_frame" and parent >= 0:
                rest_vecs.append(rest_joints[parent] - rest_joints[j])
                target_vecs.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            _add_soma77_orientation_guides(
                j,
                soma,
                rest_vecs,
                target_vecs,
                weights,
                head_guide_weight,
                leaf_guide_weight,
            )
            if not rest_vecs:
                local[t, j] = np.eye(3)
                global_r[t, j] = parent_global @ local[t, j]
                continue
            rest_unit, rest_valid = _safe_normalize(np.stack(rest_vecs, axis=0))
            target_unit, target_valid = _safe_normalize(np.stack(target_vecs, axis=0))
            valid = rest_valid & target_valid
            if not np.any(valid):
                rot_local = np.eye(3)
            else:
                dst_local = (parent_global.T @ target_unit[valid].T).T
                try:
                    rot_local = R.align_vectors(
                        dst_local,
                        rest_unit[valid],
                        weights=np.asarray(weights, dtype=np.float64)[valid],
                    )[0].as_matrix()
                except Exception:
                    rot_local = np.eye(3)
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def _load_caption(debug_dir: Path | None, sid: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    if debug_dir is None:
        return ""
    path = debug_dir / f"{sid}.npz"
    if not path.exists():
        return ""
    try:
        with np.load(path, allow_pickle=True) as data:
            if "caption" in data.files:
                return str(np.asarray(data["caption"]).item())
    except Exception:
        return ""
    return ""


def _target_files(input_dir: Path, ids: Path | None, limit: int | None) -> list[Path]:
    ext_files = sorted(input_dir.glob("*.npz")) or sorted(input_dir.glob("*.npy"))
    if ids is not None:
        wanted = [ln.strip() for ln in ids.read_text().splitlines() if ln.strip()]
        suffix = ext_files[0].suffix if ext_files else ".npz"
        ext_files = [input_dir / f"{sid}{suffix}" for sid in wanted]
    files = [p for p in ext_files if p.exists()]
    if limit:
        files = files[:limit]
    return files


def _retarget_one(
    target: np.ndarray,
    soma77: np.ndarray | None,
    model,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    batch_size: int,
    device: torch.device,
    floor_align: bool,
    refine_iters: int,
    refine_lr: float,
    orientation_mode: str,
    parent_ref_weight: float,
    pose_l2_weight: float,
    angle_prior_weight: float,
    foot_height_align: bool = True,
    smooth_weight: float = 1e-3,
    joint_accel_weight: float = 0.0,
    joint_fit_weights: np.ndarray | None = None,
    gmm_pose_prior=None,
    gmm_pose_prior_weight: float = 0.0,
    soma_orientation_guides: bool = False,
    head_guide_weight: float = 1.0,
    leaf_guide_weight: float = 0.35,
) -> dict[str, np.ndarray]:
    if target.ndim != 3 or target.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected (T,{N_JOINTS},3), got {target.shape}")
    target = np.asarray(target, dtype=np.float32).copy()
    if floor_align:
        target[..., 1] -= float(target[..., 1].min())

    if soma_orientation_guides and soma77 is not None:
        if floor_align:
            soma77 = np.asarray(soma77, dtype=np.float32).copy()
            soma77[..., 1] -= float(soma77[..., 1].min())
        local_r = estimate_local_rotations_with_soma77_guides(
            target,
            soma77,
            rest_joints,
            parents,
            orientation_mode=orientation_mode,
            parent_ref_weight=parent_ref_weight,
            head_guide_weight=head_guide_weight,
            leaf_guide_weight=leaf_guide_weight,
        )
    else:
        local_r = estimate_local_rotations(
            target,
            rest_joints,
            parents,
            orientation_mode=orientation_mode,
            parent_ref_weight=parent_ref_weight,
        )
    aa = R.from_matrix(local_r.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(len(target), N_JOINTS, 3)
    global_orient = aa[:, 0]
    body_pose = aa[:, 1:].reshape(len(target), 63)

    joints_no_trans = smpl_forward_22(model, global_orient, body_pose, None, batch_size, device)
    transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)
    global_orient, body_pose, transl, fitted = refine_smpl_fit(
        model,
        target,
        global_orient,
        body_pose,
        transl,
        refine_iters,
        refine_lr,
        pose_l2_weight,
        angle_prior_weight,
        device,
        smooth_weight,
    )
    if foot_height_align:
        target_floor_y = target[:, FOOT_HEIGHT_JOINTS, 1].min(axis=1)
        fitted_floor_y = fitted[:, FOOT_HEIGHT_JOINTS, 1].min(axis=1)
        y_delta = (target_floor_y - fitted_floor_y).astype(np.float32)
        transl = transl.copy()
        fitted = fitted.copy()
        transl[:, 1] += y_delta
        fitted[..., 1] += y_delta[:, None]
    local_r = R.from_rotvec(
        np.concatenate(
            [global_orient[:, None, :], body_pose.reshape(len(target), 21, 3)],
            axis=1,
        ).reshape(-1, 3)
    ).as_matrix().reshape(len(target), N_JOINTS, 3, 3).astype(np.float32)
    motion_135 = np.concatenate(
        [transl, matrix_to_rot6d_rowmajor(local_r).reshape(len(target), N_JOINTS * 6)],
        axis=-1,
    ).astype(np.float32)
    mpjpe_mm = (np.linalg.norm(fitted - target, axis=-1).mean(axis=1) * 1000.0).astype(np.float32)
    return {
        "motion_135": motion_135,
        "transl": transl.astype(np.float32),
        "global_orient": global_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "target_joints": target.astype(np.float32),
        "fitted_joints": fitted.astype(np.float32),
        "fit_mpjpe_mm": mpjpe_mm,
    }


def _summarize_outputs(out_dir: Path) -> dict[str, object]:
    summary = []
    failed = 0
    for path in sorted(out_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=True) as data:
                motion = data["motion_135"]
                mpjpe = np.asarray(data["fit_mpjpe_mm"], dtype=np.float32)
            summary.append({
                "sid": path.stem,
                "frames": int(len(motion)),
                "mpjpe_mm_mean": float(mpjpe.mean()),
                "mpjpe_mm_p95": float(np.percentile(mpjpe, 95)),
            })
        except Exception:
            failed += 1
    return {
        "count": len(summary),
        "failed": failed,
        "mean_mpjpe_mm": float(np.mean([x["mpjpe_mm_mean"] for x in summary])) if summary else None,
        "median_mpjpe_mm": float(np.median([x["mpjpe_mm_mean"] for x in summary])) if summary else None,
        "p95_frame_mpjpe_mm_mean": float(np.mean([x["mpjpe_mm_p95"] for x in summary])) if summary else None,
        "items": summary[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default=str(
            PROJECT_ROOT
            / "outputs/evaluation/humanml3d_t2m_kimodo_20260605_genfix"
            / "kimodo_official/debug_npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "outputs/evaluation/humanml3d_smpl135_kimodo_20260605_genfix_ik"
            / "kimodo_skeleton_ik"
        ),
    )
    parser.add_argument("--debug-dir", default=None, help="Optional caption source when --in-dir is positions22.")
    parser.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--floor-align", action="store_true", default=True)
    parser.add_argument("--no-floor-align", dest="floor_align", action="store_false")
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--refine-lr", type=float, default=2e-2)
    parser.add_argument("--smooth-weight", type=float, default=1e-3)
    parser.add_argument("--joint-accel-weight", type=float, default=0.0)
    parser.add_argument("--gmm-pose-prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--joint-fit-weight-preset",
        choices=["uniform", "relaxed_torso", "relaxed_upper"],
        default="uniform",
    )
    parser.add_argument("--pose-l2-weight", type=float, default=0.0)
    parser.add_argument("--angle-prior-weight", type=float, default=0.0)
    parser.add_argument("--foot-height-align", action="store_true", default=True)
    parser.add_argument("--no-foot-height-align", dest="foot_height_align", action="store_false")
    parser.add_argument("--orientation-mode", choices=["bone", "parent_frame"], default="bone")
    parser.add_argument("--parent-ref-weight", type=float, default=0.25)
    parser.add_argument("--soma-orientation-guides", action="store_true")
    parser.add_argument("--head-guide-weight", type=float, default=1.0)
    parser.add_argument("--leaf-guide-weight", type=float, default=0.35)
    parser.add_argument(
        "--force-position-ik",
        action="store_true",
        help="Ignore global_rot_mats and use the legacy position-only IK path.",
    )
    parser.add_argument(
        "--smpl-height-mode",
        choices=["source_root", "foot_floor"],
        default="source_root",
        help="Height policy for the SOMA-rotation transfer path.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    device = torch.device(args.device)
    rotation_retargeter = KIMODOSOMAToSMPLRetargeter(
        SOMAToSMPLIKConfig(
            model_dir=args.model_dir,
            device=device,
            batch_size=args.batch_size,
            floor_align=args.floor_align,
            foot_height_align=args.foot_height_align,
            refine_iters=args.refine_iters,
            refine_lr=args.refine_lr,
            orientation_mode=args.orientation_mode,
            parent_ref_weight=args.parent_ref_weight,
            pose_l2_weight=args.pose_l2_weight,
            angle_prior_weight=args.angle_prior_weight,
            smooth_weight=args.smooth_weight,
            joint_accel_weight=args.joint_accel_weight,
            joint_fit_weight_preset=args.joint_fit_weight_preset,
            soma_orientation_guides=args.soma_orientation_guides,
            head_guide_weight=args.head_guide_weight,
            leaf_guide_weight=args.leaf_guide_weight,
            smpl_height_mode=args.smpl_height_mode,
        )
    )
    model, rest_joints, parents = rotation_retargeter.model, rotation_retargeter.rest_joints, rotation_retargeter.parents
    joint_fit_weights = _make_joint_fit_weights(args.joint_fit_weight_preset)
    gmm_pose_prior = _load_gmm_pose_prior(device) if args.gmm_pose_prior_weight > 0 else None
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    files = _target_files(in_dir, Path(args.ids) if args.ids else None, args.limit)
    if args.num_shards > 1:
        files = [path for idx, path in enumerate(files) if idx % args.num_shards == args.shard_index]
    print(
        f"[setup] files={len(files)} in={in_dir} out={out_dir} device={device} "
        f"refine_iters={args.refine_iters} shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )

    summary = []
    failed = 0
    for i, path in enumerate(files, 1):
        dst = out_dir / f"{path.stem}.npz"
        if args.skip_existing and dst.exists():
            continue
        try:
            target, soma77, caption = _load_target(path)
            has_soma_rotations = False
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=True) as data:
                    has_soma_rotations = "global_rot_mats" in data.files
            if has_soma_rotations and not args.force_position_ik:
                ret = rotation_retargeter.retarget_file(path)
            else:
                ret = _retarget_one(
                    target,
                    soma77,
                    model,
                    rest_joints,
                    parents,
                    args.batch_size,
                    device,
                    args.floor_align,
                    args.refine_iters,
                    args.refine_lr,
                    args.orientation_mode,
                    args.parent_ref_weight,
                    args.pose_l2_weight,
                    args.angle_prior_weight,
                    args.foot_height_align,
                    args.smooth_weight,
                    args.joint_accel_weight,
                    joint_fit_weights,
                    gmm_pose_prior,
                    args.gmm_pose_prior_weight,
                    args.soma_orientation_guides,
                    args.head_guide_weight,
                    args.leaf_guide_weight,
                )
                ret["retarget_method"] = np.array("legacy_position_ik", dtype=object)
            caption = _load_caption(debug_dir, path.stem, caption)
            np.savez_compressed(
                dst,
                **ret,
                caption=np.array(caption, dtype=object),
                source_id=np.array(path.stem, dtype=object),
                source_skeleton_path=np.array(str(path), dtype=object),
                source_fps=np.array(30.0, dtype=np.float32),
                target_fps=np.array(30.0, dtype=np.float32),
                refine_iters=np.array(args.refine_iters, dtype=np.int32),
                foot_height_align=np.array(args.foot_height_align, dtype=np.bool_),
                smooth_weight=np.array(args.smooth_weight, dtype=np.float32),
                joint_accel_weight=np.array(args.joint_accel_weight, dtype=np.float32),
                gmm_pose_prior_weight=np.array(args.gmm_pose_prior_weight, dtype=np.float32),
                joint_fit_weight_preset=np.array(args.joint_fit_weight_preset, dtype=object),
                soma_orientation_guides=np.array(args.soma_orientation_guides, dtype=np.bool_),
                head_guide_weight=np.array(args.head_guide_weight, dtype=np.float32),
                leaf_guide_weight=np.array(args.leaf_guide_weight, dtype=np.float32),
            )
            mpjpe = ret["fit_mpjpe_mm"]
            summary.append({
                "sid": path.stem,
                "frames": int(len(ret["motion_135"])),
                "mpjpe_mm_mean": float(mpjpe.mean()),
                "mpjpe_mm_p95": float(np.percentile(mpjpe, 95)),
            })
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 10:
                print(f"[fail] {path.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 250 == 0 or i == len(files):
            mean = np.mean([x["mpjpe_mm_mean"] for x in summary]) if summary else float("nan")
            print(f"  {i}/{len(files)} ok={len(summary)} fail={failed} mean_mpjpe_mm={mean:.2f}", flush=True)

    stats = _summarize_outputs(out_dir)
    stats["current_run_failed"] = failed
    (out_dir / "_retarget_summary.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] {json.dumps({k: v for k, v in stats.items() if k != 'items'}, indent=2)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit KIMODO/TMR position-only outputs to SMPL motion_135 with SMPLify3D.

KIMODO's ``kimodo-soma-rp`` model emits global SOMA joint positions but no
joint rotations.  The lightweight SOMA->SMPL path in
``build_kimodo_skeleton_smpl_ik_viewer.py`` therefore solves an under-
constrained position-only IK problem.  This script is a diagnostic alternative
that uses the released MDM/FlowMDM ``joints2smpl`` SMPLify3D objective: robust
joint fitting, GMM pose prior, angle prior, and previous-frame initialization.

The saved ``motion_135`` intentionally fixes betas to zero by default because
the 135-D representation has no shape slot and the downstream viewer/evaluator
renders with a neutral zero-shape body.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "ref_repo" / "FlowMDM"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.eval.hml263_to_smpl_ik import (  # noqa: E402
    N_JOINTS,
    matrix_to_rot6d,
)

from utils.visualize.joints2smpl.src import config as j2s_config  # noqa: E402

j2s_config.SMPL_MODEL_DIR = str(REPO / "ref_repo" / "MDM" / "body_models")
j2s_config.GMM_MODEL_DIR = str(
    REPO / "ref_repo" / "FlowMDM" / "utils" / "visualize" / "joints2smpl" / "smpl_models"
)
j2s_config.SMPL_MEAN_FILE = str(Path(j2s_config.GMM_MODEL_DIR) / "neutral_smpl_mean_params.h5")

import smplx  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from utils.visualize.joints2smpl.src.smplify import SMPLify3D  # noqa: E402


SOMA77_TO_SMPL22 = [
    0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75,
    4, 11, 39, 6, 12, 40, 13, 41, 14, 42,
]


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32), None, ""
    with np.load(path, allow_pickle=True) as data:
        caption = ""
        if "caption" in data.files:
            try:
                caption = str(np.asarray(data["caption"]).item())
            except Exception:
                caption = str(data["caption"])
        if "positions" in data.files:
            target = np.asarray(data["positions"], dtype=np.float32)
        elif "posed_joints" in data.files:
            target = np.asarray(data["posed_joints"], dtype=np.float32)[:, SOMA77_TO_SMPL22]
        else:
            raise KeyError(f"{path} has neither positions nor posed_joints")
        soma77 = np.asarray(data["posed_joints"], dtype=np.float32) if "posed_joints" in data.files else None
    return target, soma77, caption


def _iter_files(in_dir: Path, ids: Path | None, limit: int | None) -> Iterable[Path]:
    files = sorted(in_dir.glob("*.npz")) or sorted(in_dir.glob("*.npy"))
    if ids is not None:
        wanted = [line.strip() for line in ids.read_text().splitlines() if line.strip()]
        suffix = files[0].suffix if files else ".npz"
        files = [in_dir / f"{sid}{suffix}" for sid in wanted]
    files = [p for p in files if p.exists()]
    return files[:limit] if limit else files


def _load_mean_pose(device: torch.device) -> torch.Tensor:
    with h5py.File(j2s_config.SMPL_MEAN_FILE, "r") as f:
        mean_pose = torch.from_numpy(f["pose"][:]).unsqueeze(0).float().to(device)
    return mean_pose


def _smooth_positions(target: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(target) < 3:
        return target.astype(np.float32)
    if window % 2 == 0:
        window += 1
    window = min(window, len(target) if len(target) % 2 == 1 else len(target) - 1)
    if window < 3:
        return target.astype(np.float32)
    try:
        from scipy.signal import savgol_filter
        flat = target.reshape(len(target), -1)
        return savgol_filter(flat, window_length=window, polyorder=2, axis=0).reshape(target.shape).astype(np.float32)
    except Exception:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        flat = np.pad(target.reshape(len(target), -1), ((pad, pad), (0, 0)), mode="edge")
        out = np.stack([
            (flat[i:i + window] * kernel[:, None]).sum(axis=0)
            for i in range(len(target))
        ], axis=0)
        return out.reshape(target.shape).astype(np.float32)


def _smpl_forward_22(model, pose72: np.ndarray, transl: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    for i in range(len(pose72)):
        pose = torch.from_numpy(pose72[i:i + 1]).float().to(device)
        tr = torch.from_numpy(transl[i:i + 1]).float().to(device)
        with torch.no_grad():
            out = model(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                transl=tr,
                betas=torch.zeros(1, 10, device=device),
            )
        chunks.append(out.joints[:, :N_JOINTS].detach().cpu().numpy()[0])
    return np.stack(chunks, axis=0).astype(np.float32)


def _confidence(preset: str, device: torch.device) -> torch.Tensor:
    conf = torch.ones(N_JOINTS, dtype=torch.float32, device=device)
    if preset == "official":
        return conf
    if preset == "fix_foot":
        conf[[7, 8, 10, 11]] = 1.5
        return conf
    if preset == "relaxed_head":
        conf[[12, 15]] = 0.35
        conf[[13, 14]] = 0.6
        return conf
    raise ValueError(f"unknown confidence preset: {preset}")


def retarget_one(
    in_path: Path,
    out_path: Path,
    model,
    smplify: SMPLify3D,
    mean_pose: torch.Tensor,
    device: torch.device,
    floor_align: bool,
    smooth_target_window: int,
    max_frames_per_clip: int | None,
    confidence_preset: str,
    optimize_shape_first_frame: bool,
) -> dict[str, object]:
    target, soma77, caption = _load_target(in_path)
    if target.ndim != 3 or target.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected (T,{N_JOINTS},3), got {target.shape}")
    target = target.astype(np.float32)
    if max_frames_per_clip is not None and max_frames_per_clip > 0:
        target = target[:max_frames_per_clip]
        if soma77 is not None:
            soma77 = soma77[:max_frames_per_clip]
    if smooth_target_window > 1:
        target = _smooth_positions(target, smooth_target_window)
    if floor_align:
        target = target.copy()
        target[..., 1] -= float(target[..., 1].min())

    confidence = _confidence(confidence_preset, device)
    pred_pose = mean_pose.clone()
    pred_betas = torch.zeros(1, 10, dtype=torch.float32, device=device)
    pred_cam_t = torch.zeros(1, 3, dtype=torch.float32, device=device)

    poses = []
    trans = []
    for idx, joints_np in enumerate(target):
        keypoints = torch.from_numpy(joints_np[None]).float().to(device)
        seq_ind = idx if optimize_shape_first_frame else idx + 1
        _, _, opt_pose, _opt_betas, opt_cam_t, _ = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints,
            conf_3d=confidence,
            seq_ind=seq_ind,
        )
        pred_pose = opt_pose.detach().float()
        pred_cam_t = opt_cam_t.detach().reshape(1, 3).float()
        if optimize_shape_first_frame and idx == 0:
            pred_betas = _opt_betas.detach().float()
        poses.append(pred_pose.detach().cpu().numpy()[0])
        trans.append(pred_cam_t.detach().cpu().numpy()[0])

    pose72 = np.stack(poses, axis=0).astype(np.float32)
    transl = np.stack(trans, axis=0).astype(np.float32)
    fitted = _smpl_forward_22(model, pose72, transl, device)
    mpjpe_mm = np.linalg.norm(fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0

    global_orient = pose72[:, :3]
    body_pose_21 = pose72[:, 3:66]
    local_r = R.from_rotvec(
        np.concatenate([global_orient[:, None], body_pose_21.reshape(len(target), 21, 3)], axis=1)
        .reshape(-1, 3)
    ).as_matrix().reshape(len(target), N_JOINTS, 3, 3).astype(np.float32)
    motion_135 = np.concatenate(
        [transl, matrix_to_rot6d(local_r, "row").reshape(len(target), N_JOINTS * 6)],
        axis=-1,
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        motion_135=motion_135,
        transl=transl.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose_21.astype(np.float32),
        target_joints=target.astype(np.float32),
        fitted_joints=fitted.astype(np.float32),
        fit_mpjpe_mm=mpjpe_mm.astype(np.float32),
        posed_joints=soma77.astype(np.float32) if soma77 is not None else np.zeros((0, 77, 3), dtype=np.float32),
        caption=np.array(caption, dtype=object),
        source_id=np.array(in_path.stem, dtype=object),
        source_skeleton_path=np.array(str(in_path), dtype=object),
        source_fps=np.array(30.0, dtype=np.float32),
        target_fps=np.array(30.0, dtype=np.float32),
        backend=np.array("flowmdm_smplify3d", dtype=object),
        smooth_target_window=np.array(smooth_target_window, dtype=np.int32),
        confidence_preset=np.array(confidence_preset, dtype=object),
        optimize_shape_first_frame=np.array(optimize_shape_first_frame, dtype=np.bool_),
    )
    return {
        "sid": in_path.stem,
        "frames": int(len(target)),
        "mpjpe_mm_mean": float(mpjpe_mm.mean()),
        "mpjpe_mm_p95": float(np.percentile(mpjpe_mm, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-smplify-iters", type=int, default=20)
    parser.add_argument("--floor-align", action="store_true", default=True)
    parser.add_argument("--no-floor-align", dest="floor_align", action="store_false")
    parser.add_argument("--smooth-target-window", type=int, default=0)
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=0,
        help="Diagnostic-only cap; 0 keeps full clips.",
    )
    parser.add_argument(
        "--confidence-preset",
        choices=["official", "fix_foot", "relaxed_head"],
        default="official",
    )
    parser.add_argument("--optimize-shape-first-frame", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = smplx.create(
        j2s_config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device)
    model.eval()
    smplify = SMPLify3D(
        smplxmodel=model,
        batch_size=1,
        joints_category="AMASS",
        num_iters=args.num_smplify_iters,
        device=device,
    )
    mean_pose = _load_mean_pose(device)
    files = list(_iter_files(Path(args.in_dir), Path(args.ids) if args.ids else None, args.limit))
    print(
        f"[setup] files={len(files)} out={out_dir} device={device} "
        f"iters={args.num_smplify_iters} conf={args.confidence_preset}",
        flush=True,
    )

    summary = []
    failed = 0
    for i, in_path in enumerate(files, 1):
        out_path = out_dir / f"{in_path.stem}.npz"
        if args.skip_existing and out_path.exists():
            continue
        try:
            item = retarget_one(
                in_path,
                out_path,
                model,
                smplify,
                mean_pose,
                device,
                args.floor_align,
                args.smooth_target_window,
                args.max_frames_per_clip if args.max_frames_per_clip > 0 else None,
                args.confidence_preset,
                args.optimize_shape_first_frame,
            )
            summary.append(item)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 20:
                print(f"[fail] {in_path.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 4 == 0 or i == len(files):
            mean = np.mean([x["mpjpe_mm_mean"] for x in summary]) if summary else float("nan")
            print(f"[progress] {i}/{len(files)} ok={len(summary)} fail={failed} mean_mpjpe_mm={mean:.2f}", flush=True)

    stats = {
        "count": len(summary),
        "failed": failed,
        "mean_mpjpe_mm": float(np.mean([x["mpjpe_mm_mean"] for x in summary])) if summary else None,
        "median_mpjpe_mm": float(np.median([x["mpjpe_mm_mean"] for x in summary])) if summary else None,
        "p95_frame_mpjpe_mm_mean": float(np.mean([x["mpjpe_mm_p95"] for x in summary])) if summary else None,
        "items": summary[:100],
    }
    (out_dir / "_retarget_summary.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] {json.dumps({k: v for k, v in stats.items() if k != 'items'}, indent=2)}", flush=True)


if __name__ == "__main__":
    main()

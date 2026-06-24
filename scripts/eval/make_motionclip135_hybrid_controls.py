#!/usr/bin/env python3
"""Build MotionCLIP-135 hybrid controls from retargeted HML263 SMPL fits.

This diagnostic separates the effect of root translation from SMPL pose:

  pred_trans_gt_rot  : retargeted translation + real SMPL rotations
  gt_trans_pred_rot  : real SMPL translation + retargeted rotations
  pred_root_gt_body  : retargeted translation/root orient + real body pose
  gt_root_pred_body  : real translation/root orient + retargeted body pose

Outputs are annotation-key ``.npy`` files consumable by
``eval_with_motionclip_evaluator.py --pred_dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)


def _yaw_align(
    transl: np.ndarray,
    global_orient: np.ndarray,
    gt_transl: np.ndarray,
    gt_global_orient: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    pred_go = torch.from_numpy(global_orient.astype(np.float32)).reshape(-1, 3)
    gt_go0 = torch.from_numpy(gt_global_orient.astype(np.float32)[:1]).reshape(1, 3)
    pred_mat = axis_angle_to_matrix(pred_go)
    gt_mat0 = axis_angle_to_matrix(gt_go0)[0]
    if mode == "full":
        delta = gt_mat0 @ pred_mat[0].transpose(0, 1)
    elif mode == "yaw":
        def yaw_from_mat(mat: torch.Tensor) -> torch.Tensor:
            fwd = mat[:, 2]
            return torch.atan2(fwd[0], fwd[2])

        yaw = yaw_from_mat(gt_mat0) - yaw_from_mat(pred_mat[0])
        c, s = torch.cos(yaw), torch.sin(yaw)
        z = torch.zeros_like(c)
        o = torch.ones_like(c)
        delta = torch.stack([
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ])
    else:
        raise ValueError(f"unknown align mode {mode!r}")
    aligned_mat = delta[None] @ pred_mat
    aligned_go = matrix_to_axis_angle(aligned_mat).numpy().astype(np.float32)
    tr = torch.from_numpy(transl.astype(np.float32))
    gt_tr0 = torch.from_numpy(gt_transl.astype(np.float32)[0])
    aligned_tr = ((delta @ (tr - tr[0]).T).T + gt_tr0).numpy().astype(np.float32)
    return aligned_tr, aligned_go


def _to_motion135(transl: np.ndarray, global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    t = int(min(len(transl), len(global_orient), len(body_pose)))
    transl = transl[:t].astype(np.float32)
    go = torch.from_numpy(global_orient[:t].astype(np.float32)).reshape(t, 3)
    bp = torch.from_numpy(body_pose[:t].astype(np.float32)).reshape(t, 21, 3)
    go6 = matrix_to_rotation_6d(axis_angle_to_matrix(go)).numpy().reshape(t, 6)
    bp6 = matrix_to_rotation_6d(axis_angle_to_matrix(bp)).numpy().reshape(t, 126)
    return np.concatenate([transl, go6, bp6], axis=-1).astype(np.float32)


def _load_gt_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(str(path), allow_pickle=True)
    transl = np.asarray(z["transl"], dtype=np.float32)
    global_orient = np.asarray(z["global_orient"], dtype=np.float32).reshape(len(transl), 3)
    body_pose = np.asarray(z["body_pose"], dtype=np.float32).reshape(len(transl), -1)[:, :63]
    return transl, global_orient, body_pose


def _load_pred_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(str(path), allow_pickle=True)
    transl = np.asarray(z["transl"], dtype=np.float32)
    global_orient = np.asarray(z["global_orient"], dtype=np.float32).reshape(len(transl), 3)
    body_pose = np.asarray(z["body_pose"], dtype=np.float32).reshape(len(transl), 63)
    return transl, global_orient, body_pose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", required=True)
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument("--pred-smpl-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--include-mirrors", action="store_true")
    ap.add_argument("--key-fallback", action="store_true")
    ap.add_argument("--align-mode", choices=["yaw", "full"], default="yaw")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    anno = json.loads(Path(args.anno_file).read_text())["data_list"]
    pred_dir = Path(args.pred_smpl_dir)
    out_root = Path(args.out_root)
    modes = [
        "pred_trans_gt_rot",
        "gt_trans_pred_rot",
        "pred_root_gt_body",
        "gt_root_pred_body",
    ]
    for mode in modes:
        (out_root / mode).mkdir(parents=True, exist_ok=True)

    written = {mode: 0 for mode in modes}
    missing = 0
    for i, (name, entry) in enumerate(anno.items()):
        if args.limit is not None and i >= args.limit:
            break
        cid = Path(str(entry.get("smplx_path") or "")).stem
        if cid.startswith("M") and not args.include_mirrors:
            continue
        pred_path = pred_dir / f"{cid}.npz"
        if args.key_fallback and not pred_path.exists():
            pred_path = pred_dir / f"{name}.npz"
        gt_rel = entry.get("smplx_path")
        gt_path = Path(args.data_dir) / gt_rel if gt_rel else None
        if not pred_path.exists() or gt_path is None or not gt_path.exists():
            missing += 1
            continue

        p_tr, p_go, p_bp = _load_pred_npz(pred_path)
        g_tr, g_go, g_bp = _load_gt_npz(gt_path)
        p_tr, p_go = _yaw_align(p_tr, p_go, g_tr, g_go, args.align_mode)
        t = min(len(p_tr), len(g_tr), len(p_go), len(g_go), len(p_bp), len(g_bp))
        combos = {
            "pred_trans_gt_rot": (p_tr[:t], g_go[:t], g_bp[:t]),
            "gt_trans_pred_rot": (g_tr[:t], p_go[:t], p_bp[:t]),
            "pred_root_gt_body": (p_tr[:t], p_go[:t], g_bp[:t]),
            "gt_root_pred_body": (g_tr[:t], g_go[:t], p_bp[:t]),
        }
        for mode, parts in combos.items():
            np.save(out_root / mode / f"{name}.npy", _to_motion135(*parts))
            written[mode] += 1

    print(json.dumps({"written": written, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()

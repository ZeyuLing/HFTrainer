#!/usr/bin/env python3
"""Create annotation-key 135D files for MotionCLIP evaluation.

The HML263 retargeter stores ``motion_135`` in row-major 6D order so that the
MotionStreamer-272 FK path decodes it correctly.  MotionCLIP uses the standard
``matrix_to_rotation_6d`` convention, so for MotionCLIP we rebuild 135D from the
saved axis-angle SMPL fields and write annotation-key ``.npy`` files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


def _align_root_frame(
    transl: np.ndarray,
    global_orient: np.ndarray,
    gt_path: Path | None,
    mode: str = "yaw",
) -> tuple[np.ndarray, np.ndarray]:
    if gt_path is None:
        return transl, global_orient
    gt = np.load(str(gt_path), allow_pickle=True)
    if "transl" not in gt.files or "global_orient" not in gt.files:
        return transl, global_orient
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
        matrix_to_axis_angle,
    )

    pred_go = torch.from_numpy(np.asarray(global_orient, dtype=np.float32)).reshape(-1, 3)
    gt_go0 = torch.from_numpy(np.asarray(gt["global_orient"], dtype=np.float32)[:1]).reshape(1, 3)
    pred_mat = axis_angle_to_matrix(pred_go)
    gt_mat0 = axis_angle_to_matrix(gt_go0)[0]
    if mode == "full":
        delta = gt_mat0 @ pred_mat[0].transpose(0, 1)
    else:
        def yaw_from_mat(mat: torch.Tensor) -> torch.Tensor:
            fwd = mat[:, 2]
            return torch.atan2(fwd[0], fwd[2])

        yaw = yaw_from_mat(gt_mat0) - yaw_from_mat(pred_mat[0])
        c, s = torch.cos(yaw), torch.sin(yaw)
        delta = torch.stack([
            torch.stack([c, torch.zeros_like(c), s]),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
            torch.stack([-s, torch.zeros_like(c), c]),
        ])
    aligned_mat = delta[None] @ pred_mat
    aligned_go = matrix_to_axis_angle(aligned_mat).numpy().astype(np.float32)
    tr = torch.from_numpy(np.asarray(transl, dtype=np.float32))
    gt_tr0 = torch.from_numpy(np.asarray(gt["transl"], dtype=np.float32)[0])
    aligned_tr = ((delta @ (tr - tr[0]).T).T + gt_tr0).numpy().astype(np.float32)
    return aligned_tr, aligned_go


def smpl_npz_to_motionclip135(
    path: Path,
    gt_path: Path | None = None,
    align_mode: str = "yaw",
) -> np.ndarray:
    z = np.load(str(path), allow_pickle=True)
    transl = np.asarray(z["transl"], dtype=np.float32)
    t = transl.shape[0]
    global_orient = np.asarray(z["global_orient"], dtype=np.float32).reshape(t, 3)
    transl, global_orient = _align_root_frame(transl, global_orient, gt_path, align_mode)
    go = torch.from_numpy(global_orient).reshape(t, 3)
    bp = torch.from_numpy(np.asarray(z["body_pose"], dtype=np.float32)).reshape(t, 21, 3)
    go6 = matrix_to_rotation_6d(axis_angle_to_matrix(go)).numpy().reshape(t, 6)
    bp6 = matrix_to_rotation_6d(axis_angle_to_matrix(bp)).numpy().reshape(t, 126)
    out = np.concatenate([transl, go6, bp6], axis=-1).astype(np.float32)
    if out.shape[-1] != 135:
        raise ValueError(f"{path}: expected 135D, got {out.shape}")
    return out


def _worker(task: tuple[str, str, str | None, str, bool]) -> tuple[str, str]:
    src, dst, gt_path, align_mode, overwrite = task
    dst_path = Path(dst)
    if dst_path.exists() and not overwrite:
        return "skip", src
    try:
        np.save(
            dst_path,
            smpl_npz_to_motionclip135(
                Path(src),
                Path(gt_path) if gt_path else None,
                align_mode=align_mode,
            ),
        )
        return "ok", src
    except Exception as exc:  # noqa: BLE001
        return f"fail:{type(exc).__name__}:{exc}", src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-mirrors", action="store_true")
    ap.add_argument("--align-to-gt-root", action="store_true",
                    help="Rigidly align prediction root translation/orientation to the GT first frame.")
    ap.add_argument("--align-root-mode", choices=["yaw", "full"], default="yaw")
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument(
        "--key-fallback",
        action="store_true",
        help="If smplx_path stem is missing, also try an NPZ named by the annotation key.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    anno = json.loads(Path(args.anno_file).read_text())["data_list"]
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped = missing = 0
    for name, entry in anno.items():
        cid = Path(str(entry.get("smplx_path") or "")).stem
        if not cid:
            missing += 1
            continue
        if cid.startswith("M") and not args.include_mirrors:
            skipped += 1
            continue
        src = src_dir / f"{cid}.npz"
        if args.key_fallback and not src.exists():
            src = src_dir / f"{name}.npz"
        if not src.exists():
            missing += 1
            continue
        dst = out_dir / f"{name}.npy"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        gt_path = None
        if args.align_to_gt_root:
            rel = entry.get("smplx_path")
            if rel:
                gt_path = str((Path(args.data_dir) / rel).resolve())
        tasks.append((str(src), str(dst), gt_path, args.align_root_mode, args.overwrite))

    written = failed = 0
    n_workers = max(1, args.workers)
    if n_workers == 1:
        iterator = map(_worker, tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_workers)
        iterator = pool.imap_unordered(_worker, tasks, chunksize=16)

    for i, (status, src) in enumerate(iterator, 1):
        if status == "ok":
            written += 1
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
            if failed <= 10:
                print(f"[fail] {src}: {status}", flush=True)
        if i % 500 == 0:
            print(
                f"[progress] {i}/{len(tasks)} written={written} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )
    if n_workers > 1:
        pool.close()
        pool.join()

    print(
        f"[done] src={src_dir} out={out_dir} written={written} "
        f"skipped={skipped} missing={missing} failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()

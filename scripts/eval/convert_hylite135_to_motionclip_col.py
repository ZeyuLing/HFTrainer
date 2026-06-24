#!/usr/bin/env python3
"""Convert HYMotion-Lite 135D outputs for MotionCLIP T2M evaluation.

HYMotion-Lite writes SMPL-22 135D motions as
``transl + 22 * rotation_6d`` but its 6D rotations follow the row-major layout.
The MotionCLIP evaluator used in the PRISM paper is validated with the standard
column-major 6D convention.  This script converts the convention and can align
the prediction's first-frame root to the GT sample frame, matching the protocol
used for trusted MotionStreamer T2M cross-evaluation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _load_data_list(anno_file: Path) -> dict:
    raw = json.loads(anno_file.read_text())
    data = raw["data_list"] if isinstance(raw, dict) and "data_list" in raw else raw
    if not isinstance(data, dict):
        raise ValueError(f"expected dict or data_list dict in {anno_file}")
    return data


def _yaw_delta(gt_mat0: torch.Tensor, pred_mat0: torch.Tensor) -> torch.Tensor:
    def yaw_from_mat(mat: torch.Tensor) -> torch.Tensor:
        fwd = mat[:, 2]
        return torch.atan2(fwd[0], fwd[2])

    yaw = yaw_from_mat(gt_mat0) - yaw_from_mat(pred_mat0)
    c, s = torch.cos(yaw), torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ]
    )


def _align_root_to_gt(
    transl: np.ndarray,
    rot_mats: np.ndarray,
    gt_path: Path | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if gt_path is None or not gt_path.exists():
        return transl, rot_mats
    gt = np.load(str(gt_path), allow_pickle=True)
    if "transl" not in gt.files or "global_orient" not in gt.files:
        return transl, rot_mats

    pred_root0 = torch.from_numpy(np.asarray(rot_mats[0, 0], dtype=np.float32))
    gt_go0 = torch.from_numpy(np.asarray(gt["global_orient"], dtype=np.float32)[:1]).reshape(1, 3)
    gt_mat0 = axis_angle_to_matrix(gt_go0)[0]
    if mode == "full":
        delta = gt_mat0 @ pred_root0.transpose(0, 1)
    else:
        delta = _yaw_delta(gt_mat0, pred_root0)

    out_rot = np.asarray(rot_mats, dtype=np.float32).copy()
    out_rot[:, 0] = np.matmul(delta.numpy()[None], out_rot[:, 0]).astype(np.float32)

    tr = torch.from_numpy(np.asarray(transl, dtype=np.float32))
    gt_tr0 = torch.from_numpy(np.asarray(gt["transl"], dtype=np.float32)[0])
    out_tr = ((delta @ (tr - tr[0]).T).T + gt_tr0).numpy().astype(np.float32)
    return out_tr, out_rot


def convert_motion(
    src: Path,
    gt_path: Path | None,
    align_to_gt_root: bool,
    align_root_mode: str,
) -> np.ndarray:
    arr = np.asarray(np.load(str(src), allow_pickle=True), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 135:
        raise ValueError(f"{src}: expected (T,135), got {arr.shape}")

    transl = arr[:, :3]
    rot6d_row = torch.from_numpy(arr[:, 3:].reshape(arr.shape[0], 22, 6))
    rot_mats = rotation_6d_to_matrix(rot6d_row, convention="row").numpy().astype(np.float32)
    if align_to_gt_root:
        transl, rot_mats = _align_root_to_gt(
            transl,
            rot_mats,
            gt_path=gt_path,
            mode=align_root_mode,
        )
    rot6d_col = matrix_to_rotation_6d(
        torch.from_numpy(rot_mats),
        convention="column",
    ).numpy().reshape(arr.shape[0], 22 * 6)
    out = np.concatenate([transl.astype(np.float32), rot6d_col.astype(np.float32)], axis=-1)
    if out.shape[-1] != 135:
        raise ValueError(f"{src}: bad converted shape {out.shape}")
    return out.astype(np.float32)


def _source_candidates(src_dir: Path, name: str, entry: dict) -> list[Path]:
    candidates = [src_dir / f"{name}.npy"]
    smplx_path = entry.get("smplx_path")
    if smplx_path:
        candidates.append(src_dir / f"{Path(str(smplx_path)).stem}.npy")
    return candidates


def _worker(task: tuple[str, str, str | None, bool, str, bool]) -> tuple[str, str]:
    src_s, dst_s, gt_s, align_to_gt_root, align_root_mode, overwrite = task
    dst = Path(dst_s)
    if dst.exists() and not overwrite:
        return "skip", src_s
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out = convert_motion(
            Path(src_s),
            Path(gt_s) if gt_s else None,
            align_to_gt_root=align_to_gt_root,
            align_root_mode=align_root_mode,
        )
        np.save(str(dst), out)
        return "ok", src_s
    except Exception as exc:  # noqa: BLE001
        return f"fail:{type(exc).__name__}:{exc}", src_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--anno-file", required=True)
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument("--align-to-gt-root", action="store_true")
    ap.add_argument("--align-root-mode", choices=["yaw", "full"], default="yaw")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    data_dir = Path(args.data_dir)
    data_list = _load_data_list(Path(args.anno_file))

    tasks = []
    missing = skipped = 0
    for name, entry in data_list.items():
        src = next((p for p in _source_candidates(src_dir, name, entry) if p.exists()), None)
        if src is None:
            missing += 1
            continue
        dst = out_dir / f"{name}.npy"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        gt_path = None
        if args.align_to_gt_root and entry.get("smplx_path"):
            gt_path = str((data_dir / entry["smplx_path"]).resolve())
        tasks.append(
            (
                str(src),
                str(dst),
                gt_path,
                args.align_to_gt_root,
                args.align_root_mode,
                args.overwrite,
            )
        )
        if args.max_samples is not None and len(tasks) >= args.max_samples:
            break

    print(
        f"[start] src={src_dir} out={out_dir} tasks={len(tasks)} "
        f"missing={missing} skipped={skipped} align={args.align_to_gt_root}:{args.align_root_mode}",
        flush=True,
    )

    ok = fail = 0
    failures: list[tuple[str, str]] = []
    n_workers = max(1, args.workers)
    if n_workers == 1:
        iterator = map(_worker, tasks)
        pool = None
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_workers)
        iterator = pool.imap_unordered(_worker, tasks, chunksize=16)
    try:
        for i, (status, src) in enumerate(iterator, 1):
            if status in {"ok", "skip"}:
                ok += int(status == "ok")
                skipped += int(status == "skip")
            else:
                fail += 1
                failures.append((src, status))
                if fail <= 10:
                    print(f"[fail] {src}: {status}", flush=True)
            if i % 500 == 0:
                print(f"[progress] {i}/{len(tasks)} ok={ok} skipped={skipped} fail={fail}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    summary = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "anno_file": args.anno_file,
        "tasks": len(tasks),
        "written": ok,
        "skipped": skipped,
        "missing": missing,
        "failed": fail,
        "align_to_gt_root": args.align_to_gt_root,
        "align_root_mode": args.align_root_mode,
        "failures": failures[:50],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

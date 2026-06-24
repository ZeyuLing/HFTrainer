#!/usr/bin/env python3
"""Prepare MotionStreamer-272 predictions for T2M cross-evaluation.

Input files are canonical HumanML3D-id ``.npy``/``.npz`` clips with shape
``(T, 272)``.  This writes:

* canonical-id ``.npz`` files with key ``motion_272`` for
  ``eval_motionstreamer_272.py``;
* annotation-key ``.npy`` files with column-major 135D SMPL-22 rotations for
  ``eval_with_motionclip_evaluator.py``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load_m272(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        z = np.load(str(path), allow_pickle=True)
        if "motion_272" not in z.files:
            raise KeyError(f"{path} has no motion_272")
        arr = z["motion_272"]
    else:
        arr = np.load(str(path), allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 272:
        raise ValueError(f"{path}: expected (T,272), got {arr.shape}")
    return arr


def _motion272_to_motionclip135(
    m272: np.ndarray,
    gt_path: Path | None = None,
    align_mode: str = "yaw",
    rot6d_convention: str = "column",
) -> np.ndarray:
    import torch

    from hftrainer.datasets.motion.representation.humanml_repr import (
        recover_local_rotations_and_root,
    )
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
        matrix_to_rotation_6d,
    )

    rot, root = recover_local_rotations_and_root(np.asarray(m272, dtype=np.float32))
    if gt_path is not None and gt_path.exists():
        gt = np.load(str(gt_path), allow_pickle=True)
        if "global_orient" in gt.files and "transl" in gt.files:
            gt_go0 = torch.from_numpy(np.asarray(gt["global_orient"], dtype=np.float32)[:1]).reshape(1, 3)
            gt_mat0 = axis_angle_to_matrix(gt_go0)[0]
            pred_mat0 = torch.from_numpy(np.asarray(rot[0, 0], dtype=np.float32))
            if align_mode == "full":
                delta = gt_mat0 @ pred_mat0.transpose(0, 1)
            else:
                def yaw_from_mat(mat: torch.Tensor) -> torch.Tensor:
                    fwd = mat[:, 2]
                    return torch.atan2(fwd[0], fwd[2])

                yaw = yaw_from_mat(gt_mat0) - yaw_from_mat(pred_mat0)
                c, s = torch.cos(yaw), torch.sin(yaw)
                delta = torch.stack([
                    torch.stack([c, torch.zeros_like(c), s]),
                    torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
                    torch.stack([-s, torch.zeros_like(c), c]),
                ])
            rot = np.asarray(rot, dtype=np.float32).copy()
            rot[:, 0] = np.matmul(delta.numpy()[None], rot[:, 0]).astype(np.float32)
            root_t = torch.from_numpy(np.asarray(root, dtype=np.float32))
            gt_tr0 = torch.from_numpy(np.asarray(gt["transl"], dtype=np.float32)[0])
            root = ((delta @ (root_t - root_t[0]).T).T + gt_tr0).numpy().astype(np.float32)
    rot_t = torch.from_numpy(np.asarray(rot, dtype=np.float32))
    if rot6d_convention not in {"row", "column"}:
        raise ValueError(f"rot6d_convention must be row/column, got {rot6d_convention!r}")
    rot6d = matrix_to_rotation_6d(rot_t, convention=rot6d_convention).numpy()
    out = np.concatenate(
        [np.asarray(root, dtype=np.float32), rot6d.reshape(rot6d.shape[0], 22 * 6)],
        axis=-1,
    ).astype(np.float32)
    if out.shape[-1] != 135:
        raise ValueError(f"bad MotionCLIP135 shape: {out.shape}")
    return out


def _annotation_map(
    anno_file: Path,
    include_mirrors: bool,
    data_dir: Path,
) -> dict[str, list[tuple[str, str | None]]]:
    raw = json.loads(anno_file.read_text())
    data = raw["data_list"] if isinstance(raw, dict) and "data_list" in raw else raw
    if not isinstance(data, dict):
        raise ValueError(f"expected dict data_list in {anno_file}")
    out: dict[str, list[tuple[str, str | None]]] = {}
    for name, entry in data.items():
        cid = Path(str(entry.get("smplx_path") or "")).stem
        if not cid:
            continue
        if cid.startswith("M") and not include_mirrors:
            continue
        gt_path = str((data_dir / entry["smplx_path"]).resolve()) if entry.get("smplx_path") else None
        out.setdefault(cid, []).append((str(name), gt_path))
        out.setdefault(str(name), []).append((str(name), gt_path))
    return out


def _worker(task: tuple[str, str | None, str | None, list[str], str | None, str, str, bool]) -> tuple[str, str]:
    src_s, ms_out_s, mc_out_s, anno_names, gt_path_s, align_mode, rot6d_convention, overwrite = task
    src = Path(src_s)
    try:
        m272 = _load_m272(src)
        if ms_out_s:
            ms_out = Path(ms_out_s)
            if overwrite or not ms_out.exists():
                ms_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(ms_out), motion_272=m272)
        if mc_out_s and anno_names:
            mc = _motion272_to_motionclip135(
                m272,
                Path(gt_path_s) if gt_path_s else None,
                align_mode=align_mode,
                rot6d_convention=rot6d_convention,
            )
            mc_out = Path(mc_out_s)
            mc_out.mkdir(parents=True, exist_ok=True)
            for name in anno_names:
                dst = mc_out / f"{name}.npy"
                if overwrite or not dst.exists():
                    np.save(str(dst), mc)
        return "ok", src.name
    except Exception as exc:  # noqa: BLE001
        return f"fail:{type(exc).__name__}:{exc}", src.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument("--ms-npz-dir", default=None)
    ap.add_argument("--motionclip-dir", default=None)
    ap.add_argument("--align-to-gt-root", action="store_true",
                    help="Rigidly align converted clips to each sample's GT first-frame root.")
    ap.add_argument("--align-root-mode", choices=["yaw", "full"], default="yaw")
    ap.add_argument("--rot6d-convention", choices=["row", "column"], default="column",
                    help="Output 6D rotation layout for MotionCLIP 135D files. "
                         "MotionCLIP paper evaluator is validated with column-major inputs.")
    ap.add_argument("--include-mirrors", action="store_true", default=True)
    ap.add_argument("--no-include-mirrors", dest="include_mirrors", action="store_false")
    ap.add_argument("--only-mapped", action="store_true",
                    help="Skip source ids that cannot be mapped through --anno-file.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    files = sorted(list(src_dir.glob("*.npy")) + list(src_dir.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npy/.npz files under {src_dir}")
    can_to_annos = _annotation_map(Path(args.anno_file), args.include_mirrors, Path(args.data_dir))
    ms_dir = Path(args.ms_npz_dir).resolve() if args.ms_npz_dir else None
    mc_dir = Path(args.motionclip_dir).resolve() if args.motionclip_dir else None
    if ms_dir:
        ms_dir.mkdir(parents=True, exist_ok=True)
    if mc_dir:
        mc_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src in files:
        ms_out = str(ms_dir / f"{src.stem}.npz") if ms_dir else None
        anno_items = can_to_annos.get(src.stem, [])
        if args.only_mapped and not anno_items:
            continue
        if args.align_to_gt_root and anno_items:
            # One source canonical id may map to multiple annotation entries.
            # Each entry can have a different arbitrary global frame.
            for anno_name, gt_path in anno_items:
                tasks.append((
                    str(src), ms_out, str(mc_dir) if mc_dir else None,
                    [anno_name], gt_path, args.align_root_mode, args.rot6d_convention, args.overwrite,
                ))
        else:
            anno_names = [x[0] for x in anno_items]
            tasks.append((
                str(src), ms_out, str(mc_dir) if mc_dir else None,
                anno_names, None, args.align_root_mode, args.rot6d_convention, args.overwrite,
            ))

    print(
        f"[start] src={src_dir} files={len(files)} ms_dir={ms_dir} "
        f"mc_dir={mc_dir} mapped={sum(1 for t in tasks if t[3])}",
        flush=True,
    )
    ok = fail = 0
    failures = []
    n_workers = max(1, args.workers)
    if n_workers == 1:
        iterator = map(_worker, tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_workers)
        iterator = pool.imap_unordered(_worker, tasks, chunksize=16)
    try:
        for i, (status, name) in enumerate(iterator, 1):
            if status == "ok":
                ok += 1
            else:
                fail += 1
                failures.append((name, status))
                if fail <= 10:
                    print(f"[fail] {name}: {status}", flush=True)
            if i % 500 == 0 or i == len(tasks):
                print(f"[progress] {i}/{len(tasks)} ok={ok} fail={fail}", flush=True)
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    summary = {
        "src_dir": str(src_dir),
        "files": len(files),
        "ok": ok,
        "failed": fail,
        "ms_npz_dir": str(ms_dir) if ms_dir else None,
        "motionclip_dir": str(mc_dir) if mc_dir else None,
        "motionclip_files": len(list(mc_dir.glob("*.npy"))) if mc_dir else 0,
        "ms_npz_files": len(list(ms_dir.glob("*.npz"))) if ms_dir else 0,
        "failures": failures[:50],
    }
    report_dir = mc_dir or ms_dir or src_dir
    (report_dir / "_convert_ms272_summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a directory of SMPL ``motion_135`` clips to HumanML3D-263.

This is the prediction-side bridge used by the KIMODO-SMPLX T2M evaluation:

    motion_135 @ 30 fps -> SMPL-22 FK -> 20 fps joints
      -> official HumanML3D ``process_file`` -> un-normalized HML263

Output files are named ``<id>.npy`` and can be scored directly by
``HumanML263Evaluator`` / ``scripts/eval/verify_evaluators.py --which hml263``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_motion_135(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "motion_135" in data.files:
            return np.asarray(data["motion_135"], dtype=np.float32)
        if {"transl", "global_orient", "body_pose"}.issubset(set(data.files)):
            import torch
            from hftrainer.models.motion.components.utils.geometry.rotation_convert import axis_angle_to_matrix
            from hftrainer.motion.skeleton.fk import rotmat_to_rot6d_row_major

            transl = np.asarray(data["transl"], dtype=np.float32)
            t = transl.shape[0]
            go = torch.from_numpy(np.asarray(data["global_orient"], dtype=np.float32)).reshape(t, 3)
            bp = torch.from_numpy(np.asarray(data["body_pose"], dtype=np.float32)).reshape(t, -1, 3)
            if bp.shape[1] < 21:
                raise ValueError(f"{path} body_pose has {bp.shape[1]} joints, expected >=21")
            aa = torch.cat([go[:, None], bp[:, :21]], dim=1)
            rot6d = rotmat_to_rot6d_row_major(axis_angle_to_matrix(aa)).reshape(t, 132)
            return torch.cat([torch.from_numpy(transl), rot6d], dim=1).numpy().astype(np.float32)
        raise KeyError(f"{path} has neither motion_135 nor SMPL transl/global_orient/body_pose")
    return np.asarray(np.load(path), dtype=np.float32)


def _init_worker() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from hftrainer.datasets.motion.representation.humanml_repr import setup_process_globals

    torch.set_num_threads(1)
    setup_process_globals()


def _valid_hml263(path: Path) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.ndim == 2 and arr.shape[1] == 263 and np.isfinite(arr[: min(len(arr), 3)]).all()
    except Exception:
        return False


def _valid_metadata(path: Path) -> bool:
    try:
        data = np.load(path, mmap_mode="r")
        return "source_motion135_transl" in data.files and "root_quat_init" in data.files
    except Exception:
        return False


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npy")
    try:
        np.save(tmp, array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _worker(task: tuple[str, str, str | None, str, float, float, float, bool]) -> tuple[str, str, dict[str, Any] | None]:
    src_s, dst_s, meta_s, rotation_space, src_fps, dst_fps, feet_thre, skip_existing = task
    src = Path(src_s)
    dst = Path(dst_s)
    meta_path = Path(meta_s) if meta_s else None
    if skip_existing and _valid_hml263(dst) and (meta_path is None or _valid_metadata(meta_path)):
        return src.stem, "skip", None
    try:
        from hftrainer.datasets.motion.representation.humanml_repr import (
            motion198_to_humanml263,
            motion198_to_humanml263_with_metadata,
        )

        motion_135 = _load_motion_135(src)
        if motion_135.ndim != 2 or motion_135.shape[1] < 135:
            raise ValueError(f"expected (T,>=135), got {motion_135.shape}")
        if len(motion_135) < 4:
            raise ValueError(f"too few frames: {len(motion_135)}")
        if meta_path is not None:
            m263, _, meta = motion198_to_humanml263_with_metadata(
                motion_135[:, :135],
                rotation_space=rotation_space,
                src_fps=src_fps,
                dst_fps=dst_fps,
                feet_thre=feet_thre,
                ensure_globals=False,
            )
            meta.update({
                "source_motion135_transl": motion_135[:, :3].astype(np.float32),
                "source_motion135_num_frames": np.array(len(motion_135), dtype=np.int32),
            })
        else:
            m263, _ = motion198_to_humanml263(
                motion_135[:, :135],
                rotation_space=rotation_space,
                src_fps=src_fps,
                dst_fps=dst_fps,
                feet_thre=feet_thre,
                ensure_globals=False,
            )
            meta = None
        if m263.ndim != 2 or m263.shape[1] != 263 or not np.isfinite(m263).all():
            raise ValueError(f"bad HML263 output {m263.shape}")
        _atomic_save_npy(dst, m263.astype(np.float32))
        if meta_path is not None and meta is not None:
            _atomic_save_npz(meta_path, {k: np.asarray(v) for k, v in meta.items()})
        return src.stem, "ok", {
            "frames_in": int(len(motion_135)),
            "frames_out": int(len(m263)),
            "metadata": str(meta_path) if meta_path is not None else None,
        }
    except Exception as exc:  # noqa: BLE001
        return src.stem, f"fail:{type(exc).__name__}:{exc}", None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rotation-space", choices=["local", "global"], default="local")
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--dst-fps", type=float, default=20.0)
    parser.add_argument("--feet-thre", type=float, default=0.002)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Optional sidecar directory for process_file canonicalization metadata.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    meta_dir = Path(args.metadata_dir) if args.metadata_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta_dir is not None:
        meta_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        ids = [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]
        suffix = ".npz" if args.pattern.endswith(".npz") else ".npy"
        files = [in_dir / f"{sid}{suffix}" for sid in ids]
    else:
        files = sorted(p for p in in_dir.glob(args.pattern) if not p.name.startswith("_"))
    files = [p for p in files if p.exists()]
    if args.limit:
        files = files[: args.limit]

    tasks = [
        (
            str(path),
            str(out_dir / f"{path.stem}.npy"),
            str(meta_dir / f"{path.stem}.npz") if meta_dir is not None else None,
            args.rotation_space,
            args.src_fps,
            args.dst_fps,
            args.feet_thre,
            args.skip_existing,
        )
        for path in files
    ]
    print(
        f"[motion135->hml263] inputs={len(tasks)} out={out_dir} "
        f"workers={args.workers} rotation_space={args.rotation_space} "
        f"metadata_dir={meta_dir}",
        flush=True,
    )

    ok = skipped = failed = 0
    rows = []
    failures = []
    if args.workers <= 1:
        _init_worker()
        iterator = map(_worker, tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_init_worker)
        iterator = pool.imap_unordered(_worker, tasks, chunksize=8)
    try:
        for i, (sid, status, info) in enumerate(iterator, 1):
            if status == "ok":
                ok += 1
                rows.append({"sid": sid, **(info or {})})
            elif status == "skip":
                skipped += 1
                rows.append({"sid": sid, "skipped": True})
            else:
                failed += 1
                failures.append({"sid": sid, "status": status})
                if failed <= 10:
                    print(f"  [fail] {sid}: {status}", flush=True)
            if i % 200 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ok={ok} skipped={skipped} failed={failed}", flush=True)
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    summary = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "inputs": len(tasks),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "rotation_space": args.rotation_space,
        "src_fps": args.src_fps,
        "dst_fps": args.dst_fps,
        "metadata_dir": str(meta_dir) if meta_dir is not None else None,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if failures:
        (out_dir / "_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

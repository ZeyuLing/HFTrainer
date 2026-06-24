#!/usr/bin/env python3
"""Convert a MotionHub annotation split's GT SMPL motions to HumanML3D-263.

This builds the representation-control row requested for the T2M table:

    GT SMPL npz -> MotionCLIP-style 135D -> SMPL-22 FK -> 20 fps joints
      -> official HumanML3D process_file -> HML3D-263

The output files are named by annotation key (for example
``humanml3d_194.npy``).  Downstream ``hml263_to_smpl_ik.py`` then writes
``humanml3d_194.npz``, and ``remap_hml3d_smpl_to_motionclip135.py
--key-fallback`` maps them back to the same annotation entries.
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

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    motion198_to_humanml263,
    setup_process_globals,
)


def _iter_annotation(path: Path):
    raw = json.loads(path.read_text())
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        yield from data.items()
        return
    for i, entry in enumerate(data):
        key = entry.get("motion_id") or entry.get("id") or str(i)
        yield str(key), entry


def _load_smpl22_row135(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        z = np.load(str(path), allow_pickle=True)
    except Exception:
        return None
    if "transl" not in z.files or "global_orient" not in z.files or "body_pose" not in z.files:
        return None
    import torch
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import axis_angle_to_matrix
    from hftrainer.pipelines.motion.differentiable_fk import rotmat_to_rot6d_row_major

    transl = np.asarray(z["transl"], dtype=np.float32)
    t = transl.shape[0]
    go = torch.from_numpy(np.asarray(z["global_orient"], dtype=np.float32)).reshape(t, 3)
    bp = torch.from_numpy(np.asarray(z["body_pose"], dtype=np.float32)).reshape(t, 21, 3)
    aa = torch.cat([go[:, None], bp], dim=1)
    rot6d = rotmat_to_rot6d_row_major(axis_angle_to_matrix(aa)).reshape(t, 132)
    return torch.cat([torch.from_numpy(transl), rot6d], dim=1).numpy().astype(np.float32)


def _worker(task):
    key, entry, data_dir, out_dir, src_fps, dst_fps, min_hml_len, layout, skip_existing = task
    out_dir = Path(out_dir)
    flat_path = out_dir / f"{key}.npy"
    hml_path = out_dir / "new_joint_vecs" / f"{key}.npy"
    check_path = hml_path if layout in {"humanml", "both"} else flat_path
    if skip_existing and check_path.exists():
        return key, "skip"
    rel = entry.get("smplx_path")
    if not rel:
        return key, "missing_smplx_path"
    motion_path = Path(data_dir) / rel
    m135 = _load_smpl22_row135(motion_path)
    if m135 is None:
        return key, f"load_failed:{motion_path}"
    if len(m135) < 4:
        return key, f"too_short:{len(m135)}"
    try:
        setup_process_globals()
        m263, _ = motion198_to_humanml263(
            m135,
            rotation_space="local",
            src_fps=float(entry.get("fps") or src_fps),
            dst_fps=dst_fps,
            ensure_globals=False,
        )
    except Exception as exc:  # noqa: BLE001
        return key, f"convert_failed:{type(exc).__name__}:{exc}"
    if len(m263) < min_hml_len or not np.isfinite(m263).all():
        return key, f"bad_m263:shape={m263.shape}"
    if layout in {"flat", "both"}:
        np.save(flat_path, m263.astype(np.float32))
    if layout in {"humanml", "both"}:
        hml_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hml_path, m263.astype(np.float32))
    return key, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", required=True)
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-person", type=int, default=1)
    ap.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="Skip motions longer than this many seconds. 0 disables filtering.",
    )
    ap.add_argument("--src-fps", type=float, default=30.0)
    ap.add_argument("--dst-fps", type=float, default=20.0)
    ap.add_argument(
        "--min-hml-len",
        type=int,
        default=0,
        help="Skip converted HML263 clips shorter than this many frames.",
    )
    ap.add_argument(
        "--layout",
        choices=["flat", "humanml", "both"],
        default="both",
        help="Output layout. 'humanml' writes new_joint_vecs/<id>.npy and test.txt.",
    )
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    skipped_person = 0
    skipped_duration = 0
    for key, entry in _iter_annotation(Path(args.anno_file)):
        path = entry.get("smplx_path")
        n_person = len(path) if isinstance(path, list) else 1
        if n_person != args.num_person:
            skipped_person += 1
            continue
        if args.max_duration > 0:
            fps = float(entry.get("fps") or args.src_fps)
            num_frames = int(entry.get("num_frames") or 0)
            duration = num_frames / fps if fps > 0 and num_frames > 0 else 0.0
            if duration > args.max_duration:
                skipped_duration += 1
                continue
        entries.append((key, entry))
    if args.limit:
        entries = entries[: args.limit]
    tasks = [
        (
            key,
            entry,
            args.data_dir,
            str(out_dir),
            args.src_fps,
            args.dst_fps,
            args.min_hml_len,
            args.layout,
            args.skip_existing,
        )
        for key, entry in entries
    ]
    print(
        f"[start] {len(tasks)} entries -> {out_dir} "
        f"skipped_person={skipped_person} skipped_duration={skipped_duration}",
        flush=True,
    )

    ok = skipped = failed = 0
    ok_keys = []
    failures = []
    if args.workers <= 1:
        iterator = map(_worker, tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers)
        iterator = pool.imap_unordered(_worker, tasks, chunksize=8)

    try:
        for i, (key, status) in enumerate(iterator, 1):
            if status == "ok":
                ok += 1
                ok_keys.append(key)
            elif status == "skip":
                skipped += 1
                ok_keys.append(key)
            else:
                failed += 1
                failures.append((key, status))
                if failed <= 10:
                    print(f"[fail] {key}: {status}", flush=True)
            if i % 200 == 0 or i == len(tasks):
                print(
                    f"[progress] {i}/{len(tasks)} ok={ok} skipped={skipped} failed={failed}",
                    flush=True,
                )
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    print(f"[done] ok={ok} skipped={skipped} failed={failed} out={out_dir}", flush=True)
    ok_set = set(ok_keys)
    ordered_ok = [key for key, _ in entries if key in ok_set]
    if args.layout in {"humanml", "both"}:
        (out_dir / "test.txt").write_text("\n".join(ordered_ok) + ("\n" if ordered_ok else ""), encoding="utf-8")
    summary = {
        "anno_file": args.anno_file,
        "data_dir": args.data_dir,
        "out_dir": str(out_dir),
        "num_person": args.num_person,
        "max_duration": args.max_duration,
        "min_hml_len": args.min_hml_len,
        "layout": args.layout,
        "num_entries_after_filters": len(entries),
        "ok": ok,
        "skipped_existing": skipped,
        "failed": failed,
        "test_ids": len(ordered_ok),
        "skipped_person": skipped_person,
        "skipped_duration": skipped_duration,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[summary] {out_dir / 'summary.json'}", flush=True)
    if args.layout in {"humanml", "both"}:
        print(f"[split] {out_dir / 'test.txt'} ({len(ordered_ok)} ids)", flush=True)
    if failures:
        report = out_dir / "_failures.json"
        report.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"[failures] {report}", flush=True)


if __name__ == "__main__":
    main()

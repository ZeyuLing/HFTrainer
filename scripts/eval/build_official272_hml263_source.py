#!/usr/bin/env python3
"""Materialize MotionStreamer official HumanML3D-272 clips as HML263.

This is used to feed 263D tokenizers (e.g. MotionLCM/MoMask-style VAEs) from
the same paired source used by the MotionStreamer-272 evaluator.  The conversion
uses the validated SMPL-H FK path in ``humanml272_to_humanml263`` rather than
MotionStreamer's stored joint positions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MS = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer"
DEFAULT_SPLIT = MS / "humanml3d_272" / "split" / "test.txt"
DEFAULT_MOTION_DIR = MS / "humanml3d_272" / "motion_data"

sys.path.insert(0, str(REPO))

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    humanml272_to_humanml263,
    setup_process_globals,
)


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--motion-dir", default=str(DEFAULT_MOTION_DIR))
    parser.add_argument("--out-dir", required=True, help="Output HumanML-style root.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    split = Path(args.split)
    motion_dir = Path(args.motion_dir)
    out_dir = Path(args.out_dir)
    out_motion = out_dir / "new_joint_vecs"
    out_motion.mkdir(parents=True, exist_ok=True)

    ids_all = _read_ids(split)
    if args.max_samples:
        ids_all = ids_all[: args.max_samples]
    ids = [sid for idx, sid in enumerate(ids_all) if idx % args.num_shards == args.shard_index]

    setup_process_globals()
    t0 = time.time()
    written: list[str] = []
    failures: list[dict[str, str]] = []
    lengths: list[int] = []
    print(
        f"[setup] official272->hml263 ids={len(ids)} shard={args.shard_index}/{args.num_shards} "
        f"motion_dir={motion_dir} out={out_dir}",
        flush=True,
    )

    for i, sid in enumerate(ids, 1):
        src = motion_dir / f"{sid}.npy"
        dst = out_motion / f"{sid}.npy"
        if args.skip_existing and dst.exists():
            written.append(sid)
            try:
                lengths.append(int(len(np.load(dst))))
            except Exception:
                pass
            continue
        try:
            m272 = np.load(src).astype(np.float32)
            m263, _joints = humanml272_to_humanml263(m272, joints_from="smpl_fk")
            if m263.ndim != 2 or m263.shape[1] != 263:
                raise ValueError(f"bad converted shape {m263.shape}")
            if not np.isfinite(m263).all():
                raise ValueError("non-finite converted feature")
            np.save(dst, m263.astype(np.float32))
            written.append(sid)
            lengths.append(int(len(m263)))
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sid, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 10:
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if i % 100 == 0 or i == len(ids):
            print(f"[progress] {i}/{len(ids)} ok={len(written)} fail={len(failures)}", flush=True)

    if args.num_shards == 1:
        (out_dir / "test.txt").write_text(
            "\n".join(written) + ("\n" if written else ""),
            encoding="utf-8",
        )
        summary_name = "_build_summary.json"
    else:
        (out_dir / f"test_s{args.shard_index:02d}_of_{args.num_shards:02d}.txt").write_text(
            "\n".join(written) + ("\n" if written else ""),
            encoding="utf-8",
        )
        summary_name = f"_build_summary_s{args.shard_index:02d}_of_{args.num_shards:02d}.json"

    summary = {
        "split": str(split),
        "motion_dir": str(motion_dir),
        "out_dir": str(out_dir),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected": len(ids),
        "written": len(written),
        "failures": failures,
        "length": {
            "mean": float(np.mean(lengths)) if lengths else None,
            "min": int(np.min(lengths)) if lengths else None,
            "max": int(np.max(lengths)) if lengths else None,
        },
        "elapsed_sec": float(time.time() - t0),
    }
    (out_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

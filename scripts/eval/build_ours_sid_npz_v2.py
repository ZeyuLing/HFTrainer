#!/usr/bin/env python3
"""Robust canonical-id remapping for ours MIB inference NPZs.

The legacy ``/tmp/ours_full_to_sid_npz.py`` matched each ours output to a
canonical HumanML3D id using only the *first frame* of ``gt_motion_135``
rounded to 2 decimals, with a greedy caption tie-break. The condition frame is
shared across many clips, so that signature collided and produced wrong sids
(e.g. canonical ``000022`` received a 186-frame motion whose true length is 174,
breaking endpoint alignment).

This rebuild matches each ours output against
``data/eval/h3d_editing/source_npz/<id>.npz`` (keyed by ``source_id`` with the
full GT ``motion_135``) using a nearest-neighbour over the first+last GT frames.
The ours ``gt_motion_135`` only differs from the source by a normalisation
round-trip (~3e-3 mean-abs), so the endpoint NN is collision-free and
length-consistent (verified 200/200 exact length match on a random sample).

By default it writes symlinks ``<out>/<canonical_id>.npz -> <ours_path>`` so the
exact inference NPZ (with intact ``gt_motion_135`` / ``src_mask`` / condition
frames) is reused without slow CephFS copies. ``--copy`` writes minimal npz
instead.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import numpy as np


def _src_feat(path: str):
    z = np.load(path, allow_pickle=True)
    m = np.asarray(z["motion_135"], np.float32)
    sid = str(z["source_id"]) if "source_id" in z.files else Path(path).stem
    return (sid, m.shape[0], m[0].copy(), m[-1].copy())


def _ours_feat(path: str):
    z = np.load(path, allow_pickle=True)
    g = np.asarray(z["gt_motion_135"], np.float32)
    return (path, g.shape[0], g[0].copy(), g[-1].copy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="data/eval/h3d_editing/source_npz")
    ap.add_argument(
        "--ours-glob",
        default="output/evaluation/ours_mib_full_cfg20/shard_*/"
        "smpl_caption_editfix_latest/E2_both_1f/npz/*.npz",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-dist", type=float, default=0.05,
                    help="reject matches whose endpoint NN distance exceeds this")
    ap.add_argument("--copy", action="store_true",
                    help="write minimal npz instead of symlinking the source npz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_files = sorted(glob.glob(os.path.join(args.src_dir, "*.npz")))
    ours_files = sorted(glob.glob(args.ours_glob))
    print(f"source={len(src_files)} ours={len(ours_files)}", flush=True)

    with mp.Pool(args.workers) as pool:
        S = pool.map(_src_feat, src_files, chunksize=16)
        O = pool.map(_ours_feat, ours_files, chunksize=16)

    sids = [s[0] for s in S]
    Ts = np.asarray([s[1] for s in S])
    S0 = np.stack([s[2] for s in S])
    SL = np.stack([s[3] for s in S])
    print(f"source table {S0.shape}", flush=True)

    plan: list[tuple[str, str]] = []
    assigned: dict[str, tuple[str, float]] = {}  # sid -> (ours_path, dist)
    rejected = bad_len = 0
    for path, T, o0, oL in O:
        d = np.abs(S0 - o0[None]).mean(1) + np.abs(SL - oL[None]).mean(1)
        # prefer length-consistent candidates to break any rare endpoint ties
        mask = Ts == T
        if mask.any():
            idx = np.where(mask)[0]
            j = idx[int(np.argmin(d[idx]))]
        else:
            j = int(np.argmin(d))
            bad_len += 1
        dist = float(d[j])
        if dist > args.max_dist:
            rejected += 1
            continue
        sid = sids[j]
        prev = assigned.get(sid)
        if prev is None or dist < prev[1]:
            assigned[sid] = (path, dist)
    for sid, (path, _d) in assigned.items():
        plan.append((path, sid))

    print(f"matched={len(plan)} unique_sids={len(assigned)} "
          f"rejected(dist>{args.max_dist})={rejected} length-fallback={bad_len}",
          flush=True)

    if args.dry_run:
        import random

        random.seed(0)
        for path, sid in random.sample(plan, min(6, len(plan))):
            z = np.load(path, allow_pickle=True)
            pm = np.asarray(z["motion_135"], np.float32)
            gm = np.asarray(z["gt_motion_135"], np.float32)
            print(f"  {sid} T={len(pm)} first|p-g|={np.abs(pm[0]-gm[0]).mean():.4f} "
                  f"last|p-g|={np.abs(pm[-1]-gm[-1]).mean():.4f}")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import json

    (out.parent / "mapping.json").write_text(
        json.dumps({sid: os.path.abspath(p) for p, sid in plan}, indent=0)
    )

    res = Counter()
    for path, sid in plan:
        dst = out / f"{sid}.npz"
        if dst.exists() or dst.is_symlink():
            res["skip"] += 1
            continue
        if args.copy:
            z = np.load(path, allow_pickle=True)
            np.savez(
                dst,
                motion_135=np.asarray(z["motion_135"], np.float32),
                gt_motion_135=np.asarray(z["gt_motion_135"], np.float32),
                src_mask=np.asarray(z["src_mask"], np.float32),
                positions=np.asarray(z["positions"], np.float32),
            )
        else:
            os.symlink(os.path.abspath(path), dst)
        res["ok"] += 1
    print("emit:", dict(res), "->", out, flush=True)


if __name__ == "__main__":
    main()

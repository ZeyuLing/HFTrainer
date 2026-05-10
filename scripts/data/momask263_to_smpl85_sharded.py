#!/usr/bin/env python3
"""Sharded version of momask263_to_smpl85.py.

Each shard processes 1/num_shards of the input files (deterministic by
sorted filename).  Designed to be launched in parallel across multiple GPUs.
Shared-storage friendly: each output file is written atomically (write to
``<name>.tmp.<pid>`` then rename).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"))

from momask263_to_smpl85 import SeqFitter, linear_resample_positions  # noqa: E402
from utils.motion_process import recover_from_ric  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir_263", required=True)
    p.add_argument("--out_dir_smpl85", required=True)
    p.add_argument("--shard_idx", type=int, required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--src_fps", type=float, default=20.0)
    p.add_argument("--dst_fps", type=float, default=30.0)
    p.add_argument("--num_iters", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    src = Path(args.pred_dir_263)
    dst = Path(args.out_dir_smpl85)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    files = files[args.shard_idx :: args.num_shards]
    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(files)} files")
    print(f"[shard {args.shard_idx}] device={args.device}, iters={args.num_iters}")

    fitter = SeqFitter(num_iters=args.num_iters, device=args.device)
    n_ok = n_skip = n_err = 0
    for f in tqdm(files, ncols=80, desc=f"shard{args.shard_idx}"):
        out_file = dst / f.name
        if out_file.exists():
            n_skip += 1
            continue
        try:
            m263 = np.load(str(f))
            if m263.ndim != 2 or m263.shape[1] != 263 or len(m263) < 4:
                n_err += 1
                continue
            joints20 = recover_from_ric(torch.from_numpy(m263).float(), 22).numpy()
            joints30 = linear_resample_positions(joints20, args.src_fps, args.dst_fps)
            params = fitter.fit(joints30)
            smpl_85 = np.concatenate(
                [params["pose"], params["trans"], params["betas"]], axis=-1
            ).astype(np.float32)
            tmp_file = str(out_file).replace(".npy", f".tmp.{os.getpid()}.npy")
            np.save(tmp_file, smpl_85)
            os.replace(tmp_file, str(out_file))
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [!] {f.name}: {e}", flush=True)

    print(f"[shard {args.shard_idx}] done: ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()

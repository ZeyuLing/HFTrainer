#!/usr/bin/env python3
"""Build a deterministic G1 replay bank from an annotation json.

This is the fixed-bank counterpart to ``build_replay_pool.py``: no random scan,
no sampling, and no pool mutation.  It converts the listed G1 npz clips through
the same canonical encode->decode->qpos path used by the hard-overfit scorer,
then writes ProtoMotions ``*.motion`` files into ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from hftrainer.models.motion.physflow.g1_repr import encode_g1_motion, decode_g1_to_qpos  # noqa
from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward  # noqa

G1_ROOT = os.path.join(REPO, "data/g1")


def load_items(path: str):
    data = json.load(open(os.path.join(REPO, path)))
    if isinstance(data, dict):
        return list(data.get("items", []))
    return list(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/annotation/_coevo_hardovf_agile_eval.json")
    ap.add_argument("--out", required=True, help="output directory for .motion replay bank")
    ap.add_argument("--prefix", default="fixed_hard_")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    if os.path.isdir(out) and args.overwrite:
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    items = load_items(args.anno)
    if args.limit:
        items = items[: args.limit]
    print(f"[fixed-replay] items={len(items)} anno={args.anno}", flush=True)

    tmp = tempfile.mkdtemp(prefix="fixed_g1_replay_", dir=os.path.join(REPO, "output"))
    csv_dir = os.path.join(tmp, "csv")
    proto_dir = os.path.join(tmp, "proto")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(proto_dir, exist_ok=True)

    meta = {}
    for i, it in enumerate(items):
        g1_path = it["g1_path"]
        p = os.path.join(G1_ROOT, g1_path)
        if not os.path.exists(p):
            print(f"  [skip] missing {g1_path}", flush=True)
            continue
        try:
            npz = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            m38 = encode_g1_motion(npz, canonicalize=True)
            qpos = decode_g1_to_qpos(torch.from_numpy(m38)).numpy()
        except Exception as e:
            print(f"  [skip] {g1_path}: {e}", flush=True)
            continue
        stem = f"{args.prefix}{i:03d}"
        np.savetxt(os.path.join(csv_dir, f"{stem}.csv"), qpos, delimiter=",", fmt="%.6f")
        meta[stem] = {
            "idx": i,
            "g1_path": g1_path,
            "agility": it.get("agility"),
            "num_frames": int(qpos.shape[0]),
        }
        if (i + 1) % 10 == 0:
            print(f"  encoded {i+1}/{len(items)}", flush=True)

    print(f"[fixed-replay] converting {len(meta)} CSV -> .motion ...", flush=True)
    reward = PhysicsJudgeReward()
    reward._convert_csv_dir(
        __import__("pathlib").Path(csv_dir),
        __import__("pathlib").Path(proto_dir),
    )

    copied = 0
    for f in sorted(os.listdir(proto_dir)):
        if f.endswith(".motion"):
            shutil.copy2(os.path.join(proto_dir, f), os.path.join(out, f))
            copied += 1
    json.dump(meta, open(os.path.join(out, "_fixed_replay_manifest.json"), "w"), indent=2)
    print(f"[fixed-replay] wrote {copied} motions -> {out}", flush=True)


if __name__ == "__main__":
    main()

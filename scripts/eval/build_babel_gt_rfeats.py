#!/usr/bin/env python3
"""Build per-composition GT rfeats for the BABEL mesh viewer.

Mirrors the GT assembly in build_babel_seq_viewer_cache.py (same deterministic
closest-length clip selection) but stores the concatenated *rfeats* [T,135]
instead of FK joints, so the mesh viewer can run SMPLH -> vertices on them.
Output: <cache>/<id>/GT.rfeats.npy  and  <id>/GT.seglens.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
FM = REPO / "ref_repo/FlowMDM"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "output/evaluation/babel_seq_viewer"))
    args = ap.parse_args()
    os.chdir(str(FM)); sys.path.insert(0, str(FM))

    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader
    fixseed(0); dist_util.setup_dist(0)

    val = json.load(open(FM / "dataset/babel_val_set.json"))
    needed = set()
    for it in val:
        for c in it["text"]:
            needed.add(str(c).strip().lower())

    gt = get_dataset_loader(name="babel", batch_size=8, num_frames=(30, 200),
                            split="val", load_mode="gt", shuffle=False,
                            drop_last=False, cropping_sampler=False, num_workers=0)
    ds = gt.dataset
    n = len(ds)
    print(f"[gt-rfeats] scanning {n} clips for {len(needed)} captions ...", flush=True)
    pool = {c: [] for c in needed}
    for i in range(n):
        try:
            it = ds[i]
        except Exception:
            continue
        cap = str(it[2]).strip().lower()
        if cap not in pool:
            continue
        motion = np.asarray(it[4], np.float32)
        mlen = int(it[5])
        pool[cap].append((mlen, motion[:mlen].copy()))
        if i % 1000 == 0:
            print(f"  [gt-rfeats] {i}/{n}", flush=True)

    out = Path(args.cache)
    for it in val:
        cid = it["id"]
        chunks, seglens, segcaps, miss = [], [], [], 0
        for cap, tgt in zip(it["text"], it["lengths"]):
            cands = pool.get(str(cap).strip().lower(), [])
            if not cands:
                miss += 1
                continue
            L, feats = min(cands, key=lambda lf: abs(lf[0] - int(tgt)))
            chunks.append(np.asarray(feats, np.float32))
            seglens.append(int(L))
            segcaps.append(str(cap).strip())
        if not chunks:
            print(f"[gt-rfeats] {cid}: no clips, skip", flush=True)
            continue
        full = np.concatenate(chunks, 0)  # [T,135]
        (out / cid).mkdir(parents=True, exist_ok=True)
        np.save(out / cid / "GT.rfeats.npy", full.astype(np.float32))
        json.dump(seglens, open(out / cid / "GT.seglens.json", "w"))
        json.dump(segcaps, open(out / cid / "GT.segcaps.json", "w"))
        print(f"[gt-rfeats] {cid}: T={full.shape[0]} miss={miss}", flush=True)
    print("[gt-rfeats] done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a joint-trajectory cache for the BABEL sequential-generation viewer.

For each of FlowMDM's 64 ``babel_val_set`` compositions we materialise SMPL-22
world joints for the ground truth and for every method we evaluated, all decoded
through FlowMDM's *own* SlimSMPL transform (rfeats -> rots -> SMPLH joints) so the
visual is exactly what the native evaluator scores.

* methods : ``{method}/{ii}/{ii}.pt`` are FlowMDM rfeats ``[1,135,1,T]`` for the
            whole composition. Per-segment lengths/captions come from ``{ii}_kwargs.json``.
* GT      : there is no per-composition GT (FlowMDM compositions are synthetic),
            so we assemble one by, for each sub-action caption, pulling a real
            BABEL val clip of matching label (closest length) and concatenating
            them with floor + trajectory continuity. This is the honest "real
            motion for each action" reference.

Output cache layout (read by ``babel_seq_multi_app.py``)::

    <out>/index.json                 # [{id, captions:[...], n_seg, methods:[...]}]
    <out>/<id>/gt.npz                # joints[T,22,3] f16, seg_ids[T] i16, seg_lens[]
    <out>/<id>/<method>.npz

Run inside the FlowMDM repo env (needs body_models/smplh)::

    python3 scripts/eval/build_babel_seq_viewer_cache.py --out output/evaluation/babel_seq_viewer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
FM = REPO / "ref_repo/FlowMDM"

# Methods: display name -> precomputed run folder under results/babel/<run>/evaluation_precomputed/<sub>/00
METHOD_RUNS = {
    "PRISM": "PRISM_e19/evaluation_precomputed/Motion_PRISM_e19_001300000_gscale1.5_debug_s10/00",
    "MotionStreamer": "MotionStreamer/evaluation_precomputed/Motion_MotionStreamer_001300000_gscale1.5_debug_s10/00",
    "FlowMDM": "FlowMDM/evaluation_precomputed/Motion_FlowMDM_001300000_gscale1.5_debug_s10/00",
}

_TR = None


def _transform():
    global _TR
    if _TR is None:
        from data_loaders.amass.transforms import SlimSMPLTransform
        _TR = SlimSMPLTransform(batch_size=32, name="SlimSMPLTransform",
                                ename="smplnh", normalization=True)
    return _TR


def feats_to_joints(feats: np.ndarray) -> np.ndarray:
    """rfeats [T,135] (transform-native normalised) -> joints [T,22,3]."""
    import torch
    t = torch.as_tensor(np.asarray(feats, np.float32))
    j = _transform().SlimDatastruct(features=t).joints.detach().cpu().numpy()
    return np.asarray(j, np.float32)[:, :22, :]


def floor_align(joints: np.ndarray) -> np.ndarray:
    out = np.asarray(joints, np.float32).copy()
    out[..., 1] -= float(np.nanmin(out[..., 1]))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def seg_ids_from_lens(lens, total):
    ids = np.zeros(total, dtype=np.int16)
    s = 0
    for k, L in enumerate(lens):
        e = min(total, s + int(L))
        ids[s:e] = k
        s = e
        if s >= total:
            break
    if s < total:
        ids[s:] = len(lens) - 1
    return ids


def load_val_set():
    items = json.load(open(FM / "dataset/babel_val_set.json"))
    return {it["id"]: it for it in items}


def build_gt_pool(needed_caps):
    """Return {caption_lower: [(length, rfeats[T,135]), ...]} for needed captions."""
    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader
    fixseed(0)
    dist_util.setup_dist(0)
    gt = get_dataset_loader(name="babel", batch_size=8, num_frames=(30, 200),
                            split="val", load_mode="gt", shuffle=False,
                            drop_last=False, cropping_sampler=False, num_workers=0)
    ds = gt.dataset
    n = len(ds)
    print(f"[gt] scanning {n} clips for {len(needed_caps)} captions ...", flush=True)
    pool: dict[str, list] = {c: [] for c in needed_caps}
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
            print(f"  [gt] {i}/{n}", flush=True)
    have = sum(1 for c in pool if pool[c])
    print(f"[gt] captions with >=1 clip: {have}/{len(needed_caps)}", flush=True)
    return pool


def assemble_gt(comp, pool):
    """Concatenate one matching GT clip per sub-action, with continuity."""
    caps = comp["text"]
    seg_lens_target = comp["lengths"]
    chunks = []
    used_lens = []
    miss = 0
    last_xz = np.zeros(2, np.float32)
    for cap, tgt in zip(caps, seg_lens_target):
        key = str(cap).strip().lower()
        cands = pool.get(key, [])
        if not cands:
            miss += 1
            used_lens.append(0)
            continue
        # pick clip whose length is closest to the target segment length
        L, feats = min(cands, key=lambda lf: abs(lf[0] - int(tgt)))
        j = floor_align(feats_to_joints(feats))  # [t,22,3], feet on floor
        # trajectory continuity: start this clip's root xz where previous ended
        root0 = j[0, 0, [0, 2]].copy()
        j[..., 0] += last_xz[0] - root0[0]
        j[..., 2] += last_xz[1] - root0[1]
        last_xz = j[-1, 0, [0, 2]].copy()
        chunks.append(j)
        used_lens.append(int(j.shape[0]))
    if not chunks:
        return None, None, miss
    full = np.concatenate(chunks, 0)
    full = floor_align(full)
    seg = seg_ids_from_lens(used_lens, full.shape[0])
    return full, used_lens, miss


def method_joints(run_dir: Path, ii: int):
    import torch
    pt = run_dir / f"{ii:02d}.pt"
    kj = run_dir / f"{ii:02d}_kwargs.json"
    if not pt.exists() or not kj.exists():
        return None
    u = torch.load(pt, map_location="cpu", weights_only=False).float()  # [1,135,1,T]
    feats = u[0, :, 0, :].permute(1, 0).contiguous().numpy()
    j = floor_align(feats_to_joints(feats))
    kw = json.load(open(kj))
    kw = kw["y"] if "y" in kw else kw
    lens = [int(x) for x in kw["lengths"]]
    seg = seg_ids_from_lens(lens, j.shape[0])
    return j, lens, kw.get("text", [])


def save_npz(path: Path, joints, seg_ids, seg_lens):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joints=np.asarray(joints, np.float16),
        seg_ids=np.asarray(seg_ids, np.int16),
        seg_lens=np.asarray(seg_lens, np.int32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "output/evaluation/babel_seq_viewer"))
    ap.add_argument("--skip-gt", action="store_true")
    ap.add_argument("--ids", default="", help="comma list of comp ids to (re)build; empty=all")
    args = ap.parse_args()

    os.chdir(str(FM))
    sys.path.insert(0, str(FM))

    val = load_val_set()
    ids = sorted(val.keys())
    if args.ids:
        wanted = set(args.ids.split(","))
        ids = [i for i in ids if i in wanted]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- methods ---
    index = []
    for cid in ids:
        comp = val[cid]
        ii = int(cid)
        entry = {"id": cid, "captions": comp["text"], "n_seg": len(comp["text"]),
                 "methods": []}
        for mname, sub in METHOD_RUNS.items():
            rd = REPO / "ref_repo/FlowMDM/results/babel" / sub
            r = method_joints(rd, ii)
            if r is None:
                continue
            j, lens, _txt = r
            save_npz(out / cid / f"{mname}.npz", j, seg_ids_from_lens(lens, j.shape[0]), lens)
            entry["methods"].append(mname)
        index.append(entry)
        print(f"[method] {cid}: {entry['methods']}", flush=True)

    # --- GT ---
    if not args.skip_gt:
        needed = set()
        for cid in ids:
            for c in val[cid]["text"]:
                needed.add(str(c).strip().lower())
        pool = build_gt_pool(needed)
        for e in index:
            cid = e["id"]
            comp = val[cid]
            full, used_lens, miss = assemble_gt(comp, pool)
            if full is None:
                print(f"[gt] {cid}: no clips matched, skip", flush=True)
                continue
            save_npz(out / cid / "GT.npz", full,
                     seg_ids_from_lens(used_lens, full.shape[0]), used_lens)
            if "GT" not in e["methods"]:
                e["methods"].insert(0, "GT")
            print(f"[gt] {cid}: assembled T={full.shape[0]} miss={miss}/{e['n_seg']}",
                  flush=True)

    json.dump(index, open(out / "index.json", "w"), ensure_ascii=False, indent=0)
    print(f"[done] wrote {len(index)} comps to {out}", flush=True)


if __name__ == "__main__":
    main()

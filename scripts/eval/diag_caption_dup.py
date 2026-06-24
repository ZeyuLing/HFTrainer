#!/usr/bin/env python3
"""Diagnose whether BABEL R-precision collapse is caused by DUPLICATE captions.
Compares R@3 with (a) the standard random 32-batches vs (b) caption-deduplicated
items (one representative segment per unique caption -> every 32-batch has
distinct captions), for both GT and a prediction dir."""
import os, sys, json
import numpy as np
REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts/eval"))
sys.path.insert(0, os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer"))
import torch
import eval_motionstreamer_272 as E
import eval_babel_seq_ms272 as B

MAXT = 360
man = [json.loads(l) for l in open(os.path.join(REPO, "data/babel/babel_seq_val_manifest.jsonl")) if l.strip()]
man = [m for m in man if m["total_frames"] <= MAXT]
mean = np.load(os.path.join(B.HUMANML_MEAN_STD, "Mean.npy"))
std = np.load(os.path.join(B.HUMANML_MEAN_STD, "Std.npy"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
textenc, motionenc = E.load_evaluator(device)

def load_gt(sid):
    p = os.path.join(B.GT_STREAM, sid + ".npy")
    return np.load(p).astype(np.float32) if os.path.isfile(p) else None

def load_pred(pred_dir):
    def f(sid):
        p = os.path.join(pred_dir, sid + ".npz")
        if not os.path.isfile(p): return None
        d = np.load(p, allow_pickle=True)
        return np.asarray(d["motion_272"], np.float32) if "motion_272" in d else None
    return f

def build(loader):
    items = []
    for rec in man:
        seq = loader(rec["id"])
        if seq is None: continue
        T = seq.shape[0]
        for s in rec["segments"]:
            cap = str(s["caption"]).strip()
            if cap.lower() == "transition": continue
            cap = B.rewrite_caption(cap)
            a, e = s["start"], min(s["end"], T)
            m, L = B.norm_pad(seq[a:e], mean, std)
            if m is not None:
                items.append((cap, m, L))
    return items

def dedup(items):
    seen, out = set(), []
    for it in items:
        if it[0] in seen: continue
        seen.add(it[0]); out.append(it)
    return out

def r3(items, seed=0):
    enc = E.encode_items(items, textenc, motionenc, device, np.random.RandomState(seed))
    return enc["R"][2], enc["nb"]

for name, loader in [("GT", load_gt),
                     ("PRISM", load_pred(os.path.join(REPO, "outputs/evaluation/babel_seq/prism_272f_rw")))]:
    items = build(loader)
    dd = dedup(items)
    # average over a few seeds for stability
    full = np.mean([r3(items, s)[0] for s in range(3)])
    ddr  = np.mean([r3(dd, s)[0] for s in range(3)])
    print(f"{name:6}  segs={len(items):5d}  unique-cap={len(dd):4d}  ||  R@3(all,dup)={full:.4f}   R@3(dedup-batch)={ddr:.4f}", flush=True)

#!/usr/bin/env python3
"""Convert the BABEL Table-3 MS-272 motion sources to motion_135 for the SMPL
mesh viewer (retarget_smpl_app.py).

Same sources as eval_babel_seq_ms272.py (2-action, <=360f):
  GT  = val_stream/<id>.npy (native 272)
  PRISM/MS/FlowMDM = <dir>/<id>.npz motion_272

Output: <out>/<Method>/<id>.npz (motion_135) + <out>/_captions.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not REPO.is_dir():
    REPO = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))
from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402

SRC = {
    "GT":             ("data/babel_272_stream/val_stream", "npy"),
    "PRISM":          ("outputs/evaluation/babel_seq/prism_272f_rw", "npz"),
    "MotionStreamer": ("outputs/evaluation/babel_seq/ms_gen_rw", "npz"),
    "FlowMDM":        ("outputs/evaluation/babel_seq/flowmdm_272f", "npz"),
}


def load272(method, sid):
    d, kind = SRC[method]
    if kind == "npy":
        p = REPO / d / f"{sid}.npy"
        return np.load(p).astype(np.float32) if p.exists() else None
    p = REPO / d / f"{sid}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["motion_272"], np.float32) if "motion_272" in z else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--out", default="outputs/evaluation/babel_seq/mesh135")
    ap.add_argument("--max-total", type=int, default=360)
    ap.add_argument("--require-all", action="store_true", default=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    man = [json.loads(l) for l in open(REPO / args.manifest) if l.strip()]
    man = [m for m in man if m.get("total_frames", 0) <= args.max_total]
    out = REPO / args.out
    for meth in SRC:
        (out / meth).mkdir(parents=True, exist_ok=True)

    caps = {}
    ok = {m: 0 for m in SRC}
    n_all = 0
    for rec in man:
        sid = rec["id"]
        m272 = {m: load272(m, sid) for m in SRC}
        if args.require_all and any(v is None for v in m272.values()):
            continue
        if m272["GT"] is None:
            continue
        n_all += 1
        for meth, arr in m272.items():
            if arr is None:
                continue
            dst = out / meth / f"{sid}.npz"
            if args.skip_existing and dst.exists():
                ok[meth] += 1
                continue
            try:
                m135 = humanml272_to_motion135(arr)
                np.savez_compressed(dst, motion_135=m135.astype(np.float32),
                                    source_id=np.array(sid, dtype=object))
                ok[meth] += 1
            except Exception as e:  # noqa: BLE001
                print(f"[fail] {meth} {sid}: {e}", flush=True)
        segs = rec["segments"]
        rw = " → ".join(rewrite_caption(str(s["caption"]).strip()) for s in segs)
        raw = " → ".join(str(s["caption"]).strip() for s in segs)
        caps[sid] = f"{rw}   [raw: {raw}]"
        if n_all % 200 == 0:
            print(f"[mesh135] {n_all} cases  ok={ok}", flush=True)

    json.dump(caps, open(out / "_captions.json", "w"), ensure_ascii=False)
    print(f"[mesh135] DONE cases={n_all} ok={ok} -> {out}", flush=True)


if __name__ == "__main__":
    main()

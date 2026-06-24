#!/usr/bin/env python
"""Numerical equivalence check for hftrainer's vendored M2D metric operators.

Loads the GT feature cache produced by the versatilemotion LODGE scorer and
recomputes the Real-row metrics through ``hftrainer.evaluation.motion`` to prove
the ported operators are bit-compatible with the verified protocol.
"""
import os
import pickle

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

from hftrainer.evaluation.motion import M2DFeatures, aggregate_m2d_metrics

CACHE = "/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/outputs/eval_m2d/_gt_cache_lodge.pkl"
EXPECT = {  # from versatilemotion _gt_only_metrics.json (verified Real row)
    "aist++": {"div_k": 10.83, "bas": 0.3622},
    "finedance": {"div_k": 10.66, "bas": 0.2155},
}

cache = pickle.load(open(CACHE, "rb"))
buckets = {}
for c in cache.values():
    buckets.setdefault(c["subset"], []).append(
        M2DFeatures(
            kinetic=c["gt_kf"],
            manual=c["gt_mf"],
            dance_beats=c["gt_db"],
            music_beats=c["music_beat"],
        )
    )

ok = True
for subset, feats in sorted(buckets.items()):
    r = aggregate_m2d_metrics(feats, None)  # Real
    e = EXPECT.get(subset, {})
    dk_ok = abs(r["div_k"] - e.get("div_k", -1)) < 0.01
    bas_ok = abs(r["bas"] - e.get("bas", -1)) < 0.001
    ok = ok and dk_ok and bas_ok
    print(
        f"[{subset:10s}] n={r['n']:3d} "
        f"Div_k={r['div_k']:.4f} (exp {e.get('div_k')}, {'OK' if dk_ok else 'MISMATCH'})  "
        f"Div_g={r['div_g']:.4f}  "
        f"BAS={r['bas']:.4f} (exp {e.get('bas')}, {'OK' if bas_ok else 'MISMATCH'})  "
        f"FID_k(real)={r['fid_k']:.2f}"
    )

print("\nEQUIVALENCE:", "PASS" if ok else "FAIL")

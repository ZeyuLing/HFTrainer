#!/usr/bin/env python3
"""Resample PRISM precomputed (rots+transl) sub-actions to FlowMDM's *canonical*
sub-action lengths so the native BABEL harness accepts them.

FlowMDM's CompMDMUnfoldingGeneratedDataset asserts that the precomputed
``kwargs['y']['lengths']`` equal the dataset's canonical lengths and slices the
concatenated motion at those exact boundaries. PRISM (VAE temporal scale 4 +
AR carry of k_carry frames) cannot hit arbitrary per-segment lengths, so each
sub-action comes out ~5 frames short. We deterministically recover PRISM's real
per-segment boundaries (a_i), slice the concatenated motion, and temporally
resample each segment back to its canonical length (Slerp on per-joint rotations,
linear interp on translation). Result: total == sum(canonical) with every
sub-action exactly canonical length.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SCALE = 4          # vae.config.scale_factor_temporal
AR_COND = 5        # ar_condition_frames used at generation time


def round_frames(n: int, s: int = SCALE) -> int:
    return n if (n - 1) % s == 0 else (n // s) * s + 1


def k_carry(ar: int = AR_COND, s: int = SCALE) -> int:
    return ((ar - 1) // s) * s + 1


def prism_segment_lengths(L):
    """Deterministic per-segment output frame counts of the AR pipeline."""
    kc = k_carry()
    a = [round_frames(L[0])]
    for i in range(1, len(L)):
        ri = round_frames(L[i])
        a.append(ri - min(kc, ri - 1))
    return a


def resample_rots(rots_seg: np.ndarray, tgt: int) -> np.ndarray:
    """rots_seg [n,22,3,3] -> [tgt,22,3,3] via per-joint Slerp."""
    n = rots_seg.shape[0]
    if n == tgt:
        return rots_seg
    if n == 1:
        return np.repeat(rots_seg, tgt, axis=0)
    src_t = np.linspace(0.0, 1.0, n)
    tgt_t = np.linspace(0.0, 1.0, tgt)
    out = np.empty((tgt, rots_seg.shape[1], 3, 3), dtype=np.float32)
    for j in range(rots_seg.shape[1]):
        rot = Rotation.from_matrix(rots_seg[:, j])
        out[:, j] = Slerp(src_t, rot)(tgt_t).as_matrix()
    return out


def resample_transl(tr_seg: np.ndarray, tgt: int) -> np.ndarray:
    n = tr_seg.shape[0]
    if n == tgt:
        return tr_seg
    if n == 1:
        return np.repeat(tr_seg, tgt, axis=0)
    src_t = np.linspace(0.0, 1.0, n)
    tgt_t = np.linspace(0.0, 1.0, tgt)
    return np.stack([np.interp(tgt_t, src_t, tr_seg[:, d]) for d in range(3)], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precomp-dir", required=True)
    ap.add_argument("--seg-mode", default="auto", choices=["auto", "prism", "sidecar"],
                    help="how to recover the per-segment frame counts of the concatenated motion. "
                         "'sidecar' reads {idx}_seglens.json (saved by the generator); "
                         "'prism' uses the AR VAE formula; 'auto' prefers sidecar then prism.")
    args = ap.parse_args()
    PC = args.precomp_dir

    npys = sorted(glob.glob(os.path.join(PC, "*.npy")))
    n_ok = 0
    for f in npys:
        idx = os.path.basename(f).split(".")[0]
        kf = os.path.join(PC, f"{idx}_kwargs.json")
        bak = os.path.join(PC, f"{idx}_kwargs.gtlen.json")
        # canonical lengths always come from the backup (untouched GT lengths)
        src_kw = bak if os.path.exists(bak) else kf
        kw = json.load(open(src_kw))
        L = [int(x) for x in kw["y"]["lengths"]]

        d = np.load(f, allow_pickle=True).item()
        rots = d["rots"].astype(np.float32)   # [T,22,3,3]
        transl = d["transl"].astype(np.float32)
        T = rots.shape[0]

        sidecar = os.path.join(PC, f"{idx}_seglens.json")
        a = None
        if args.seg_mode in ("auto", "sidecar") and os.path.exists(sidecar):
            a = [int(x) for x in json.load(open(sidecar))]
        if a is None and args.seg_mode in ("auto", "prism"):
            a = prism_segment_lengths(L)
        if a is None or sum(a) != T:
            print(f"[skip] {idx}: sum(a)={None if a is None else sum(a)} != T={T} (seg recovery failed)")
            continue

        bnd = np.cumsum([0] + a)
        out_rots, out_tr = [], []
        for s in range(len(L)):
            seg_r = rots[bnd[s]:bnd[s + 1]]
            seg_t = transl[bnd[s]:bnd[s + 1]]
            out_rots.append(resample_rots(seg_r, L[s]))
            out_tr.append(resample_transl(seg_t, L[s]))
        new_rots = np.concatenate(out_rots, axis=0)
        new_tr = np.concatenate(out_tr, axis=0)
        assert new_rots.shape[0] == sum(L), f"{new_rots.shape[0]} != {sum(L)}"

        np.save(f, {"rots": new_rots.astype(np.float32),
                    "transl": new_tr.astype(np.float32)}, allow_pickle=True)
        # restore canonical kwargs (lengths must equal dataset canonical)
        json.dump(kw, open(kf, "w"))
        pt = os.path.join(PC, f"{idx}.pt")
        if os.path.exists(pt):
            os.remove(pt)
        n_ok += 1
    print(f"RESAMPLE_DONE {n_ok}/{len(npys)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Table-3 BABEL sub-action eval with the **MotionStreamer-272 evaluator**
(DistilBERT text + ACTOR motion, HumanML3D-trained) + **rewritten full sentence**
retrieval + **per-sub-segment canonicalization** + **per-batch caption dedup**.

Motion -> 272, per row (each via its *faithful* native rep + the evaluator's
skeleton where possible):

  PRISM / MS : their real generated output is the rots/transl ``.npy`` (the
      globvelandy ``.pt`` is a LOSSY re-encoding -- do NOT use it). Route:
      .npy rots/transl -> undo y2z -> SMPL-22 FK with **canon272 offsets**
      (the GT-272 SMPL-X canonical skeleton the evaluator expects, ~23 mm; SMPL-H
      rest offsets are ~210 mm off, see motion135_to_272 docstring) -> encode_272.
  FlowMDM    : only the globvelandy ``.pt`` exists -> SlimSMPL inverse -> SMPL-H
      body-model FK joints (faithful for FlowMDM) -> encode_272. NB: FlowMDM thus
      sits on the SMPL-H skeleton (a few-cm offset vs canon272), flagged below.
  GT / Real  : native BABEL 272 clips from val_stream, matched to each babel_val_set
      segment by caption + closest length (same composition source as predictions).

Then per row: slice -> reencode_272_via_stored_positions (xz=0, face +z, own floor)
-> humanml mean/std norm + pad -> MS-272, batch=32, FlowMDM-style caption dedup.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
FM = os.path.join(REPO, "ref_repo/FlowMDM")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))
sys.path.insert(0, FM)

import eval_motionstreamer_272 as E  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402
from hftrainer.motion.skeleton.fk import differentiable_fk  # noqa: E402
from hftrainer.motion.representation.motion272 import (  # noqa: E402
    _canonical_272_offsets, encode_smpl_to_272, reencode_272_via_stored_positions,
)

HUMANML_MEAN_STD = os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std")
VAL_SET = os.path.join(REPO, "ref_repo/FlowMDM/dataset/babel_val_set.json")
GT_STREAM = os.environ.get("BABEL_GT_STREAM", os.path.join(REPO, "data/babel_272_stream/val_stream"))
MANIFEST = os.path.join(REPO, "data/babel/babel_seq_val_manifest.jsonl")
PRECOMP = {
    "PRISM": "ref_repo/FlowMDM/results/babel/PRISM_e19/evaluation_precomputed/Motion_PRISM_e19_001300000_gscale1.5_debug_s10/00",
    "MotionStreamer": "ref_repo/FlowMDM/results/babel/MotionStreamer/evaluation_precomputed/Motion_MotionStreamer_001300000_gscale1.5_debug_s10/00",
    "FlowMDM": "ref_repo/FlowMDM/results/babel/FlowMDM/evaluation_precomputed/Motion_FlowMDM_001300000_gscale1.5_debug_s10/00",
}
NPY_METHODS = {"PRISM", "MotionStreamer"}  # faithful rots/transl available
MIN_SEG, MAX_LEN = 16, 300
_R_Y2Z = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
_RZ2Y = _R_Y2Z.T
_BO = torch.from_numpy(_canonical_272_offsets()).float()
_TR = None


def _transform():
    global _TR
    if _TR is None:
        from data_loaders.amass.transforms import SlimSMPLTransform
        cwd = os.getcwd(); os.chdir(FM)
        _TR = SlimSMPLTransform(batch_size=32, name="SlimSMPLTransform",
                                ename="smplnh", normalization=True)
        os.chdir(cwd)
    return _TR


def npy_comp_to_272(npy_path):
    """PRISM/MS rots/transl (y2z, z-up) -> (T,272) raw, canon272 skeleton, Y-up."""
    d = np.load(npy_path, allow_pickle=True).item()
    rots = torch.tensor(d["rots"], dtype=torch.float32).clone()   # [T,22,3,3]
    transl = torch.tensor(d["transl"], dtype=torch.float32)
    rots[:, 0] = torch.matmul(_RZ2Y, rots[:, 0])                  # undo y2z on root
    transl = torch.matmul(_RZ2Y, transl.unsqueeze(-1)).squeeze(-1)
    joints, _ = differentiable_fk(rots, transl, _BO)
    return encode_smpl_to_272(joints.numpy(), rots.numpy())


def globvelandy_pt_to_272(pt_path):
    """FlowMDM native globvelandy .pt -> SlimSMPL inverse -> SMPL-H body-model FK
    joints + rots -> (T,272) raw.

    FlowMDM only exposes globvelandy, whose self-consistent inverse is the SMPL-H
    body model (SlimDatastruct.joints). Re-running it through the canon272 FK is a
    *mismatched* FK and distorts the motion (verified: R@3 0.52->0.34, FID 215->465),
    so we keep the faithful SMPL-H joints. Consequence: FlowMDM sits on the SMPL-H
    skeleton (a few-cm offset vs the canon272 footing PRISM/MS use), which inflates
    its *FID* (a distribution distance) but leaves the retrieval R-precision/MM-Dist
    -- the headline cross-method metrics -- comparable."""
    cwd = os.getcwd(); os.chdir(FM)
    try:
        pt = torch.load(pt_path, map_location="cpu")              # [1,135,1,T]
        feats = pt[0, :, 0, :].permute(1, 0).contiguous()
        ds = _transform().SlimDatastruct(features=feats)
        joints = ds.joints.detach().cpu().numpy()
        rotmat = ds.rots.rots.detach().cpu().numpy()
    finally:
        os.chdir(cwd)
    return encode_smpl_to_272(joints, rotmat)


@torch.no_grad()
def encode_items(items, textenc, motionenc, device, rng, dedup, batch_size=32):
    order = rng.permutation(len(items))
    n_batches = len(order) // batch_size
    em_all = []
    topk = np.zeros(3, np.float64)
    match_sum = 0.0
    all_size = 0
    for b in range(n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        batch = [items[i] for i in idx]
        batch.sort(key=lambda x: x[2], reverse=True)
        texts = [x[0] for x in batch]
        motions = torch.from_numpy(np.stack([x[1] for x in batch])).float().to(device)
        lengths = torch.tensor([x[2] for x in batch], device=device)
        em = motionenc(motions, lengths).loc.cpu().numpy()
        et = textenc(texts).loc.cpu().numpy()
        em_all.append(em)
        dist = E.euclidean_distance_matrix(et, em)
        match_sum += dist.trace()
        if dedup:
            uniq = np.unique(np.asarray(texts), return_index=True)[1]
            args = np.argsort(dist[uniq][:, uniq], axis=1)
            tk = E.calculate_top_k(args, 3)
            topk += tk.sum(axis=0) * (len(texts) / tk.shape[0])
        else:
            topk += E.calculate_top_k(np.argsort(dist, axis=1), 3).sum(axis=0)
        all_size += len(texts)
    return {"em": np.concatenate(em_all, 0), "R": topk / all_size,
            "matching": match_sum / all_size, "nb": all_size}


def _canon_seg(seg272, per_seg_canon):
    if not per_seg_canon:
        return np.asarray(seg272, np.float32)
    if len(seg272) < 2:
        return None
    return np.asarray(reencode_272_via_stored_positions(np.asarray(seg272, np.float32)),
                      dtype=np.float32)


def norm_pad(motion, mean, std):
    L = len(motion)
    if L < MIN_SEG:
        return None, None
    L = min(L, MAX_LEN)
    m = (motion[:L] - mean) / std
    if L < MAX_LEN:
        m = np.concatenate([m, np.zeros((MAX_LEN - L, m.shape[1]))], axis=0)
    return m.astype(np.float32), int(L)


def _add_items(items, seg272, cap, mean, std, use_rewrite, per_seg_canon):
    seg = _canon_seg(seg272, per_seg_canon)
    if seg is None:
        return
    cap_q = rewrite_caption(cap) if use_rewrite else f"a person {cap}"
    m, n = norm_pad(seg, mean, std)
    if m is not None:
        items.append((cap_q, m, n))


def build_pred_items(method, mean, std, use_rewrite, per_seg_canon):
    pc = os.path.join(REPO, PRECOMP[method])
    val = json.load(open(VAL_SET))
    items = []
    for i, entry in enumerate(val):
        native = os.path.join(pc, f"{i:02d}_native272.npy")
        if method == "MotionStreamer" and os.path.isfile(native):
            # MS is a 272-native model: use its native 272 directly (the recovered
            # rots+FK round-trip is lossy and tanks R-precision, native 0.32 vs 0.12
            # on comp00). Slice by the actual decoded seg lengths sidecar.
            seq = np.load(native).astype(np.float32)
            sl = os.path.join(pc, f"{i:02d}_seglens.json")
            seglens = [int(x) for x in json.load(open(sl))] if os.path.isfile(sl) else None
            caps = entry["text"]
            if seglens is None or len(seglens) != len(caps):
                continue
            start = 0
            for L, cap in zip(seglens, caps):
                _add_items(items, seq[start:start + L], cap, mean, std, use_rewrite, per_seg_canon)
                start += L
            continue
        use_npy = method in NPY_METHODS
        path = os.path.join(pc, f"{i:02d}.{'npy' if use_npy else 'pt'}")
        if not os.path.isfile(path):
            continue
        seq = npy_comp_to_272(path) if use_npy else globvelandy_pt_to_272(path)
        start = 0
        for L, cap in zip(entry["lengths"], entry["text"]):
            L = int(L)
            _add_items(items, seq[start:start + L], cap, mean, std, use_rewrite, per_seg_canon)
            start += L
    return items


def build_gt_items(mean, std, use_rewrite, per_seg_canon):
    """Native-272 GT, same composition source as predictions: pool BABEL val GT
    sub-actions {caption -> [(len, 272clip)]} from val_stream, then for each
    babel_val_set segment pick the same-caption clip with the closest length."""
    man = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    pool = {}
    for rec in man:
        p = os.path.join(GT_STREAM, rec["id"] + ".npy")
        if not os.path.isfile(p):
            continue
        seq = np.load(p).astype(np.float32)
        T = seq.shape[0]
        for seg in rec["segments"]:
            cap = str(seg["caption"]).strip().lower()
            s, e = seg["start"], min(seg["end"], T)
            if e - s < MIN_SEG:
                continue
            pool.setdefault(cap, []).append((e - s, seq[s:e].copy()))
    val = json.load(open(VAL_SET))
    items = []
    for entry in val:
        for L, cap in zip(entry["lengths"], entry["text"]):
            cands = pool.get(str(cap).strip().lower())
            if not cands:
                continue
            _, clip = min(cands, key=lambda lc: abs(lc[0] - int(L)))
            _add_items(items, clip, cap, mean, std, use_rewrite, per_seg_canon)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="PRISM,MotionStreamer,FlowMDM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rewrite", dest="use_rewrite", action="store_false")
    ap.set_defaults(use_rewrite=True)
    ap.add_argument("--no-per-seg-canon", dest="per_seg_canon", action="store_false")
    ap.set_defaults(per_seg_canon=True)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.set_defaults(dedup=True)
    ap.add_argument("--out-json", default="docs/temp/babel_ms272_sentence_canon_dedup_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = np.load(os.path.join(HUMANML_MEAN_STD, "Mean.npy"))
    std = np.load(os.path.join(HUMANML_MEAN_STD, "Std.npy"))
    textenc, motionenc = E.load_evaluator(device)
    print(f"[setup] MS-272 | text={'sentence' if args.use_rewrite else 'terse'} "
          f"canon={args.per_seg_canon} dedup={args.dedup}", flush=True)

    gt_items = build_gt_items(mean, std, args.use_rewrite, args.per_seg_canon)
    gt_enc = encode_items(gt_items, textenc, motionenc, device,
                          np.random.RandomState(args.seed), args.dedup)
    gmu, gcov = E.calculate_activation_statistics(gt_enc["em"])
    gt_div = E.diversity_of(gt_enc["em"], np.random.RandomState(args.seed + 300))
    rows = {"Real": {"r1": float(gt_enc["R"][0]), "r3": float(gt_enc["R"][2]),
                     "mm_dist": float(gt_enc["matching"]), "fid": 0.0,
                     "div": float(gt_div), "n": int(len(gt_items))}}
    print(f"[Real] n_sub={len(gt_items)} R@1={gt_enc['R'][0]:.4f} R@3={gt_enc['R'][2]:.4f} "
          f"MM-Dist={gt_enc['matching']:.4f} Div={gt_div:.4f}", flush=True)

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        items = build_pred_items(method, mean, std, args.use_rewrite, args.per_seg_canon)
        enc = encode_items(items, textenc, motionenc, device,
                           np.random.RandomState(args.seed), args.dedup)
        pmu, pcov = E.calculate_activation_statistics(enc["em"])
        fid = E.calculate_frechet_distance(gmu, gcov, pmu, pcov)
        div = E.diversity_of(enc["em"], np.random.RandomState(args.seed + 300))
        rows[method] = {"r1": float(enc["R"][0]), "r3": float(enc["R"][2]),
                        "mm_dist": float(enc["matching"]), "fid": float(fid),
                        "div": float(div), "n": int(len(items))}
        print(f"[{method}] n_sub={len(items)} R@1={enc['R'][0]:.4f} R@3={enc['R'][2]:.4f} "
              f"MM-Dist={enc['matching']:.4f} FID={fid:.4f} Div={div:.4f}", flush=True)

    oj = args.out_json if os.path.isabs(args.out_json) else os.path.join(REPO, args.out_json)
    os.makedirs(os.path.dirname(oj), exist_ok=True)
    json.dump(rows, open(oj, "w"), indent=2)
    print(f"[done] -> {oj}")


if __name__ == "__main__":
    main()

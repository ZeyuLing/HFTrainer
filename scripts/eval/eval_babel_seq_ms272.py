#!/usr/bin/env python3
"""BABEL sequential-action generation evaluation (Table 3) with the public
MotionStreamer-272 evaluator.

Two metric blocks (per the paper / FlowMDM protocol, 30-frame transition window):

  Subseq. Quality  (per sub-action segment, text-conditioned):
      R@3 (up), FID (down, vs GT segments), Div (-> Real), MM-D (down)
  Transition Smoothness  (per boundary, 30-frame motion-only window):
      FID (down, vs GT windows), Div (-> Real),
      Peak Jerk (-> Real), Area Jerk (down)  -- on FK joints

Inputs
------
--manifest   data/babel/babel_seq_val_manifest.jsonl   (build_babel_seq_manifest.py)
--pred-dir   dir of <id>.npz with motion_272 (or motion_135) full sequences,
             OR omit / --real to evaluate the GT val_stream as the Real row.
The BABEL t2m mean/std are used (NOT HumanML3D).

The Real row uses the GT val_stream motions sliced by the same manifest, so it is
the upper bound under the identical segment / transition windowing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

import eval_motionstreamer_272 as E  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402
from hftrainer.motion.representation.motion272 import (  # noqa: E402
    reencode_272_via_stored_positions,
)


def per_seg_canon(seg272):
    """Re-canonicalize one 272 segment: first frame -> floor + xz origin + face
    +z (re-encode from stored positions + local rotations).

    Sub-segments sliced out of a continuous val_stream start at an arbitrary
    heading/translation, so they are OOD for the HumanML3D-trained evaluator
    (which expects each clip to begin canonical). Generated PRISM/MS segments are
    naturally canonical at frame 0, so without this step GT is unfairly deflated
    below the generators. Applying it uniformly to every method (idempotent for
    already-canonical clips) is the fair, in-distribution protocol.
    """
    seg = np.asarray(seg272, np.float32)
    if len(seg) < 2:
        return None
    return np.asarray(reencode_272_via_stored_positions(seg), dtype=np.float32)

BABEL_MEAN_STD = os.path.join(REPO, "data/babel/babel_272/t2m_babel_mean_std")
# The released Evaluator_272 (epoch=99.ckpt) was trained on HumanML3D-272 with
# these stats; feeding BABEL motions normalized with HumanML3D stats keeps the
# evaluator input distribution consistent (required to match the MS paper GT).
HUMANML_MEAN_STD = os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std")
# Allow pointing GT at a /dev/shm cache to bypass CephFS contention.
GT_STREAM = os.environ.get("BABEL_GT_STREAM", os.path.join(REPO, "data/babel_272_stream/val_stream"))
TRANSITION_LEN = 30
MAX_LEN = 300
MIN_SEG = 16          # relaxed (BABEL sub-actions are often <60 frames)


def norm_pad(motion, mean, std, length=None):
    """(T,272) raw -> ((MAX_LEN,272) normalized+zero-padded, used_len)."""
    L = len(motion) if length is None else min(len(motion), length)
    if L < MIN_SEG:
        return None, None
    L = min(L, MAX_LEN)
    m = (motion[:L] - mean) / std
    if L < MAX_LEN:
        m = np.concatenate([m, np.zeros((MAX_LEN - L, m.shape[1]))], axis=0)
    return m.astype(np.float32), int(L)


@torch.no_grad()
def encode_motion_only(items, motionencoder, device, batch_size=32):
    """items: list of (motion(MAX_LEN,272), len). Returns embeddings (N,256)."""
    em = []
    for b in range(0, len(items), batch_size):
        batch = items[b:b + batch_size]
        batch.sort(key=lambda x: x[1], reverse=True)
        motions = torch.from_numpy(np.stack([x[0] for x in batch])).float().to(device)
        lengths = torch.tensor([x[1] for x in batch], device=device)
        em.append(motionencoder(motions, lengths).loc.cpu().numpy())
    if not em:
        return np.zeros((0, 256), np.float32)
    return np.concatenate(em, 0)


def fk_272(m272):
    """(T,272) -> (T,22,3) world joints (uses the stored positions in 272)."""
    from hftrainer.datasets.motion.representation.humanml_repr import (
        recover_272_stored_positions,
    )
    return np.asarray(recover_272_stored_positions(np.asarray(m272, np.float32)),
                      dtype=np.float32)


def per_frame_jerk(pos):
    """pos (T,22,3) -> per-frame jerk scalar curve (T,), max over joints of
    |d^3 pos/dt^3| summed over xyz (FlowMDM metrics.calculate_jerk)."""
    if pos.shape[0] < 4:
        return None
    vel = np.diff(pos, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)               # (T-3,22,3)
    j = np.abs(jerk).sum(-1)                   # (T-3,22)
    j = j.max(-1)                              # (T-3,) max over joints
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--pred-dir", default=None,
                    help="dir of <id>.npz (motion_272 or motion_135); omit/--real for GT")
    ap.add_argument("--real", action="store_true", help="evaluate GT val_stream (Real row)")
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--min-total", type=int, default=0,
                    help="keep only episodes with total_frames >= this")
    ap.add_argument("--max-total", type=int, default=0,
                    help="keep only episodes with total_frames <= this (0=no cap); "
                         "use to match the generation length filter so Real/ours "
                         "are scored on identical episodes")
    ap.add_argument("--caption-template", default="a person {cap}",
                    help="fallback reformat for terse BABEL labels when --no-rewrite. "
                         "Use '{cap}' to keep the raw label.")
    ap.add_argument("--no-rewrite", dest="use_rewrite", action="store_false",
                    help="disable the faithful LLM rewrite for the retrieval query and "
                         "fall back to --caption-template. By DEFAULT the same faithful "
                         "rewrite_caption() used to condition generation is also used as "
                         "the retrieval query for ALL rows (incl. Real), so the text-motion "
                         "alignment metric is consistent across generation and evaluation.")
    ap.set_defaults(use_rewrite=True)
    ap.add_argument("--keep-transition-caps", action="store_true",
                    help="by default action-agnostic 'transition' sub-segments are "
                         "dropped from the Subseq-Quality R-precision/FID (they have no "
                         "retrievable action); pass this to keep them.")
    ap.add_argument("--subseq-proto", default="raw", choices=["raw", "standard"],
                    help="raw=slice+pad (keeps short clips); standard=dataset_eval_t2m "
                         "crop_and_norm (drop <60f, random unit-rounded crop) to match "
                         "the MS/Guo R-precision protocol.")
    ap.add_argument("--mean-std", default="humanml", choices=["babel", "humanml"],
                    help="normalization stats fed to the (HumanML3D-trained) evaluator. "
                         "The released Evaluator_272 was trained on HumanML3D-272 stats, "
                         "so 'humanml' is required to reproduce the paper's GT numbers.")
    ap.add_argument("--dedup", dest="dedup", action="store_true", default=True,
                    help="FlowMDM-style per-batch caption dedup for R-precision "
                         "(default ON; BABEL terse labels repeat -> GT otherwise "
                         "unfairly deflated below the generators).")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.add_argument("--rprec-batching", default="balanced",
                    choices=["balanced", "random", "unique"],
                    help="batch construction for R-precision. balanced spreads "
                         "identical BABEL action captions across mini-batches "
                         "before drop-last; random reproduces the legacy shuffled "
                         "MS/Guo batching; unique additionally enforces zero "
                         "duplicate captions inside every batch.")
    ap.add_argument("--rprec-batch-size", type=int, default=32,
                    help="mini-batch size for R-precision. BABEL unique batching "
                         "requires a small enough value; for the common manifest, "
                         "8 is the largest duplicate-free size.")
    ap.add_argument("--per-seg-canon", dest="per_seg_canon", action="store_true",
                    default=True,
                    help="re-canonicalize every sub-segment (floor + xz origin + "
                         "face +z) before eval (default ON). Sliced GT segments "
                         "are otherwise OOD vs naturally-canonical generated ones.")
    ap.add_argument("--no-per-seg-canon", dest="per_seg_canon", action="store_false")
    ap.add_argument("--gt-stream-dir", default=None,
                    help="override GT stream source (default val_stream native 272). "
                         "Point at gt_fk272 / gt_ik272 (.npz motion_272) for a "
                         "same-source GT encoded through the same chain as preds.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ms_dir = BABEL_MEAN_STD if args.mean_std == "babel" else HUMANML_MEAN_STD
    mean = np.load(os.path.join(ms_dir, "Mean.npy"))
    std = np.load(os.path.join(ms_dir, "Std.npy"))
    print(f"[setup] mean_std source={args.mean_std} dir={ms_dir}", flush=True)

    man = [json.loads(l) for l in open(os.path.join(REPO, args.manifest)) if l.strip()]
    if args.min_total:
        man = [m for m in man if m["total_frames"] >= args.min_total]
    if args.max_total:
        man = [m for m in man if m["total_frames"] <= args.max_total]
    if args.max_episodes:
        man = man[:args.max_episodes]

    pred_dir = None
    if args.pred_dir:
        pred_dir = args.pred_dir if os.path.isabs(args.pred_dir) else os.path.join(REPO, args.pred_dir)

    def load_seq_pred(sid):
        p = os.path.join(pred_dir, sid + ".npz")
        if not os.path.isfile(p):
            return None
        d = np.load(p, allow_pickle=True)
        if "motion_272" in d:
            return np.asarray(d["motion_272"], np.float32)
        if "motion_135" in d:
            from motionstreamer_272_encoder import motion135_to_272
            return motion135_to_272(np.asarray(d["motion_135"], np.float32))
        return None

    gt_stream_dir = GT_STREAM
    if args.gt_stream_dir:
        gt_stream_dir = (args.gt_stream_dir if os.path.isabs(args.gt_stream_dir)
                         else os.path.join(REPO, args.gt_stream_dir))

    def load_seq_gt(sid):
        # native val_stream (.npy) OR a same-source re-encoded stream (.npz with
        # motion_272), e.g. gt_fk272 / gt_ik272 to match the prediction chain.
        p_npy = os.path.join(gt_stream_dir, sid + ".npy")
        if os.path.isfile(p_npy):
            return np.load(p_npy).astype(np.float32)
        p_npz = os.path.join(gt_stream_dir, sid + ".npz")
        if os.path.isfile(p_npz):
            return np.asarray(np.load(p_npz, allow_pickle=True)["motion_272"],
                              np.float32)
        return None

    use_gt = args.real or pred_dir is None
    load_pred = load_seq_gt if use_gt else load_seq_pred

    print(f"[setup] tag={args.tag} episodes={len(man)} source={'GT' if use_gt else pred_dir}", flush=True)
    print(f"[setup] rprec_batching={args.rprec_batching} "
          f"batch_size={args.rprec_batch_size} dedup={args.dedup}", flush=True)
    textenc, motionenc = E.load_evaluator(device)
    print("[setup] evaluator loaded", flush=True)

    # ---- collect items (predictions/GT-eval-source AND the GT reference) ----
    rng = np.random.RandomState(args.seed)

    def collect(loader):
        sub_items, trans_items, jerk_curves = [], [], []
        n_ep = n_seg = n_tr = 0
        for rec in man:
            seq = loader(rec["id"])
            if seq is None:
                continue
            T = seq.shape[0]
            n_ep += 1
            for seg in rec["segments"]:
                cap_raw = str(seg["caption"]).strip()
                if (not args.keep_transition_caps) and cap_raw.lower() == "transition":
                    continue
                cap = (rewrite_caption(cap_raw) if args.use_rewrite
                       else args.caption_template.format(cap=cap_raw))
                s, e = seg["start"], min(seg["end"], T)
                raw_seg = seq[s:e].astype(np.float32)
                if args.per_seg_canon:
                    raw_seg = per_seg_canon(raw_seg)
                    if raw_seg is None:
                        continue
                if args.subseq_proto == "standard":
                    # validated dataset_eval_t2m protocol: drop <60f, random
                    # unit-rounded crop, normalize, pad (matches MS/Guo eval).
                    m, L = E.crop_and_norm(raw_seg, mean, std, rng)
                else:
                    m, L = norm_pad(raw_seg, mean, std)
                if m is None:
                    continue
                sub_items.append((cap, m, L))
                n_seg += 1
            for b in rec["boundaries"]:
                a, c = b - TRANSITION_LEN // 2, b + (TRANSITION_LEN - TRANSITION_LEN // 2)
                a, c = max(0, a), min(T, c)
                win = seq[a:c]
                if win.shape[0] < 8:
                    continue
                m, L = norm_pad(win, mean, std, length=TRANSITION_LEN)
                if m is not None:
                    trans_items.append((m, L))
                    n_tr += 1
                jc = per_frame_jerk(fk_272(seq[a:c]))
                if jc is not None and jc.size:
                    jerk_curves.append(jc)
        return sub_items, trans_items, jerk_curves, (n_ep, n_seg, n_tr)

    pred_sub, pred_tr, pred_jerk, pstat = collect(load_pred)
    print(f"[pred] ep={pstat[0]} seg={pstat[1]} trans={pstat[2]}", flush=True)
    gt_sub, gt_tr, gt_jerk, gstat = collect(load_seq_gt)
    print(f"[gt]   ep={gstat[0]} seg={gstat[1]} trans={gstat[2]}", flush=True)

    # ---- Subseq Quality (text-conditioned) ----
    pred_enc = E.encode_items(pred_sub, textenc, motionenc, device,
                              np.random.RandomState(args.seed), dedup=args.dedup,
                              batch_size=args.rprec_batch_size,
                              batch_mode=args.rprec_batching)
    gt_enc = E.encode_items(gt_sub, textenc, motionenc, device,
                            np.random.RandomState(args.seed), dedup=args.dedup,
                            batch_size=args.rprec_batch_size,
                            batch_mode=args.rprec_batching)
    pred_div = E.diversity_of(pred_enc["em"], np.random.RandomState(args.seed + 100))
    pmu, pcov = E.calculate_activation_statistics(pred_enc["em"])
    gmu, gcov = E.calculate_activation_statistics(gt_enc["em"])
    sub_fid = E.calculate_frechet_distance(gmu, gcov, pmu, pcov)

    # ---- Transition Smoothness ----
    pe = encode_motion_only(pred_tr, motionenc, device)
    ge = encode_motion_only(gt_tr, motionenc, device)
    tr_div = E.diversity_of(pe, np.random.RandomState(args.seed + 200)) if len(pe) > 2 else float("nan")
    tpmu, tpcov = E.calculate_activation_statistics(pe)
    tgmu, tgcov = E.calculate_activation_statistics(ge)
    tr_fid = E.calculate_frechet_distance(tgmu, tgcov, tpmu, tpcov)

    # Peak/Area Jerk: average the per-frame jerk curves across episodes (pad to
    # common length = min curve length), Peak = max(mean curve); Area Jerk =
    # sum|mean_curve - GT_mean_curve| (GT reference recomputed in this 272+FK
    # space, not FlowMDM's 135-dim constant).
    def mean_curve(curves):
        if not curves:
            return None
        Lmin = min(len(c) for c in curves)
        arr = np.stack([c[:Lmin] for c in curves], 0)
        return arr.mean(0)
    pcurve = mean_curve(pred_jerk)
    gcurve = mean_curve(gt_jerk)
    peak_jerk = float(pcurve.max()) if pcurve is not None else float("nan")
    gt_peak = float(gcurve.max()) if gcurve is not None else float("nan")
    if pcurve is not None and gcurve is not None:
        Lm = min(len(pcurve), len(gcurve))
        area_jerk = float(np.abs(pcurve[:Lm] - gcurve[:Lm]).sum())
    else:
        area_jerk = float("nan")

    res = {
        "tag": args.tag,
        "eval_args": {
            "manifest": args.manifest,
            "pred_dir": args.pred_dir,
            "source": "GT" if use_gt else pred_dir,
            "gt_stream_dir": gt_stream_dir,
            "real": bool(use_gt),
            "seed": int(args.seed),
            "device": str(device),
            "max_episodes": int(args.max_episodes),
            "min_total": int(args.min_total),
            "max_total": int(args.max_total),
            "caption_template": args.caption_template,
            "use_rewrite": bool(args.use_rewrite),
            "keep_transition_caps": bool(args.keep_transition_caps),
            "subseq_proto": args.subseq_proto,
            "mean_std": args.mean_std,
            "dedup": bool(args.dedup),
            "rprec_batching": args.rprec_batching,
            "rprec_batch_size": int(args.rprec_batch_size),
            "per_seg_canon": bool(args.per_seg_canon),
        },
        "n_episodes": pstat[0], "n_segments": pstat[1], "n_transitions": pstat[2],
        "subseq": {
            "r_precision": pred_enc["R"].tolist(),
            "r3": float(pred_enc["R"][2]),
            "fid": float(sub_fid),
            "diversity": float(pred_div),
            "mm_dist": float(pred_enc["matching"]),
            "nb": int(pred_enc["nb"]),
            "rprec_batching": pred_enc["batch_mode"],
            "rprec_batch_size": int(args.rprec_batch_size),
            "batch_stats": pred_enc["batch_stats"],
        },
        "transition": {
            "fid": float(tr_fid),
            "diversity": float(tr_div),
            "peak_jerk": peak_jerk,
            "area_jerk": area_jerk,
            "gt_peak_jerk": gt_peak,
            "n_pred": int(len(pe)), "n_gt": int(len(ge)),
        },
        "gt_ref_subseq": {
            "r3": float(gt_enc["R"][2]),
            "diversity": float(E.diversity_of(gt_enc["em"], np.random.RandomState(args.seed + 300))),
            "mm_dist": float(gt_enc["matching"]),
            "rprec_batching": gt_enc["batch_mode"],
            "rprec_batch_size": int(args.rprec_batch_size),
            "batch_stats": gt_enc["batch_stats"],
        },
    }
    print("\n==== Subseq:  R@3=%.4f FID=%.3f Div=%.3f MM-D=%.3f ===="
          % (res["subseq"]["r3"], res["subseq"]["fid"], res["subseq"]["diversity"], res["subseq"]["mm_dist"]))
    print("==== Trans:   FID=%.3f Div=%.3f PeakJerk=%.4f AreaJerk=%.3f (GTpeak=%.4f) ===="
          % (tr_fid, tr_div, peak_jerk, area_jerk, gt_peak))

    if args.out_json:
        oj = args.out_json if os.path.isabs(args.out_json) else os.path.join(REPO, args.out_json)
        os.makedirs(os.path.dirname(oj), exist_ok=True)
        json.dump(res, open(oj, "w"), indent=2)
        print(f"[done] -> {oj}")


if __name__ == "__main__":
    main()

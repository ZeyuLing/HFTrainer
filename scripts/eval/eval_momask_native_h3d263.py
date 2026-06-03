#!/usr/bin/env python3
"""Native MoMask evaluator on a reconstructed HumanML3D-263 (20 fps) test set.

Loads MoMask's ``text_mot_match`` evaluator (Comp_v6_KLD005 mean/std + glove)
and computes FID / R-Precision / MM-Dist / Diversity following MoMask's own
``utils.eval_t2m`` protocol.

Two modes:
    --mode gt-only : real motions only (sanity check; FID should be ~0).
    --mode pred    : real vs predicted motions (read from --pred_dir).

The reconstructed test set is expected to be the output of
``tools/build_h3d263_test_from_h3d272.py`` -- i.e. a directory with::

    new_joint_vecs/<id>.npy   (T, 263)
    new_joints/<id>.npy        (T, 22, 3)
    Mean.npy / Std.npy         (263,)
    test.txt                   list of <id> with valid reconstructions

Captions are read from ``<src_h3d272>/texts/<id>.txt`` (HumanML3D-style).

Usage::

    python3 tools/eval_momask_native_h3d263.py \
        --recon_root work_dirs/momask_eval/h3d263_test_recon \
        --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --momask_root ref_repo/Momask/momask-codes \
        --mode gt-only \
        --num_repeats 1 \
        --output work_dirs/momask_eval/momask_native_gt_only.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# tokenisation helpers (mirror Text2MotionDatasetEval.__getitem__)
# ---------------------------------------------------------------------------

def _tokenise(tokens: List[str], max_text_len: int):
    if len(tokens) < max_text_len:
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
    return tokens, sent_len


def _read_h3d_texts(text_file: Path) -> List[Dict]:
    """Read HumanML3D-style ``texts/<id>.txt``. Returns list of dicts with
    ``caption``, ``tokens``, ``f_tag``, ``to_tag`` (only the f_tag==0,to_tag==0
    full-clip captions are useful for our 'whole motion' eval)."""
    out = []
    if not text_file.exists():
        return out
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap, toks, ftag, ttag = parts[0], parts[1].split(), parts[2], parts[3]
        try:
            ftag_v = float(ftag)
            ttag_v = float(ttag)
        except ValueError:
            continue
        out.append({
            "caption": cap,
            "tokens": toks,
            "f_tag": 0.0 if (np.isnan(ftag_v) if ftag_v != ftag_v else False) else ftag_v,
            "to_tag": 0.0 if (np.isnan(ttag_v) if ttag_v != ttag_v else False) else ttag_v,
        })
    return out


# ---------------------------------------------------------------------------
# core eval (mirrors evaluation_mask_transformer_test, no MM channel)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recon_root", required=True)
    p.add_argument("--src_h3d272", required=True)
    p.add_argument("--momask_root", required=True)
    p.add_argument("--pred_dir", default=None,
                   help="Directory with <id>.npy 263-dim predictions (un-standardised).")
    p.add_argument("--mode", choices=["gt-only", "pred"], default="gt-only")
    p.add_argument("--num_repeats", type=int, default=20)
    p.add_argument("--max_motion_length", type=int, default=196)
    p.add_argument("--unit_length", type=int, default=4)
    p.add_argument("--max_text_len", type=int, default=20)
    p.add_argument("--diversity_times", type=int, default=300,
                   help="random pair count for Diversity (MoMask uses 300).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--split", default=None,
                   help="Optional path to a custom id list (overrides recon_root/test.txt).")
    p.add_argument("--drop_mirrored", action="store_true",
                   help="Exclude HumanML3D mirrored clips (ids starting with 'M'). The "
                        "official published 'Real' metrics use unique (non-mirrored) "
                        "motions; including mirror pairs deflates Diversity.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.momask_root).resolve()))
    from utils.word_vectorizer import WordVectorizer  # noqa: E402
    from models.t2m_eval_wrapper import EvaluatorModelWrapper  # noqa: E402
    from utils.metrics import (
        calculate_R_precision,
        euclidean_distance_matrix,
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
    )

    device = args.device
    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)

    # ----------- evaluator wrapper -----------
    class Opt:
        pass
    opt = Opt()
    opt.dataset_name = "t2m"
    opt.device = device
    opt.checkpoints_dir = str(Path(args.momask_root).resolve() / "checkpoints")
    opt.dim_movement_enc_hidden = 512
    opt.dim_movement_latent = 512
    opt.unit_length = args.unit_length
    eval_wrapper = EvaluatorModelWrapper(opt)
    print(f"[+] evaluator loaded")

    # ----------- glove word vectoriser -----------
    glove_dir = Path(args.momask_root).resolve() / "glove"
    w_vectorizer = WordVectorizer(str(glove_dir), 'our_vab')

    # ----------- mean / std -----------
    mean = np.load(str(Path(args.recon_root) / "Mean.npy"))
    std = np.load(str(Path(args.recon_root) / "Std.npy"))

    # ----------- build sample list -----------
    split_file = Path(args.split) if args.split else (Path(args.recon_root) / "test.txt")
    test_ids = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]
    if args.drop_mirrored:
        n0 = len(test_ids)
        test_ids = [s for s in test_ids if not s.startswith("M")]
        print(f"[+] drop_mirrored: {n0} -> {len(test_ids)} ids (removed {n0 - len(test_ids)} mirrored)")
    src = Path(args.src_h3d272)

    samples = []  # each: dict(name, motion_gt(263), text_dict, m_length)
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    skipped_no_text = skipped_no_pred = skipped_too_short = 0
    min_len = 40

    for sid in tqdm(test_ids, desc="loading", ncols=80):
        m = np.load(str(Path(args.recon_root) / "new_joint_vecs" / f"{sid}.npy"))
        if len(m) < min_len or len(m) >= 200:
            skipped_too_short += 1
            continue
        text_list = _read_h3d_texts(src / "texts" / f"{sid}.txt")
        text_list = [t for t in text_list if t["f_tag"] == 0.0 and t["to_tag"] == 0.0]
        if not text_list:
            skipped_no_text += 1
            continue
        if args.mode == "pred":
            assert pred_dir is not None
            pp = pred_dir / f"{sid}.npy"
            if not pp.exists():
                skipped_no_pred += 1
                continue
            m_pred = np.load(str(pp))
            if m_pred.ndim != 2 or m_pred.shape[1] != 263 or len(m_pred) < min_len:
                skipped_too_short += 1
                continue
            # Match length: clip to min(len(gt), len(pred))
            t_eff = min(len(m), len(m_pred))
            t_eff = (t_eff // args.unit_length) * args.unit_length
            if t_eff < min_len:
                skipped_too_short += 1
                continue
            samples.append({
                "name": sid,
                "motion_gt": m[:t_eff],
                "motion_pred": m_pred[:t_eff],
                "text_list": text_list,
                "length": t_eff,
            })
        else:
            t_eff = (len(m) // args.unit_length) * args.unit_length
            if t_eff < min_len:
                skipped_too_short += 1
                continue
            samples.append({
                "name": sid,
                "motion_gt": m[:t_eff],
                "text_list": text_list,
                "length": t_eff,
            })
        if args.max_samples and len(samples) >= args.max_samples:
            break

    print(f"[+] {len(samples)} valid samples; skipped: too_short={skipped_too_short}, "
          f"no_text={skipped_no_text}, no_pred={skipped_no_pred}")

    # ----------- run repeats -----------
    rprec_list = []
    fid_list = []
    div_list = []
    div_real_list = []
    mm_dist_list = []
    mm_dist_real_list = []
    rprec_real_list = []

    for repeat in range(args.num_repeats):
        random.seed(args.seed + repeat)
        # reset rng for caption sampling

        # Build the per-sample tensors for this repeat (random caption choice).
        word_embs_all = []
        pos_oh_all = []
        sent_len_all = []
        motion_gt_all = []
        motion_pred_all = []
        m_length_all = []

        for s in samples:
            t_eff = s["length"]
            text_data = random.choice(s["text_list"])
            tokens = text_data["tokens"]
            tokens, sent_len = _tokenise(tokens, args.max_text_len)
            word_embs = []
            pos_one_hots = []
            for tok in tokens:
                we, po = w_vectorizer[tok]
                word_embs.append(we)
                pos_one_hots.append(po)
            word_embs = np.stack(word_embs)
            pos_one_hots = np.stack(pos_one_hots)

            mg = (s["motion_gt"] - mean) / std
            if t_eff < args.max_motion_length:
                pad = np.zeros((args.max_motion_length - t_eff, mg.shape[1]), dtype=mg.dtype)
                mg = np.concatenate([mg, pad], axis=0)
            motion_gt_all.append(mg)
            if args.mode == "pred":
                mp = (s["motion_pred"] - mean) / std
                if t_eff < args.max_motion_length:
                    pad = np.zeros((args.max_motion_length - t_eff, mp.shape[1]), dtype=mp.dtype)
                    mp = np.concatenate([mp, pad], axis=0)
                motion_pred_all.append(mp)
            word_embs_all.append(word_embs)
            pos_oh_all.append(pos_one_hots)
            sent_len_all.append(sent_len)
            m_length_all.append(t_eff)

        word_embs_all = torch.from_numpy(np.stack(word_embs_all)).float()
        pos_oh_all = torch.from_numpy(np.stack(pos_oh_all)).float()
        sent_len_all = torch.tensor(sent_len_all)
        motion_gt_all = torch.from_numpy(np.stack(motion_gt_all)).float()
        m_length_all = torch.tensor(m_length_all)
        if args.mode == "pred":
            motion_pred_all = torch.from_numpy(np.stack(motion_pred_all)).float()

        # batched embedding
        em_gt_chunks = []
        et_chunks = []
        em_pred_chunks = []
        bsz = 32
        for i in tqdm(range(0, len(samples), bsz), desc=f"rep{repeat} embed", ncols=80):
            sl = slice(i, i + bsz)
            we = word_embs_all[sl]
            po = pos_oh_all[sl]
            sn = sent_len_all[sl]
            mg = motion_gt_all[sl]
            ml = m_length_all[sl]
            # MoMask's TextEncoderBiGRUCo uses pack_padded_sequence with
            # enforce_sorted=True. Sort the batch by sent_len descending so
            # the text encoder is happy; then undo with cap_inv.
            cap_order = torch.argsort(sn, descending=True).cpu().numpy()
            cap_inv = np.empty_like(cap_order)
            cap_inv[cap_order] = np.arange(len(cap_order))
            we = we[cap_order]
            po = po[cap_order]
            sn = sn[cap_order]
            mg = mg[cap_order]
            ml = ml[cap_order]
            et, em_gt = eval_wrapper.get_co_embeddings(we, po, sn, mg, ml)
            # Wrapper's internal align_idx scrambles both et and em the same
            # way, so they remain paired. Across batches we don't need to
            # restore the cap_order: paired (et, em) already form a valid
            # pairing within the batch, which is all that R-Precision and
            # MM-Dist need.
            em_gt_chunks.append(em_gt.cpu().numpy())
            et_chunks.append(et.cpu().numpy())
            if args.mode == "pred":
                mp = motion_pred_all[sl][cap_order]
                _, em_pred = eval_wrapper.get_co_embeddings(we, po, sn, mp, ml)
                em_pred_chunks.append(em_pred.cpu().numpy())

        em_gt = np.concatenate(em_gt_chunks, axis=0)
        et = np.concatenate(et_chunks, axis=0)
        em_pred = np.concatenate(em_pred_chunks, axis=0) if args.mode == "pred" else em_gt

        # compute metrics across the whole pool (MoMask sums per-batch and
        # divides by nb_sample; with a single-pool eval we can do the same
        # via batched chunks of bsz=32, which is exactly MoMask's protocol).
        # Drop the final incomplete batch if smaller than top_k=3 -- mirrors
        # MoMask's typical eval setup (drop_last=True).
        TOP_K = 3
        n = len(samples)
        nb_sample = 0
        rprec_real_acc = np.zeros(TOP_K)
        rprec_acc = np.zeros(TOP_K)
        mm_dist_real_acc = 0.0
        mm_dist_acc = 0.0
        for i in range(0, n, bsz):
            sl = slice(i, i + bsz)
            et_b = et[sl]
            em_b = em_gt[sl]
            em_p_b = em_pred[sl]
            bs_actual = len(et_b)
            if bs_actual <= TOP_K:
                # too small for top-3 R-precision; drop the residual (matches
                # MoMask's drop_last=True dataloader behaviour).
                continue
            rprec_real_acc += calculate_R_precision(et_b, em_b, top_k=TOP_K, sum_all=True)
            mm_dist_real_acc += euclidean_distance_matrix(et_b, em_b).trace()
            rprec_acc += calculate_R_precision(et_b, em_p_b, top_k=TOP_K, sum_all=True)
            mm_dist_acc += euclidean_distance_matrix(et_b, em_p_b).trace()
            nb_sample += bs_actual

        rprec_real = rprec_real_acc / nb_sample
        rprec = rprec_acc / nb_sample
        mm_dist_real = mm_dist_real_acc / nb_sample
        mm_dist = mm_dist_acc / nb_sample

        gt_mu, gt_cov = calculate_activation_statistics(em_gt)
        if args.mode == "pred":
            mu, cov = calculate_activation_statistics(em_pred)
            try:
                fid = float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov))
            except ValueError as e:
                # Small-sample degeneracy (singular covariance -> sqrtm imaginary
                # blow-up). FID needs n >> embedding-dim (512); for pilot n<~512
                # FID is undefined. R-Precision/MM-Dist/Diversity remain valid.
                print(f"  [warn] FID undefined (n={len(samples)} too small): {e}")
                fid = float("nan")
        else:
            fid = 0.0  # by construction GT vs GT is 0

        div_t = min(args.diversity_times, n - 1)  # calculate_diversity needs n > div_t
        div_real = float(calculate_diversity(em_gt, div_t))
        div_pred = float(calculate_diversity(em_pred, div_t))

        rprec_list.append([float(x) for x in rprec])
        rprec_real_list.append([float(x) for x in rprec_real])
        fid_list.append(float(fid))
        div_list.append(div_pred)
        div_real_list.append(div_real)
        mm_dist_list.append(float(mm_dist))
        mm_dist_real_list.append(float(mm_dist_real))

        print(f"[rep {repeat}] FID={fid:.4f} R-P={rprec.round(4).tolist()} "
              f"MM-Dist={mm_dist:.4f} Div={div_pred:.4f} | GT R-P={rprec_real.round(4).tolist()} "
              f"GT MM-Dist={mm_dist_real:.4f} GT Div={div_real:.4f}")

    def _summary(name, vals):
        a = np.array(vals)
        if a.ndim == 1:
            return {"mean": float(a.mean()), "std": float(a.std())}
        return {"mean": [float(x) for x in a.mean(0)], "std": [float(x) for x in a.std(0)]}

    summary = {
        "mode": args.mode,
        "num_repeats": args.num_repeats,
        "n_samples": len(samples),
        "fid": _summary("fid", fid_list),
        "r_precision": _summary("rprec", rprec_list),
        "matching_score": _summary("mm_dist", mm_dist_list),
        "diversity": _summary("div", div_list),
        "r_precision_real": _summary("rprec_real", rprec_real_list),
        "matching_score_real": _summary("mm_dist_real", mm_dist_real_list),
        "diversity_real": _summary("div_real", div_real_list),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"[+] wrote {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

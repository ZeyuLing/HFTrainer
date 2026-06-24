#!/usr/bin/env python3
"""Cross-evaluate indexed 272-dim T2M predictions on the **HumanML3D-263**
evaluator.

Predictions come from a generation script that keys outputs by the deterministic
``MotionStreamer272Evaluator.load_test_pairs()`` index (e.g.
``motionmillion_h3d272.py`` / ``hymotion_t2m_h3d272.py``). Each ``<idx>.npy`` is
a raw 272-dim motion; we convert it to HumanML3D-263 with
``hftrainer.motion.representation.convert.motion272_to_hml263``. By default the
script keeps the older pred-only sanity mode (retrieval / matching / diversity
with ``motion_gt = motion_pred``). Pass ``--with_fid`` to convert the paired
MS272 GT clips too and score full pred-vs-real HumanML263 metrics, including
FID, without collapsing duplicate captions or sub-clips by ``name``.

Example:
    python3 scripts/eval/eval_272dir_h3d263.py \
        --pred_dir outputs/evaluation/motionmillion_h3d272/mm_272 \
        --out_json outputs/evaluation/motionmillion_h3d272/metrics_h3d263.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEFAULT_DATA = REPO / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"
MIN_LEN = 40
UNIT = 4


def _simple_caption_tokens(caption: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+|[0-9]+", caption.lower())
    return [f"{w}/OTHER" for w in words] or ["unk/OTHER"]


def _valid_hml263(m263: np.ndarray) -> bool:
    return (
        m263.ndim == 2
        and m263.shape[1] == 263
        and len(m263) >= MIN_LEN
        and np.isfinite(m263).all()
    )


def _as_hml263(converted) -> np.ndarray:
    # motion272_to_hml263 delegates to humanml272_to_humanml263, whose public
    # return is (m263, joints263). Older callers sometimes expected only m263.
    if isinstance(converted, tuple):
        converted = converted[0]
    return np.asarray(converted, dtype=np.float32)


def _convert_one(args):
    idx, name, caption, pred_path, gt272, with_fid = args
    try:
        from hftrainer.motion.representation.convert import motion272_to_hml263

        m272 = np.load(pred_path).astype(np.float32)
        if m272.ndim != 2 or m272.shape[1] != 272 or len(m272) < MIN_LEN:
            return None
        pred263 = _as_hml263(motion272_to_hml263(m272))
        if not _valid_hml263(pred263):
            return None
        gt263 = None
        if with_fid:
            gt272 = np.asarray(gt272, dtype=np.float32)
            if gt272.ndim != 2 or gt272.shape[1] != 272 or len(gt272) < MIN_LEN:
                return None
            gt263 = _as_hml263(motion272_to_hml263(gt272))
            if not _valid_hml263(gt263):
                return None
        return idx, name, caption, pred263, gt263
    except Exception as e:  # noqa: BLE001
        return ("__error__", f"{idx}:{name}: {e}")


def _text_list_for_pair(data_root: Path, name: str, caption: str, read_h3d_texts):
    texts = read_h3d_texts(data_root / "texts" / f"{name}.txt")
    matched = [t for t in texts if t["caption"] == caption]
    if matched:
        return [matched[0]]
    return [{
        "caption": caption,
        "tokens": _simple_caption_tokens(caption),
        "f_tag": 0.0,
        "to_tag": 0.0,
    }]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="dir of <idx:06d>.npy raw-272 preds")
    p.add_argument("--data_root", default=str(DEFAULT_DATA))
    p.add_argument("--ckpt_dir", default=None, help="263 evaluator dir (text_mot_match.tar + glove/ + meta/)")
    p.add_argument("--n_repeats", type=int, default=20)
    p.add_argument("--caption_selection", default="random", choices=["first", "random"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--with_fid", action="store_true",
                   help="also convert paired MS272 GT and compute full pred-vs-real FID")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator, read_h3d_texts
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    pred_dir = Path(args.pred_dir)
    data_root = Path(args.data_root)

    ev272 = MotionStreamer272Evaluator(device="cpu")
    pairs = ev272.load_test_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    jobs = []
    for idx, (name, caption, gt, ml) in enumerate(pairs):
        pf = pred_dir / f"{idx:06d}.npy"
        if pf.exists():
            jobs.append((idx, name, caption, str(pf), gt if args.with_fid else None, args.with_fid))
    print(
        f"[263] converting {len(jobs)} indexed preds 272->263 "
        f"(with_fid={args.with_fid}, workers={args.workers})...",
        flush=True,
    )

    converted = []
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_convert_one, jobs):
            if res is None:
                continue
            if res[0] == "__error__":
                errors += 1
                if errors <= 5:
                    print(f"[263] convert FAIL {res[1]}", flush=True)
                continue
            converted.append(res)
    print(f"[263] converted={len(converted)} errors={errors}", flush=True)

    # Build indexed samples. Keeping one sample per pair preserves duplicate
    # full-clip captions and time-tagged sub-clips from the MS272 evaluator.
    samples = []
    for idx, name, caption, pred263, gt263 in converted:
        tl = _text_list_for_pair(data_root, name, caption, read_h3d_texts)
        ref = gt263 if args.with_fid else pred263
        t_eff = (min(len(pred263), len(ref)) // UNIT) * UNIT
        if t_eff < MIN_LEN:
            continue
        motion_gt = ref[:t_eff]
        samples.append({
            "name": f"{idx:06d}:{name}",
            "motion_pred": pred263[:t_eff],
            "motion_gt": motion_gt,
            "text_list": tl, "length": t_eff,
        })
    print(f"[263] samples = {len(samples)}", flush=True)

    ev = HumanML263Evaluator(ckpt_dir=args.ckpt_dir)
    ev._ensure_loaded()
    drop_last = len(samples) >= 32
    res = ev.evaluate(samples, mode="pred", n_repeats=args.n_repeats,
                      caption_selection=args.caption_selection,
                      drop_last=drop_last)

    out = {
        "mode": "pred_vs_ms272_gt" if args.with_fid else "pred_only_self_gt",
        "with_fid": args.with_fid,
        "n_samples": res["n_samples"],
        "n_repeats": res["n_repeats"],
        "caption_selection": args.caption_selection,
        "drop_last": drop_last,
        "fid": res["fid"] if args.with_fid else None,
        "r_precision": res["r_precision"],
        "matching_score": res["matching_score"],
        "diversity": res["diversity"],
        "r_precision_real": res["r_precision_real"],
        "matching_score_real": res["matching_score_real"],
        "diversity_real": res["diversity_real"],
        # Backward-compatible aliases used by older handoff notes.
        "r_precision_vald": res["r_precision"]["mean"],
        "r_precision_vald_std": res["r_precision"]["std"],
        "matching_score_vald": res["matching_score"]["mean"],
        "diversity_vald": res["diversity"]["mean"],
        "note": (
            "Full HumanML263 cross-eval from indexed MS272 pairs."
            if args.with_fid
            else "Pred-only sanity metrics on 272->263 converted preds; pass --with_fid for FID."
        ),
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[+] wrote {args.out_json}")


if __name__ == "__main__":
    main()

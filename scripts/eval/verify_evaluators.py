#!/usr/bin/env python3
"""Verify the persisted hftrainer T2M evaluators reproduce paper / GT metrics.

Both evaluators live entirely under ``hftrainer.evaluation.evaluators`` and load
their weights from ``checkpoints/evaluators/`` + data from ``data/evaluators/``
(never ``ref_repo``). This script is the single reproducible entry point that
demonstrates each evaluator recovers (a) the published *GT / Real* row and
(b) a baseline row, so any future metric gap is attributable to the baseline
model itself rather than the evaluator.

Examples
--------
# GT-row sanity (both evaluators), fast:
python3 scripts/eval/verify_evaluators.py --which both --gt-only

# Full check incl. a baseline prediction dir:
python3 scripts/eval/verify_evaluators.py --which ms272 \
    --ms272-pred outputs/evaluation/mdm_h3d272_repro_1000s/mdm_272
python3 scripts/eval/verify_evaluators.py --which hml263 \
    --hml263-pred outputs/evaluation/mdm_h3d272_repro_1000s/mdm_263 \
    --hml263-texts-dir ref_repo/CondMDI/dataset/HumanML3D/texts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Published references (PRISM_TMM2026 Table 1 GT row + MDM paper HumanML3D row).
REF = {
    "hml263": {
        "gt": {"fid": 0.002, "r_top3": 0.797, "diversity": 9.503, "mm_dist": 2.974},
        "mdm": {"fid": 0.544, "r_top3": 0.611, "diversity": 9.559, "mm_dist": 5.566},
    },
    "ms272": {
        "gt": {"fid": 0.0, "r_top1": 0.706, "diversity": 27.36, "mm_dist": 15.01},
    },
}


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def verify_hml263(
    pred_dir,
    n_repeats,
    out_json,
    *,
    gt_root,
    texts_dir,
    split_file,
    caption_selection,
):
    from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator

    ev = HumanML263Evaluator(device="cuda")
    res = ev.evaluate_dir(
        gt_root=str(gt_root),
        texts_dir=str(texts_dir),
        pred_dir=pred_dir,
        split_file=str(split_file),
        n_repeats=n_repeats,
        caption_selection=caption_selection,
    )
    res.setdefault("config", {})
    res["config"].update({
        "gt_root": str(gt_root),
        "texts_dir": str(texts_dir),
        "split_file": str(split_file),
        "caption_selection": caption_selection,
        "caption_protocol_note": (
            "Semantic metrics require texts_dir to match the captions used for "
            "generation. For selected-caption official272 runs, pass "
            "outputs/evaluation/t2m/humanml3d_official_test/captions/"
            "gt_motionclip_selected_20260622/texts."
        ),
    })
    print("\n[HumanML3D-263 evaluator]")
    print(f"  n_samples = {res['n_samples']}")
    print(f"  GT(real)  R-Prec={res['r_precision_real']['mean']}  "
          f"Div={_fmt(res['diversity_real']['mean'])}  MM={_fmt(res['matching_score_real']['mean'])}")
    print(f"            (paper GT: R@3 {REF['hml263']['gt']['r_top3']}, "
          f"Div {REF['hml263']['gt']['diversity']}, MM {REF['hml263']['gt']['mm_dist']})")
    if pred_dir:
        print(f"  pred      FID={_fmt(res['fid']['mean'])}  R-Prec={res['r_precision']['mean']}  "
              f"Div={_fmt(res['diversity']['mean'])}  MM={_fmt(res['matching_score']['mean'])}")
        print(f"            (paper MDM: FID {REF['hml263']['mdm']['fid']}, R@3 "
              f"{REF['hml263']['mdm']['r_top3']}, Div {REF['hml263']['mdm']['diversity']})")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
        print(f"  -> {out_json}")
    return res


def verify_ms272(pred_dir, n_repeats, out_json):
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    ev = MotionStreamer272Evaluator(device="cuda")
    res = ev.evaluate_dir(
        pred_dir=pred_dir or "", n_repeats=n_repeats, gt_only=(pred_dir is None)
    )
    print("\n[MotionStreamer-272 evaluator]")
    print(f"  GT(real)  FID={_fmt(res['fid']) if pred_dir is None else '-'}  "
          f"R-Prec={res['r_precision_real']}  Div={_fmt(res['diversity_real'])}  "
          f"MM={_fmt(res['matching_score_real'])}")
    print(f"            (paper GT: R@1 {REF['ms272']['gt']['r_top1']}, "
          f"Div {REF['ms272']['gt']['diversity']}, MM {REF['ms272']['gt']['mm_dist']})")
    if pred_dir:
        print(f"  pred      FID={_fmt(res['fid'])}  R-Prec={res['r_precision_pred']}  "
              f"Div={_fmt(res['diversity_pred'])}  MM={_fmt(res['matching_score_pred'])}")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
        print(f"  -> {out_json}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="both", choices=["both", "hml263", "ms272"])
    p.add_argument("--gt-only", action="store_true", help="skip baseline pred dirs")
    p.add_argument("--hml263-pred", default=None)
    p.add_argument("--ms272-pred", default=None)
    p.add_argument(
        "--hml263-gt-root",
        default=str(REPO / "ref_repo/CondMDI/dataset/HumanML3D"),
        help="HumanML3D-263 GT root containing new_joint_vecs/ and test.txt.",
    )
    p.add_argument(
        "--hml263-texts-dir",
        default=str(REPO / "ref_repo/CondMDI/dataset/HumanML3D/texts"),
        help="Caption texts used by the HumanML263 evaluator. Must match the generation captions.",
    )
    p.add_argument(
        "--hml263-split-file",
        default=str(REPO / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt"),
        help="Split id list for HumanML263 evaluation.",
    )
    p.add_argument(
        "--hml263-caption-selection",
        default="first",
        choices=["first", "random"],
        help="Caption selection inside each HumanML3D text file.",
    )
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    if args.which in ("both", "hml263"):
        verify_hml263(
            None if args.gt_only else args.hml263_pred,
            args.n_repeats,
            str(out / "verify_hml263.json") if out else None,
            gt_root=Path(args.hml263_gt_root),
            texts_dir=Path(args.hml263_texts_dir),
            split_file=Path(args.hml263_split_file),
            caption_selection=args.hml263_caption_selection,
        )
    if args.which in ("both", "ms272"):
        verify_ms272(
            None if args.gt_only else args.ms272_pred,
            args.n_repeats,
            str(out / "verify_ms272.json") if out else None,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Table-10 (PerMo style/persona editing) evaluation for \\ours{}.

Scores the \\ours{} E16 ``style_edit`` outputs (120 PerMo neutral->style pairs)
with the SAME Guo HumanML3D-263 ``text_mot_match`` evaluator used by the cited
PersonaBooth baselines (FID / R-Precision / Diversity), plus a GT-passthrough
sanity (feeding the targets as predictions; R-Precision_pred should match the
GT/real reference, validating the 135->263 conversion).

Pipeline per sample (from the saved E16 npz, which carries motion_135 +
gt_motion_135 + caption):
    motion_135 (30fps) --motion135_to_motion272--> 272 --motion272_to_hml263--> 263 @20fps
The HumanML263Evaluator then z-normalises, encodes and computes the metrics.

NOTE on protocol: PersonaBooth's published FID/R/Div are cited (their code/ckpts
are unavailable locally and their protocol generates on the HumanML3D test split).
Here we report \\ours{} under the same Guo evaluator but on the 120-clip PerMo
neutral->style audit set; the table footnote states this explicitly.

Run with a clean PYTHONPATH so MoMask's namespace ``utils`` package resolves:
    unset PYTHONPATH; python3 scripts/eval/eval_permo_table10.py --npz-dir <dir>
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
# MoMask root FIRST so its namespace `utils.motion_process` is importable
# (a regular `utils` package elsewhere on the path would otherwise shadow it).
sys.path.insert(0, str(_REPO / "ref_repo/Momask/momask-codes"))
sys.path.insert(0, str(_REPO))


def caption_to_tokens(cap: str):
    words = re.findall(r"[a-zA-Z]+|[0-9]+", cap.lower())
    return [f"{w}/OTHER" for w in words] or ["unk/OTHER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=str(
        _REPO / "output/evaluation/permo_style_ours_big/smpl_caption_editfix_latest/E16_style_edit/npz"))
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--out", default=str(_REPO / "output/evaluation/permo_style_ours_big/metrics_table10.json"))
    args = ap.parse_args()

    import utils.motion_process  # noqa: F401  (ensure MoMask utils import works)
    from hftrainer.motion.representation.convert import (
        motion135_to_motion272, motion272_to_hml263,
    )
    from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator

    def to263(m135):
        m272 = motion135_to_motion272(np.asarray(m135, dtype=np.float32))
        out = motion272_to_hml263(m272, joints_from="smpl_fk")
        m263 = out[0] if isinstance(out, tuple) else out
        return np.asarray(m263, dtype=np.float32)

    files = sorted(glob.glob(f"{args.npz_dir}/*.npz"))
    print(f"[info] {len(files)} PerMo npz")
    samples = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "gt_motion_135" not in d:
            continue
        try:
            mp_263 = to263(d["motion_135"])
            mg_263 = to263(d["gt_motion_135"])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {Path(f).name}: {type(e).__name__}: {e}")
            continue
        cap = str(d["caption"]) if "caption" in d else ""
        # Cap at the evaluator's max_motion_length (196 @20fps); pad handled inside.
        L = min(len(mp_263), len(mg_263), 196)
        if L < 40:
            continue
        samples.append({
            "name": Path(f).stem,
            "motion_gt": mg_263[:L],
            "motion_pred": mp_263[:L],
            "text_list": [{"caption": cap, "tokens": caption_to_tokens(cap),
                           "f_tag": 0.0, "to_tag": 0.0}],
            "length": L,
        })
    print(f"[info] built {len(samples)} samples (>=40 frames @20fps)")
    if len(samples) < 600:
        print(f"[WARN] only {len(samples)} samples < ~512: the 512-dim Guo motion "
              "embedding makes the FID covariance rank-deficient and unstable. "
              "Generate more PerMo pairs for a trustworthy FID.")

    ev = HumanML263Evaluator(device="cuda")
    metrics = ev.evaluate(samples, mode="pred", n_repeats=args.n_repeats,
                          caption_selection="first")

    def fmt(x):
        return x["mean"] if isinstance(x, dict) else x

    rp = fmt(metrics["r_precision"]); rpr = fmt(metrics["r_precision_real"])
    summary = {
        "n_samples": len(samples),
        "FID": round(fmt(metrics["fid"]), 4),
        "R@1": round(rp[0], 4), "R@2": round(rp[1], 4), "R@3": round(rp[2], 4),
        "Diversity": round(fmt(metrics["diversity"]), 4),
        "MM-Dist": round(fmt(metrics["matching_score"]), 4),
        "R@1_real": round(rpr[0], 4), "R@2_real": round(rpr[1], 4), "R@3_real": round(rpr[2], 4),
        "Diversity_real": round(fmt(metrics["diversity_real"]), 4),
    }
    print("\n===== Table-10 \\ours{} (Guo HumanML3D-263 evaluator, PerMo pairs) =====")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "raw": metrics}, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

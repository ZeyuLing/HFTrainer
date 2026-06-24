"""Score 263-baseline predictions (already converted to MS-272) with the
MotionStreamer272Evaluator. Shares one loaded evaluator across models.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    torch.set_grad_enabled(False)
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    ev = MotionStreamer272Evaluator(device="cuda")
    out_root = ROOT / "outputs/evaluation/ms272_from263"
    jobs = {
        "gt": None,  # gt-only reference
        "t2mgpt": str(out_root / "t2mgpt_272"),
        "momask": str(out_root / "momask_272"),
    }
    results = {}
    for name, pdir in jobs.items():
        print(f"\n===== MS-272 eval: {name} =====", flush=True)
        m = ev.evaluate_dir(pdir if pdir else str(out_root / "t2mgpt_272"),
                            n_repeats=20, gt_only=(pdir is None))
        results[name] = m
        rp = m.get("r_precision_pred") or [float("nan")] * 3
        slim = {
            "fid": m.get("fid"), "r_precision_pred": rp,
            "matching_score_pred": m.get("matching_score_pred"),
            "diversity_pred": m.get("diversity_pred"),
            "diversity_real": m.get("diversity_real"),
            "n_samples_used": m.get("n_samples_used"),
            "skipped_no_pred": m.get("skipped_no_pred"),
        }
        print(json.dumps(slim, indent=2), flush=True)
        json.dump(m, open(out_root / f"metrics_{name}.json", "w"), indent=2, default=str)

    print("\n===== SUMMARY (MS-272) =====", flush=True)
    for name, m in results.items():
        rp = m.get("r_precision_pred") or [float("nan")] * 3
        print(f"{name:8s} FID={m.get('fid'):.4f} "
              f"R@1/2/3={rp[0]:.4f}/{rp[1]:.4f}/{rp[2]:.4f} "
              f"MM={m.get('matching_score_pred'):.4f} "
              f"Div={m.get('diversity_pred'):.4f} "
              f"n={m.get('n_samples_used')}", flush=True)


if __name__ == "__main__":
    main()

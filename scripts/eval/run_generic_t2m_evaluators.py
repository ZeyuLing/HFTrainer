#!/usr/bin/env python3
"""Run bridge/generic T2M evaluators over the current HumanML3D baselines.

The script is deliberately status-preserving: missing weights, unsupported
representations, and per-method failures are written as JSON records instead of
aborting the whole batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

DEFAULT_MANIFEST = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/viewer_methods_all_motion135.json"
)
DEFAULT_OUT = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/generic_evaluators_20260621"
)
HML_ROOT = REPO / "outputs/evaluation/t2m/humanml3d_official_test/hml263"


HML263_CANDIDATES: Dict[str, list[str]] = {
    "GT": ["gt_official_test_from_motion135/pred_hml263"],
    "PRISM": [
        "prism_epoch31_smooth_exactlen_0617_vermo/pred_hml263",
        "prism_epoch31_smooth_20260617/pred_hml263",
    ],
    "HYMotion": ["hymotion_1b_exactlen_0617_vermo/pred_hml263"],
    "MotionStreamer": ["motionstreamer_exactlen_0617_vermo/pred_hml263"],
    "FlowMDM": ["flowmdm_official/predictions/hml263"],
    "MotionLab": ["motionlab_official/predictions/hml263"],
    "MDM": ["mdm_official/predictions/hml263", "mdm_official/mdm_263"],
    "MLD": [
        "mld_standard_pipeline_20260621/predictions/hml263",
        "mld_official/predictions/hml263",
    ],
    "T2M-GPT": ["t2mgpt_official/predictions/hml263"],
    "MoMask": ["momask_official/predictions/hml263", "momask_official/momask_263"],
    "MotionGPT3": ["motiongpt3_official/predictions/hml263"],
    "MoGenTS": ["mogents_ts10_cfg4_rescfg5_seed0"],
    "KIMODO": ["kimodo_official_from_motion135/pred_hml263"],
    "GoToZero": ["gotozero_official_from_motion135/pred_hml263"],
}


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "_")
        .replace(".", "_")
    )


def _json_safe(obj):
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _count_files(path: Optional[Path], suffix: str) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for _ in path.glob(f"*{suffix}"))


def _first_with_files(base: Path, rels: Iterable[str], suffix: str) -> Optional[Path]:
    for rel in rels:
        p = base / rel
        if _count_files(p, suffix) > 0:
            return p
    return None


def load_methods(manifest: Path) -> list[dict]:
    data = json.load(open(manifest))
    methods = data.get("methods", data if isinstance(data, list) else [])
    out = []
    for m in methods:
        label = m["label"]
        motion_dir = (REPO / m["dir"]).resolve() if not Path(m["dir"]).is_absolute() else Path(m["dir"])
        hml_dir = _first_with_files(HML_ROOT, HML263_CANDIDATES.get(label, []), ".npy")
        out.append({"label": label, "motion135_dir": motion_dir, "hml263_dir": hml_dir})
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_safe))


def run_tmr(method: dict, args) -> dict:
    if method["hml263_dir"] is None:
        return {
            "status": "unsupported_representation",
            "reason": "no hml263 prediction directory registered for this method",
        }
    from hftrainer.evaluation.evaluators.tmr_humanml3d import (
        MissingEvaluatorAssets,
        TMRHumanML3DEvaluator,
    )

    try:
        ev = TMRHumanML3DEvaluator(
            model_dir=args.tmr_model_dir,
            tmr_root=args.tmr_root,
            device=args.device,
            batch_size=args.batch_size,
        )
        return ev.evaluate_dir(str(method["hml263_dir"]), max_samples=args.max_samples or None)
    except MissingEvaluatorAssets as exc:
        return {"status": "missing_weights", "error": str(exc), "asset_help": TMRHumanML3DEvaluator.asset_help()}


def flatten_for_csv(label: str, evaluator: str, res: dict) -> dict:
    row = {
        "method": label,
        "evaluator": evaluator,
        "status": res.get("status", "unknown"),
        "n_samples": res.get("n_samples", ""),
    }
    for key in [
        "fid_latent",
        "diversity_pred",
        "score_mean",
        "score_std",
        "score_median",
        "score_p05",
        "score_p95",
        "skipped",
    ]:
        if key in res:
            row[key] = res[key]
    skipped_reasons = res.get("skipped_reasons")
    if isinstance(skipped_reasons, dict):
        row["skipped_reasons"] = json.dumps(skipped_reasons, sort_keys=True)
    pred = res.get("retrieval_pred")
    if isinstance(pred, dict):
        for k in ["t2m/R01", "t2m/R02", "t2m/R03", "t2m/R05", "t2m/R10", "t2m/MedR"]:
            if k in pred:
                row[f"tmr_{k.replace('/', '_')}"] = pred[k]
    if "error" in res:
        row["error"] = res["error"][:300]
    if "reason" in res:
        row["error"] = res["reason"]
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--evaluators", default="tmr")
    p.add_argument("--methods", default="", help="comma-separated labels; default all manifest methods")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--tmr-root", default=str(REPO / "ref_repo/TMR"))
    p.add_argument("--tmr-model-dir", default=None)
    p.add_argument("--io-workers", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_evaluators = [e.strip() for e in args.evaluators.split(",") if e.strip()]
    wanted_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    methods = load_methods(Path(args.manifest))
    if wanted_methods:
        methods = [m for m in methods if m["label"] in wanted_methods]

    rows = []
    all_results = {}
    for method in methods:
        label = method["label"]
        all_results[label] = {
            "paths": {
                "motion135_dir": str(method["motion135_dir"]),
                "hml263_dir": str(method["hml263_dir"]) if method["hml263_dir"] else None,
                "motion135_count": _count_files(method["motion135_dir"], ".npz"),
                "hml263_count": _count_files(method["hml263_dir"], ".npy"),
            },
            "evaluators": {},
        }
        for evaluator in wanted_evaluators:
            try:
                if evaluator == "tmr":
                    res = run_tmr(method, args)
                else:
                    res = {"status": "unknown_evaluator", "error": evaluator}
            except Exception as exc:
                res = {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=20),
                }
            res["method"] = label
            res["paths"] = all_results[label]["paths"]
            all_results[label]["evaluators"][evaluator] = res
            write_json(out_dir / evaluator / f"{_slug(label)}.json", res)
            rows.append(flatten_for_csv(label, evaluator, res))
            print(f"[{evaluator}] {label}: {res.get('status')} n={res.get('n_samples', '')}")

    write_json(out_dir / "summary.json", all_results)
    csv_path = out_dir / "summary.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {out_dir}")
    print(f"[done] {csv_path}")


if __name__ == "__main__":
    main()

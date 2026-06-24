#!/usr/bin/env python3
"""Build a 4-column PhysFlow overfit comparison manifest for /physflow_quad.

Columns per case:
  1. base T2M reference        (un-optimized KIMODO-G1 generated motion)
  2. base -> frozen tracker     (MuJoCo rollout of column 1)
  3. optimized T2M reference   (PhysFlow a1 checkpoint generated motion)
  4. optimized -> frozen tracker(MuJoCo rollout of column 3)

Inputs are the two single-arm eval runs produced by physflow_coevolve_viz.py
(via tools/physflow_eval_trainset.sh):
  <eval>/base/...            + <eval>/manifest_base/robot_frames_reference/*.json
  <eval>/opt_iter1000/...    + <eval>/manifest_opt_iter1000/robot_frames_reference/*.json
"""
import argparse, json, os
from pathlib import Path


def load_summary(p):
    d = json.load(open(p))
    recs = d if isinstance(d, list) else (d.get("records") or list(d.values()))
    return {r["output_stem"]: r for r in recs if isinstance(r, dict) and "output_stem" in r}


def kin(r, k):
    return (r.get("kinematic") or {}).get(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="work_dirs/physflow_overfit_eval")
    ap.add_argument("--base-arm", default="base")
    ap.add_argument("--opt-arm", default="opt_iter1000")
    ap.add_argument("--out", default="work_dirs/physflow_overfit_eval/quad_manifest.json")
    args = ap.parse_args()

    root = Path(args.eval_dir).resolve()
    base = load_summary(root / args.base_arm / "summary.json")
    opt = load_summary(root / args.opt_arm / "summary.json")
    base_ref = root / f"manifest_{args.base_arm}" / "robot_frames_reference"
    opt_ref = root / f"manifest_{args.opt_arm}" / "robot_frames_reference"

    stems = sorted(set(base) & set(opt))
    rows = []
    for i, stem in enumerate(stems):
        b, o = base[stem], opt[stem]
        col = lambda title, p: {"title": title, "path": str(p) if p and Path(p).is_file() else ""}
        rows.append({
            "case": i,
            "prompt_id": b.get("prompt_id", stem),
            "prompt": b.get("prompt", ""),
            "columns": [
                col("Base T2M", base_ref / f"{stem}.raw_reference.json"),
                col("Base \u2192 Tracker", root / args.base_arm / "json" / f"{stem}.json"),
                col("Optimized T2M", opt_ref / f"{stem}.raw_reference.json"),
                col("Optimized \u2192 Tracker", root / args.opt_arm / "json" / f"{stem}.json"),
            ],
            "metrics": {
                "base_adv": b.get("adversarial_score"), "opt_adv": o.get("adversarial_score"),
                "d_adv": (o.get("adversarial_score") - b.get("adversarial_score"))
                          if b.get("adversarial_score") is not None and o.get("adversarial_score") is not None else None,
                "base_skate": kin(b, "foot_skate_speed"), "opt_skate": kin(o, "foot_skate_speed"),
                "d_skate": (kin(o, "foot_skate_speed") - kin(b, "foot_skate_speed"))
                            if kin(b, "foot_skate_speed") is not None and kin(o, "foot_skate_speed") is not None else None,
                "base_root": b.get("root_trajectory_error_mean_m"), "opt_root": o.get("root_trajectory_error_mean_m"),
                "d_root": (o.get("root_trajectory_error_mean_m") - b.get("root_trajectory_error_mean_m"))
                           if b.get("root_trajectory_error_mean_m") is not None and o.get("root_trajectory_error_mean_m") is not None else None,
                "base_fall": bool(b.get("fall_detected")), "opt_fall": bool(o.get("fall_detected")),
            },
        })

    out = {"title": f"PhysFlow overfit 4-column (base vs {args.opt_arm})", "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    n_full = sum(1 for r in rows if all(c["path"] for c in r["columns"]))
    print(f"wrote {args.out}: {len(rows)} cases ({n_full} with all 4 columns present)")


if __name__ == "__main__":
    main()

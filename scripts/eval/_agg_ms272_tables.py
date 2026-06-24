#!/usr/bin/env python3
"""Aggregate per-method MotionStreamer-272 eval JSONs into a compact table."""
import argparse
import json
from pathlib import Path


def g(o, k, i=None, nd=4):
    v = o.get(k)
    if i is not None and isinstance(v, list):
        v = v[i] if len(v) > i else None
    if isinstance(v, (int, float)):
        return round(v, nd)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = Path(args.res_dir)
    rows = {}
    gt_ref = None
    for p in sorted(res.glob("*.json")):
        if p.stem.startswith(("phys", "physical", "physics")):
            continue
        d = json.load(open(p))
        tag = d.get("tag", p.stem)
        if str(tag).startswith(("phys", "physical", "physics")):
            continue
        pred = d.get("pred") or {}
        gr = d.get("gt_real") or {}
        rows[tag] = {
            "samples": d.get("ids_with_required_files"),
            "nb_pred": g(pred, "nb"),
            "R1": g(pred, "r_precision", 0),
            "R2": g(pred, "r_precision", 1),
            "R3": g(pred, "r_precision", 2),
            "FID_native": g(pred, "fid_vs_gt_native"),
            "FID_refk": g(pred, "fid_vs_gt_refk"),
            "MM": g(pred, "matching_score"),
            "Div": g(pred, "diversity"),
            "gt_R3": g(gr, "r_precision", 2),
            "gt_MM": g(gr, "matching_score"),
            "gt_Div": g(gr, "diversity"),
            "gt_selfFID": g(gr, "self_fid_split_halves"),
        }
        # keep a representative GT-real (largest sample count)
        if gr and (gt_ref is None or (gr.get("nb", 0) or 0) > gt_ref[1]):
            gt_ref = ({
                "R1": g(gr, "r_precision", 0), "R3": g(gr, "r_precision", 2),
                "MM": g(gr, "matching_score"), "Div": g(gr, "diversity"),
                "selfFID": g(gr, "self_fid_split_halves"),
                "nb": gr.get("nb"),
            }, gr.get("nb", 0) or 0)

    out = {"gt_real_reference": gt_ref[0] if gt_ref else None, "methods": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))

    hdr = f"{'method':18s} {'n':>5s} {'R1':>7s} {'R3':>7s} {'FIDnat':>9s} {'FIDrefk':>9s} {'MM':>7s} {'Div':>7s}"
    print(hdr)
    print("-" * len(hdr))
    if gt_ref:
        r = gt_ref[0]
        print(f"{'Real(native272)':18s} {str(r['nb']):>5s} {str(r['R1']):>7s} {str(r['R3']):>7s} "
              f"{'~0':>9s} {'-':>9s} {str(r['MM']):>7s} {str(r['Div']):>7s}")
    order = ["real_conv", "mdm", "mld", "momask", "motiongpt3", "t2mgpt", "motiongpt",
             "gotozero",
             "flowmdm", "motionlab", "vimogen", "hymotion", "motionstreamer", "ours",
             "ours_c1", "ours_c5", "ours_c9",
             "flowmdm_c1", "flowmdm_c5", "flowmdm_c9",
             "motionlab_c1", "motionlab_c5", "motionlab_c9",
             "motionstreamer_c1", "motionstreamer_c5", "motionstreamer_c9"]
    seen = set()
    for k in order + [k for k in rows if k not in order]:
        if k not in rows or k in seen:
            continue
        seen.add(k)
        r = rows[k]
        def s(x):
            return "-" if x is None else str(x)
        print(f"{k:18s} {s(r['samples']):>5s} {s(r['R1']):>7s} {s(r['R3']):>7s} "
              f"{s(r['FID_native']):>9s} {s(r['FID_refk']):>9s} {s(r['MM']):>7s} {s(r['Div']):>7s}")


if __name__ == "__main__":
    main()

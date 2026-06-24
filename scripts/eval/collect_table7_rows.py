#!/usr/bin/env python3
"""Collect Table 7 (trajectory) rows from the _metrics/{tag}__{fid,new,ric}.json
triplets into one print + JSON. Column mapping (validated against \\ours):
  FID      <- fid.FID
  R@3      <- fid.R@3
  Div      <- fid.Diversity
  Traj.Err <- new.trajectory_err_m
  Foot     <- new.foot_skating_ratio
  Jitter   <- ric.jitter_mean
"""
import glob
import json
import os
import sys

MD = sys.argv[1] if len(sys.argv) > 1 else \
    "output/evaluation/table7_traj/_metrics"


def _first(d):
    if isinstance(d, dict) and len(d) == 1 and isinstance(next(iter(d.values())), dict):
        return next(iter(d.values()))
    return d


def load(tag, kind):
    p = os.path.join(MD, f"{tag}__{kind}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


tags = sorted({os.path.basename(f).rsplit("__", 1)[0]
               for f in glob.glob(os.path.join(MD, "*__*.json"))})
rows = {}
hdr = f"{'tag':28} {'n':>5} {'FID':>8} {'R@3':>6} {'Div':>6} {'TrajErr':>8} {'Foot':>6} {'Jitter':>7}"
print(hdr)
print("-" * len(hdr))
for tag in tags:
    fid = load(tag, "fid")
    new = load(tag, "new")
    ric = load(tag, "ric")
    fid = _first(fid) if fid else {}
    new = _first(new) if new else {}
    ric = ric or {}
    row = {
        "n_fid": fid.get("n_records"),
        "n_new": new.get("n"),
        "n_ric": ric.get("n"),
        "FID": fid.get("FID"),
        "R@3": fid.get("R@3"),
        "Div": fid.get("Diversity"),
        "TrajErr_m": new.get("trajectory_err_m"),
        "Foot": new.get("foot_skating_ratio"),
        "Jitter": ric.get("jitter_mean"),
    }
    rows[tag] = row

    def f(x, p=2):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  -  "
    print(f"{tag:28} {str(row['n_fid'] or row['n_new'] or row['n_ric']):>5} "
          f"{f(row['FID']):>8} {f(row['R@3'],3):>6} {f(row['Div']):>6} "
          f"{f(row['TrajErr_m'],4):>8} {f(row['Foot'],3):>6} {f(row['Jitter'],1):>7}")

out = os.path.join(MD, "_table7_rows.json")
json.dump(rows, open(out, "w"), indent=2)
print(f"\n-> {out}")

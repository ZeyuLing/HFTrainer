#!/usr/bin/env python3
"""Build splits for the DATA-ENGINE benefit experiment on the 29-DOF ProtoMotions
G1 stack.

Goal: show that augmenting a *data-limited* tracker's training with T2M-generator
motions improves tracking on a STANDARD held-out AMASS-G1 benchmark, vs a no-
generator baseline (and approaching a real-broad-data upper bound).

Splits (all real AMASS-derived G1 clips from the "Academic" source = standard AMASS
retargeted to 29-DOF G1; all mutually disjoint and disjoint from the agile held-out):
  - H_bench      : broad standard held-out eval set (representative, NOT agility-biased)
  - B_narrow     : narrow-domain base set (low-agility locomotion/stand) -> specialize
                   the SOTA into a data-limited tracker with head-room on the broad H
  - real_broad   : broad real motions (upper-bound control: B_narrow + real_broad)

The generator-augmentation pool is reused from the co-evolution run (already broad,
quality-filtered generated motions). Outputs annotation JSONs; .motion / CSV
materialization is done by the node script (py3.8 converter).
"""
import argparse, json, os, random
import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
G1_ROOT = os.path.join(REPO, "data/g1")
import sys
sys.path.insert(0, REPO)
from scripts.embodied.build_heldout_agile import agility  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json")
    ap.add_argument("--agile", default="data/annotation/_heldout_agile_ground_only.json")
    ap.add_argument("--scan", type=int, default=12000)
    ap.add_argument("--n-hbench", type=int, default=120)
    ap.add_argument("--n-narrow", type=int, default=300)
    ap.add_argument("--n-realbroad", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--out", default="data/annotation")
    args = ap.parse_args()

    items = json.load(open(os.path.join(REPO, args.anno)))["items"]
    agile_excl = {it["g1_path"] for it in json.load(open(os.path.join(REPO, args.agile)))}
    # standard AMASS-derived source only
    pool = [it for it in items if it["g1_path"].split("/")[0] == "Academic"
            and it["g1_path"] not in agile_excl]
    random.seed(args.seed)
    random.shuffle(pool)

    scored = []
    for it in pool[: args.scan]:
        p = os.path.join(G1_ROOT, it["g1_path"])
        if not os.path.exists(p):
            continue
        a = agility(p)
        if a is None:
            continue
        scored.append({**it, **a})
    print(f"[splits] scored {len(scored)} valid Academic clips", flush=True)

    # sort by agility: low = narrow locomotion/stand; build narrow from the low tail
    scored.sort(key=lambda r: r["agility"])
    n = len(scored)
    lo_third = scored[: n // 3]            # low-agility (walk/stand) candidates
    # H_bench: representative broad sample across the WHOLE agility range (standard)
    step = max(1, n // args.n_hbench)
    hbench = scored[::step][: args.n_hbench]
    hbench_ids = {r["g1_path"] for r in hbench}

    narrow = [r for r in lo_third if r["g1_path"] not in hbench_ids][: args.n_narrow]
    narrow_ids = {r["g1_path"] for r in narrow}

    realbroad = [r for r in scored
                 if r["g1_path"] not in hbench_ids and r["g1_path"] not in narrow_ids]
    random.shuffle(realbroad)
    realbroad = realbroad[: args.n_realbroad]

    def dump(name, rows):
        outp = os.path.join(REPO, args.out, name)
        json.dump(rows, open(outp, "w"), indent=1)
        ag = [r["agility"] for r in rows]
        print(f"[splits] {name}: {len(rows)} clips | agility {min(ag):.2f}..{max(ag):.2f} "
              f"mean={np.mean(ag):.2f}", flush=True)

    dump("_de_hbench.json", hbench)
    dump("_de_narrow.json", narrow)
    dump("_de_realbroad.json", realbroad)
    print("[splits] done. H_bench=eval, narrow=specialize base, realbroad=upper-bound ctrl")


if __name__ == "__main__":
    main()

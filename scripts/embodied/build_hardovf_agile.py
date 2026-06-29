#!/usr/bin/env python3
"""Build an AGILE hard-overfit prompt set for validating the direction-B frontier
mechanism. Unlike the easy `_coevo_overfit8.json` (which the SOTA-warm-started
trainee already tracks perfectly -> no frontier), these are the MOST AGILE real
clips, which the released G1 tracker is most likely to *drop* -- i.e. a genuine
learnability frontier exists, so we can watch n_frontier_mean>0 and the trainee's
completion on these clips rise round-by-round.

Selection: reuse build_heldout_agile.agility() with the SAME scan seed, EXCLUDE
the 80 held-out clips (truly disjoint from the key-capability eval), and take the
next top-K most agile. Output uses the dataloader's {meta_info, items} schema.
"""
import argparse, json, os, random
from build_heldout_agile import agility, REPO, G1_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json")
    ap.add_argument("--heldout", default="data/annotation/_heldout_agile_ground_only.json")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--out", default="data/annotation/_coevo_hardovf_agile.json")
    ap.add_argument("--seed", type=int, default=20260612)  # same as held-out scan
    args = ap.parse_args()

    heldout_paths = {it["g1_path"] for it in json.load(open(os.path.join(REPO, args.heldout)))}
    items = json.load(open(os.path.join(REPO, args.anno)))["items"]
    random.seed(args.seed)
    idx = random.sample(range(len(items)), min(args.sample, len(items)))
    scored = []
    for n, i in enumerate(idx):
        it = items[i]
        if it["g1_path"] in heldout_paths:
            continue
        p = os.path.join(G1_ROOT, it["g1_path"])
        if not os.path.exists(p):
            continue
        a = agility(p)
        if a is None:
            continue
        scored.append({**it, **a})
        if (n + 1) % 500 == 0:
            print(f"  scanned {n+1}/{len(idx)} | kept {len(scored)}", flush=True)
    scored.sort(key=lambda r: r["agility"], reverse=True)
    top = scored[: args.topk]
    # keep only the dataloader fields in items; stash agility in meta for reference
    keep = ("g1_path", "caption_rel", "emb_rel")
    out = {
        "meta_info": {
            "dataset": "g1_t2m co-evo AGILE hard-overfit (frontier validation)",
            "n": len(top),
            "src": args.anno,
            "excluded_heldout": len(heldout_paths),
            "agility_range": [top[-1]["agility"], top[0]["agility"]] if top else None,
        },
        "items": [{k: r[k] for k in keep if k in r} for r in top],
    }
    outp = os.path.join(REPO, args.out)
    json.dump(out, open(outp, "w"), indent=1)
    print(f"\n[hardovf] scanned {len(idx)} | valid {len(scored)} | wrote top-{len(top)} -> {outp}")
    if top:
        print("[hardovf] agility range: %.3f .. %.3f" % (top[-1]["agility"], top[0]["agility"]))
        for r in top[:8]:
            print(f"   ag={r['agility']:.2f} jv={r['joint_speed']:.1f} dz={r['z_excursion']:.2f} "
                  f"rv={r['root_speed']:.2f} T={r['num_frames']} | {os.path.basename(r['g1_path'])[:56]}")


if __name__ == "__main__":
    main()
